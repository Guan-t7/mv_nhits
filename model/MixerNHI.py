import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, BaseFinetuning, LearningRateMonitor
import nni
import torch as t
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from model.mixer import LKA
from model.nhits import NHITS
from utils.metrics import inv_transform, MAELoss


class GlblPred0(pl.LightningModule):
    def __init__(
        self,
        num_series=370,
        n_time_in=24*7,
        n_time_out=24,
        Xseq_kwargs=dict(),
        lr=1e-3,
        lr_decay=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()  #
        # before nhits
        depth = 3
        self.encoder = LKA(num_series, depth)
        # temporal model
        # Tc -> Tp for each variable
        Xseq = NHITS(**Xseq_kwargs)
        self.Xseq = Xseq.model

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "hparams" in name:
                raise AttributeError()
            return getattr(self.hparams, name)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        (_, inpY), (_, tgtY), _, wnd_start = batch  # use normed data
        num_series, pred_steps = tgtY.shape
        assert num_series == self.num_series
        assert pred_steps == self.n_time_out  # shouldn't be batched
        # RNNs need teacher-forcing input
        assert inpY.shape[-1] == self.n_time_in + (pred_steps-1)

        Y_in = inpY[:, :self.n_time_in]
        Y_h = self(Y_in, None, None)

        l1loss = nn.L1Loss()
        loss = l1loss(Y_h, tgtY)
        self.log("loss/train", loss,
                 on_step=True, on_epoch=False, batch_size=1)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.hparams.lr)

        min_lr = self.lr * (self.lr_decay ** 3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', self.lr_decay, patience=1, cooldown=1, min_lr=min_lr)

        return {
            'optimizer': optimizer, 'lr_scheduler': {
                "scheduler": lr_scheduler,
                "monitor": "loss/val", }
        }

    def forward(self, inpY, wnd_start, pred_steps):
        num_series, cond_steps = inpY.shape
        assert num_series == self.num_series
        assert cond_steps == self.n_time_in

        Y_mixed = self.encoder(inpY)
        Y_h = self.Xseq(None, Y_mixed)

        return Y_h

    def validation_step(self, batch, batch_idx):
        (_, inpY_n), (tgtY, tgtY_n), scaler, wnd_start = batch
        num_series, pred_steps = tgtY.shape
        assert pred_steps == self.n_time_out  # shouldn't be batched
        assert inpY_n.shape[-1] == self.n_time_in + (pred_steps-1)

        Y_in = inpY_n[:, :self.n_time_in]
        Y_h = self(Y_in, None, None)
        outY = inv_transform(scaler, Y_h)
        assert outY.isnan().any() == False

        l1loss = nn.L1Loss()
        loss = l1loss(Y_h, tgtY_n)
        self.log("loss/val", loss,
                 on_step=False, on_epoch=True, batch_size=1)

        return {
            'loss': loss,
            'pred': outY, 'scaled_pred': Y_h,
            'tgt': tgtY, 'scaled_tgt': tgtY_n
        }

    def validation_epoch_end(self, outputs) -> None:
        losses = []
        for d in outputs:
            losses.append(d['loss'])
        loss = t.stack(losses).mean().item()  # approx
        nni.report_intermediate_result(loss)

    def configure_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            monitor='loss/val', mode='min', save_last=True)
        self.earlystop_callback = EarlyStopping(
            monitor='loss/val', patience=5, mode='min')

        cbs = [self.earlystop_callback, self.checkpoint_callback, ]

        if self.trainer.logger is not None:
            self.loglr_cb = LearningRateMonitor("epoch")
            cbs.append(self.loglr_cb)

        return cbs

    def on_train_epoch_end(self):
        if self.trainer.should_stop:
            hp_metric = self.checkpoint_callback.best_model_score
            self.log('hp_metric', hp_metric)
            final_result = hp_metric.item()
            nni.report_final_result(final_result)
