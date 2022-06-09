import math
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, BaseFinetuning, LearningRateMonitor
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.optimizer import Optimizer

from model.nhits import NHITS
from utils.metrics import *


def basisAttn(basis: torch.Tensor):
    '''(#entities, #features) -> (#entities, #entities)
    '''
    Nt, E = basis.shape
    attn = basis @ basis.T
    attn = attn / math.sqrt(E)
    attn = F.softmax(attn, dim=-1)
    return attn


class TLAE(pl.LightningModule):
    def __init__(
        self,
        num_series=370,
        n_time_in=24*7,
        n_time_out=24,
        hbsize=2*24*7,
        enc_channels=[64, 16],
        Xseq_kwargs=dict(),
        reg=0.5,
        lr=0.0001,
        lr_decay=0.5,
    ):
        super().__init__()
        self.save_hyperparameters()  #
        # enc
        num_layers = len(enc_channels)
        assert num_layers > 1
        enc_channels = [num_series] + enc_channels
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(enc_channels[i], enc_channels[i+1]))
            if i != num_layers-1:  # act except last layer
                layers.append(nn.ELU())
        self.encoder = nn.Sequential(*layers)
        # temporal model
        # Tc -> Tp for each variable
        Xseq = NHITS(**Xseq_kwargs)
        self.Xseq = Xseq.model
        # decoder
        dec_channels = enc_channels[::-1]
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(dec_channels[i], dec_channels[i+1]))
            if i != num_layers-1:  # act except last layer
                layers.append(nn.ELU())
        self.decoder = nn.Sequential(*layers)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            if "hparams" in name:
                raise AttributeError()
            return getattr(self.hparams, name)

    @property
    def num_latent(self):
        return self.hparams.enc_channels[-1]

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        (_, inpY), (_, tgtY), _, wnd_start = batch  # use normed data
        num_series, pred_steps = tgtY.shape
        assert num_series == self.num_series
        # batched along time dim for balanced recov loss
        assert pred_steps == self.hbsize - self.n_time_in
        assert pred_steps % self.n_time_out == 0  # convenience
        # RNNs need teacher-forcing input
        assert inpY.shape[-1] == self.n_time_in + (pred_steps-1)
        
        Y = torch.cat((inpY, tgtY[:, [-1]]), -1)  # the whole window
        # TLAE alg
        X = self.encoder(Y.T).T  # N_latent, L
        # model needs to predict multiple windows for each variable
        n_slices = pred_steps // self.n_time_out
        slices = [X[:, i*self.n_time_out: i*self.n_time_out+self.n_time_in]
                  for i in range(n_slices)]
        Xseq_in = torch.vstack(slices)  # n*N_latent, L_in
        forecasts = self.Xseq(None, Xseq_in)
        slices = torch.vsplit(forecasts, n_slices)
        X_h = torch.cat(slices, -1)  # N_latent, L_out
        # calc recovered Y
        dec_in = torch.cat((X[:, :self.n_time_in], X_h), -1)
        Y_h = self.decoder(dec_in.T).T  # N_input, L

        l1loss = nn.L1Loss()
        recov_loss = l1loss(Y_h, Y)
        l2loss = nn.MSELoss()
        reg_loss = l2loss(X_h, X[:, self.n_time_in:])
        loss = recov_loss + self.hparams.reg * reg_loss
        self.log("loss/train", loss, 
                 on_step=True, on_epoch=False, batch_size=1)
        self.log("loss/train_recov", recov_loss,
                 on_step=True, on_epoch=False, batch_size=1)
        self.log("loss/train_reg", reg_loss,
                 on_step=True, on_epoch=False, batch_size=1)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), self.hparams.lr)

        min_lr = self.lr * (self.lr_decay ** 3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', self.lr_decay, patience=3, cooldown=2, min_lr=min_lr)

        return {
            'optimizer': optimizer, 'lr_scheduler': {
                "scheduler": lr_scheduler,
                "monitor": "loss/val", }
        }

    def forward(self, inpY, wnd_start, pred_steps):
        '''TLAE alg with intermediates for loss compute
        '''
        num_series, cond_steps = inpY.shape
        assert num_series == self.num_series
        assert cond_steps == self.n_time_in

        X = self.encoder(inpY.T).T  # N_latent, L_in
        X_h = self.Xseq(None, X)  # N_latent, L_out
        Y_h = self.decoder(X_h.T).T  # N_input, L_out

        return Y_h, X_h, 

    def validation_step(self, batch, batch_idx):
        (_, inpY_n), (tgtY, tgtY_n), scaler, wnd_start = batch
        num_series, pred_steps = tgtY.shape
        assert pred_steps == self.n_time_out  # shouldn't be batched
        assert inpY_n.shape[-1] == self.n_time_in + (pred_steps-1)

        Y = torch.cat((inpY_n, tgtY_n[:, [-1]]), -1)  # the whole window
        # get prediction
        Y_h1, X_h = self(Y[..., :self.n_time_in], wnd_start, pred_steps)
        # for recov loss
        X0 = self.encoder(Y[..., :self.n_time_in].T)
        Y_h0 = self.decoder(X0).T
        Y_h = torch.cat((Y_h0, Y_h1), -1)
        # for reg loss
        X1 = self.encoder(Y[..., self.n_time_in:].T).T
        # for metrics
        outY = inv_transform(scaler, Y_h1)
        assert outY.isnan().any() == False

        l1loss = nn.L1Loss()
        recov_loss = l1loss(Y_h, Y)
        l2loss = nn.MSELoss()
        reg_loss = l2loss(X_h, X1)
        loss = recov_loss + self.hparams.reg * reg_loss
        assert loss < 1e3
        self.log("loss/val", loss,
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("loss/val_recov", recov_loss,
                 on_step=False, on_epoch=True, batch_size=1)
        self.log("loss/val_reg", reg_loss,
                 on_step=False, on_epoch=True, batch_size=1)
        """@nni.report_intermediate_result(...)"""

        return loss, outY, tgtY

    def validation_epoch_end(self, outputs) -> None:
        outYs = []
        tgtYs = []
        for loss, outY, tgtY in outputs:
            outYs.append(outY)
            tgtYs.append(tgtY)
        outY = torch.cat(outYs, -1)
        tgtY = torch.cat(tgtYs, -1)
        wape, mape, smape = xape3(outY, tgtY)
        self.log("metrics/wape", wape, batch_size=1)
        self.log("metrics/mape", mape, batch_size=1)
        self.log("metrics/smape", smape, batch_size=1)

    def configure_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            monitor='loss/val', mode='min', save_last=True)
        self.earlystop_callback = EarlyStopping(
            monitor='loss/val', patience=10, mode='min')
        self.loglr_cb = LearningRateMonitor("epoch")

        return [self.loglr_cb, self.earlystop_callback, self.checkpoint_callback]  #

    def on_train_epoch_end(self):
        if self.trainer.should_stop:
            hp_metric = self.checkpoint_callback.best_model_score
            self.log('hp_metric', hp_metric)
            final_result = hp_metric.item()
            """@nni.report_final_result(final_result)"""
