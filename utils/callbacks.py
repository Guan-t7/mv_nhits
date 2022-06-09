from functools import partial
from typing import Optional
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics import MeanAbsoluteError, MeanSquaredError, MeanAbsolutePercentageError, SymmetricMeanAbsolutePercentageError


class LogMetricsCallback(Callback):
    """Log requested regression metrics for predictor evaluation
    """
    cls_dict = {
        'mae': MeanAbsoluteError,
        'mse': MeanSquaredError,
        'mape': MeanAbsolutePercentageError,
        'smape': SymmetricMeanAbsolutePercentageError,
        'rmse': partial(MeanSquaredError, squared=False),
    }

    def __init__(self, metrics=['mae', 'mse'], scaled=True, free_mem=False) -> None:
        '''
        @scaled: whether to compute metrics from normalized data.
        @free_mem: remove pred results from batch outputs. Cannot store them in big datasets.
        '''
        self.metrics = nn.ModuleDict()
        for name in metrics:
            self.metrics[name] = self.cls_dict[name]()
        self.rm = free_mem
        self.prefix = "scaled_" if scaled else ""

    def on_validation_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.metrics.to(pl_module.device)
    
    def on_validation_batch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", 
                                outputs: dict, batch, batch_idx: int, dataloader_idx: int) -> None:
        outY, tgtY = outputs[self.prefix+'pred'], outputs[self.prefix+'tgt']
        for name, metric in self.metrics.items():
            metric.update(outY, tgtY)
        if self.rm:
            outputs['pred'] = None; outputs['tgt'] = None
            outputs['scaled_pred'] = None; outputs['scaled_tgt'] = None
    
    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        for name, metric in self.metrics.items():
            pl_module.log(f"metrics/{name}", metric.compute())
            metric.reset()
