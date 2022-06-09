# Cell
import math
import random
import numpy as np

import torch as t
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, BaseFinetuning, LearningRateMonitor
import nni

from typing import Tuple
from functools import partial

from utils.metrics import inv_transform, MAELoss
from utils.callbacks import LogMetricsCallback

# TODO statics unused for now
class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [nn.Dropout(p=0.5),
                  nn.Linear(in_features=in_features, out_features=out_features),
                  nn.ReLU()]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class IdentityBasis(nn.Module):
    '''interpolation back to input size'''
    def __init__(self, backcast_size: int, forecast_size: int, interpolation_mode: str):
        super().__init__()
        assert (interpolation_mode in ['linear',])
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size
        self.interpolation_mode = interpolation_mode

    def forward(self, theta: t.Tensor, ) -> Tuple[t.Tensor, t.Tensor]:
        backcast = theta[:, :self.backcast_size]
        knots = theta[:, self.backcast_size:]

        if self.interpolation_mode=='linear':
            knots = knots[:,None,:]
            forecast = F.interpolate(knots, size=self.forecast_size, mode=self.interpolation_mode) #, align_corners=True)
            forecast = forecast[:,0,:]

        return backcast, forecast


ACTIVATIONS = ['ReLU',
               'Softplus',
               'Tanh',
               'SELU',
               'LeakyReLU',
               'PReLU',
               'Sigmoid']


class _NHITSBlock(nn.Module):
    """N-HiTS block which takes a basis function as an argument.

    - x_t (local covariate series) was disabled, which is now removed
    """
    def __init__(self, n_time_in: int, n_time_out: int, 
                 n_s: int, n_s_hidden: int, n_theta: int, n_theta_hidden: list,
                 n_pool_kernel_size: int, pooling_mode: str, basis: nn.Module,
                 n_layers: int,  batch_normalization: bool, dropout_prob: float, activation: str):
        super().__init__()

        assert (pooling_mode in ['max',])

        n_time_in_pooled = int(np.ceil(n_time_in/n_pool_kernel_size))

        if n_s == 0:
            n_s_hidden = 0
        # n_s is computed with data, n_s_hidden is provided by user, if 0 no statics are used
        if n_s > 0 and n_s_hidden > 0:
            self.static_encoder = _StaticFeaturesEncoder(in_features=n_s, out_features=n_s_hidden)

        n_theta_hidden = [n_time_in_pooled + n_s_hidden] + n_theta_hidden

        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.n_pool_kernel_size = n_pool_kernel_size
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob

        assert activation in ACTIVATIONS, f'{activation} is not in {ACTIVATIONS}'
        activ_t = getattr(nn, activation)

        if pooling_mode == 'max':
            self.pooling_layer = nn.MaxPool1d(kernel_size=self.n_pool_kernel_size,
                                              stride=self.n_pool_kernel_size, ceil_mode=True)

        hidden_layers = []
        for i in range(n_layers):
            hidden_layers.append(nn.Linear(in_features=n_theta_hidden[i], out_features=n_theta_hidden[i+1]))
            hidden_layers.append(activ_t())

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=n_theta_hidden[i+1]))

            if self.dropout_prob>0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [nn.Linear(in_features=n_theta_hidden[-1], out_features=n_theta)]
        layers = hidden_layers + output_layer

        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(self, insample_y: t.Tensor, x_s: t.Tensor) -> Tuple[t.Tensor, t.Tensor]:
        # Pooling layer to downsample input
        insample_y = self.pooling_layer(insample_y)

        # Static exogenous
        if (self.n_s > 0) and (self.n_s_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta)

        return backcast, forecast


class _NHITS(nn.Module):
    """Modified N-HiTS impl.

    - lecun_normal init was ignored, which is now removed
    - remove unused mask:
        * insample_mask=available_mask, which is always 1 and redundant
        * outsample_mask=sample_mask, which is a flag indicating tr/val/te set
    - remove outsample_y from model input
    """
    def __init__(self,
                 n_time_in,
                 n_time_out,
                 n_s,
                 n_s_hidden,
                 stack_types: list,
                 n_blocks: list,
                 n_layers: list,
                 n_theta_hidden: list,
                 n_pool_kernel_size: list,
                 n_freq_downsample: list,
                 pooling_mode,
                 interpolation_mode,
                 dropout_prob_theta,
                 activation,
                 batch_normalization,
                 shared_weights):
        super().__init__()

        self.n_time_out = n_time_out

        blocks = self.create_stack(stack_types=stack_types,
                                   n_blocks=n_blocks,
                                   n_time_in=n_time_in,
                                   n_time_out=n_time_out,
                                   n_s=n_s,
                                   n_s_hidden=n_s_hidden,
                                   n_layers=n_layers,
                                   n_theta_hidden=n_theta_hidden,
                                   n_pool_kernel_size=n_pool_kernel_size,
                                   n_freq_downsample=n_freq_downsample,
                                   pooling_mode=pooling_mode,
                                   interpolation_mode=interpolation_mode,
                                   batch_normalization=batch_normalization,
                                   dropout_prob_theta=dropout_prob_theta,
                                   activation=activation,
                                   shared_weights=shared_weights,
                                   )
        self.blocks = t.nn.ModuleList(blocks)  # across stacks

    def create_stack(self, stack_types, n_blocks,
                     n_time_in, n_time_out,
                     n_s, n_s_hidden,
                     n_layers, n_theta_hidden,
                     n_pool_kernel_size, n_freq_downsample, pooling_mode, interpolation_mode,
                     batch_normalization, dropout_prob_theta,
                     activation, shared_weights):

        block_list = []
        for i in range(len(stack_types)):
            for block_id in range(n_blocks[i]):
                # Batch norm only on first block
                if batch_normalization and len(block_list) == 0:
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Shared weights
                if shared_weights and block_id>0:
                    nbeats_block = block_list[-1]
                else:
                    if stack_types[i] == 'identity':
                        n_theta = (n_time_in + max(n_time_out//n_freq_downsample[i], 1) )
                        basis = IdentityBasis(backcast_size=n_time_in,
                                              forecast_size=n_time_out,
                                              interpolation_mode=interpolation_mode)
                    else:
                        raise ValueError(f'Block type not found!')
                    nbeats_block = _NHITSBlock(n_time_in=n_time_in,
                                               n_time_out=n_time_out,
                                               n_s=n_s,
                                               n_s_hidden=n_s_hidden,
                                               n_theta=n_theta,
                                               n_theta_hidden=n_theta_hidden[i],
                                               n_pool_kernel_size=n_pool_kernel_size[i],
                                               pooling_mode=pooling_mode,
                                               basis=basis,
                                               n_layers=n_layers[i],
                                               batch_normalization=batch_normalization_block,
                                               dropout_prob=dropout_prob_theta,
                                               activation=activation)
                block_list.append(nbeats_block)
        return block_list

    def forward(self, S: t.Tensor, Y: t.Tensor, return_decomposition=False):
        # only Y_df is not empty from callers
        insample_y = Y # [:, :-self.n_time_out]

        if return_decomposition:
            forecast, block_forecasts = self.forecast_decomposition(insample_y=insample_y,
                                                                    x_s=S)
            return forecast, block_forecasts
        else:
            forecast = self.forecast(insample_y=insample_y, x_s=S)
            return forecast

    def forecast(self, insample_y: t.Tensor, x_s: t.Tensor):
        # ? about pooling?
        residuals = insample_y.flip(dims=(-1,))
        # Naive: predict last step
        forecast = insample_y[:, -1:].repeat(1, self.n_time_out)
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, x_s=x_s)
            residuals = residuals - backcast
            forecast = forecast + block_forecast

        return forecast

    def forecast_decomposition(self, insample_y: t.Tensor, x_s: t.Tensor):
        residuals = insample_y.flip(dims=(-1,))
        forecast = insample_y[:, -1:].repeat(1, self.n_time_out)
        block_forecasts = [ forecast ]

        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(insample_y=residuals, x_s=x_s)
            residuals = residuals - backcast
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        return forecast, block_forecasts





class NHITS(pl.LightningModule):
    """N-HiTS training and prediction procedure

    Parameters
    ----------
    n_time_in: int
        Lookback period.
    n_time_out: int
        Forecast horizon.
    shared_weights: bool
        If True, repeats first block.
    activation: str
        Activation function.
        An item from `ACTIVATIONS`.
    stack_types: List[str]
        List of stack types.
        Subset from ['identity'].
    n_blocks: List[int]
        Number of blocks for each stack type.
        Note that len(n_blocks) = len(stack_types).
    n_layers: List[int]
        Number of layers for each stack type.
        Note that len(n_layers) = len(stack_types).
    n_theta_hidden: List[List[int]]
        Structure of hidden layers for each stack type.
        Each internal list should contain the number of units of each hidden layer.
        Note that len(n_theta_hidden) = len(stack_types).
    n_pool_kernel_size List[int]:
        Pooling size for input for each stack.
        Note that len(n_pool_kernel_size) = len(stack_types).
    n_freq_downsample List[int]:
        Downsample multiplier of output for each stack.
        Note that len(n_freq_downsample) = len(stack_types).
    batch_normalization: bool
        Whether perform batch normalization.
    dropout_prob_theta: float
        Float between (0, 1).
        Dropout for Nbeats basis.
    learning_rate: float
        Learning rate between (0, 1).
    lr_decay: float
        Decreasing multiplier for the learning rate.
    lr_decay_step_size: int
        Steps between each lerning rate decay.
    weight_decay: float
        L2 penalty for optimizer.
    loss_train: str
        Loss to optimize.
        An item from ['MAPE', 'MASE', 'SMAPE', 'MSE', 'MAE', 'PINBALL', 'PINBALL2'].
    loss_hypar:
        Hyperparameter for chosen loss.
    loss_valid:
        Validation loss.
        An item from ['MAPE', 'MASE', 'SMAPE', 'RMSE', 'MAE', 'PINBALL'].
    frequency: str
        Time series frequency.
    random_seed: int
        random_seed for pseudo random pytorch initializer and
        numpy random generator.
    seasonality: int
        Time series seasonality.
        Usually 7 for daily data, 12 for monthly data and 4 for weekly data.
    """
    def __init__(self,
                 n_time_in=5*96,
                 n_time_out=96,
                 n_s=0,
                 n_s_hidden=0,
                 shared_weights=False,
                 activation='ReLU',
                 stack_types=3*['identity'],
                 n_blocks=3*[1],
                 n_layers=3*[2],
                 n_theta_hidden=3*[[512, 512]],
                 n_pool_kernel_size=[8, 4, 1],
                 n_freq_downsample=[24, 12, 1],
                 pooling_mode='max',
                 interpolation_mode='linear',
                 batch_normalization=False,
                 dropout_prob_theta=0,
                 learning_rate=1e-3,
                 lr_decay=0.5,
                 weight_decay=0,
                 loss_train='MAE',
                 loss_hypar=24,
                 loss_valid='MAE',
                 frequency='H',
                 seasonality=24):
        super(NHITS, self).__init__()
        self.save_hyperparameters()  # todo object.__getattr__(self, name)
        #------------------------ Model Attributes ------------------------#
        # Architecture parameters
        self.n_time_in = n_time_in
        self.n_time_out = n_time_out
        self.n_s = n_s
        self.n_s_hidden = n_s_hidden
        self.shared_weights = shared_weights
        self.activation = activation
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        self.n_theta_hidden = n_theta_hidden
        self.n_pool_kernel_size = n_pool_kernel_size
        self.n_freq_downsample = n_freq_downsample
        self.pooling_mode = pooling_mode
        self.interpolation_mode = interpolation_mode

        # Loss functions
        self.loss_train = loss_train
        self.loss_hypar = loss_hypar
        self.loss_valid = loss_valid
        assert loss_train == loss_valid == "MAE"
        self.loss_fn_train = MAELoss
        self.loss_fn_valid = MAELoss

        # Regularization and optimization parameters
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob_theta
        self.learning_rate = learning_rate
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay

        # Data parameters
        self.frequency = frequency
        self.seasonality = seasonality
        self.return_decomposition = False

        self.model = _NHITS(n_time_in=self.n_time_in,
                            n_time_out=self.n_time_out,
                            n_s=self.n_s,
                            n_s_hidden=self.n_s_hidden,
                            stack_types=self.stack_types,
                            n_blocks=self.n_blocks,
                            n_layers=self.n_layers,
                            n_theta_hidden=self.n_theta_hidden,
                            n_pool_kernel_size=self.n_pool_kernel_size,
                            n_freq_downsample=self.n_freq_downsample,
                            pooling_mode=self.pooling_mode,
                            interpolation_mode=self.interpolation_mode,
                            dropout_prob_theta=self.dropout_prob_theta,
                            activation=self.activation,
                            batch_normalization=self.batch_normalization,
                            shared_weights=self.shared_weights)

    def training_step(self, batch, batch_idx):
        (_, inpY), (_, tgtY), _, wnd_start = batch  # use normed data
        n_series = inpY.shape[0]
        S = t.zeros((n_series,), device=self.device)

        forecast = self.model(S=S, Y=inpY[..., :-self.n_time_out+1])

        loss = self.loss_fn_train(y=tgtY, y_hat=forecast, )

        self.log("loss/train", loss,
                 on_step=True, on_epoch=False, batch_size=1)

        return loss

    def validation_step(self, batch, idx):
        (_, inpY_n), (tgtY, tgtY_n), scaler, wnd_start = batch
        n_series = inpY_n.shape[0]
        S = t.zeros((n_series,), device=self.device)

        forecast = self.model(S=S, Y=inpY_n[..., :-self.n_time_out+1])
        outY = inv_transform(scaler, forecast)

        loss = self.loss_fn_valid(y=tgtY_n, y_hat=forecast, )
        self.log("loss/val", loss,
                 on_step=False, on_epoch=True, batch_size=1)

        return {
            'loss': loss,
            'pred': outY, 'scaled_pred': forecast,
            'tgt': tgtY, 'scaled_tgt': tgtY_n
        }

    def validation_epoch_end(self, outputs) -> None:
        losses = []
        for d in outputs:
            losses.append(d['loss'])
        loss = t.stack(losses).mean().item()  # approx
        nni.report_intermediate_result(loss)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)

        min_lr = self.learning_rate * (self.lr_decay ** 3)
        lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 'min', self.lr_decay, patience=1, cooldown=1, min_lr=min_lr)

        return {
            'optimizer': optimizer, 'lr_scheduler': {
                "scheduler": lr_scheduler,
                "monitor": "loss/val", }
        }

    def on_train_epoch_end(self):
        if self.trainer.should_stop:
            hp_metric = self.checkpoint_callback.best_model_score
            self.log('hp_metric', hp_metric)
            final_result = hp_metric.item()
            nni.report_final_result(final_result)

    def configure_callbacks(self):
        self.checkpoint_callback = ModelCheckpoint(
            monitor='loss/val', mode='min', save_last=True)
        self.earlystop_callback = EarlyStopping(
            monitor='loss/val', patience=5, mode='min')
        self.loglr_cb = LearningRateMonitor("epoch")

        return [self.loglr_cb, self.earlystop_callback, self.checkpoint_callback]  #
