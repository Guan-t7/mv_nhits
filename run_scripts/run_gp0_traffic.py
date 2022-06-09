import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import nni
from nni.utils import merge_parameter

sys.path.append('./')

from model.MixerNHI import GlblPred0
from data.TS import TSDataModule
from utils.callbacks import LogMetricsCallback


exp_df = pd.DataFrame.from_records(
    data=[
        ("PeMSD8", 12, 12, [1, 1, 1], [4, 2, 1]),
        # ("TrafficL", 5*96, 96, [1, 1, 1], [24, 12, 1]),
        # ("traffic", 7*24, 24, [8, 4, 1], [12, 6, 1]),
    ], columns=['dataset', 'history', 'horizon', 'pool', 'freq']
)
n_series_dict = {
    "TrafficL": 862,
    "traffic": 963,
    "PeMSD8": 170,
}


def main_gp0(*args):
    tuner_params, = args
    for rec in exp_df.itertuples(index=False):
        dataset, n_time_in, n_time_out, n_pool_kernel_size, n_freq_downsample = rec

        dm = TSDataModule(dataset, n_time_in, n_time_out, hbstep=1,
                          mode="M", bs=256)

        n_theta_hidden = [[512, 512], [512, 512], [512, 512]]
        nhits_kwargs = dict(n_time_in=n_time_in, n_time_out=n_time_out,
                            n_theta_hidden=n_theta_hidden,
                            n_pool_kernel_size=n_pool_kernel_size,
                            n_freq_downsample=n_freq_downsample,)
        merge_parameter(nhits_kwargs, tuner_params)
        model = GlblPred0(num_series=n_series_dict[dataset],
                          n_time_in=n_time_in, n_time_out=n_time_out,
                          Xseq_kwargs=nhits_kwargs,
                          lr=1e-3,
                          lr_decay=0.5,
                          )
        logmetric_cb = LogMetricsCallback(metrics=['mae', 'rmse', ],  # 'mape'
                                          scaled=False, free_mem=True)
        logger = True  #(tuner_params == {})
        trainer = pl.Trainer(gpus=1, log_every_n_steps=37, logger=logger,  # fast_dev_run=5,
                             callbacks=[logmetric_cb, ]
                             )  # min_epochs=5 max_epochs=10
        trainer.fit(model, datamodule=dm,)


warnings.filterwarnings(
    "ignore", ".*Consider increasing the value of the `num_workers` argument*")
warnings.filterwarnings(
    "ignore", '.*Running .* code without runtime.*')

if __name__ == "__main__":
    tuner_params: dict = nni.get_next_parameter()
    seed = tuner_params.get("seed", 2021)
    pl.seed_everything(seed)
    if "seed" in tuner_params:
        del tuner_params["seed"]

    main_gp0(tuner_params)
