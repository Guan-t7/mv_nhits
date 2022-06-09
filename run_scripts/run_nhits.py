import argparse
import sys
import warnings

import numpy as np
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import PyTorchProfiler
import nni
from nni.utils import merge_parameter

sys.path.append('./')

from model import nhits_mv
# from model.nhits_mv import NHITS
from model.nhits import NHITS
from data.TS import TSDataModule
from utils.callbacks import LogMetricsCallback


exp_df = pd.DataFrame.from_records(
    data=[
        ("PeMSD8", 12, [1, 1, 1], [12, 6, 1]),
        # ("TrafficL", 96, [1, 1, 1], [24, 12, 1]),
        # ("TrafficL", 192, [2, 2, 2], [60, 8, 1]),
        # ("ECL", 96, [1, 1, 1], [60, 8, 1]),
        # ("ECL", 192, [16, 8, 1], [60, 8, 1]),
    ], columns=['dataset', 'horizon', 'pool', 'freq']
)

def main_NHI(*args):
    tuner_params, = args
    for rec in exp_df.itertuples(index=False):
        dataset, n_time_out, n_pool_kernel_size, n_freq_downsample = rec
        
        n_time_in = n_time_out*5  # LTSF/PeMS dataset: 5/1
        dm = TSDataModule(dataset, n_time_in, n_time_out, hbstep=1,
                          mode="M", bs=1024)
        if dataset == "PeMSD8":
            adj_mx = torch.from_numpy(np.load('data/adj_08.npy'))
            # nhits_mv.adj_mx = adj_mx
        
        nhits_kwargs = dict(
            n_pool_kernel_size=n_pool_kernel_size, n_freq_downsample=n_freq_downsample)
        merge_parameter(nhits_kwargs, tuner_params)
        n_theta_hidden = [[512, 512], [512, 512], [512, 512]]
        model = NHITS(n_time_in=n_time_in, n_time_out=n_time_out,
                      n_theta_hidden=n_theta_hidden,
                      **nhits_kwargs
                      )
        # LTSF dataset: metrics=['mae', 'mse', ], scaled=True
        # PeMS dataset: metrics=['mae', 'rmse', ], scaled=False
        logmetric_cb = LogMetricsCallback(metrics=['mae', 'mse', ],  # todo 'mape'
                                          scaled=True, free_mem=True)
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
    
    main_NHI(tuner_params)
