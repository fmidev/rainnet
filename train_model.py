import os
from pathlib import Path
import argparse
import random

import numpy as np

from utils.config import load_config
from utils.logging import setup_logging
from utils.radar_image_plots import plot_1h_plus_1h_timeseries

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, LearningRateMonitor, DeviceStatsMonitor
from pytorch_lightning.profiler import PyTorchProfiler

from datamodules import Rainnet_FMICompositeDataModule
from models import RainNet
from callbacks import NowcastMetrics, LogNowcast

def main(configpath, checkpoint=None):
    # load configuration from configuration files
    confpath = Path("config") / configpath
    dsconf = load_config(confpath / "FMIComposite.yaml")
    outputconf = load_config(confpath / "output.yaml")
    modelconf = load_config(confpath / "rainnet.yaml")
    metricsconf = load_config(confpath / "nowcast_metrics_callback.yaml")
    lognowcastconf = load_config(confpath / "log_nowcast_callback.yaml")

    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)
    
    setup_logging(outputconf.logging)
    
    # pl datamodule
    datamodel = Rainnet_FMICompositeDataModule(
            dsconf, modelconf.train_params)
    
    # pl module
    model = RainNet(modelconf)

    profiler = PyTorchProfiler(profile_memory=False)
    
    # callbacks
    model_ckpt = ModelCheckpoint(
        dirpath = 'checkpoint/',
        save_top_k = 3,
        monitor = "val_loss", 
        save_on_train_epoch_end = False,
    )
    nowcast_image_logger = LogNowcast(config=lognowcastconf)
    nowcast_metrics = NowcastMetrics(config=metricsconf)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    #device_monitor = DeviceStatsMonitor()

    # tensorboard logger
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir="logs",
        name=f"train_{confpath}"
        )

    trainer = pl.Trainer(
        profiler=profiler,
        logger=tb_logger,
        val_check_interval=modelconf.train_params.val_check_interval,
        max_epochs=modelconf.train_params.max_epochs,
        max_time=modelconf.train_params.max_time,
        gpus=modelconf.train_params.gpus,
        limit_train_batches=modelconf.train_params.train_batches, 
        limit_val_batches = modelconf.train_params.val_batches,
        callbacks=[
            EarlyStopping(**modelconf.train_params.early_stopping),
            model_ckpt,
            lr_monitor,
            nowcast_image_logger,
            nowcast_metrics
            ],
    )
    trainer.fit(model=model, datamodule=datamodel, ckpt_path=checkpoint)
    torch.save(model.state_dict(), f"state_dict_{modelconf.train_params.savefile}.ckpt")
    trainer.save_checkpoint(f"{modelconf.train_params.savefile}.ckpt")    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration folder")
    argparser.add_argument("-c", "--continue_training", type=str, default=None,
                           help="Path to checkpoint for model that is continued.")

    args = argparser.parse_args()
    main(args.config, args.continue_training)
