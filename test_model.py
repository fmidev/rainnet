import os
from pathlib import Path
import argparse

from utils.config import load_config
from utils.logging import setup_logging
from utils.radar_image_plots import plot_1h_plus_1h_timeseries

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers

from datamodules import Rainnet_FMICompositeDataModule
from models import RainNet

def main(configpath, checkpoint=None):
    confpath = Path("config") / configpath
    dsconf = load_config(confpath / "FMIComposite.yaml")
    outputconf = load_config(confpath / "output.yaml")
    modelconf = load_config(confpath / "rainnet.yaml")

    setup_logging(outputconf.logging)
    
    datamodel = Rainnet_FMICompositeDataModule(
            dsconf, modelconf.train_params)
    
    model = RainNet(modelconf)
    
    tb_logger = pl_loggers.TensorBoardLogger("logs", name=f"train_{confpath}")
    trainer = pl.Trainer(
        profiler="pytorch",
        logger=tb_logger,
        val_check_interval=modelconf.train_params.val_check_interval,
        max_epochs=modelconf.train_params.max_epochs,
        max_time=modelconf.train_params.max_time,
        gpus=modelconf.train_params.gpus,
        limit_train_batches = 10,
        limit_val_batches = 5,
        limit_test_batches = 75
    )
    
    trainer.test(model=model, datamodule=datamodel, ckpt_path=checkpoint)
    
    
if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("config", type=str, help="Configuration folder")
    argparser.add_argument("-c", "--checkpoint", type=str, default=None,
                           help="Path to checkpoint for model that is tested.")

    args = argparser.parse_args()
    main(args.config, args.checkpoint)
