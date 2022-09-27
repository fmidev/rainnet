"""This script will run nowcasting prediction for the L-CNN model implementation
"""
import argparse
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

from utils import load_config, setup_logging
from utils import RainnetHDF5Writer
from models import RainNet
from datamodules import Rainnet_FMICompositeDataModule


def run(checkpointpath, configpath) -> None:

    confpath = Path("config") / configpath
    dsconf = load_config(confpath / "FMIComposite.yaml")
    outputconf = load_config(confpath / "output.yaml")
    modelconf = load_config(confpath / "rainnet.yaml")

    setup_logging(outputconf.logging)

    datamodel = Rainnet_FMICompositeDataModule(dsconf, modelconf.train_params)

    model = RainNet(modelconf).load_from_checkpoint(checkpointpath, config=modelconf)

    output_writer = RainnetHDF5Writer(**modelconf.prediction_output)

    tb_logger = pl_loggers.TensorBoardLogger("logs", name=f"predict_{confpath}")
    trainer = pl.Trainer(
        profiler="pytorch",
        logger=tb_logger,
        gpus=modelconf.train_params.gpus,
        callbacks=[output_writer],
    )

    # Predictions are written in HDF5 file
    trainer.predict(model, datamodel, return_predictions=False)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    argparser.add_argument("checkpoint", type=str, help="Path to checkpoint file")
    argparser.add_argument(
        "config",
        type=str,
        help="Configuration folder path",
    )
    args = argparser.parse_args()

    run(args.checkpoint, args.config)