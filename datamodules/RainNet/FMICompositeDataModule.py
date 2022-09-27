"""Datamodule for FMI radar composite."""
import h5py
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from datasets import Rainnet_FMIComposite


class FMICompositeDataModule(pl.LightningDataModule):
    def __init__(self, dsconfig, train_params):
        super().__init__()
        self.save_hyperparameters()
        self.dsconfig = dsconfig
        self.train_params = train_params

        self.db_path = dsconfig.pop("hdf5_path")
        if self.dsconfig["importer"] == "hdf5":
            self.db = h5py.File(self.db_path, 'r')
        else:
            self.db = None
        

    def prepare_data(self):
        # called only on 1 GPU
        pass

    def setup(self, stage):
        # called on every GPU
        self.train_dataset = Rainnet_FMIComposite(split="train", db=self.db,  **self.dsconfig)
        self.valid_dataset = Rainnet_FMIComposite(split="valid", db=self.db, **self.dsconfig)
        self.test_dataset = Rainnet_FMIComposite(split="test", db=self.db, **self.dsconfig)
        self.predict_dataset = Rainnet_FMIComposite(split="predict",db = self.db, **self.dsconfig)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.train_params.train_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=True,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.train_params.valid_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            pin_memory=True,
        )
        return valid_loader

    def test_dataloader(self):
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.train_params.test_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
        )
        return test_loader
    
    def predict_dataloader(self):
        predict_loader = DataLoader(
            self.predict_dataset,
            batch_size=self.train_params.predict_batch_size,
            num_workers=self.train_params.num_workers,
            shuffle=False,
            collate_fn=_collate_fn,
        )
        return predict_loader
    
    def teardown(self, stage=None):
        if self.dsconfig.importer == "hdf5":
            self.db.close()

    def get_train_size(self):
        if hasattr(self, "train_dataset"):
            return len(self.train_dataset)
        else:
            dummy_train_ds = Rainnet_FMIComposite(split="train", db=self.db,  **self.dsconfig)
            train_size = len(dummy_train_ds)
            del dummy_train_ds
            return train_size


def _collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)
