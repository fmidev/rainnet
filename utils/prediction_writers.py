import torch
from pytorch_lightning.callbacks import BasePredictionWriter
from typing import Any, List
import numpy as np
from datetime import timedelta
import h5py
from pathlib import Path


class RainnetHDF5Writer(BasePredictionWriter):
    def __init__(
        self,
        db_name: str,
        group_format: str,
        method_name: str,
        what_attrs: dict = {},
        write_interval: str = "batch",
        **kwargs,
    ):
        super().__init__(write_interval)
        self.db_name = db_name
        self.what_attrs = what_attrs
        self.group_format = group_format
        self.method_name = method_name

    def write_on_batch_end(
        self,
        trainer,
        pl_module : 'LightningModule',
        prediction: Any,
        batch_indices: List[int],
        batch: Any,
        batch_idx: int,
        dataloader_idx: int,
    ):
        _, _, batch_indices_ = batch
        batch_indices_ = batch_indices_.tolist()

        with h5py.File(self.db_name, 'a') as db:

            for bi, b_idx in enumerate(batch_indices_):
                common_time = trainer.datamodule.predict_dataset.get_common_time(b_idx)
                group_name = self.group_format.format(
                    timestamp=common_time,
                    method = self.method_name)
                group = db.require_group(group_name)

                n_leadtimes = (
                    prediction.shape[1] if prediction.ndim == 4 # B, T, W, H
                    else prediction.shape[2] # S, B, T, W, H
                )

                for i in range(n_leadtimes):
                    data_unpacked = (
                        prediction[bi, i, ...] if prediction.ndim == 4 # B, T, W, H
                        else prediction[:, bi, i, ...] # S, B, T, W, H
                    )
                    date = common_time + timedelta(
                        minutes=i * trainer.datamodule.predict_dataset.timestep
                    )
                    dname = f"{i + 1}"
                    ds_group = group.require_group(dname)
                    what_attrs = self.what_attrs.copy()
                    what_attrs["validtime"] = np.string_(f"{date:%Y-%m-%d %H:%M:%S}")
                    packed = arr_compress_uint8(data_unpacked.detach().cpu().numpy(), missing_val = 0)
                    write_image(
                        group=ds_group,
                        ds_name="data",
                        data=packed,
                        what_attrs=what_attrs,
                    )

    def write_on_epoch_end(
        self,
        trainer,
        pl_module: "LightningModule",
        predictions: List[Any],
        batch_indices: List[Any],
    ):
        pass


def arr_compress_uint8(
    dBZ_array: np.ndarray, missing_val: np.uint8 = 255
) -> np.ndarray:
    masked = np.ma.masked_where(~np.isfinite(dBZ_array), dBZ_array)
    max_value_dBZ = -32 + 0.5 * 254  # not 255 to ensure no real value gets lost!
    mask_big_values = dBZ_array[...] >= max_value_dBZ
    arr = ((2.0 * masked) + 64).astype(np.uint8)
    arr[arr.mask] = missing_val
    arr[mask_big_values] = 254
    return arr.data


def write_image(group: h5py.Group, ds_name: str, data: np.ndarray, what_attrs: dict):
    try:
        del group[ds_name]
    except:
        pass
    dataset = group.create_dataset(
        ds_name, data=data, dtype="uint8", compression="gzip", compression_opts=9
    )
    dataset.attrs["CLASS"] = np.string_("IMAGE")
    dataset.attrs["IMAGE_VERSION"] = np.string_("1.2")

    ds_what = group.require_group("what")
    for k, val in what_attrs.items():
        ds_what.attrs[k] = val
