'''
Script and functions for writing measurements and predictions to an hdf5 file in 
a compressed scale-offset uint8 format.
'''
from datetime import datetime, timedelta
import warnings
import os
import sys
import argparse
from attrdict import AttrDict
import yaml

import h5py
from tqdm import tqdm
import numpy as np

from pysteps import motion, nowcasts, io, rcparams
from pysteps.utils import conversion, transformation, dimension

import io_tools
import torch

data_source = rcparams.data_sources["fmi"]
root_path = data_source["root_path"]
path_fmt = data_source["path_fmt"]
fn_pattern = data_source["fn_pattern"]
fn_ext = data_source["fn_ext"]
importer_name = data_source["importer"]
importer_kwargs = data_source["importer_kwargs"]
timestep = data_source["timestep"]
importer = io.get_method(importer_name, "importer")


def run(config):
    datelist_path = config.datelist_path
    downsampling = config.downsampling
    hdf5_path = config.hdf5_path
    builder = config.builder
    save_indexes = config.save_indexes
    n_leadtimes = config.n_leadtimes

    with h5py.File(hdf5_path, 'a') as db:
        if "measurements" in builder:
            print("building observations' db")
            days = io_tools.read_file(datelist_path, start_idx=1)
            n_days = len(days)
            for i in tqdm(range(n_days)):
                if i % 10 == 0:
                    process_one_rainday_measurements(db=db,
                        rainday=days[i],
                        downsampling=downsampling
                        )
            return 
        
        timestamps = io_tools.read_file(datelist_path,start_idx=0)

        if "advection" in builder:
            print("building extrapolation prediction db")
            process_avd_predictions(timestamps = timestamps, db=db,
                                    n_leadtimes=n_leadtimes, save_indexes=save_indexes)
        if "sprog" in builder:
            print("building S-PROG prediction db")
            process_sprog_predictions(timestamps = timestamps, db=db,
                                    n_leadtimes=n_leadtimes, save_indexes=save_indexes)
        if "linda" in builder:
            print("building LINDA prediction db")
            process_linda_predictions(timestamps = timestamps, db=db,
                                    n_leadtimes=n_leadtimes, save_indexes=save_indexes)
        if "rainnet_keras" in builder:
            print("building Keras RainNet implementation prediction db")
            model_path = config.model_path
            process_rainnet_keras_predictions(timestamps = timestamps, db=db,
                                              model_path = model_path,
                                    n_leadtimes=n_leadtimes, save_indexes=save_indexes)
        if "rainnet_pytorch" in builder:
            print("building PyTorch RainNet implementation prediction db")
            model_path = config.model_path
            process_rainnet_pytorch_predictions(timestamps = timestamps, db=db,
                                              model_path = model_path,
                                    n_leadtimes=n_leadtimes, save_indexes=save_indexes)

    


def process_one_rainday_measurements(db : h5py.File,
                                     rainday: str,
                                     downsampling: bool):
    what_attrs = dict()
    what_attrs["quantity"] = 'DBZH'       
    day_obj = datetime.strptime(rainday, '%Y-%m-%d')

    for instant in tqdm(range(1,324)): # 00:05 .. 23:55 (288 timestamp) + 35 remaining prediction 
        timenow = day_obj + timedelta(minutes=5*instant)
        t_label = timenow.strftime('%Y-%m-%d %H:%M:%S')
        t_grp = db.require_group(t_label)
        grp = t_grp.require_group("measurement")
        if str(t_label + "/" + "measurement/data") not in db:
            try:
                fns = io.archive.find_by_date(
                    timenow, root_path, path_fmt, fn_pattern, fn_ext, timestep
            )
            except IOError:
                warnings.warn(f"No file for time {t_label}, continuing")
                continue
            try:
                Z, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
            except:
                warnings.warn(f"! An error occured in reading {fns} in... continuing")
                continue
            bbox = (125 * metadata["xpixelsize"], 637 *  metadata["xpixelsize"],
                    604 * metadata["ypixelsize"],  1116 * metadata["ypixelsize"])
            metadata["yorigin"] = "lower"
            Z, metadata = dimension.clip_domain(Z, metadata, extent=bbox)
            if downsampling:
                metadata["xpixelsize"] = metadata["ypixelsize"]
                Z, metadata = dimension.aggregate_fields_space(Z, metadata, metadata["xpixelsize"]*2.0)
            Z_compressed = io_tools.arr_compress_uint8(Z)
            what_attrs["gain"] = 0.5
            what_attrs["offset"] = metadata["zerovalue"]
            what_attrs["nodata"] = metadata["zerovalue"]
            what_attrs["undetect"] = 255
            io_tools.write_image(grp, "data", Z_compressed, what_attrs)

def process_avd_predictions(timestamps : list,
                                     db : h5py.File,
                                     n_leadtimes : int,
                                     save_indexes : list = None):
    what_attrs = dict()
    what_attrs["quantity"] = 'DBZH'
    lk = motion.get_method("LK")
    advep = nowcasts.get_method("lagrangian")

    if save_indexes is None:
        save_indexes = range(n_leadtimes)


    for t_label in tqdm(timestamps):
        time = datetime.strptime(t_label, '%Y-%m-%d %H:%M:%S')
        t_grp = db.require_group(t_label)
        grp = t_grp.require_group("advection")
        if str(t_label + "/" + "advection/36/data") not in db:
            try:
                fns = io.archive.find_by_date(
                    time, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_next_files=3
            )
            except IOError:
                warnings.warn(f"No file for time {t_label}, continuing")
                continue
            try:
                Z, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
            except:
                warnings.warn(f"! An error occured in reading {fns} in... continuing")
                continue
            R, metadata = conversion.to_rainrate(Z,metadata)
            bbox = (125 * metadata["xpixelsize"], 637 *  metadata["xpixelsize"],
                    604 *metadata["ypixelsize"], 1116 * metadata["ypixelsize"])
            metadata["yorigin"] = "lower"
            R, metadata = dimension.clip_domain(R, metadata, extent=bbox)
            R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)
            R[~np.isfinite(R)] = metadata["zerovalue"]
            lk_flow = lk(R[0:4,:,:])
            nowcast = advep(R[3,:,:], lk_flow, n_leadtimes)
            #nowcast[~np.isfinite(nowcast)] = metadata["zerovalue"]
            nowcast, _ = transformation.dB_transform(nowcast, threshold=-10, zerovalue=0, inverse=True)
            nowcast = io_tools.rainrate_to_dBZ(nowcast)


            for i in save_indexes:
                leadtime_grp = grp.require_group(str(i+1))
                ni_compressed = io_tools.arr_compress_uint8(nowcast[i,:,:])
                what_attrs["gain"] = 0.5
                what_attrs["offset"] = -32
                what_attrs["nodata"] = 255
                what_attrs["undetect"] = 255
                io_tools.write_image(leadtime_grp, "data", ni_compressed, what_attrs)

def process_sprog_predictions(timestamps : list,
                                     db : h5py.File,
                                     n_leadtimes : int,
                                     save_indexes : list = None):
    what_attrs = dict()
    what_attrs["quantity"] = 'DBZH'
    lk = motion.get_method("LK")
    sprog = nowcasts.get_method("sprog")

    if save_indexes is None:
        save_indexes = range(n_leadtimes)

    for t_label in tqdm(timestamps):
        time = datetime.strptime(t_label, '%Y-%m-%d %H:%M:%S')
        t_grp = db.require_group(t_label)
        grp = t_grp.require_group("sprog")

        if str(t_label + "/" + "sprog/36/data") not in db:

            try:
                fns = io.archive.find_by_date(
                    time, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_next_files=3
            )
            except IOError:
                warnings.warn(f"No file for time {t_label}, continuing")
                continue

            try:
                Z, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
            except:
                warnings.warn(f"! An error occured in reading {fns} in... continuing")
                continue

            R, metadata = conversion.to_rainrate(Z,metadata)
            bbox = (125 * metadata["xpixelsize"], 637 *  metadata["xpixelsize"],
                    604 *metadata["ypixelsize"], 1116 * metadata["ypixelsize"])
            metadata["yorigin"] = "lower"
            R, metadata = dimension.clip_domain(R, metadata, extent=bbox)
            R, metadata = transformation.dB_transform(R, metadata, threshold=0.1, zerovalue=-15.0)
            R[~np.isfinite(R)] = metadata["zerovalue"]
            # block printing
            sys.stdout = open(os.devnull, 'w')
            lk_flow = lk(R[0:4,:,:])
            nowcast = sprog(R[1:4,:,:], lk_flow, n_leadtimes, R_thr=-10)
            # re-enable printing
            sys.stdout = sys.__stdout__
            #nowcast[~np.isfinite(nowcast)] = metadata["zerovalue"]
            nowcast,_ = transformation.dB_transform(nowcast, threshold=-10, zerovalue=0, inverse=True)
            nowcast = io_tools.rainrate_to_dBZ(nowcast)


            for i in save_indexes:
                leadtime_grp = grp.require_group(str(i+1))
                ni_compressed = io_tools.arr_compress_uint8(nowcast[i,:,:])
                what_attrs["gain"] = 0.5
                what_attrs["offset"] = -32
                what_attrs["nodata"] = 255
                what_attrs["undetect"] = 255
                io_tools.write_image(leadtime_grp, "data", ni_compressed, what_attrs)

def process_linda_predictions(timestamps : list,
                                     db : h5py.File,
                                     n_leadtimes : int,
                                     save_indexes : list = None):
    what_attrs = dict()
    what_attrs["quantity"] = 'DBZH'
    lk = motion.get_method("LK")
    linda = nowcasts.get_method("linda")

    if save_indexes is None:
        save_indexes = range(n_leadtimes)

    for t_label in tqdm(timestamps):
        time = datetime.strptime(t_label, '%Y-%m-%d %H:%M:%S')
        t_grp = db.require_group(t_label)
        grp = t_grp.require_group("linda")

        if str(t_label + "/" + "linda/36/data") not in db:

            try:
                fns = io.archive.find_by_date(
                    time, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_next_files=3
            )
            except IOError:
                warnings.warn(f"No file for time {t_label}, continuing")
                continue

            try:
                Z, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
            except:
                warnings.warn(f"! An error occured in reading {fns} in... continuing")
                continue

            R, metadata = conversion.to_rainrate(Z,metadata)
            
            bbox = (125 * metadata["xpixelsize"], 637 *  metadata["xpixelsize"],
                    604 *metadata["ypixelsize"], 1116 * metadata["ypixelsize"])
            metadata["yorigin"] = "lower"
            R, metadata = dimension.clip_domain(R, metadata, extent=bbox)
            #metadata["xpixelsize"] = metadata["ypixelsize"]
            #R, metadata = dimension.aggregate_fields_space(R, metadata, metadata["xpixelsize"]*2.0)
            R[~np.isfinite(R)] = metadata["zerovalue"]
            sys.stdout = open(os.devnull, 'w')
            lk_flow = lk(R[0:4,:,:])
            nowcast = linda(R[1:4,:,:], lk_flow, n_leadtimes,
                                                 num_workers=32,
                                                 use_multiprocessing=True,
                                                 max_num_features=25,
                                                 add_perturbations = False)
            #nowcast[~np.isfinite(nowcast)] = metadata["zerovalue"]
            sys.stdout = sys.__stdout__
            nowcast = io_tools.rainrate_to_dBZ(nowcast)


            for i in save_indexes:
                leadtime_grp = grp.require_group(str(i+1))
                ni_compressed = io_tools.arr_compress_uint8(nowcast[i,:,:])
                what_attrs["gain"] = 0.5
                what_attrs["offset"] = -32
                what_attrs["nodata"] = 255
                what_attrs["undetect"] = 255
                io_tools.write_image(leadtime_grp, "data", ni_compressed, what_attrs)


def _predict_keras(data : np.ndarray, model, n_leadtimes : int) : 

    def scaler(data):
        return np.log(data + 0.01) 
    def invScaler(data):
        return (np.exp(data) - 0.01) 
    
    out = np.empty((n_leadtimes,512,512))
    in_data = scaler(data[:4,:,:])
    in_data = np.swapaxes(in_data[np.newaxis, ...], 1,3)
    in_data = np.swapaxes(in_data, 1, 2)
    
    for i in tqdm(range(n_leadtimes)):
        pred = model.predict(x = in_data)
        out[i,:,:] = invScaler(np.squeeze(pred))
        in_data = np.roll(in_data, -1, axis=3)
        in_data[:,:,:,3] = pred[:,:,:,0]
        
    return out

def process_rainnet_keras_predictions(timestamps : list,
                                    db : h5py.File,
                                    model_path : str,
                                    n_leadtimes : int,
                                    save_indexes : list = None):
    import tensorflow as tf
    import keras

    model = keras.models.load_model(model_path)
    what_attrs = dict()
    what_attrs["quantity"] = 'DBZH'

    if save_indexes is None:
        save_indexes = range(n_leadtimes)

    for t_label in tqdm(timestamps):
        time = datetime.strptime(t_label, '%Y-%m-%d %H:%M:%S')
        t_grp = db.require_group(t_label)
        grp = t_grp.require_group("rainnet_keras")

        if str(t_label + "/" + "rainnet_keras/36/data") not in db:
            try:
                fns = io.archive.find_by_date(
                    time, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_next_files=3)
            except IOError:
                warnings.warn(f"No file for time {t_label}, continuing")
                continue
            try:
                Z, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
            except:
                warnings.warn(f"! An error occured in reading {fns} in... continuing")
                continue
            
            R, metadata = conversion.to_rainrate(Z, metadata)
            bbox = (125 * metadata["xpixelsize"], 637 *  metadata["xpixelsize"],
                    604 *metadata["ypixelsize"], 1116 * metadata["ypixelsize"])
            metadata["yorigin"] = "lower"
            R, metadata = dimension.clip_domain(R, metadata, extent=bbox)
            R[~np.isfinite(R)] = metadata["zerovalue"]
            nowcast = _predict_keras(R, model, n_leadtimes)
            nowcast = io_tools.rainrate_to_dBZ(nowcast)

            for i in save_indexes:
                    leadtime_grp = grp.require_group(str(i+1))
                    ni_compressed = io_tools.arr_compress_uint8(nowcast[i,:,:])
                    what_attrs["gain"] = 0.5
                    what_attrs["offset"] = -32
                    what_attrs["nodata"] = 255
                    what_attrs["undetect"] = 255
                    io_tools.write_image(leadtime_grp, "data", ni_compressed, what_attrs)

def _predict_pytorch(data : torch.Tensor, model, n_leadtimes : int):

    def scaler(data: torch.Tensor):
        return torch.log(data+0.01)
    def invScaler(data: torch.Tensor):
        return torch.exp(data) - 0.01
    
    out = torch.empty((n_leadtimes,256,256))
    in_data = scaler(torch.Tensor(data))
    in_data = in_data[None, ...]

    for i in tqdm(range(n_leadtimes)):
        pred = model(in_data)

        out[i,:,:] = invScaler(pred.squeeze())
        in_data = torch.roll(in_data, -1, dims=0)
        in_data[:,3,:,:] = pred
        
    return out.numpy()

def process_rainnet_pytorch_predictions(timestamps : list,
                                    db : h5py.File,
                                    model_path : str,
                                    n_leadtimes : int,
                                    save_indexes : list = None):
    from models import RainNet
    from utils.config import load_config

    modelconf = load_config("config/athras-bent/rainnet.yaml")
    model = RainNet.load_from_checkpoint(model_path, config=modelconf)
    what_attrs = dict()
    what_attrs["quantity"] = 'DBZH'

    if save_indexes is None:
        save_indexes = range(n_leadtimes)

    for t_label in tqdm(timestamps):
        time = datetime.strptime(t_label, '%Y-%m-%d %H:%M:%S')
        t_grp = db.require_group(t_label)
        grp = t_grp.require_group("rainnet_pytorch")

        if str(t_label + "/" + "rainnet_pytorch/36/data") not in db:
            try:
                fns = io.archive.find_by_date(
                    time, root_path, path_fmt, fn_pattern, fn_ext, timestep, num_next_files=3)
            except IOError:
                warnings.warn(f"No file for time {t_label}, continuing")
                continue
            try:
                Z, _, metadata = io.read_timeseries(fns, importer, **importer_kwargs)
            except:
                warnings.warn(f"! An error occured in reading {fns} in... continuing")
                continue
            
            R, metadata_R = conversion.to_rainrate(Z, metadata)
            bbox = (125 * metadata_R["xpixelsize"], 637 *  metadata_R["xpixelsize"],
                    604 *metadata_R["ypixelsize"], 1116 * metadata_R["ypixelsize"])
            metadata["yorigin"] = "lower"
            R, metadata_R = dimension.clip_domain(R, metadata_R, extent=bbox)
            metadata_R["xpixelsize"] = metadata_R["ypixelsize"]
            R, metadata_R = dimension.aggregate_fields_space(R, metadata_R, metadata_R["xpixelsize"]*2.0)
            with torch.no_grad():
                R[~np.isfinite(R)] = metadata_R["zerovalue"]
                nowcast = _predict_pytorch(R, model, n_leadtimes)
            nowcast = io_tools.rainrate_to_dBZ(nowcast)

            for i in save_indexes:
                    leadtime_grp = grp.require_group(str(i+1))
                    ni_compressed = io_tools.arr_compress_uint8(nowcast[i,:,:])
                    what_attrs["gain"] = 0.5
                    what_attrs["offset"] = -32
                    what_attrs["nodata"] = 255
                    what_attrs["undetect"] = 255
                    io_tools.write_image(leadtime_grp, "data", ni_compressed, what_attrs)




if __name__ == "__main__":

    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("config_path", type=str, help="Configuration file path")
    args = argparser.parse_args()

    with open(args.config_path, "r") as f:
        config = AttrDict(yaml.safe_load(f))

    run(config)