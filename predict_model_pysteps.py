"""
    This script will run nowcasting predictions
    for the pytorch RainNet implementation 
"""
import argparse
from attrdict import AttrDict
import yaml
from typing import Sequence
import argparse
from pathlib import Path 

import h5py
from tqdm import tqdm

from verification.pincast_verif import PytorchIterativePrediction
from verification.pincast_verif import io_tools


def run(builders : Sequence[PytorchIterativePrediction]) -> None:
    
    date_paths = [builder.date_path for builder in builders]
    if any(path != date_paths[0] for path in date_paths):
        raise ValueError("The datelists used must be the same for all runs,\
                        Please check that the paths given match.")

    timesteps = io_tools.read_file(date_paths[0])
    output_dbs = [h5py.File(builder.hdf5_path, 'a') 
                  for builder in builders]

    for t in tqdm(timesteps):
        for i, builder in enumerate(builders):
            
            group_name = builder.save_params.group_format.format(
                timestamp = io_tools.get_neighbor(time=t, distance=builder.input_params.num_next_files),
                method = builder.nowcast_params.nowcast_method
            )
            group = output_dbs[i].require_group(group_name)
            if len(group.keys()) == builder.nowcast_params.n_leadtimes:
                continue
            try:
                nowcast = builder.run(t)
                builder.save(nowcast=nowcast,group=group,
                        save_parameters=builder.save_params)
            except IOError:
                print("IO error")
                continue
            except ValueError:
                print("Value error")
                continue

    for db in output_dbs:
        db.close()


def load_config(path : str):
    with open(path, "r") as f:
        config = AttrDict(yaml.safe_load(f))
    return config


if __name__ == "__main__" : 

    argparser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    argparser.add_argument("config", type=str, help=
                           "Configuration folder path, contains \
                           one YAML configuration file per forecast \
                           type that is to be computed."
                           )
    args = argparser.parse_args()

    config_dir = Path("config") / args.config
    config_filenames = config_dir.glob("*.yaml")
    configurations = [load_config(filename) for filename in config_filenames]
    for c in config_filenames:
        print(f"Loaded {c} from {config_dir}.")
    predictor_builders = [PytorchIterativePrediction(config = config)
                         for config in configurations]
    run(builders=predictor_builders)
