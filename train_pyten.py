import random
from cv2 import data
import fire

import numpy as np
import wandb

from utils import get_tensor_entries, compute_rse
import config
from config import pyten_configs
from pyten.method import cp_als, tucker_als, silrtc, halrtc, TNCP
from pyten.tenclass import Tensor

import pdb

def train(experiment="pyten_target1", seed=13):
    pyten_config = pyten_configs[experiment]
    
    dataset_name = pyten_config["dataset"]
    ranks = pyten_config["ranks"]
    portions = pyten_config["portions"]
    init = pyten_config["init"]
    
    n_frames = pyten_config["n_frames"]
    dim_y = pyten_config["dim_y"]
    dim_z = pyten_config["dim_z"]
    
    n_val_entries = pyten_config["n_val_entries"]
    n_test_entries = pyten_config["n_test_entries"]
    predict_frames = pyten_config["predict_frames"]
    max_iter = pyten_config["max_iter"]
    
    dataset_full = config.datasets[dataset_name]
    for dim_x in n_frames:
        dataset = dataset_full[:dim_x]
        for rank, true_rank in ranks:
            for portion in portions:
                np.random.seed(seed)
                random.seed(seed)

                n = (dim_x, dim_y, dim_z)
                n_entries = int(dim_x*dim_y*dim_z*portion)
                wandb_config = {
                    "method": "pyten",
                    "num_frames": dim_x,
                    "height": dim_y,
                    "width": dim_z,
                    "dataset": dataset_name,
                    "rank": rank,
                    "true_rank": true_rank,
                    "portion": portion,
                    "n_val_entries": n_val_entries,
                    "n_test_entries": n_test_entries,
                    "predict_frames": predict_frames,
                    "init": init,
                    "max_iter": max_iter,
                }
                group_name = "Dim-{}x{}x{} dataset-{} rank-{} portion-{}".format(
                    dim_x, dim_y, dim_z, dataset_name, rank, portion
                )
                logger = wandb.init(
                    project='tensor-completion', entity='tensor-completion', group=group_name, reinit=True
                    )
                logger.config.update(wandb_config)
                run_name = "method: {}; init: {}".format(
                    "pyten", init
                )
                logger.name = run_name
                
                entries_arr = get_tensor_entries(dataset, size=n_entries)
                val_entries = get_tensor_entries(dataset, size=n_val_entries)
                test_entries = get_tensor_entries(dataset, size=n_test_entries)
                Omega = np.zeros(n)
                Omega[entries_arr[0].astype(np.int), entries_arr[1].astype(np.int), entries_arr[2].astype(np.int)] = 1
                R = [rank, rank, rank]
                data_observed = dataset.copy()
                data_observed[Omega==0] = 0

                # CP ALS
                X = Tensor(data_observed.copy())
                [T1, rX1] = cp_als(X, rank, Omega, maxiter=max_iter, init=init)
                cp_als_rse = compute_rse(rX1.data, dataset, Omega)

                # Tucker ALS
                X = Tensor(data_observed.copy())
                pdb.set_trace()
                [T2, rX2] = tucker_als(X, R, Omega, max_iter=max_iter, init=init)
                tucker_als_rse = compute_rse(rX2.data, dataset, test_entries)

                # Silrtc
                X = Tensor(data_observed.copy())
                rX3 = silrtc(X, Omega, max_iter=max_iter)
                silrtc_rse = compute_rse(rX3.data, dataset, test_entries)

                # Halrtc
                X = Tensor(data_observed.copy())
                rX4 = halrtc(X, Omega, max_iter=max_iter)
                halrtc_rse = compute_rse(rX4.data, dataset, test_entries)

                # TNCP
                X = Tensor(data_observed.copy())
                solver = TNCP(X, Omega, rank=rank, max_iter=max_iter)
                solver.run()
                tncp_rse = compute_rse(solver.X.data, dataset, test_entries)

                columns = ["Init", "CP ALS", "Tucker ALS", "Silrtc", "Halrtc", "TNCP"]
                table = wandb.Table(columns=columns)
                table.add_data(init, cp_als_rse, tucker_als_rse, silrtc_rse, halrtc_rse, tncp_rse)
                logger.log({"rse_errors": table})

                images = []
                for idx_frame in predict_frames:
                    image = rX1.data[idx_frame]
                    images.append(wandb.Image(image, caption="Frame #{}; method: {}; rank: {}; portion:{}".format(
                        idx_frame, "CP ALS", rank, portion)))

                    image = rX2.data[idx_frame]
                    images.append(wandb.Image(image, caption="Frame #{}; method: {}; rank: {}; portion:{}".format(
                        idx_frame, "Tucker ALS", R, portion
                    )))

                    image = rX3.data[idx_frame]
                    images.append(wandb.Image(image, caption="Frame #{}; method: {}; rank: {}; portion:{}".format(
                        idx_frame, "Silrtc", rank, portion
                    )))

                    image = rX4.data[idx_frame]
                    images.append(wandb.Image(image, caption="Frame #{}; method: {}; rank: {}; portion:{}".format(
                        idx_frame, "Halrtc", rank, portion
                    )))

                    image = solver.X.data[idx_frame]
                    images.append(wandb.Image(image, caption="Frame #{}; method: {}; rank: {}; portion:{}".format(
                        idx_frame, "TNCP", rank, portion
                    )))

                logger.finish()
                

if __name__=="__main__":
    fire.Fire(train)

