import sys, os, time
import random, math
import fire

import numpy as np
import matplotlib.pyplot as plt
from sklearn import decomposition

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

import wandb

from utils import elapsed, get_tensor_entries
import config
from config import experiment_configs

import pdb


def main(experiment="experiment1", seed=13):
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    experiment_config = experiment_configs[experiment]
    method = experiment_config["method"]
    max_iter = experiment_config["max_iter"]
    
    dataset_name = experiment_config["dataset"]
    ranks = experiment_config["ranks"]
    portions = experiment_config["portions"]
    
    n_frames = experiment_config["n_frames"]
    dim_y = experiment_config["dim_y"]
    dim_z = experiment_config["dim_z"]
    
    lambda_ = experiment_config.get("lambda", None)
    n_val_entries = experiment_config["n_val_entries"]
    n_test_entries = experiment_config["n_test_entries"]
    predict_frames = experiment_config["predict_frames"]
    
    dataset_full = config.datasets[dataset_name]
    
    for dim_x in n_frames:
        for rank in ranks:
            for portion in portions:
                n = (dim_x, dim_y, dim_z)
                n_entries = int(dim_x * dim_y * dim_z * portion)
                dataset = dataset_full[:dim_x]
                
                # Set logger
                wandb_configs = {
                    "method": method,
                    "num_frames": dim_x,
                    "height": dim_y,
                    "width": dim_z,
                    "dataset": dataset_name,
                    "rank": rank,
                    "portion": portion,
                    "n_entries": n_entries,
                    "lambda": lambda_,
                    "n_val_entries": n_val_entries,
                    "n_test_entries": n_test_entries,
                    "predict_frames": predict_frames,
                }
                logger = wandb.init(project='tensor-completion', entity='tensor-completion', group=method, reinit=True)
                logger.config.update(wandb_configs)
                run_name =  "frames: {};dataset: {}, rank: {}, portion: {}".format(
                    dim_x, dataset_name, rank, portion
                )
                logger.name = run_name
                
                
                entries_arr = get_tensor_entries(dataset, size=n_entries, seed=seed)
                val_entries = get_tensor_entries(dataset, size=n_val_entries)
                test_entries = get_tensor_entries(dataset, size=n_test_entries)
                
                solver = config.methods[method]
                solution = solver(
                    n, rank, n_entries,
                    max_iter, entries_arr,
                    val_entries, test_entries, logger
                )
                
                predictor = config.predictors[method]
                pred = predictor(solution, predict_frames)
                images = []
                for image, idx_frame in zip(pred, predict_frames):
                    images.append(wandb.Image(image, caption=f"Frame #{idx_frame}"))
                wandb.log({"Visualize prediction:": images})
                

if __name__=="__main__":
    fire.Fire(main)
