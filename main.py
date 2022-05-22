import random
import fire

import numpy as np
import wandb

from utils import get_tensor_entries
import config
from config import experiment_configs

import pdb


def main(experiment="experiment1", seed=13, num_run=None):
    experiment_config = experiment_configs[experiment]
    methods = experiment_config["methods"]
    max_iter = experiment_config["max_iter"]
    
    dataset_name = experiment_config["dataset"]
    ranks = experiment_config["ranks"]
    portions = experiment_config["portions"]
    noisy = experiment_config["noisy"]
    randominit = experiment_config["randominit"]
    
    n_frames = experiment_config["n_frames"]
    dim_y = experiment_config["dim_y"]
    dim_z = experiment_config["dim_z"]
    
    lambda_ = experiment_config.get("lambda", None)
    fix_mu = experiment_config.get("fix_mu", False)
    momentum = experiment_config.get("momentum", None)
    n_val_entries = experiment_config["n_val_entries"]
    n_test_entries = experiment_config["n_test_entries"]
    predict_frames = experiment_config["predict_frames"]
    
    dataset_full = config.datasets[dataset_name]
    
    
    
    for dim_x in n_frames:
        if dataset_full is not None:
            dataset = dataset_full[:dim_x]
        else:
            dataset = None
        for rank, true_rank in ranks:
            for portion in portions:
                for method in methods:
                    # Set seeds
                    seed_solver = None
                    if num_run is None:
                        np.random.seed(num_run)
                        seed = np.random.randint(int(1e6))
                        seed_solver = np.random.randint(int(1e6))
                    np.random.seed(seed)
                    random.seed(seed)
                    
                    if method=="ALS":
                        lambda_ = 0
                    else:
                        lambda_ = experiment_config.get("lambda", None)
                    n = (dim_x, dim_y, dim_z)
                    n_entries = int(dim_x * dim_y * dim_z * portion)

                    # Set logger config
                    wandb_configs = {
                        "methods": method,
                        "num_frames": dim_x,
                        "height": dim_y,
                        "width": dim_z,
                        "dataset": dataset_name,
                        "rank": rank,
                        "true_rank": true_rank,
                        "portion": portion,
                        "n_entries": n_entries,
                        "lambda": lambda_,
                        "n_val_entries": n_val_entries,
                        "n_test_entries": n_test_entries,
                        "predict_frames": predict_frames,
                        "noisy": noisy,
                        "randominit": randominit,
                        "fix_mu": fix_mu,
                        "momentum": momentum,
                        "seed": seed,
                        "seed_solver":seed_solver,
                        "num_run": num_run,
                        "stage": "final"
                    }
                    group_name = "Dim-{}x{}x{} dataset-{} rank-{} portion-{}".format(dim_x, dim_y, dim_z, dataset_name, rank, portion)
                    if dataset is None:
                        group_name += " true_rank-{}".format(true_rank)
                    logger = wandb.init(project='tensor-completion', entity='ykivva', group=group_name, reinit=True)
                    logger.config.update(wandb_configs)
                    run_name =  "method: {}; randinit:{}; noisy:{}".format(
                        method, randominit, noisy
                    )
                    logger.name = run_name

                    entries_arr = get_tensor_entries(dataset, size=n_entries)
                    val_entries = get_tensor_entries(dataset, size=n_val_entries)
                    test_entries = get_tensor_entries(dataset, size=n_test_entries)
                    
                    solver = config.solvers[method]
                    
                    #init data
                    solver = solver(
                        n=n, rank=rank, n_entries=n_entries,
                        entries_arr=entries_arr, noisy=noisy,
                        randominit=randominit, true_rank=true_rank,
                        seed=seed_solver
                    )
                    
                    if dataset is None:
                        val_entries = solver.get_entries(n_val_entries)
                        test_entries = solver.get_entries(n_test_entries)

                    solution = solver.fit(
                        test_entries=test_entries,
                        val_entries=val_entries,
                        logger=logger, max_iter=max_iter,
                        lam=lambda_, fix_mu=fix_mu,
                        momentum=momentum
                    )

                    pred = solver.predict(solution, predict_frames)
                    images = []
                    for image, idx_frame in zip(pred, predict_frames):
                        images.append(
                            wandb.Image(
                                image, caption="Frame #{}; method: {}; rank: {}; portion:{};".format(idx_frame, method, rank, portion)))
                    logger.log({"Visualize prediction:": images})
                    logger.finish()


if __name__=="__main__":
    fire.Fire(main)
