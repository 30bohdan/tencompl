import numpy as np

from functools import partial
from models import ALS_NN, LM_completion
from utils import read_yuv2rgb, read_yuv2gray


def generate_config(
    dataset, ranks, n_frames, 
    dim_y, dim_z, max_iter, lambda_=1,
    n_entries=None, portions=None,
    n_val_entries=10000, n_test_entries=10000,
    methods="Kron-Altmin-LiuMoitra",
    predict_frames=None, noisy=False,
    randominit=True
):
    if portions is None and n_enries is None:
        raise Exception("Config is unvalid")
    
#     if p is not None:
#         n_entries = [int(dim_x*dim_y*dim_z*p) for p in portions]
        
    config = {
        "dataset": dataset,
        "ranks": ranks,
        "n_frames": n_frames,
        "dim_y": dim_y,
        "dim_z": dim_z,
        "portions": portions,
        "n_val_entries": n_val_entries,
        "n_test_entries": n_test_entries,
        "predict_frames": predict_frames,
        "methods": methods,
        "max_iter": max_iter,
        "noisy": noisy,
        "randominit": randominit,
        "lambda": lambda_
    }
    return config


datasets = {
    "artificial":None,
    "akiyo": read_yuv2gray(
        height=144,
        width=176,
        n_frames=300,
        file_name='akiyo_qcif.yuv',
        file_dir='data/'
    ),
}

experiment_configs = {
    "experiment1": generate_config(
        dataset="akiyo",
        methods=["ALS_NN"],
        ranks=[5, 8],
        n_frames=[50, 70],
        dim_y=144,
        dim_z=176,
        portions=[0.02, 0.03, 0.05, 0.075],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[0, 10, 20],
        max_iter=50,
        noisy=False,
        randominit=True,
        lambda_=1
    ),
    "experiment2": generate_config(
        dataset="akiyo",
        methods=["Kron-Altmin-LiuMoitra"],
        ranks=[5, 8],
        n_frames=[50, 70],
        dim_y=144,
        dim_z=176,
        portions=[0.02, 0.03, 0.05, 0.075],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[0, 10, 20],
        max_iter=50,
        noisy=False,
        randominit=True,
        lambda_=1
    ),
    "experiment3": generate_config(
        dataset="akiyo",
        methods=["ALS_NN"],
        ranks=[5, 8],
        n_frames=[50, 70],
        dim_y=144,
        dim_z=176,
        portions=[0.02, 0.03, 0.05, 0.075],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[0, 10, 20],
        max_iter=50,
        noisy=False,
        randominit=False,
        lambda_=1
    ),
    "experiment4": generate_config(
        dataset="akiyo",
        methods=["Kron-Altmin-LiuMoitra", "ALS_NN"],
        ranks=[5, 8],
        n_frames=[50, 70],
        dim_y=144,
        dim_z=176,
        portions=[0.02, 0.03, 0.05, 0.075],
        n_val_entries=5000,
        n_test_entries=5000,
        predict_frames=[0, 10, 20],
        max_iter=50,
        noisy=False,
        randominit=False,
        lambda_=0
    ),
}


solvers = {
    "Kron-Altmin-LiuMoitra": LM_completion,
    "ALS_NN": ALS_NN,
}