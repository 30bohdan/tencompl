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
    predict_frames=None, noisy=None,
    randominit=True, true_rank=None,
    fix_mu=False, momentum=None
):
    if portions is None and n_enries is None:
        raise Exception("Config is unvalid")
        
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
        "lambda": lambda_,
        "fix_mu": fix_mu,
        "momentum": momentum,
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
    "bus": read_yuv2gray(
        height=288,
        width=352,
        n_frames=150,
        file_name='bus_cif.yuv',
        file_dir='data/'
    ),
    "bridge": read_yuv2gray(
        height=144,
        width=176,
        n_frames=2001,
        file_name='bridge-close_qcif.yuv',
        file_dir='data/'
    ),
}

solvers = {
    "Kron-Altmin-LiuMoitra": LM_completion,
    "ALS_NN": ALS_NN,
    "ALS": ALS_NN,
}

experiment_configs = {
    "akiyo_settings_v1": generate_config(
        dataset="akiyo",
        methods=["ALS", "ALS_NN", "Kron-Altmin-LiuMoitra"],
        ranks=[(10, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.05, 0.2],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.9,
        lambda_=1
    ),
    "bridge_settings_v1": generate_config(
        dataset="bridge",
        methods=["ALS", "ALS_NN"],
        ranks=[(5, None), (10, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.02, 0.05, 0.1, 0.2],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.9,
        lambda_=1
    ),
    "bridge_settings_v2": generate_config(
        dataset="bridge",
        methods=["ALS", "ALS_NN"],
        ranks=[(15, None), (20, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.05, 0.1, 0.2],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.9,
        lambda_=1
    ),
    "artificial_undercomplete_5": generate_config(
        dataset="artificial",
        methods=["ALS", "ALS_NN"],
        ranks=[(5, 5)],
        n_frames=[50],
        dim_y=50,
        dim_z=50,
        portions=[0.04, 0.06],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.75,
        lambda_=1
    ),
    "artificial_undercomplete_10": generate_config(
        dataset="artificial",
        methods=["ALS", "ALS_NN"],
        ranks=[(10, 10)],
        n_frames=[50],
        dim_y=50,
        dim_z=50,
        portions=[0.08, 0.12],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.75,
        lambda_=1
    ),
    "artificial_undercomplete_20": generate_config(
        dataset="artificial",
        methods=["ALS", "ALS_NN"],
        ranks=[(20, 20)],
        n_frames=[50],
        dim_y=50,
        dim_z=50,
        portions=[0.08, 0.15],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.7,
        lambda_=1
    ),
    "artificial_undercomplete_200": generate_config(
        dataset="artificial",
        methods=["ALS", "ALS_NN"],
        ranks=[(20, 20)],
        n_frames=[200],
        dim_y=200,
        dim_z=200,
        portions=[0.02, 0.03],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.75,
        lambda_=1
    ),
    "artificial_overcomplete_50": generate_config(
        dataset="artificial",
        methods=["ALS", "ALS_NN"],
        ranks=[(50, 50), (75, 50)],
        n_frames=[50],
        dim_y=50,
        dim_z=50,
        portions=[0.15, 0.2],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.9,
        lambda_=1
    ),
    "artificial_overcomplete_75": generate_config(
        dataset="artificial",
        methods=["ALS", "ALS_NN"],
        ranks=[(75, 75), (100, 75)],
        n_frames=[75],
        dim_y=75,
        dim_z=75,
        portions=[0.15, 0.2, 0.3],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        momentum=0.9,
        lambda_=1
    ),
}

pyten_configs = {
    "bridge_settings": {
        "dataset": "akiyo",
        "ranks": [(5, None), (10, None), (15, None), (20, None)],
        "portions": [0.02, 0.05, 0.1, 0.2],
        "init": "random",
        "n_frames": [50],
        "dim_y": 144,
        "dim_z": 176,
        "n_val_entries": 5000,
        "n_test_entries": 10000,
        "predict_frames": [10, 20, 30],
        "max_iter": 75,
    },
    "akiyo_settings": {
        "dataset": "akiyo",
        "ranks": [(10, None)],
        "portions": [0.05, 0.2],
        "init": "random",
        "n_frames": [50],
        "dim_y": 144,
        "dim_z": 176,
        "n_val_entries": 5000,
        "n_test_entries": 10000,
        "predict_frames": [10, 20, 30],
        "max_iter": 75,
    },
}