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
    "bridge_settings_v1": generate_config(
        dataset="bridge",
        methods=["Kron-Altmin-LiuMoitra", "ALS_NN", "ALS"],
        ranks=[(5, None), (10, None), (15, None), (20, None)],
        n_frames=[50, 75, 100],
        dim_y=144,
        dim_z=176,
        portions=[0.05, 0.075, 0.1, 0.2],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        lambda_=1
    ),
    "bridge_settings_v2": generate_config(
        dataset="bridge",
        methods=["Kron-Altmin-LiuMoitra", "ALS_NN", "ALS"],
        ranks=[(15, None), (20, None)],
        n_frames=[50, 75, 100],
        dim_y=144,
        dim_z=176,
        portions=[0.05, 0.075, 0.1],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        lambda_=1
    ),
    "bridge_settings_v3": generate_config(
        dataset="bridge",
        methods=["Kron-Altmin-LiuMoitra", "ALS_NN", "ALS"],
        ranks=[(5, None), (10, None), (15, None), (20, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.025],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        lambda_=1
    ),
    "bridge_settings_v4": generate_config(
        dataset="bridge",
        methods=["ALS_NN", "ALS"],
        ranks=[(5, None), (10, None), (15, None), (20, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.02],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        lambda_=1
    ),
    "bridge_settings_v5": generate_config(
        dataset="bridge",
        methods=["ALS_NN", "ALS"],
        ranks=[(5, None), (10, None), (15, None), (20, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.015],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        lambda_=1
    ),
    "bridge_settings_v6": generate_config(
        dataset="bridge",
        methods=["Kron-Altmin-LiuMoitra"],
        ranks=[(5, None), (10, None), (15, None), (20, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.015],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        lambda_=1
    ),
    "bridge_settings_v7": generate_config(
        dataset="bridge",
        methods=["Kron-Altmin-LiuMoitra", "ALS_NN", "ALS"],
        ranks=[(20, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.05, 0.075, 0.1, 0.2],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        lambda_=1
    ),
    "bridge_settings_v7": generate_config(
        dataset="bridge",
        methods=["Kron-Altmin-LiuMoitra", "ALS_NN", "ALS"],
        ranks=[(5, None)],
        n_frames=[50],
        dim_y=144,
        dim_z=176,
        portions=[0.02],
        n_val_entries=5000,
        n_test_entries=10000,
        predict_frames=[10, 20, 30],
        max_iter=75,
        noisy=None,
        randominit=True,
        lambda_=1
    ),

}

pyten_configs = {
    "bridge_settings_v1": {
        "dataset": "bridge",
        "ranks": [(5, None), (10, None), (15, None), (20, None)],
        "portions": [0.05, 0.075, 0.1, 0.2],
        "init": "eigs",
        "n_frames": [50, 75, 100],
        "dim_y": 144,
        "dim_z": 176,
        "n_val_entries": 5000,
        "n_test_entries": 10000,
        "predict_frames": [10, 20, 30],
        "max_iter": 75,
    },
    "bridge_settings_v2": {
        "dataset": "bridge",
        "ranks": [(5, None), (10, None), (15, None), (20, None)],
        "portions": [0.015],
        "init": "eigs",
        "n_frames": [50],
        "dim_y": 144,
        "dim_z": 176,
        "n_val_entries": 5000,
        "n_test_entries": 10000,
        "predict_frames": [10, 20, 30],
        "max_iter": 75,
    },
    "bridge_settings_v3": {
        "dataset": "bridge",
        "ranks": [(5, None),],
        "portions": [0.02],
        "init": "eigs",
        "n_frames": [50],
        "dim_y": 144,
        "dim_z": 176,
        "n_val_entries": 5000,
        "n_test_entries": 10000,
        "predict_frames": [10, 20, 30],
        "max_iter": 75,
    },
}

# experiment_configs = {
#     "experiment1": generate_config(
#         dataset="akiyo",
#         methods=["ALS_NN"],
#         ranks=[(5, None), (8, None)],
#         n_frames=[50, 70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.02, 0.03, 0.05, 0.075],
#         n_val_entries=5000,
#         n_test_entries=10000,
#         predict_frames=[0, 10, 20],
#         max_iter=50,
#         noisy=None,
#         randominit=True,
#         lambda_=1
#     ),
#     "experiment2": generate_config(
#         dataset="akiyo",
#         methods=["Kron-Altmin-LiuMoitra"],
#         ranks=[(5, None), (8, None)],
#         n_frames=[50, 70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.02, 0.03, 0.05, 0.075],
#         n_val_entries=5000,
#         n_test_entries=10000,
#         predict_frames=[0, 10, 20],
#         max_iter=50,
#         noisy=None,
#         randominit=True,
#         lambda_=1
#     ),
#     "experiment3": generate_config(
#         dataset="akiyo",
#         methods=["ALS_NN"],
#         ranks=[(5, None), (8, None)],
#         n_frames=[50, 70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.02, 0.03, 0.05, 0.075],
#         n_val_entries=5000,
#         n_test_entries=10000,
#         predict_frames=[0, 10, 20],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "experiment4": generate_config(
#         dataset="akiyo",
#         methods=["Kron-Altmin-LiuMoitra", "ALS_NN"],
#         ranks=[(5, None), (8, None)],
#         n_frames=[50, 70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.02, 0.03, 0.05, 0.075],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 10, 20],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target1": generate_config(
#         dataset="akiyo",
#         methods=["Kron-Altmin-LiuMoitra", "ALS_NN", "ALS"],
#         ranks=[(5, None), (8, None)],
#         n_frames=[50, 70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.02, 0.03, 0.05, 0.075, 0.1],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target2": generate_config(
#         dataset="artificial",
#         methods=["Kron-Altmin-LiuMoitra", "ALS_NN", "ALS"],
#         ranks=[(10, 10), (15, 10), (20, 10)],
#         n_frames=[100],
#         dim_y=100,
#         dim_z=100,
#         portions=[0.02, 0.025, 0.03, 0.035, 0.04],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target3": generate_config(
#         dataset="artificial",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(15, 15), (20, 15), (25, 15)],
#         n_frames=[100],
#         dim_y=100,
#         dim_z=100,
#         portions=[0.02, 0.025, 0.03, 0.035, 0.04],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target4": generate_config(
#         dataset="akiyo",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(15, None), (20, None), (25, None)],
#         n_frames=[50, 70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.3],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target5": generate_config(
#         dataset="artificial",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(15, 10), (20, 10)],
#         n_frames=[100],
#         dim_y=100,
#         dim_z=100,
#         portions=[0.02, 0.025, 0.03, 0.035, 0.04],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=50,
#         noisy=None,
#         randominit=True,
#         lambda_=1
#     ),
#     "target6": generate_config(
#         dataset="artificial",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(15, 15), (20, 20), (20, 15), (25, 20)],
#         n_frames=[200],
#         dim_y=150,
#         dim_z=100,
#         portions=[0.03, 0.035, 0.04, 0.045, 0.05],
#         n_val_entries=5000,
#         n_test_entries=15000,
#         predict_frames=[0],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target7": generate_config(
#         dataset="artificial",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(70, 70), (80, 70), (70, 60)],
#         n_frames=[50],
#         dim_y=50,
#         dim_z=50,
#         portions=[0.15, 0.2, 0.225, 0.25, 0.3],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=50,
#         noisy=None,
#         randominit=True,
#         lambda_=1
#     ),
#     #TODO
#     "target8": generate_config(
#         dataset="artificial",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(10, 10), (15, 10), (20, 20), (25, 20)],
#         n_frames=[100],
#         dim_y=100,
#         dim_z=100,
#         portions=[0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=100,
#         noisy=0.1,
#         randominit=True,
#         lambda_=1
#     ),
#     "target9": generate_config(
#         dataset="artificial",
#         methods=["Kron-Altmin-LiuMoitra"],
#         ranks=[(10, 10), (15, 10)],
#         n_frames=[100],
#         dim_y=100,
#         dim_z=100,
#         portions=[0.05, 0.075, 0.1, 0.125, 0.15, 0.175],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=100,
#         noisy=0.1,
#         randominit=True,
#         lambda_=1
#     ),
#     #TODO
#     "target10": generate_config(
#         dataset="artificial",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(70, 70), (80, 70), (70, 60)],
#         n_frames=[50],
#         dim_y=50,
#         dim_z=50,
#         portions=[0.15, 0.2, 0.225, 0.25, 0.3],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=100,
#         noisy=0.5,
#         randominit=True,
#         lambda_=1
#     ),
#     #TODO
#     "target11": generate_config(
#         dataset="bus",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(15, None), (20, None), (25, None)],
#         n_frames=[50, 70],
#         dim_y=288,
#         dim_z=352,
#         portions=[0.01, 0.02, 0.03, 0.05, 0.075, 0.1],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=75,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target12": generate_config(
#         dataset="bus",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(30, None), (40, None), (50, None)],
#         n_frames=[50, 70],
#         dim_y=288,
#         dim_z=352,
#         portions=[0.2, 0.3, 0.4],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=75,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target13": generate_config(
#         dataset="bridge",
#         methods=["Kron-Altmin-LiuMoitra", "ALS_NN", "ALS"],
#         ranks=[(5, None), (8, None)],
#         n_frames=[50, 70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.02, 0.03, 0.05, 0.075, 0.1],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target14": generate_config(
#         dataset="bridge",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(20, None), (25, None), (30, None)],
#         n_frames=[70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.05, 0.075, 0.15, 0.2, 0.25, 0.275, 0.3],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target15": generate_config(
#         dataset="bridge",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(40, None), (50, None)],
#         n_frames=[70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.05, 0.075, 0.15, 0.2, 0.25, 0.275, 0.3],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0, 20, -20, -1],
#         max_iter=50,
#         noisy=None,
#         randominit=False,
#         lambda_=1
#     ),
#     "target16": generate_config(
#         dataset="artificial",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(100, 100), (150, 150), (110, 100), (160, 150)],
#         n_frames=[70],
#         dim_y=70,
#         dim_z=70,
#         portions=[0.2, 0.3, 0.4],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=100,
#         noisy=0.5,
#         randominit=True,
#         lambda_=1
#     ),
#     "target17": generate_config(
#         dataset="artificial",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(110, 100), (160, 150)],
#         n_frames=[70],
#         dim_y=70,
#         dim_z=70,
#         portions=[0.2, 0.3, 0.4],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=100,
#         noisy=None,
#         randominit=True,
#         lambda_=1
#     ),
#     "target18": generate_config(
#         dataset="akiyo",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(25, None), (30, None)],
#         n_frames=[70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.025, 0.05, 0.075],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=50,
#         noisy=None,
#         randominit=True,
#         lambda_=1,
#         fix_mu=True
#     ),
#     "target19": generate_config(
#         dataset="akiyo",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(25, None), (30, None)],
#         n_frames=[70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.025, 0.05, 0.075],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=50,
#         noisy=None,
#         randominit=True,
#         lambda_=1,
#         fix_mu=False,
#         momentum=0.8
#     ),

#     "final": generate_config(
#         dataset="akiyo",
#         methods=["ALS_NN", "ALS"],
#         ranks=[(25, None), (30, None)],
#         n_frames=[70],
#         dim_y=144,
#         dim_z=176,
#         portions=[0.025, 0.05, 0.075],
#         n_val_entries=5000,
#         n_test_entries=5000,
#         predict_frames=[0],
#         max_iter=50,
#         noisy=None,
#         randominit=True,
#         lambda_=1,
#         fix_mu=False,
#         momentum=0.8
#     ),
# }

# pyten_configs = {
#     "target1": {
#         "dataset": "akiyo",
#         "ranks": [(20, None)],
#         "portions": [0.05, 0.075],
#         "init": "eigs",
#         "n_frames": [70],
#         "dim_y": 144,
#         "dim_z": 176,
#         "n_val_entries": 5000,
#         "n_test_entries": 5000,
#         "predict_frames": [0],
#         "max_iter": 50,
#     },

#     "test1": {
#         "dataset": "akiyo",
#         "ranks": [(25, None)],
#         "portions": [0.025, 0.05, 0.075],
#         "init": "random",
#         "n_frames": [70],
#         "dim_y": 144,
#         "dim_z": 176,
#         "n_val_entries": 5000,
#         "n_test_entries": 5000,
#         "predict_frames": [0],
#         "max_iter": 100,
#     },
# }