import numpy as np

from models import ML_completion
from utils import read_yuv2rgb, read_yuv2gray


def generate_config(
    dataset, ranks, n_frames, 
    dim_y, dim_z, max_iter, lambda_=1,
    n_entries=None, portions=None,
    n_val_entries=10000, n_test_entries=10000,
    method="Moitra-subspace-powering",
    predict_frames=None,
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
        "method": method,
        "max_iter": max_iter
    }
    return config


datasets = {
    "akiyo": read_yuv2gray(
        height=144,
        width=176,
        n_frames=300,
        file_name='akiyo_qcif.yuv',
        file_dir='data/'
    )
}

experiment_configs = {
    "experiment1": generate_config(
        dataset="akiyo",
        method="Moitra-subspace-powering",
        ranks=[5, 8],
        n_frames=[50, 70],
        dim_y=144,
        dim_z=176,
        portions=[0.02, 0.03, 0.05, 0.075],
        n_val_entries=10000,
        n_test_entries=10000,
        predict_frames=[0, 10, 20],
        max_iter=50
    ),
}


def moitra_solve(
    n, rank, n_entries,
    max_iter, entries_arr,
    val_entries, test_entries, logger
):
    solver = ML_completion(
        n=n, rank=rank, n_entries=n_entries,
        entries_arr=entries_arr
    )
    return solver.run_for_tensor(max_iter, test_entries, val_entries, logger=logger)


def moitra_predict(solution, predict_frames):
    V_x, V_y, V_z, coeffs = solution
    recovered_frames = []
    for idx_frame in predict_frames:
        recovered_frames.append(ML_completion.recover_frame(idx_frame, coeffs, V_x, V_y, V_z))
    return recovered_frames
    

methods = {
    "Moitra-subspace-powering": moitra_solve,
}

predictors = {
    "Moitra-subspace-powering": moitra_predict,
}
    