# Tensor Completion
##Introduction
In our code we use implemention of the Kron. AltMin algorithm from the https://github.com/cliu568/tensor_completion. This repository contains code with implemented Kron. AltMin algorithm used in the work Allen Liu and Ankur Moitra. Tensor completion made practical.

Also we use HALRTC, TNCP, Silrtc, Tucker ALS algorithms implemented in https://github.com/datamllab/pyten.

## Quickstart

- Set up environment to run ALS, ALS_NN, Kron. AltMin experiments.
  -  Create separate virtual environment: ```conda create -n tencomp python=3.9```
  -  Install all dependencies (numpy, wandb, opencv, fire, scipy, scikit-learn, matplotlib)

- Set up environment to run HALRTC, TNCP, Silrtc, Tucker ALS algorithms.
  - Create separate virtual environment: ```conda create -n tencomp python=2.7```. **Make sure to install correct version of python.**
  -  Install all dependencies (numpy, fire, scipy, scikit-learn, matplotlib pyspark)
  -  Install any version of wandb and opencv compatible with python2.7. For example, wandb of version 0.10.19 and  opencv of version 4.2.0.32 from https://pypi.org/project/opencv-python/4.2.0.32/
  -  Install pyten library from https://github.com/datamllab/pyten by the following command:
     
     ```pip install git+https://github.com/datamllab/pyten.git```
     
 - To run an experiments one can use next commands:

   ```python main.py --experiment="experiment_name" --seed=13```
   
   for ALS, ALS_NN, Kron. AltMin algorithms, and:
   
   ```python train_pyten.py --experiment="experiment_name"```
   
   Also you can look into ```run_visual.sh``` for the example of ```"experiment_name"```.
   
 - ```"experiment_name"``` is responsible for the config defined in the file ```config.py``` where one can create any setting and run the algorithms for them 
   
   
 
     
     

