import numpy as np 
import tensorflow as tf
import pyNeuralEMPC as nEMPC
import os
import argparse
from controller import *
from progress.bar import IncrementalBar

from boptest import * 
from env import * 

# Define the argument parser
parser = argparse.ArgumentParser(description="Choose a model between linear and convex neural network.")

# Add the 'model' argument
parser.add_argument("-m", "--model", type=str, choices=["linear", "convex_nn"],
                    required=True, help="The model to be used for analysis. Choose between 'linear' and 'convex_nn'")
parser.add_argument("-p", "--path", type=str,
                    help="The path to the model file")
args = parser.parse_args()

HORIZON = 10
SAMPLING_RATE = 900
PWM_FREQ = 3
# Load model 
MODEL_PATH = None

if args.path is None:
    MODEL_PATH = os.path.join("_cache_models", "arx_model") if args.model=="linear" else  os.path.join("_cache_models", "convex_model")
else:
    MODEL_PATH = args.path




# Initializing the simulation environment
envBoptest = BestestHydronicPwm("test", SAMPLING_RATE, PWM_FREQ)
init_y, init_u, init_forcast  = envBoptest.reset()
experiment_size = (envBoptest.scenario_bound[1]-envBoptest.scenario_bound[0])/SAMPLING_RATE
curr_temp = init_y[-1,:]

# Initializing the controller
mpc_controller = ARXMpc(MODEL_PATH, HORIZON) if args.model=="linear" else CNNMpc(MODEL_PATH, HORIZON)
mpc_controller.reset(prev_y=init_y[:-1, :], prev_u=init_u[:, :])
1/0

# Setup for simulation loop
terminated = False
progress_bar = IncrementalBar("Simulation ", suffix='%(percent)d%% [ elapsed time %(elapsed_td)s / ETA %(eta_td)s]', max=experiment_size)

# Main simulation loop
i = 0
forcast = init_forcast
while not terminated:
    i += 1
    u = mpc_controller.update(curr_temp, forcast)
    terminated, y, forcast, info = envBoptest.step(np.array([u]))
    if terminated: continue


    curr_temp = float(y[0,0])
    progress_bar.next()

# Clean up progress bar
progress_bar.clearln()
del progress_bar