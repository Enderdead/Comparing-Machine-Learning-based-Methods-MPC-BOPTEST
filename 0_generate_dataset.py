from boptest import * 
from env import * 
from controller.pid import PID
import matplotlib.pyplot as plt 
import numpy as np 
import argparse
from progress.bar import IncrementalBar
import pandas as pd
import datetime as dt

# Command line argument parser setup
parser = argparse.ArgumentParser(description="cript for dataset generation.")

parser.add_argument("-s", "--sampling_rate", type=int, default=900, help="ampling rate for dataset generation.")
parser.add_argument("-p", "--pwm", type=int, default=3, help="Number of PWM cycle in one time step")
parser.add_argument("--plot", action="store_true", help="Plot the results.", default=False)
args = parser.parse_args()


# Assigning argument values to variables
SAMPLING_RATE = args.sampling_rate
PWM_FREQ      = args.pwm
PLOT          = args.plot


print("Initiate testset simulation")

# Initializing the simulation environment
envBoptest = BestestHydronicPwm("test", SAMPLING_RATE, PWM_FREQ)
init_data, _, _  = envBoptest.reset()

curr_temp = init_data[0,0]
experiment_size = (envBoptest.scenario_bound[1]-envBoptest.scenario_bound[0])/SAMPLING_RATE


# Setpoints and controller initialization
set_point_list = [295.5,296.5]
controller = PID(Kp=300.0/SAMPLING_RATE, Ki=0.0, Kd=0.0, dt=SAMPLING_RATE, setpoint=set_point_list[0])


# Setup for simulation loop
terminated = False
progress_bar =IncrementalBar("Simulation ", suffix='%(percent)d%% [ elapsed time %(elapsed_td)s / ETA %(eta_td)s]', max=experiment_size)

# Main simulation loop

while not terminated:
    i += 1
    u = controller.update(curr_temp)
    terminated, y, forcast, info = envBoptest.step(np.array([u]))
    if terminated: continue

    if info["time"]%(3600*2)<3600:
        controller.setpoint = set_point_list[0]
    else:
        controller.setpoint = set_point_list[1]
    curr_temp = float(y[0,0])
    progress_bar.next()

# Clean up progress bar
del progress_bar


# Extraction and combination of simulation history data
print("Save test_set to csv")

history_y = np.squeeze(np.array([ element[1] for element in envBoptest.history_y][:-1]))
history_u = np.array([ element[1] for element in envBoptest.history_u])
history_t = np.array([ element[0] for element in envBoptest.history_y[:-1]]).reshape(-1, 1)
data_raw = np.concatenate([history_t,history_u, history_y] , axis=1)

data_columns_name = ["t", ] + ["u",] + BestestHydronicPwm._OBS_LIST 
datapd = pd.DataFrame( data=dict( zip(data_columns_name, data_raw.T)))
datapd.set_index("t").to_csv(os.path.join("dataset", "test_set.csv"))



# Plot test set
if PLOT:
    res_data = envBoptest.get_full_data()
    x_index = np.linspace(0,int(20*len(res_data["ovePum_u"])),len(res_data["ovePum_u"]))/60
    extended_u = np.repeat(np.stack(envBoptest.history_u[3:]), int(len(x_index)/len(envBoptest.history_u)) )
    def moving_average(a, n=3) :
        return np.convolve(a, np.ones(n)/n, mode="same")

    toto = moving_average(res_data["reaQHea_y"], n=30*15)

    fig, axs = plt.subplots(2, 1,sharex=True)
    fig.suptitle('Test dataset', fontsize=16)
    axs[0].plot(x_index[len(x_index)-len(extended_u):], extended_u)
    axs_duo = axs[0].twinx()
    axs_duo.plot(x_index, toto,color="red", label="reaQHea_smooth")

    axs[0].legend(loc=0)
    axs_duo.legend(loc=1)
    axs[1].plot(x_index, res_data["oveTSetCoo_u"], label="oveTSetCoo_u")
    axs[1].plot(x_index, res_data["oveTSetHea_u"], label="oveTSetHea_u")
    axs[1].plot(x_index, res_data["reaTRoo_y"], label="reaTRoo_y")
    axs[1].legend()
    plt.show()

# delete environement
del envBoptest

print("Initiate trainset simulation")

# Initializing the simulation environment
envBoptest = BestestHydronicPwm("train_big", SAMPLING_RATE, PWM_FREQ)
init_data, _, _  = envBoptest.reset()

curr_temp = init_data[0,0]
experiment_size = (envBoptest.scenario_bound[1]-envBoptest.scenario_bound[0])/SAMPLING_RATE


# Setpoints and controller initialization
set_point_list = [295.5,296.5]
controller = PID(Kp=300.0/SAMPLING_RATE, Ki=0.0, Kd=0.0, dt=SAMPLING_RATE, setpoint=set_point_list[0])


# Setup for simulation loop
terminated = False
progress_bar =IncrementalBar("Simulation ", suffix='%(percent)d%% [ elapsed time %(elapsed_td)s / ETA %(eta_td)s]', max=experiment_size)

# Main simulation loop
while not terminated:
    u = controller.update(curr_temp)
    terminated, y, forcast, info = envBoptest.step(np.array([u]))
    if terminated: continue
    if info["time"]%(3600*2)<3600:
        controller.setpoint = set_point_list[0]
    else:
        controller.setpoint = set_point_list[1]
    curr_temp = float(y[0,0])
    progress_bar.next()

# Clean up progress bar
progress_bar.clearln()
del progress_bar

# Extraction and combination of simulation history data
print("Save training_set to csv")

history_y = np.squeeze(np.array([ element[1] for element in envBoptest.history_y][:-1]))
history_u = np.array([ element[1] for element in envBoptest.history_u])
history_t = np.array([ element[0] for element in envBoptest.history_y[:-1]]).reshape(-1, 1)
data_raw = np.concatenate([history_t,history_u, history_y] , axis=1)

data_columns_name = ["t", ] + ["u",] + BestestHydronicPwm._OBS_LIST 
datapd = pd.DataFrame( data=dict( zip(data_columns_name, data_raw.T)))
datapd.set_index("t").to_csv(os.path.join("dataset", "training_full.csv"))


# Split to medium and short training
origin_dt =envBoptest._SCENARIO_LIST["test"][0]
end_medium_dataset_dt = envBoptest._SCENARIO_LIST["train_medium"][1]-origin_dt
end_little_dataset_dt = envBoptest._SCENARIO_LIST["train_little"][1]-origin_dt

# TODO Fix bug ratio time
a = envBoptest._SCENARIO_LIST["train_medium"][1] - envBoptest._SCENARIO_LIST["train_medium"][0]
b = envBoptest._SCENARIO_LIST["train_big"][1] - origin_dt

1/0

medium_datapd = datapd[datapd["t"]<=end_medium_dataset_dt.total_seconds()]
little_datapd = datapd[datapd["t"]<=end_little_dataset_dt.total_seconds()]

medium_datapd.set_index("t").to_csv(os.path.join("dataset", "training_medium.csv"))
little_datapd.set_index("t").to_csv(os.path.join("dataset", "training_little.csv"))



if PLOT:
    res_data = envBoptest.get_full_data()
    x_index = np.linspace(0,int(20*len(res_data["ovePum_u"])),len(res_data["ovePum_u"]))/60
    extended_u = np.repeat(np.stack(envBoptest.history_u[3:]), int(len(x_index)/len(envBoptest.history_u)) )
    def moving_average(a, n=3) :
        return np.convolve(a, np.ones(n)/n, mode="same")

    toto = moving_average(res_data["reaQHea_y"], n=30*15)

    fig, axs = plt.subplots(2, 1,sharex=True)

    axs[0].plot(x_index[len(x_index)-len(extended_u):], extended_u)
    axs_duo = axs[0].twinx()
    axs_duo.plot(x_index, toto,color="red", label="reaQHea_smooth")

    axs[0].legend(loc=0)
    axs_duo.legend(loc=1)
    axs[1].plot(x_index, res_data["oveTSetCoo_u"], label="oveTSetCoo_u")
    axs[1].plot(x_index, res_data["oveTSetHea_u"], label="oveTSetHea_u")
    axs[1].plot(x_index, res_data["reaTRoo_y"], label="reaTRoo_y")
    axs[1].legend()
    plt.show()


