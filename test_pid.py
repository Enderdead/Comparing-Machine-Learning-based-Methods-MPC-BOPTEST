from boptest import * 
from env import * 
import matplotlib.pyplot as plt 

TIMESTEP = 600
EXPERIMENTS_DURATION = 4*24*3600 # 7 jours

scenario_bound=[1000,1000+EXPERIMENTS_DURATION]
env = MpcEnv("bestest_hydronic", control_list=["ovePum_u", "oveTSetSup_u"], observation_list=["reaTRoo_y", "reaQHea_y"], forcast_list=[], regressive_period=2, timestep=600, scenario_bound=scenario_bound)

env.reset()

freq_PWM = 300
alpha = 0.5
finished = False
curr_day = 0

while not finished:
    if env.t>curr_day*12*3600:
        alpha = np.random.random()
        curr_day+=1
    print("A", env.t)
    env.simulator.set_timestep(freq_PWM*alpha)
    finished, obs, forcast = env.step(np.array([1,353.15]))
    if finished: continue
    env.simulator.set_timestep(freq_PWM*(1-alpha))
    finished, obs, forcast = env.step(np.array([1,288.15]))
    if finished: continue




res_data = env.simulator.get_simulation_data(["time",]+env.simulator.available_input+env.simulator.available_measurements, scenario_bound[0],scenario_bound[1] )

x_index = np.array(res_data["time"])/60.0
#x_index = np.linspace(0,int(20*len(res_data["ovePum_u"])),len(res_data["ovePum_u"]))/600

def moving_average(a, n=3) :
    return np.convolve(a, np.ones(n)/n, mode="same")

toto = moving_average(res_data["reaQHea_y"], n=30)

fig, axs = plt.subplots(2, 1,sharex=True)

axs[0].plot(x_index, res_data["oveTSetSup_u"], label="oveTSetSup_u")
axs_duo = axs[0].twinx()
axs_duo.plot(x_index, res_data["reaQHea_y"],color="orange", label="reaQHea_y")
axs_duo.plot(x_index, toto,color="red", label="reaQHea_smooth")

axs[0].legend(loc=0)
axs_duo.legend(loc=1)
axs[1].plot(x_index, res_data["oveTSetCoo_u"], label="oveTSetCoo_u")
axs[1].plot(x_index, res_data["oveTSetHea_u"], label="oveTSetHea_u")
axs[1].plot(x_index, res_data["reaTRoo_y"], label="reaTRoo_y")
axs[1].legend()
plt.show()
