from boptest import * 
from env import * 
from controller.pid import PID
import matplotlib.pyplot as plt 
import numpy as np 

SAMPLING_RATE = 900
PWM_FREQ = 3
envBoptest = BestestHydronicPwm("test", SAMPLING_RATE, PWM_FREQ)
init_data  = envBoptest.reset()

curr_temp = init_data[0,0]

set_point_list = [295.5,23.5]


controller = PID(Kp=300.0/SAMPLING_RATE, Ki=0.0, Kd=0.0, dt=SAMPLING_RATE, setpoint=295.5)

terminated = False

while not terminated:
    u = controller.update(curr_temp)
    terminated, y, forcast, info = envBoptest.step(np.array([u]))
    print(info["time"])
    if info["time"]%(3600*2)<3600:
        controller.setpoint = 295.5
        print("A")
    else:
        controller.setpoint = 296.5
        print("B")
    if terminated: continue
    curr_temp = float(y[0,0])
    print("curr_temp : ", curr_temp, " u : ", u)





res_data = envBoptest.get_full_data()

#x_index = np.array(res_data["time"])/60.0
x_index = np.linspace(0,int(20*len(res_data["ovePum_u"])),len(res_data["ovePum_u"]))/600
extended_u = np.repeat(np.stack(envBoptest.history_u[3:]), int(len(x_index)/len(envBoptest.history_u)) )
def moving_average(a, n=3) :
    return np.convolve(a, np.ones(n)/n, mode="same")

toto = moving_average(res_data["reaQHea_y"], n=600)

fig, axs = plt.subplots(2, 1,sharex=True)

axs[0].plot(x_index[len(x_index)-len(extended_u):], extended_u)
axs_duo = axs[0].twinx()
#axs_duo.plot(x_index, res_data["reaQHea_y"],color="orange", label="reaQHea_y")
axs_duo.plot(x_index, toto,color="red", label="reaQHea_smooth")

axs[0].legend(loc=0)
axs_duo.legend(loc=1)
axs[1].plot(x_index, res_data["oveTSetCoo_u"], label="oveTSetCoo_u")
axs[1].plot(x_index, res_data["oveTSetHea_u"], label="oveTSetHea_u")
axs[1].plot(x_index, res_data["reaTRoo_y"], label="reaTRoo_y")
axs[1].legend()
plt.show()


