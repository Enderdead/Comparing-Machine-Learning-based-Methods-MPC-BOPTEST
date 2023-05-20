from boptest import * 
import matplotlib.pyplot as plt 

TIMESTEP = 600
EXPERIMENTS_DURATION = 40*24*3600 # 7 jours
simulator = Boptest("bestest_hydronic")


print(simulator.available_measurements)
# Remove all input
simulator.set_activated_input([])

# Set time step
simulator.set_timestep(TIMESTEP)


# Set Scenario 
#simulator.set_scenario(electricity_price="constant", time_period="peak_heat_day")

# Initialize
simulator.initialize(1200.0, 1200.0)
1/0

data = list()


for _ in range(int(EXPERIMENTS_DURATION/TIMESTEP)):
    y, _ = simulator.advance(np.array([]))
    if y is None:
        break
    data.append(y)

data = np.array(data)



#res_data = simulator.get_simulation_data(["time",]+simulator.available_input + simulator.available_measurements,0,EXPERIMENTS_DURATION)


res_data = simulator.get_simulation_data(["time",]+simulator.available_input+simulator.available_measurements, 311*24*3600,325*24*3600 )

#x_index = np.array(res_data["time"])/60.0
x_index = np.linspace(0,int(20*len(res_data["ovePum_u"])),len(res_data["ovePum_u"]))/600
fig, axs = plt.subplots(2, 1,sharex=True)

axs[0].plot(x_index, res_data["oveTSetSup_u"], label="oveTSetSup_u")
axs_duo = axs[0].twinx()
axs_duo.plot(x_index, res_data["reaQHea_y"],color="orange", label="reaQHea_y")
axs[0].legend(loc=0)
axs_duo.legend(loc=1)
axs[1].plot(x_index, res_data["oveTSetCoo_u"], label="oveTSetCoo_u")
axs[1].plot(x_index, res_data["oveTSetHea_u"], label="oveTSetHea_u")
axs[1].plot(x_index, res_data["reaTRoo_y"], label="reaTRoo_y")
axs[1].legend()
plt.show()


plt.plot(res_data["nTot"])
#plt.plot(res_data["weaSta_reaWeaSolTim_y"])
plt.show()
