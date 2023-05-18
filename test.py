from boptest import * 
import matplotlib.pyplot as plt 

TIMESTEP = 600
EXPERIMENTS_DURATION = 1*24*3600 # 7 jours
simulator = Boptest("bestest_hydronic")


print(simulator.available_measurements)
# Remove all input
simulator.set_activated_input([])

# Set time step
simulator.set_timestep(TIMESTEP)

# Initialize
simulator.initialize(EXPERIMENTS_DURATION, 0.0)


data = list()


for _ in range(int(EXPERIMENTS_DURATION/TIMESTEP)):
    y = simulator.advance(np.array([]))
    data.append(y)

data = np.array(data)

plt.plot(data[:,3])
plt.show()

