import numpy as np 
from scipy import interpolate
from boptest import *


def dict_to_matrix(d, column_names):
    return np.column_stack([d[name] for name in column_names if name in d])



class MpcEnv():
    # TODO ajouter prix volatilité
    # TODO ajouter le warmup mecanisme
    # TODO revoir le shift des données de controle
    def __init__(self, testcase_name, control_list, observation_list, forcast_list, timestep, regressive_period = 1, predictive_period=1, scenario_name=None, scenario_bound=None, remote_url=None):
        assert remote_url is None, "Remote Boptest is not available yet !"
        
        self.simulator = Boptest(testcase_name)
        self.timestep = timestep

        self.simulator.set_activated_input(control_list)
        self.simulator.set_activated_measurements(observation_list)
        self.simulator.set_timestep(timestep)

        self.history_y = list()
        self.history_u = list()

        assert not (scenario_bound is None and scenario_name is None), "Please provide either scenario name or scenario bound"
        self.scenario_bound = scenario_bound
        self.scenario_name = scenario_name
        self.experience_duration = 14*24*3600/self.timestep if not scenario_name is None else scenario_bound[1]-scenario_bound[0]
        self.t = 0.0
        self.regressive_period = regressive_period
        self.predictive_period = predictive_period
        self._ready = False



    def reset(self):
        if not self.scenario_name is None:
            info = self.simulator.set_scenario(time_period=self.scenario_name)
            print(info)
            self.t = info["time_period"]["time"]
            self.scenario_bound = [self.t, self.experience_duration]
        else:
            self.simulator.initialize(self.scenario_bound[0]+self.regressive_period*self.timestep, self.regressive_period*self.timestep)
            self.t = self.scenario_bound[0]+self.regressive_period*self.timestep

        
        # récup les infos
        past_obs  = self._get_observation(self.t, win_size=self.regressive_period) 
        past_ctrl = self._get_control(self.t, win_size=self.regressive_period-1)
        self.history_y, self.history_u = list(past_obs), list(past_ctrl)
        self._ready = True
        return self._get_observation(self.t, win_size=self.regressive_period)


    def _get_control(self, end_time, win_size):
        if win_size==0:
            return list()
        input_dict = self.simulator.get_simulation_data(self.simulator.activated_input, end_time-win_size*self.timestep, end_time)
        target_time_index = np.linspace(end_time - self.timestep*(win_size-1), end_time, win_size)
        res_dict = {"time" : target_time_index}
        for key, value in input_dict.items():

            f = interpolate.interp1d(input_dict['time'],
                    value, kind='linear', fill_value='extrapolate')
            res_dict[key] = f(target_time_index)

        return dict_to_matrix(res_dict, self.simulator.activated_input)


    def _get_observation(self, end_time, win_size):
        input_dict = self.simulator.get_simulation_data(self.simulator.activated_measurements, end_time-win_size*self.timestep, end_time)
        target_time_index = np.linspace(end_time - self.timestep*(win_size-1), end_time, win_size)
        res_dict = {"time" : target_time_index}
        for key, value in input_dict.items():

            f = interpolate.interp1d(input_dict['time'],
                    value, kind='linear', fill_value='extrapolate')
            res_dict[key] = f(target_time_index)

        return dict_to_matrix(res_dict, self.simulator.activated_measurements)

    def _get_forcast(self):
        return np.array([])

    def step(self, u):
        assert self._ready, "Please initiate the simulator"
        msr_array, info = self.simulator.advance(u)

        if msr_array is None:
            self._ready = False
            return True, None, None

        self.t = info["time"]
        self.history_u.append(u)
        self.history_y.append(dict_to_matrix(info, self.simulator.activated_measurements))

        if self.scenario_bound[1]<self.t:
            return True, None, None

        self.forcast = self._get_forcast()
        return False, dict_to_matrix(info, self.simulator.activated_measurements), self.forcast

    def get_full_data(self):
        return self.simulator.get_simulation_data(["time",]+self.simulator.available_input+env.simulator.available_measurements, self.scenario_bound[0],self.scenario_bound[1] )



class BestestHydronicPwm(MpcEnv):
    SAMPLING_RATE = 900
    PWM_PERIOD = 300
    _OBS_DESCRIPTION = ["Température interieur", "Température exterieur", "normal radiation (sky cover included)","sky cover", "humidity", "elevation_soleil"]#"heur_cos", "heure_sin","saison_cos", "saison_sin"]
    _OBS_LIST = ["reaTRoo_y", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y", "weaSta_reaWeaNTot_y", "weaSta_reaWeaRelHum_y", "weaSta_reaWeaSolAlt_y"]
    _CTRL_LIST = ["ovePum_u", "oveTSetSup_u"]
    _SCENARIO_LIST = ["test":[], "train_little", "train_medium", "train_big"]

    def __init__(self):
        pass    
