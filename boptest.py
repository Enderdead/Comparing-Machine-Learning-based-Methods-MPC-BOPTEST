import socket
import os
import sys
import requests
import subprocess
import random
import time
import numpy as np
from contextlib import closing
import copy
from fuzzywuzzy import process
from project1_boptest.examples.python.controllers.controller import Controller
from project1_boptest.examples.python.custom_kpi.custom_kpi_calculator import CustomKPI

def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def _get_all_testcases():
    testcases_list = os.listdir(os.path.join("project1_boptest", "testcases"))
    return list(filter( lambda x: os.path.isdir(os.path.join("project1_boptest", "testcases", x)), testcases_list))


def _check_response(response):
    if isinstance(response, requests.Response):
        status = response.status_code
    if status == 200:
        response = response.json()['payload']
        return response
    print("Unexpected error: {}".format(response.text))
    print("Exiting!")
    sys.exit()

def _run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()

    if process.returncode != 0:
        print(f"Unexpected error:\n {stderr.decode('utf-8')}")
        print("Exiting!")
        sys.exit()


def _try_function(function, max_attempts, delay):
    attempts = 0
    while attempts < max_attempts:
        try:
            return function()
        except:
            pass
        attempts += 1
        if attempts < max_attempts:
            time.sleep(delay)
        print(".", end="", flush=True)
    return None


class Boptest:
    _DOCKER_IMAGE_NAME = "boptest_base_cmp"

    def __init__(self, testcase_name, basename='127.0.0.1', port=None):
        assert testcase_name in _get_all_testcases()
        
        self.port = _find_free_port() if port is None else port
        self.testcase_name  = testcase_name

        self.url = f"http://{basename}:{self.port}"
        self._random_tag = f"boptest_base_{random.randint(0,1000)}"

        self._start_server()

        try:
            print("wait for connection.", end="", flush=True)
            _check_response(_try_function(lambda : requests.get(f"{self.url}/name"),5, 2.0))
            print("\nConnected to Boptest server", end="\n", flush=True)
        except UnboundLocalError:
            print(f"Unable to connect to the simulator (with url={self.url})")
            print("Exiting!")
            sys.exit()

        self.inputs_info = self.get_inputs_info()
        self.available_input =  [x for x in list(self.inputs_info.keys()) if not x.endswith("_activate")]
        self.available_measurements  = list(self.get_measurements_info().keys()) + self.available_input
        
        self._mask_available_input = [process.extractOne(item, [x for x in list(self.inputs_info.keys()) if x.endswith("_activate")] )[0] for item in self.available_input]
        self.activated_input = self.available_input 
        self.activated_measurements = self.available_measurements

    def set_activated_input(self, input_list: list) -> None:
        assert np.all(np.array([x in self.available_input for x in input_list]))
        self.activated_input = input_list

    def set_activated_measurements(self, measurements_list: list) -> None:
        assert np.all(np.array([x in self.available_measurements+["time"] for x in measurements_list]))
        self.activated_measurements = measurements_list

    def _start_server(self):
        _run_command(f"docker build -t {Boptest._DOCKER_IMAGE_NAME} ./project1_boptest")
        _run_command(f"""docker run --name {self._random_tag} \
                        --network boptest-net \
                        -e APP_PATH='/home/developer' \
                        -e TESTCASE='{self.testcase_name}'\
                        -v $(pwd)/project1_boptest/testcases/{self.testcase_name}/models/wrapped.fmu:/home/developer/models/wrapped.fmu \
                        -v $(pwd)/project1_boptest/testcases/{self.testcase_name}/doc/:/home/developer/doc/ \
                        -v $(pwd)/project1_boptest/restapi.py:/home/developer/restapi.py \
                        -v $(pwd)/project1_boptest/testcase.py:/home/developer/testcase.py \
                        -v $(pwd)/project1_boptest/version.txt:/home/developer/version.txt \
                        -v $(pwd)/project1_boptest/data:/home/developer/data/ \
                        -v $(pwd)/project1_boptest/forecast:/home/developer/forecast/ \
                        -v $(pwd)/project1_boptest/kpis:/home/developer/kpis/ \
                        -p 127.0.0.1:{self.port}:5000 \
                        -d \
                        {Boptest._DOCKER_IMAGE_NAME}""")


    def get_name(self) -> str:
        name = _check_response(requests.get(f"{self.url}/name")) 
        return name["name"]

    def get_measurements_info(self) -> dict:
        measurements = _check_response(requests.get(f"{self.url}/measurements"))
        return measurements

    def get_inputs_info(self) -> dict:
        inputs = _check_response(requests.get(f"{self.url}/inputs"))
        return inputs

    def get_timestep(self) -> float:
        step = _check_response(requests.get(f"{self.url}/step"))
        return step

    def set_timestep(self, step: float) -> None:
        _check_response(requests.put(f"{self.url}/step", json={"step" : step}))
    
    def initialize(self, start_time: float=0.0, warmup_period: float=0.0):
        _check_response(requests.put(f"{self.url}/initialize", json={"start_time" : start_time, "warmup_period":warmup_period}))
        
    
    def set_scenario(self, electricity_price: str=None, time_period: str=None) -> None:
        scenario_dict = {"electricity_price":electricity_price} if not electricity_price is None else  {}
        if not time_period is None: scenario_dict["time_period"] = time_period 
        data_res = _check_response(requests.put(f"{self.url}/scenario", json=scenario_dict))
        return data_res

    def get_scenario(self) -> dict:
        scenario_dict = _check_response(requests.get(f"{self.url}/scenario"))
        return scenario_dict

    def get_forcast_list(self) -> dict:
        forcast_data = _check_response(requests.get(f"{self.url}/forecast_points"))
        return forcast_data

    def get_forcast(self, names, horizon:float, interval:float) -> np.array:
        if isinstance(names, str):
            names = [names,]
        forcast_data = _check_response(requests.put(f"{self.url}/forecast", json={"point_names": [element for element in names if element!="time"], "horizon":horizon, "interval": interval}))
        result_array = np.array([])
        for name in names:
            result_array = np.concatenate([result_array, np.array(forcast_data[name])], axis=0)
        return result_array.reshape(len(names), -1), forcast_data["time"]
 
    def get_simulation_data(self, data_names, start_time: float, end_time: float):
        simu_data = _check_response(requests.put(f"{self.url}/results", json={"point_names": data_names, "start_time": start_time, "final_time": end_time}))
        return simu_data

    def advance(self, u: np.array) -> dict:
        u = u.reshape(-1)
        assert len(u) == len(self.activated_input)
        
        #TODO ajouter check de borne

        input_dict = {key : 0.0 for key in self.inputs_info.keys()}
        for label, value in zip(self.activated_input, u):
            input_dict[label] = value
            input_dict[self._mask_available_input[self.available_input.index(label)]] = 1.0

        try:
            measurements_info = _check_response(requests.post(f"{self.url}/advance", json=input_dict))
            measurements_array = np.array([ measurements_info[label] for label in self.activated_measurements])
        except KeyError:
            return None, None
        return measurements_array, measurements_info

    def get_kpi(self) -> dict:
        kpi_res = _check_response(requests.get(f"{self.url}/kpi")) 
        return kpi_res
    
    def results(self) -> dict:
        result = _check_response(requests.get(f"{self.url}/results"))
        return result

    def setKpi(self, kpi : CustomKPI):
        pass        


    def __del__(self):
        _run_command(f"docker kill {self._random_tag}")
        try:
            pass
            #requests.put(f"{self.url}/stop", json={})
        except:
            print("Boptest is not closed properly (check your running containers)")


if "__main__"==__name__:
    a = Boptest("bestest_hydronic")