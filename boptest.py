import socket
import os
import sys
import requests
import subprocess
import random
import time
from contextlib import closing
from project1_boptest.examples.python.controllers.controller import Controller
from project1_boptest.examples.python.custom_kpi.custom_kpi_calculator import CustomKPI

def _find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def _get_all_scenarios():
    scenario_list = os.listdir(os.path.join("project1_boptest", "testcases"))
    return list(filter( lambda x: os.path.isdir(os.path.join("project1_boptest", "testcases", x)), scenario_list))


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

    def __init__(self, scenario_name, basename='127.0.0.1', port=None):
        assert scenario_name in _get_all_scenarios()
        
        self.port = _find_free_port() if port is None else port
        self.scenario_name  = scenario_name

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

    def _start_server(self):
        _run_command(f"docker build -t {Boptest._DOCKER_IMAGE_NAME} ./project1_boptest")
        _run_command(f"""docker run --name {self._random_tag} \
                        --network boptest-net \
                        -e APP_PATH='/home/developer' \
                        -e TESTCASE='{self.scenario_name}'\
                        -v $(pwd)/project1_boptest/testcases/{self.scenario_name}/models/wrapped.fmu:/home/developer/models/wrapped.fmu \
                        -v $(pwd)/project1_boptest/testcases/{self.scenario_name}/doc/:/home/developer/doc/ \
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

    def get_input_list(self) -> dict:
        measurements = _check_response(requests.get(f"{self.url}/measurements"))
        return measurements

    def get_timestep(self) -> float:
        step = _check_response(requests.get(f"{self.url}/step"))
        return step

    def set_timestep(self, step: float) -> None:
        _check_response(requests.put(f"{self.url}/step", json={"step" : step}))
    
    def initialize(self, start_time: float=0.0, warmup_period: float=0.0):
        pass

    def get_kpi(self) -> dict:
        kpi_res = _check_response(requests.get(f"{self.url}/kpi")) 
        return kpi_res
        
    def setKpi(self, kpi : CustomKPI):
        pass        


    def __del__(self):
        try:
            requests.put(f"{self.url}/quit", json={})
        except:
            print("Boptest is not closed properly (check your running containers)")


if "__main__"==__name__:
    a = Boptest("testcase1")