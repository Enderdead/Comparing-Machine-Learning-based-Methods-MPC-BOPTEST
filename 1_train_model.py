import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import sys

from models import train_ARX, extend_dataset_Ivert, train_CNLARX
from deepSI_tools import * 


# Load dataset
training_dataset = pd.read_csv(os.path.join("dataset", "training_full.csv"))
test_dataset = pd.read_csv(os.path.join("dataset", "test_set.csv"))


# Extend the dataset variables
T_vertN = 6
extended_training_dataset = extend_dataset_Ivert(training_dataset, T_vertN)
extended_test_dataset     = extend_dataset_Ivert(test_dataset, T_vertN)


# Train and save the convex neural network 
cnn_model = train_CNLARX(extended_training_dataset, na=3, nb=3, u_label=["u", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y",]+[f"Tvert_{i}" for i in range(T_vertN)], y_label=["reaTRoo_y"])
cnn_model.save(os.path.join(".","_cache_models", "convex_model"))



test_dataset_si = deepSI.System_data(y = extended_test_dataset["reaTRoo_y"], u=extended_test_dataset[["u", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y",]+[f"Tvert_{i}" for i in range(T_vertN)]] , dt=900)
training_dataset_si = deepSI.System_data(y = extended_training_dataset["reaTRoo_y"], u=extended_training_dataset[["u", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y",]+[f"Tvert_{i}" for i in range(T_vertN)]] , dt=900)


arx_model = train_ARX(extended_training_dataset, na=3, nb=3, u_label=["u", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y",]+[f"Tvert_{i}" for i in range(T_vertN)], y_label=["reaTRoo_y"])

arx_model.save(os.path.join(".", "_cache_models", "arx_model"))
#compare_deepSI(test_dataset_si , arx_model,K=12)
