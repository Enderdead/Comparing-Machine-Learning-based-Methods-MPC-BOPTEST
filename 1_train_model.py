import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import sys

from models import train_ARX, extend_dataset_Ivert, train_CNLARX, ArxSolar, CNLArx
from deepSI_tools import *
from models.arx import TVP_LABEL
from models.cnlarx import CNLArx 


# Load dataset
training_dataset = pd.read_csv(os.path.join("dataset", "training_full.csv"))
test_dataset = pd.read_csv(os.path.join("dataset", "test_set.csv"))


# Extend the dataset variables
T_vertN = 6
extended_training_dataset = extend_dataset_Ivert(training_dataset, T_vertN)
extended_test_dataset     = extend_dataset_Ivert(test_dataset, T_vertN)


# Train and save the convex neural network 
cnn_model = train_CNLARX(extended_training_dataset, na=3, nb=3)
cnn_model.save(os.path.join(".","_cache_models", "convex_model"))



arx_model = train_ARX(extended_training_dataset, na=3, nb=3, n_tvert=6)

arx_model.save(os.path.join(".", "_cache_models", "arx_model"))
#=["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","time", "weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y"]
tvp = training_dataset[CNLArx._TVP_LABEL].values
res = cnn_model._process_tvp(tvp)