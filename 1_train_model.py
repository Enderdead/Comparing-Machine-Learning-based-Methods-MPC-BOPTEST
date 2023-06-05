import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import sys

from models import train_ARX, extend_dataset_Ivert, train_CNLARX, ArxSolar, CNLArx
from models.cnlarx import CNLArx 


# Load dataset
training_dataset_full = pd.read_csv(os.path.join("dataset", "training_full.csv"))
training_dataset_medium = pd.read_csv(os.path.join("dataset", "training_medium.csv"))
training_dataset_short = pd.read_csv(os.path.join("dataset", "training_short.csv"))

test_dataset = pd.read_csv(os.path.join("dataset", "test_set.csv"))


# Train and save the convex neural network 
cnn_model_full = train_CNLARX(training_dataset_full, na=3, nb=3)
cnn_model_full.save(os.path.join(".","_cache_models", "convex_model_full"))


cnn_model_medium = train_CNLARX(training_dataset_medium, na=3, nb=3)
cnn_model_medium.save(os.path.join(".","_cache_models", "convex_model_medium"))

cnn_model_short = train_CNLARX(training_dataset_short, na=3, nb=3)
cnn_model_short.save(os.path.join(".","_cache_models", "convex_model_short"))


arx_model_full = train_ARX(training_dataset_full, na=9, nb=9, n_tvert=6)
arx_model_full.save(os.path.join(".", "_cache_models", "arx_model_full"))

arx_model_medium = train_ARX(training_dataset_medium, na=9, nb=9, n_tvert=6)
arx_model_medium.save(os.path.join(".", "_cache_models", "arx_model_medium"))

arx_model_short = train_ARX(training_dataset_short, na=9, nb=9, n_tvert=6)
arx_model_short.save(os.path.join(".", "_cache_models", "arx_model_short"))