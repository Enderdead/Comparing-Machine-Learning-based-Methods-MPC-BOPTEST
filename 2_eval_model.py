import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import sys

from models import train_ARX, extend_dataset_Ivert, train_CNLARX, ArxSolar, CNLArx
from models.arx import TVP_LABEL
from models.cnlarx import CNLArx 



model = CNLArx.load("./_cache_models/convex_model")



test_dataset = pd.read_csv("./dataset/test_set.csv")


y = test_dataset[CNLArx._Y_LABEL].values[:-1]
u = test_dataset[CNLArx._U_LABEL].values[:-1]
tvp = test_dataset[CNLArx._TVP_LABEL].values[:-1]

dy = test_dataset[CNLArx._Y_LABEL].values[1:]-test_dataset[CNLArx._Y_LABEL].values[:-1]


dy_hat = model.predict(y, u, tvp)

plt.plot(dy)
plt.plot(dy_hat)
plt.show()


