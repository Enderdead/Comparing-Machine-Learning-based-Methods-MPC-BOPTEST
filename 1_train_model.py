import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt

import os
import sys

from models import train_ARX, extend_dataset_Ivert



training_dataset = pd.read_csv(os.path.join("dataset", "test_set.csv"))

extended_training_dataset = extend_dataset_Ivert(training_dataset, 6)



fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(training_dataset["t"]/900.0,".")
ax2.plot(training_dataset["u"])
plt.show()