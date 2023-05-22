import pandas as pd
import numpy as np 

def extend_dataset_Ivert(dataset, N):
    
    I_vert = dataset["weaSta_reaWeaHGloHor_y"]*np.cos(np.pi/2 - dataset["weaSta_reaWeaSolAlt_y"])/np.sin(np.pi/2 - dataset["weaSta_reaWeaSolAlt_y"])
    dataset["relative_time_day"] = dataset["t"]%86400
    categorie = np.floor((dataset["t"]%86400)/(86400/N))
    one_hot_mat = np.zeros((len(categorie),N))
    one_hot_mat[np.arange(0, len(categorie)),categorie.values.astype(np.int)] = 1.0
    one_hot_mat = one_hot_mat*I_vert.values.reshape(-1,1)
    dataset[[f"Tvert_{i}" for i in range(N)]] = one_hot_mat

    return dataset
