import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import sys
from tools import plot_table
from models import train_ARX, extend_dataset_Ivert, train_CNLARX, ArxSolar, CNLArx
from models.cnlarx import CNLArx 


def r2_score(y, yhat, **wkargs):
    return 1 - (np.sum((y-yhat)**2, axis=0))/(np.sum((y-np.mean(y, axis=0))**2 , axis=0) )





model_cnn_full = CNLArx.load("./_cache_models/convex_model_full")
model_cnn_medium = CNLArx.load("./_cache_models/convex_model_medium")
model_cnn_short = CNLArx.load("./_cache_models/convex_model_short")

model_arx_full   = ArxSolar.load("./_cache_models/arx_model_full")
model_arx_medium = ArxSolar.load("./_cache_models/arx_model_medium")
model_arx_short  = ArxSolar.load("./_cache_models/arx_model_short")




test_dataset = pd.read_csv("./dataset/test_set.csv")#[0:6]
training_full = pd.read_csv("./dataset/training_full.csv")
training_medium = pd.read_csv("./dataset/training_medium.csv")
training_short = pd.read_csv("./dataset/training_short.csv")


def perform_r2_k_step_ahead(model, k):
    y   = test_dataset[model.__class__()._Y_LABEL]  .values
    u   = test_dataset[model.__class__()._U_LABEL]  .values
    tvp = test_dataset[model.__class__()._TVP_LABEL].values

    offset = model.get_obs_size()
    nb_step = y.shape[0]-offset-k+1
    ys   = np.array([y  [i:offset+i+k-1,:]  for i in range(nb_step) ])
    us   = np.array([u  [i:offset+i+k-1,:]  for i in range(nb_step) ])
    tvps = np.array([tvp[i:offset+i+k-1,:]  for i in range(nb_step) ])
    curr_y = ys[:,0:offset,:]

    for t in range(k):
        new_y = model.predict(curr_y[:,-offset:,:], us[:,t:t+offset,:],tvps[:,t:t+offset,:])
        print(t,curr_y.shape, new_y.shape)
        curr_y = np.concatenate([curr_y, np.expand_dims(new_y,-1)], axis=1)

    pred_y_k =curr_y[:,-1,0].reshape(-1)# MISO from here

    real_y_k = y[offset+k-1:]

    return float(r2_score(real_y_k.reshape(-1, 1), pred_y_k.reshape(-1, 1)))


def perform_test(model, H,dataset=test_dataset):
    y   = dataset[model.__class__()._Y_LABEL]  .values[:-1]
    u   = dataset[model.__class__()._U_LABEL]  .values[:-1]
    tvp = dataset[model.__class__()._TVP_LABEL].values[:-1]

    offset = model.get_obs_size()
    nb_step = int((y.shape[0]-offset)//H)
    ys   = np.array([y  [i*H:offset+(i+1)*H,:]  for i in range(nb_step) ])
    us   = np.array([u  [i*H:offset+(i+1)*H,:]  for i in range(nb_step) ])
    tvps = np.array([tvp[i*H:offset+(i+1)*H,:]  for i in range(nb_step) ])

    curr_y = ys[:,0:offset,:]
    for t in range(H):
        new_y = model.predict(curr_y[:,-offset:,:], us[:,t:t+offset,:],tvps[:,t:t+offset,:])
        print(curr_y.shape, new_y.shape)
        curr_y = np.concatenate([curr_y, np.expand_dims(new_y,-1)], axis=1)


    pred_y =curr_y[:,offset:,:].reshape(-1)

    plt.plot(pred_y)
    plt.plot(y[offset:])
    plt.show()




perf_cnn_full = [perform_r2_k_step_ahead(model_cnn_full, 2), perform_r2_k_step_ahead(model_cnn_full, 6), perform_r2_k_step_ahead(model_cnn_full, 12)]
perf_cnn_medium = [perform_r2_k_step_ahead(model_cnn_medium, 2), perform_r2_k_step_ahead(model_cnn_medium, 6), perform_r2_k_step_ahead(model_cnn_medium, 12)]
perf_cnn_short = [perform_r2_k_step_ahead(model_cnn_short, 2), perform_r2_k_step_ahead(model_cnn_short, 6), perform_r2_k_step_ahead(model_cnn_short, 12)]

perf_arx_full = [perform_r2_k_step_ahead(model_arx_full, 2), perform_r2_k_step_ahead(model_arx_full, 6), perform_r2_k_step_ahead(model_arx_full, 12)]
perf_arx_medium = [perform_r2_k_step_ahead(model_arx_medium, 2), perform_r2_k_step_ahead(model_arx_medium, 6), perform_r2_k_step_ahead(model_arx_medium, 12)]
perf_arx_short = [perform_r2_k_step_ahead(model_arx_short, 2), perform_r2_k_step_ahead(model_arx_short, 6), perform_r2_k_step_ahead(model_arx_short, 12)]


data_arx =  [
            [         'ARX full (me)', 'ARX full (paper)', 'ARX medium (me)', 'ARX medium (paper)', "ARX short (me)", "ARX short (paper)"],
            [ 'N=2', f"{perf_arx_full[0]:.2f}", 0.99 , f"{perf_arx_medium[0]:.2f}" , 0.99 , f"{perf_arx_short[0]:.2f}"  , 0.99    ],
            ['N=6' , f"{perf_arx_full[1]:.2f}", 0.93 , f"{perf_arx_medium[1]:.2f}" , 0.94 , f"{perf_arx_short[1]:.2f}"  , 0.92 ],
            ['N=12', f"{perf_arx_full[2]:.2f}", 0.79 , f"{perf_arx_medium[2]:.2f}" , 0.79 , f"{perf_arx_short[2]:.2f}"  , 0.71 ],
        ]

plot_table("Resultat ARX", data_arx)


data_cnn =  [
            [         'ICNN full (me)', 'ICNN full (paper)', 'ICNN medium (me)', 'ICNN medium (paper)', "ICNN short (me)", "ICNN short (paper)"],
            [ 'N=2', f"{perf_cnn_full[0]:.2f}", 0.94 , f"{perf_cnn_medium[0]:.2f}" , 0.92 , f"{perf_cnn_short[0]:.2f}"  , 0.77    ],
            ['N=6' , f"{perf_cnn_full[1]:.2f}", 0.29 , f"{perf_cnn_medium[1]:.2f}" , 0.05 , f"{perf_cnn_short[1]:.2f}"  ,  -0.27 ],
            ['N=12', f"{perf_cnn_full[2]:.2f}", 0.13 , f"{perf_cnn_medium[2]:.2f}" , 0.12 , f"{perf_cnn_short[2]:.2f}"  , 0.03 ],
        ]

plot_table("Resultat ICNN", data_cnn)