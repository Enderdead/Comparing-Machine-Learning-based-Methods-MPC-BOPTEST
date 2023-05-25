
import deepSI
import numpy as np 
from .tools import *
from .convex_nn import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf

def train_CNLARX(dataset_pd, na=3, nb=3, nb_layer=2, units=24, y_label=["reaTRoo_y"], u_label=["u","weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y", "weaSta_reaWeaNTot_y", "weaSta_reaWeaRelHum_y", "weaSta_reaWeaSolAlt_y"], dt=900):
    assert len(dataset_pd.dropna()) == len(dataset_pd), "NAN rows"

    y = dataset_pd[y_label].values
    u=dataset_pd[u_label].values

    scaler_y = StandardScaler()
    scaler_u = StandardScaler()

    # Normaliser les données dans la matrice X
    u = scaler_u.fit_transform(u)
    y = scaler_y.fit_transform(y)


    u_extended, y_extended, y_t_1, d_y = IO_transform(y, u, na=na, nb=nb)

    d_y = d_y * 10.0

    print(u_extended.shape, y_extended.shape, y_t_1.shape, d_y.shape)
    flat_y_extended = np.transpose(y_extended, (0,2,1)).reshape(y_extended.shape[0],-1)
    flat_u_extended = np.transpose(u_extended, (0,2,1)).reshape(u_extended.shape[0],-1)

    model_input  = np.concatenate([flat_y_extended, flat_u_extended], axis=1)
    model_output = d_y
    
    model = PICNN(nb_input=model_input.shape[1], nb_input_conv=na+nb, nb_output=1, act_func=tf.nn.relu, nb_layer=nb_layer, units=units, recursive_convexity=True, kernel_initializer=tf.keras.initializers.GlorotUniform())
    model.train(model_output, model_input, nb_epoch=1000, optimizer=tf.keras.optimizers.Adam(learning_rate=0.008))

    d_y_pred = model.core.predict(model_input)

    plt.plot(model_output)
    plt.plot(d_y_pred)
    plt.show()
    return model