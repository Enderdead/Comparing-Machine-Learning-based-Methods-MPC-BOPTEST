
import deepSI
import numpy as np 
from sklearn.linear_model import LinearRegression
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from .tools import *
import pickle


class Arx():

    @classmethod
    def load(cls, path):
        res = pickle.load(open(path, "rb"))
        return res


    def __init__(self, na=3, nb=3):
        self.na = na
        self.nb = nb
        self.core = LinearRegression(fit_intercept=False)


    def train(self, y, u):

        u_extended, y_extended, y_t_1, d_y = IO_transform(y, u, na=self.na, nb=self.nb)

        flat_y_extended = np.transpose(y_extended, (0,2,1)).reshape(y_extended.shape[0],-1)
        flat_u_extended = np.transpose(u_extended, (0,2,1)).reshape(u_extended.shape[0],-1)

        model_input  = np.concatenate([flat_y_extended, flat_u_extended], axis=1)
        model_output = y_t_1

        self.core.fit(y_t_1, model_input)


    def save(self, path):
        pickle.dump(self, open(path, "wb"))

    def get_jax_predict_func():

        def predict(x):
            pass

        return predict


    def predict(self, y, x):
        model_input = np.concatenate([y[-self.na:].reshape(-1), x[-self.nb:].T.reshape(-1)]).reshape(1,-1) 
        print(model_input)
        return self.core.predict(model_input)


def train_ARX(dataset_pd, na=3, nb=3,  y_label=["reaTRoo_y"], u_label=["u","weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y", "weaSta_reaWeaNTot_y", "weaSta_reaWeaRelHum_y", "weaSta_reaWeaSolAlt_y"], dt=900):
    assert len(dataset_pd.dropna()) == len(dataset_pd), "NAN rows"

    y = dataset_pd[y_label].values
    u=dataset_pd[u_label].values

    scaler_y = StandardScaler()
    scaler_u = StandardScaler()

    # Normaliser les données dans la matrice X
    u = scaler_u.fit_transform(u)
    y = scaler_y.fit_transform(y)

    arx_model = Arx(na=na, nb=nb)

    arx_model.train(y, u)

    return arx_model