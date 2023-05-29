
import deepSI
import numpy as np 
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from .tools import *
import pickle
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

U_LABEL=["u",]
TVP_LABEL=["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","time", "weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y"]
y_LABEL=["reaTRoo_y"]

class ArxSolar():

    @classmethod
    def load(cls, path):
        res = pickle.load(open(path, "rb"))
        return res


    def __init__(self, na=3, nb=3, n_tvert=6):
        self.na = na
        self.nb = nb
        self.n_tvert = n_tvert
        self.core = Ridge(alpha=0.0001, fit_intercept=False)

        self.scaler_y   = StandardScaler()
        self.scaler_u   = StandardScaler(with_mean=False, with_std=False)
        self.scaler_tvp = StandardScaler() 
    
        self._fitted = False

    def _process_tvp(self, tvp):
        I_vert = tvp[:,3:4]*np.cos(np.pi/2  - tvp[:,4:5])/np.sin(np.pi/2 - tvp[:,4:5])
        categories = np.floor(( tvp[:,2:3]%86400)/(86400/self.n_tvert))
        one_hot_mat = np.zeros((len(categories),self.n_tvert))
        one_hot_mat[list(range(tvp.shape[0])),categories.reshape(-1).astype(np.int32)] = 1.0*I_vert.reshape(-1)
        tvp = np.concatenate([tvp[:,0:2],categories, one_hot_mat] , axis=1)
        return tvp

    def train(self, dataset):
        y, u, tvp = dataset[y_LABEL].values, dataset[U_LABEL].values, dataset[TVP_LABEL].values

        tvp = self._process_tvp(tvp)

        u   = self.scaler_u.fit_transform(u)
        tvp = self.scaler_tvp.fit_transform(tvp)
        y   = self.scaler_y.fit_transform(y)


        y_extended, u_extended, tvp_extended, y_t_1, d_y = IO_transform(y, u, tvp=tvp, na=self.na, nb=self.nb)

        flat_y_extended   = np.transpose(y_extended, (0,2,1)).reshape(y_extended.shape[0]  ,-1)
        flat_u_extended   = np.transpose(u_extended, (0,2,1)).reshape(u_extended.shape[0]  ,-1)
        flat_tvp_extended = np.transpose(tvp_extended, (0,1,2)).reshape(tvp_extended.shape[0],-1)

        model_input  = np.concatenate([flat_y_extended, flat_u_extended, flat_tvp_extended], axis=1)
        print(model_input[0:3,:])
        model_output = y_t_1

        self.core.fit(model_input, model_output)
        plt.plot(model_output)
        plt.plot(self.core.predict(model_input))
        plt.show()
        self._coef_jnp = jnp.array(self.core.coef_)
        self._fitted = True

    def save(self, path):
        pickle.dump(self, open(path, "wb"))


    def predict(self, y, u, tvp):
        assert self._fitted, "Please fit the model"

        y, u, tvp = self.scaler_y.transform(y), self.scaler_u.transform(u), self.scaler_tvp.transform(tvp)
        model_input = np.concatenate([y[-self.na:].reshape(-1), u[-self.nb:].reshape(-1), tvp[-self.nb:].reshape(-1)]).reshape(1,-1) 
        pred =  self.core.predict(model_input)

        return self.scaler_y.inverse_transform(pred)




def train_ARX(dataset_pd, na=3, nb=3, n_tvert=6):
    assert len(dataset_pd.dropna()) == len(dataset_pd), "NAN rows"



    arx_model = ArxSolar(na=na, nb=nb, n_tvert=6)

    arx_model.train(dataset_pd)

    return arx_model