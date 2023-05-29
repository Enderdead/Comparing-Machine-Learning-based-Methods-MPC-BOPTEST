
import deepSI
import numpy as np 
from sklearn.linear_model import LinearRegression
import jax.numpy as jnp
from sklearn.preprocessing import StandardScaler
from .tools import *
import pickle



U_LABEL=["u", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","time", "weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y"]#+[f"Tvert_{i}" for i in range(T_vertN)]
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
        self.core = LinearRegression(fit_intercept=False)

        self.scaler_y = StandardScaler()
        self.scaler_u = StandardScaler()
    
        self._fitted = False
        self._coef_jnp = None

    def _process_input(self, u):
        I_vert = u[:,4:5]*np.cos(np.pi/2  - u[:,5:6])/np.sin(np.pi/2 - u[:,5:6])
        categories = np.floor(( u[:,3:4]%86400)/(86400/self.n_tvert))
        one_hot_mat = np.zeros((len(categories),self.n_tvert))*I_vert

        u = np.concatenate([u[:,0:3], one_hot_mat] , axis=1)
        return u


    def _jnp_process_input(self, u):
        I_vert = u[:,4:5]*jnp.cos(np.pi/2  - u[:,5:6])/np.jsin(np.pi/2 - u[:,5:6])
        categories = jnp.floor(( u[:,3:4]%86400)/(86400/self.n_tvert))
        one_hot_mat = jnp.zeros((len(categories),self.n_tvert))*I_vert

        u = jnp.concatenate([u[:,0:3], one_hot_mat] , axis=1)

        return u


    def train(self, y, u):
        u = self._process_input(u)

        u = self.scaler_u.fit_transform(u)
        y = self.scaler_y.fit_transform(y)


        u_extended, y_extended, y_t_1, d_y = IO_transform(y, u, na=self.na, nb=self.nb)

        flat_y_extended = np.transpose(y_extended, (0,2,1)).reshape(y_extended.shape[0],-1)
        flat_u_extended = np.transpose(u_extended, (0,2,1)).reshape(u_extended.shape[0],-1)

        model_input  = np.concatenate([flat_y_extended, flat_u_extended], axis=1)
        model_output = y_t_1

        self.core.fit(y_t_1, model_input)
        self._coef_jnp = jnp.array(self.core.coef_)
        self._fitted = True

    def save(self, path):
        pickle.dump(self, open(path, "wb"))


    def predict(self, y, u):
        assert self._fitted, "Please fit the model"

        u = self._process_input(u)
        y, u = self.scaler_y.transform(y), self.scaler_u.transform(u)

        model_input = np.concatenate([y[-self.na:].reshape(-1), u[-self.nb:].T.reshape(-1)]).reshape(1,-1) 
        pred =  self.core.predict(model_input)

        return self.scaler_y.inverse_transform(pred)


    def predict_jnp(self, y, u):
        u = self._process_input(u)
        y = y*self.scaler_y._scale + self.scaler_y._mean 
        u = u*self.scaler_u._scale + self.scaler_u._mean 

        model_input = np.concatenate([y[-self.na:].reshape(-1), u[-self.nb:].T.reshape(-1)]).reshape(1,-1) 
        pred =( jnp.dot(model_input, self._coef_jnp)-self.scaler_y._mean )/self.scaler_y._scale
        return pred



def train_ARX(dataset_pd, na=3, nb=3, n_tvert=6):
    assert len(dataset_pd.dropna()) == len(dataset_pd), "NAN rows"

    y = dataset_pd[y_LABEL].values
    u = dataset_pd[U_LABEL].values


    arx_model = ArxSolar(na=na, nb=nb, n_tvert=6)

    arx_model.train(y, u)

    return arx_model