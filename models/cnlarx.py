
from sklearn.linear_model import LinearRegression, Lasso, Ridge
import numpy as np 
from .tools import *
from .convex_nn import *
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import tensorflow as tf
import pickle



class CNLArx():

    _Y_LABEL   = ["reaTRoo_y"]
    _U_LABEL   = ["u"]
    _TVP_LABEL = ["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","time", "weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y"]
    _TVP_LABEL_EXTENDED = ["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y", "cos_t", "sin_t"]

    @classmethod
    def load(cls, path):
        result = pickle.load(os.path.join(path, "obj.pickle"))
        result.core = PICNN.load(path)

    def __init__(self, na=3, nb=3, nb_layer=2, nb_unit=32):
        self.na = na
        self.nb = nb

        self.scaler_y = StandardScaler()
        self.scaler_u = StandardScaler()
        self.scaler_tvp = StandardScaler()

        self._fitted = False
        self.core = PICNN(nb_input=na*len(CNLArx._Y_LABEL) + nb*len(CNLArx._U_LABEL)+nb*len(CNLArx._TVP_LABEL_EXTENDED),
                          input_conv_list=list(range(na*len(CNLArx._Y_LABEL) )),#+ nb*len(CNLArx._U_LABEL)
                          nb_output=len(CNLArx._Y_LABEL),
                          nb_layer=nb_layer, 
                          units=nb_unit, 
                          recursive_convexity= False)


    def _process_tvp(self, tvp):
        cos_t = np.cos(2.0*np.pi*tvp[:,2:3]/86400.0  )
        sin_t = np.sin(2.0*np.pi*tvp[:,2:3]/86400.0  )

        return np.concatenate([tvp[:,[0,1,3,4]], cos_t, sin_t   ], axis=1)

    def train(self, dataset, nb_epoch=2000):
        y, u, tvp = dataset[CNLArx._Y_LABEL].values, dataset[CNLArx._U_LABEL].values, dataset[CNLArx._TVP_LABEL].values

        tvp = self._process_tvp(tvp)

        u   = self.scaler_u.fit_transform(u)
        tvp = self.scaler_tvp.fit_transform(tvp)
        y   = self.scaler_y.fit_transform(y)


        y_extended, u_extended, tvp_extended, y_t_1, d_y = IO_transform(y, u, tvp=tvp, na=self.na, nb=self.nb)


        flat_y_extended   = np.transpose(y_extended, (0,2,1)).reshape(y_extended.shape[0]  ,-1)
        flat_u_extended   = np.transpose(u_extended, (0,2,1)).reshape(u_extended.shape[0]  ,-1)
        flat_tvp_extended = np.transpose(tvp_extended, (0,1,2)).reshape(tvp_extended.shape[0],-1)

        model_input  = np.concatenate([flat_y_extended, flat_u_extended, flat_tvp_extended], axis=1)
        model_output = y_t_1

        self.core.train( model_output, model_input, nb_epoch=nb_epoch, optimizer=tf.keras.optimizers.Adam(learning_rate=0.08))

        plt.plot(model_output)
        plt.plot(self.core.core.predict(model_input))
        plt.show()
        self._fitted = True

    def save(self, path):
        if not os.path.isdir(path):
            if os.path.isfile(path):
                raise RuntimeError("Bad path")

            os.mkdir(path)

        pickle.dump(self, open(os.path.join(path, "obj.pickle"), "wb"))

    def __getstate__(self):
        state = self.__dict__.copy()
        if 'core' in state:
            del state['core']
        return state

    def predict(self, y, u, tvp):
        assert self._fitted, "Please fit the model"

        y, u, tvp = self.scaler_y.transform(y), self.scaler_u.transform(u), self.scaler_tvp.transform(tvp)
        model_input = np.concatenate([y[-self.na:].reshape(-1), u[-self.nb:].reshape(-1), tvp[-self.nb:].reshape(-1)]).reshape(1,-1) 
        pred =  self.core.core.predict(model_input)

        return self.scaler_y.inverse_transform(pred)



def train_CNLARX(dataset_pd, na=3, nb=3, nb_layer=4, units=64):
    assert len(dataset_pd.dropna()) == len(dataset_pd), "NAN rows"
    
    model = CNLArx(na=na, nb=nb, nb_layer=nb_layer, nb_unit=units)
    model.train(dataset_pd)

    return model