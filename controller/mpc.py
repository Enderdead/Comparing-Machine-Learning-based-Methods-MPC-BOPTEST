import numpy as np 
import tensorflow as tf
import pyNeuralEMPC as nEMPC
import os
import pickle
from models import *
import jax.numpy as jnp
import jax


class ARXMpc():
    def __init__(self, model_path, horizon, comfort_balance=1, tvp_dim=9):
        self.model = ArxSolar.load(model_path)
        self.comfort_balance = comfort_balance
        self.horizon = horizon
        self.tvp_dim = tvp_dim
        self.rolling_window = max(self.model.na, self.model.nb)
        self.prev_u = np.array([[]])
        self.prev_tvp = np.array([[]])
        self.prev_y = np.array([[]])
        self._ready = False

    def _get_forward_func(self):
        def forward(x_slided, u_slided,p=None,tvp=None):
            input_model = jnp.concatenate([x_slided, u_slided, tvp], axis=1)
            res = jnp.matmul(input_model , self.model.core.coef_.T )
            return res

        return forward

    def _get_cost_func(self, y_limit):
        
        def cost(x, u, p=None, tvp=None):
            lower_error = jax.nn.relu(y_limit-x)
            return jnp.sum(lower_error) + 0.1*jnp.sum(u)

        return cost

    def _treat_measurment(self, measurment):
        # input order => ["reaTRoo_y", "time", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y","weaSta_reaWeaHGloHor_y", "weaSta_reaWeaNTot_y", "weaSta_reaWeaRelHum_y", "weaSta_reaWeaSolAlt_y"]
        # target order => ["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","time", "weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y"]
        tvp = self.model._process_tvp(measurment[:,[2,6,1,4,7]])
        return measurment[:,0:1], tvp

    def _treat_forcast(self, forcast):
        # Input forcast => ["LowerSetp[1]",  "time", "TDryBul","HDirNor","HGloHor", "nTot", "relHum", "solAlt" ]
        # target tvp => ["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","time", "weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y"]
        tvp = forcast[:, [2, 6, 1, 4, 7]]
        return forcast[:,0:1], self.model._process_tvp(tvp)

    def reset(self, prev_measurment, prev_u):

        prev_y, prev_tvp = self._treat_measurment(prev_measurment)

        assert prev_y.shape[0]==(self.model.na-1), "please provide a suitable past view of y"
        self.prev_y = self.model.scaler_y.transform(prev_y)

        assert prev_u.shape[0]==(self.model.nb-1), "please provide a suitable past view of u"
        self.prev_u = self.model.scaler_u.transform(prev_u)
    
        assert prev_tvp.shape[0]==(self.model.nb-1), "please provide a suitable past view of u"
        self.prev_tvp = self.model.scaler_tvp.transform(prev_tvp)

        self._ready = True
    
    def set_cost_balance(self, setpoint):
        pass



    def update(self, measurements, forcasts):
        assert self._ready, "please reset the controller"


        # Treat input data
        curr_y, curr_tvp = self._treat_measurment(measurements)
        limit_y, futur_tvp = self._treat_forcast(forcasts)

        tvp_horizon = np.concatenate([curr_tvp, futur_tvp[1:,:]], axis=0) # We take the real value for t zero
        limit_y_horizon = limit_y

        # Scale input data
        
        curr_y          = self.model.scaler_y.transform(curr_y)
        tvp_horizon     = self.model.scaler_tvp.transform(tvp_horizon)
        limit_y_horizon = self.model.scaler_y.transform(limit_y_horizon)
        print(tvp_horizon.shape)

        # Init problem structure

        proxy_model = nEMPC.model.jax.DiffDiscretJaxModelRollingWindow(self._get_forward_func(), 1, 1, tvp_dim=self.tvp_dim,
               rolling_window=self.rolling_window, forward_rolling=True, safe_mode=False)
        integrator = nEMPC.integrator.unity.UnityIntegrator(proxy_model, self.horizon)

        proxy_model.set_prev_data(x_prev=self.prev_y, u_prev=self.prev_u, tvp_prev=self.prev_tvp) 

        constraints_nmpc = [nEMPC.constraints.DomainConstraint(
                    states_constraint=[[-np.inf, np.inf],],
                    control_constraint=[[0.0,1.0],])]

        objective_func  =  nEMPC.objective.jax.JAXObjectifFunc(self._get_cost_func(limit_y_horizon))

        optimizer = nEMPC.optimizer.Slsqp(max_iteration=350, verbose=5, tolerance=1e-3)

        controller = nEMPC.controller.NMPC(integrator, objective_func, constraints_nmpc, self.horizon, 900, optimizer=optimizer, use_hessian=False)

        res_y, res_u = controller.next(x0=curr_y.reshape(-1), tvp=tvp_horizon)


        self.prev_y = np.concatenate([self.prev_y[1:, :], curr_y], axis=0)
        self.prev_u = np.concatenate([self.prev_u[1:, :], res_u[0:1,:]], axis=0)
        self.prev_tvp = np.concatenate([self.prev_tvp[1:, :], tvp_horizon[0:1,:]], axis=0)

        return res_u[0], {"temp" : self.model.scaler_y.inverse_transform(res_y), "u" : self.model.scaler_u.inverse_transform(res_u)}


class CNNMpc():
    def __init__(self, model_path, horizon, confort_balance=1.0, tvp_dim=6):
        self.model = CNLArx.load(model_path)
        self.confort_balance = confort_balance
        self.horizon = horizon
        self.tvp_dim = tvp_dim
        self.rolling_window = max(self.model.na, self.model.nb)
        self.prev_u   = np.array([[]])
        self.prev_y   = np.array([[]])
        self.prev_tvp = np.array([[]])
        self._ready = False

    def _get_cost_func(self, y_limit):
        
        def cost(x, u, p=None, tvp=None):
            lower_error = jax.nn.relu(y_limit - x)
            return jnp.sum(lower_error) + 0.1*jnp.sum(u)
            
        return cost
    
    def _treat_measurment(self, measurment):
        # input order => ["reaTRoo_y", "time", "weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y","weaSta_reaWeaHGloHor_y", "weaSta_reaWeaNTot_y", "weaSta_reaWeaRelHum_y", "weaSta_reaWeaSolAlt_y"]
        # target order => ["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","time", "weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y"]
        tvp = self.model._process_tvp(measurment[:,[2,6,1,4,7]])
        return measurment[:,0:1], tvp

    def _treat_forcast(self, forcast):
        # Input forcast => ["LowerSetp[1]",  "time", "TDryBul","HDirNor","HGloHor", "nTot", "relHum", "solAlt" ]
        # target tvp => ["weaSta_reaWeaTDryBul_y", "weaSta_reaWeaRelHum_y","time", "weaSta_reaWeaHGloHor_y", "weaSta_reaWeaSolAlt_y"]
        tvp = forcast[:, [2, 6, 1, 4, 7]]
        return forcast[:,0:1], self.model._process_tvp(tvp)
    
    def reset(self, prev_measurment, prev_u):

        prev_y, prev_tvp = self._treat_measurment(prev_measurment)

        assert prev_y.shape[0]==(self.model.na-1), "please provide a suitable past view of y"
        self.prev_y = self.model.scaler_y.transform(prev_y)

        assert prev_u.shape[0]==(self.model.nb-1), "please provide a suitable past view of u"
        self.prev_u = self.model.scaler_u.transform(prev_u)
    
        assert prev_tvp.shape[0]==(self.model.nb-1), "please provide a suitable past view of u"
        self.prev_tvp = self.model.scaler_tvp.transform(prev_tvp)

        self._ready = True
    
    def set_cost_balance(self, setpoint):
        pass




    def update(self, measurements, forcasts):
        assert self._ready, "please reset the controller"


        # Treat input data
        curr_y, curr_tvp = self._treat_measurment(measurements)
        limit_y, futur_tvp = self._treat_forcast(forcasts)

        tvp_horizon = np.concatenate([curr_tvp, futur_tvp[1:,:]], axis=0) # We take the real value for t zero
        limit_y_horizon = limit_y

        # Scale input data
        
        curr_y          = self.model.scaler_y.transform(curr_y)
        tvp_horizon     = self.model.scaler_tvp.transform(tvp_horizon)
        limit_y_horizon = self.model.scaler_y.transform(limit_y_horizon)
        print(tvp_horizon.shape)

        # Init problem structure  
        proxy_model = nEMPC.model.tensorflow.KerasTFModelRollingInput(self.model.core.core, x_dim=1, u_dim=1, p_dim=0, tvp_dim=self.tvp_dim,
                rolling_window=self.rolling_window, forward_rolling=True)

        integrator = nEMPC.integrator.discret.DiscretIntegrator(proxy_model, self.horizon)

        proxy_model.set_prev_data(x_prev=self.prev_y, u_prev=self.prev_u, tvp_prev=self.prev_tvp) 

        constraints_nmpc = [nEMPC.constraints.DomainConstraint(
                    states_constraint=[[-np.inf, np.inf],],
                    control_constraint=[[0.0,1.0],])]

        objective_func  =  nEMPC.objective.jax.JAXObjectifFunc(self._get_cost_func(limit_y_horizon))

        optimizer = nEMPC.optimizer.Slsqp(max_iteration=350, verbose=5, tolerance=1e-3)

        controller = nEMPC.controller.NMPC(integrator, objective_func, constraints_nmpc, self.horizon, 900, optimizer=optimizer, use_hessian=False)

        res_y, res_u = controller.next(x0=curr_y.reshape(-1), tvp=tvp_horizon)


        self.prev_y = np.concatenate([self.prev_y[1:, :], curr_y], axis=0)
        self.prev_u = np.concatenate([self.prev_u[1:, :], res_u[0:1,:]], axis=0)
        self.prev_tvp = np.concatenate([self.prev_tvp[1:, :], tvp_horizon[0:1,:]], axis=0)

        return res_u[0], {"temp" : self.model.scaler_y.inverse_transform(res_y), "u" : self.model.scaler_u.inverse_transform(res_u)}
