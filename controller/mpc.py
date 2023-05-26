import numpy as np 
import tensorflow as tf
import pyNeuralEMPC as nEMPC
import os
import pickle
from models import *
import jax.numpy as jnp
import jax


class ARXMpc():
    def __init__(self, model_path, horizon, comfort_balance=1, tvp_dim=8):
        self.model = Arx.load(model_path)
        self.comfort_balance = comfort_balance
        self.horizon = horizon
        self.tvp_dim = tvp_dim
        self.rolling_window = max(self.model.na, self.model.nb)
        self.past_u = np.array([[]])
        self.past_y = np.array([[]])
        self._ready = False

    def _get_forward_func(self):
        def forward(x_slided, u_slided,p=None,tvp=None):
            proj_tvp = tvp[:,1:]
            input_model = jnp.concatenate([x_slided, u_slided, proj_tvp], axis=1)
            res = jnp.matmul(input_model , mpc_controller.model.core.coef_ )
            return res

        return forward

    def _get_cost_func(self):
        
        def cost(x, u, p=None, tvp=None):
            lower_error = jax.nn.relu(tvp[:,0:1]-x)
            return jnp.sum(lower_error) + 0.1*jnp.sum(u)

        return cost

    def reset(self, prev_y, prev_u):
        assert prev_y.shape[0]==(self.model.na-1), "please provide a suitable past view of y"
        self.past_y = prev_y

        assert prev_u.shape[0]==(self.model.nb-1), "please provide a suitable past view of u"
        self.past_u = prev_u

        self._ready = True
    
    def set_cost_balance(self, setpoint):
        pass


    def update(self, measurements, forcasts):
        assert self._ready, "please reset the controller"

        curr_tvl = measurements[1:]
        curr_x   = measurements[0:1]
        # TODO extraire les données passées du messurements

        proxy_model = nEMPC.model.jax.DiffDiscretJaxModelRollingWindow(self._get_forward_func(), 1, 1, tvp_dim=self.tvp_dim, rolling_window=self.rolling_window, forward_rolling=True)
        integrator = nEMPC.integrator.discret.DiscretIntegrator(proxy_model, self.horizon)

        proxy_model.set_prev_data(x_prev=self.prev_y[:,0:1], u_prev=self.u_prev, tvp_prev=self.prev_y[:,1:]) # TODO

        constraints_nmpc = [nEMPC.constraints.DomainConstraint(
                    states_constraint=[[-np.inf, np.inf],],
                    control_constraint=[[0.0,1.0],]),]

        objective_func  =  nEMPC.objective.jax.JAXObjectifFunc(self._get_cost_func(forcast))

        optimizer = nEMPC.optimizer.Slsqp(max_iteration=350, verbose=5, tolerance=1e-3)

        res_y, res_u = controller.next(x0=curr_y.reshape(-1))

        self.past_y = np.concatenate([self.past_y[1:, :], measurements], axis=0)
        self.past_u = np.concatenate([self.past_u[1:, :], res_u[0:1,:]], axis=0)
        return res[0]



class CNNMpc():
    def __init__(self, model_path, horizon):
        pass