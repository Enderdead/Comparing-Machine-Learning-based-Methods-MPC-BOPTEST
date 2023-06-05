import pandas as pd
import numpy as np 
import jax.numpy as jnp

def extend_dataset_Ivert(dataset, N_Tvert):
    
    I_vert = dataset["weaSta_reaWeaHGloHor_y"]*np.cos(np.pi/2 - dataset["weaSta_reaWeaSolAlt_y"])/np.sin(np.pi/2 - dataset["weaSta_reaWeaSolAlt_y"])
    dataset["relative_time_day"] = dataset["time"]%86400
    categorie = np.floor((dataset["time"]%86400)/(86400/N_Tvert))
    one_hot_mat = np.zeros((len(categorie),N_Tvert))
    one_hot_mat[np.arange(0, len(categorie)),categorie.values.astype(np.int)] = 1.0
    one_hot_mat = one_hot_mat*I_vert.values.reshape(-1,1)
    dataset[[f"Tvert_{i}" for i in range(N_Tvert)]] = one_hot_mat

    dataset["cos_day"] = np.cos(2*np.pi*(dataset["time"]%86400.0)/86400.0)
    dataset["sin_day"] = np.sin(2*np.pi*(dataset["time"]%86400.0)/86400.0)

    return dataset



def IO_transform(y, u, tvp=None, na=2, nb=3):
    assert y.shape[0] == u.shape[0]

    if not (tvp is None):
        assert y.shape[0] == tvp.shape[0]

    # We do not use u[t] for y[t] prediction
    offset = max(nb, na)# +1 enlevé
    extended_u = np.stack([u[i-nb:i] for i in range( offset, u.shape[0])])
    extended_y = np.stack([y[i-na:i] for i in range( offset, u.shape[0])])

    if not tvp is None: extended_tvp = np.stack([tvp[i-nb:i] for i in range( offset, u.shape[0])])

    y_t_1 = y[offset:]
    y_t   = y[offset-1:-1]

    if tvp is None:
        return extended_u, extended_y, y_t_1, y_t_1-y_t
    else:
        return extended_y, extended_u, extended_tvp, y_t_1, y_t_1-y_t

def IO_transform_jax(y, u, na=2, nb=3):
    assert y.shape[0] == u.shape[0]

    # We do not use u[t] for y[t] prediction
    offset = max(nb, na+1)
    extended_u = jnp.stack([u[i-nb:i] for i in range( offset, u.shape[0])])
    extended_y = jnp.stack([y[i-na:i] for i in range( offset, u.shape[0])])
    y_t_1 = y[offset:]
    y_t   = y[offset-1:-1]

    return extended_u, extended_y, y_t_1,  y_t_1-y_t