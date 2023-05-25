
import matplotlib.pyplot as plt
from typeguard import typechecked
import pureSysId as psi
import matplotlib.pyplot as plt
import numpy as np
import deepSI


def compare_deepSI(dataset, model, K=48, save_img=None, quiet=False, dt=600):
    assert K!=0
        # We need to cut to sub part the data
    offset = model.init_state(dataset)

    total_size = dataset.N_samples
    nb_step = int((total_size-offset)/K)
    dataset_normed = model.norm.transform(dataset)

    model.init_state_multi(dataset_normed, nf=K, stride=K)

    pred = list()
    for i in range(K):
        pred.append(model.measure_act_multi(np.array([ dataset_normed[offset+i+k*K:offset+i+k*K+1].u[0]   for k in range(nb_step)])))
            
        #np.array([[test_normed[10+i:10+i+1].u[0]],[test_normed[20+i:20+i+1].u[0]],[test_normed[30+i:30+i+1].u[0]]]      )))
    
    pred_y = np.array(pred).transpose(1,0,2).reshape(-1,1)
    real_y = dataset_normed.y[offset:offset+K*nb_step]

    #real_y, pred_y = model.forcast(datas)
    #real_y, pred_y = np.concatenate(real_y), np.concatenate(pred_y)
    t_range = np.arange(0, real_y.shape[0])*dt
    if not quiet:
        fig, axs = plt.subplots(1, 1, sharex=True, figsize=(8,4)) 
        axs = axs if 1 > 1 else [axs]
        for y_dim, ax in enumerate(axs):
            ax.plot(t_range, real_y, label="Real")
            ax.plot(t_range, pred_y, label="Fake")
            ax.vlines(x=[dt*i*K for i in range(nb_step)], ymin=real_y.min(), ymax=real_y.max(), color="red", linestyles="dashed")
            ax.legend()

        plt.tight_layout()
        if not (save_img is None):
            plt.savefig(save_img)
        plt.show()
    return real_y, pred_y

def acc_times_full_series_deepSI(dataset, model, K=48):
    assert K!=0
        # We need to cut to sub part the data
    if isinstance(dataset, deepSI.system_data.system_data.System_data_list):
        offset = model.init_state(dataset[0])
    elif isinstance(dataset, deepSI.system_data.system_data.System_data):
        offset = model.init_state(dataset)
    else:
        raise RuntimeError("Please provide a dataset : System_data or System_data_list")

    if not isinstance(dataset, deepSI.system_data.system_data.System_data_list):
        dataset = deepSI.system_data.system_data.System_data_list([dataset])

    diff_list = list()
    for dataset_idx in range(len(dataset)):
        total_size = dataset[dataset_idx].N_samples
        nb_step = total_size-offset-K+1#int((total_size-offset)/K)
        dataset_local = dataset[dataset_idx]

        dataset_normed = model.norm.transform(dataset_local)
        model.init_state_multi(dataset_normed, nf=K, stride=1)

        pred = list()
        for i in range(K):

            pred.append(model.measure_act_multi(np.array([ dataset_normed[offset+i+k:offset+i+k+1].u[0]   for k in range(nb_step)])))
                

        pred_y = np.array(pred).transpose(1,0,2)
        real_y = np.array([dataset_normed.y[offset+i:offset+i+K] for i in range(nb_step)])

        diff = real_y-pred_y
        diff_list.append(diff)

    final_diff = np.concatenate(diff_list, axis=0)
    final_diff = np.mean(np.square(np.linalg.norm(final_diff, axis=2)), axis=0)
    final_diff = np.sqrt(final_diff)

    return final_diff
    
def acc_times_series_deepSI(dataset, model, K=48):
    assert K!=0
        # We need to cut to sub part the data
    if isinstance(dataset, deepSI.system_data.system_data.System_data_list):
        offset = model.init_state(dataset[0])
    elif isinstance(dataset, deepSI.system_data.system_data.System_data):
        offset = model.init_state(dataset)
    else:
        raise RuntimeError("Please provide a dataset : System_data or System_data_list")

    if not isinstance(dataset, deepSI.system_data.system_data.System_data_list):
        dataset = deepSI.system_data.system_data.System_data_list([dataset])

    diff_list = list()
    for dataset_idx in range(len(dataset)):
        total_size = dataset[dataset_idx].N_samples
        nb_step = total_size-offset-K+1#int((total_size-offset)/K)
        dataset_local = dataset[dataset_idx]

        dataset_normed = model.norm.transform(dataset_local)
        model.init_state_multi(dataset_normed, nf=K, stride=1)

        pred = list()
        for i in range(K):
            pred.append(model.measure_act_multi(np.array([ dataset_normed[offset+i+k:offset+i+k+1].u[0]   for k in range(nb_step)])))
                

        pred_y = np.array(pred).transpose(1,0,2)
        real_y = np.array([dataset_normed.y[offset+i:offset+i+K] for i in range(nb_step)])
        diff = real_y-pred_y
        diff_list.append(diff)

    final_diff = np.concatenate(diff_list, axis=0)
    final_diff = np.mean(np.square(final_diff), axis=0)
    final_diff = np.sqrt(final_diff)

    return final_diff
    
"""

def compare_deepSI(dataset, model, K=48):
    assert K!=0
    assert isinstance(dataset,  deepSI.system_data.system_data.System_data) and not isinstance(dataset,  deepSI.system_data.system_data.System_data_list)

    offset = model.init_state(dataset)

    nb_step = int((dataset.N_samples-offset)/K)

    for idx_pred in range(nb_step):
        model.init_state(dataset[idx_pred*K:idx_pred*K+offset])
        pred = []
        for i in range(K):
            pred.append(model.measure_act(dataset_normed[offset+idx_pred*K+i:offset+idx_pred*K+i+1].u))
                
    return pred

"""