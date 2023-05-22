
import deepSI
import numpy as np 


def train_ARX(dataset_pd, na=3, nb=3,  y_label=["reaTRoo_y"], u_label=["u","weaSta_reaWeaTDryBul_y", "weaSta_reaWeaHDirNor_y", "weaSta_reaWeaNTot_y", "weaSta_reaWeaRelHum_y", "weaSta_reaWeaSolAlt_y"], dt=900):
    assert len(dataset_pd.dropna()) == len(dataset_pd), "NAN rows"

    sys_data_train =  deepSI.System_data(y = dataset_pd[y_label], u=dataset_pd[u_label] , dt=dt)
    fit_sys_IO = deepSI.fit_systems.Sklearn_io_linear(na=3,nb=3)
    fit_sys_IO.fit(sys_data_train)


    return fit_sys_IO