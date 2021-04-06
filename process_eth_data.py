import numpy as np
import pandas as pd

filename="biwi_eth_test.txt"

columns = ['t', 'ped id', 'x', 'y']

data=pd.read_csv(filename,header=None,delimiter="\t",names=columns, dtype={'t': np.float64, 'ped id': np.int32, 'x': np.float64, 'y': np.float64})


timestamps = data['t'].unique()

len_tstamps = len(timestamps) 

val_tstamps = timestamps[:int(0.8*len_tstamps)]
test_tstamps = timestamps[int(0.8*len_tstamps):] 

data_val = data.loc[data['t'].isin(val_tstamps)]
data_val.to_csv('biwi_eth_train.txt', header=None, index=False, sep="\t") 

data_test = data.loc[data['t'].isin(test_tstamps)]
data_test.to_csv('biwi_eth_val.txt', header=None, index=False, sep="\t")




