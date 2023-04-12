import h5py
import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn import preprocessing

infilename = 'data.h5'
outfilename = 'data_out.h5'
datain = h5py.File(infilename, 'r')


for group in datain.keys():
    print(group)
    for dataset in datain[group].keys():
        print(dataset)
        sc = preprocessing.StandardScaler()
        df = pd.DataFrame(np.array(datain[group][dataset]))
        df.columns = ['Time', 'Linear acc x', 'Linear acc y',	'Linear acc z', 'Absolute acc']
        dfa = df.rolling(25).mean().dropna()
        data = dfa.iloc[:, 1:]
        data = sc.fit_transform(data)
        print(df)
        print(data)
