import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn import preprocessing

infilename = 'data.h5'
datain = h5py.File(infilename, 'r')

# Visualization
# js = pd.DataFrame(np.array(datain['Jump']['jump1-4']))
# js.columns = ['Time', 'Linear acc x', 'Linear acc y', 'Linear acc z', 'Absolute acc']
# jsa = js.rolling(50).mean().dropna()
# jdata = jsa.iloc[:, 1:]
# jdata = preprocessing.StandardScaler().fit_transform(jdata)
# plt.plot(jdata)
# plt.show()
# ws = pd.DataFrame(np.array(datain['Walk']['walk1-4']))
# ws.columns = ['Time', 'Linear acc x', 'Linear acc y', 'Linear acc z', 'Absolute acc']
# wsa = ws.rolling(50).mean().dropna()
# wdata = wsa.iloc[:, 1:]
# wdata = preprocessing.StandardScaler().fit_transform(wdata)
# plt.plot(wdata)
# plt.show()

features = []

for group in datain.keys():
    print(group)
    for subgroup in datain[group].keys():
        for dataset in datain[group][subgroup].keys():
            sc = preprocessing.StandardScaler()
            df = pd.DataFrame(np.array(datain[group][subgroup][dataset]))
            df.columns = ['Time', 'Linear acc x', 'Linear acc y', 'Linear acc z', 'Absolute acc']

            # Pre-processing
            df_data = df.iloc[:, 1:]
            df_clean = df_data.rolling(50).mean().dropna()
            df_clean = pd.DataFrame(sc.fit_transform(df_clean))

            # Feature extraction
            data_out = df_clean.min().tolist()
            data_out += df_clean.max().tolist()
            data_out += df_clean.mean().tolist()
            data_out += df_clean.median().tolist()
            data_out += df_clean.skew().tolist()
            data_out += [subgroup]

            features.append(data_out)

feat = pd.DataFrame(features, columns=['Min X', 'Min Y', 'Min Z', 'Min Abs', 'Max X', 'Max Y', 'Max Z', 'Max Abs',
                                       'Mean X', 'Mean Y', 'Mean Z', 'Mean Abs', 'Median X', 'Median Y', 'Median Z',
                                       'Median Abs', 'Skew X', 'Skew Y', 'Skew Z', 'Skew Abs', 'Label'])

print(feat)
