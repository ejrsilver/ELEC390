import h5py
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

infilename = 'raw_data.h5'
outfilename = 'train_test.h5'
datain = h5py.File(infilename, 'r')
dataout = h5py.File(outfilename, 'w')

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
# Top Level Groups (Each Member)
for group in datain.keys():
    # Subgroups (Jump and Walk)
    for subgroup in datain[group].keys():
        # 5-second segments
        for dataset in datain[group][subgroup].keys():
            sc = preprocessing.StandardScaler()
            df = pd.DataFrame(np.array(datain[group][subgroup][dataset]))

            # Pre-processing and Feature Extraction
            data_clean = pd.DataFrame(sc.fit_transform(df.iloc[:, 1:].rolling(5).mean().dropna()))
            data_out = data_clean.max().tolist()
            data_out += data_clean.min().tolist()
            data_out += data_clean.mean().tolist()
            data_out += data_clean.skew().tolist()
            data_out += data_clean.std().tolist()
            data_out += data_clean.kurt().tolist()
            # 0 for walking, 1 for jumping
            data_out += [0 if subgroup == 'Walk' else 1]

            features.append(data_out)

feat = pd.DataFrame(features, columns=['Max X', 'Max Y', 'Max Z', 'Max Abs', 'Min X', 'Min Y', 'Min Z', 'Min Abs',
                                       'Mean X', 'Mean Y', 'Mean Z', 'Mean Abs', 'Skew X', 'Skew Y', 'Skew Z',
                                       'Skew Abs', 'STD X', 'STD Y', 'STD Z', 'STD Abs', 'Kurt X', 'Kurt Y',
                                       'Kurt Z', 'Kurt Abs', 'Label'])

train, test = train_test_split(feat, test_size=.10, random_state=5, shuffle=True)
dataout.create_group('dataset')
GTest = dataout.create_group('dataset/Test')
GTrain = dataout.create_group('dataset/Train')

GTest.create_dataset(name='test', data=test)

GTrain.create_dataset(name='train', data=train)
