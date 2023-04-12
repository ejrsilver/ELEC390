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

features = pd.DataFrame()
# Top Level Groups (Each Member)
for group in datain.keys():
    # Subgroups (Jump and Walk)
    for subgroup in datain[group].keys():
        # 5-second segments
        for dataset in datain[group][subgroup].keys():
            sc = preprocessing.StandardScaler()
            df = pd.DataFrame(np.array(datain[group][subgroup][dataset]))

            # Pre-processing and Feature Extraction
            df_data = df.iloc[:, 1:]

            data_max = pd.DataFrame(sc.fit_transform(df_data.rolling(50).max().dropna())).astype('float32')
            data_min = pd.DataFrame(sc.fit_transform(df_data.rolling(50).min().dropna())).astype('float32')
            data_mean = pd.DataFrame(sc.fit_transform(df_data.rolling(50).mean().dropna())).astype('float32')
            data_median = pd.DataFrame(sc.fit_transform(df_data.rolling(50).median().dropna())).astype('float32')
            data_skew = pd.DataFrame(sc.fit_transform(df_data.rolling(50).skew().dropna())).astype('float32')
            # 0 for walking, 1 for jumping
            data_labels = pd.DataFrame([0 if subgroup == 'Walk' else 1] * len(data_max)).astype('int32')

            data_out = pd.concat([data_max, data_min, data_mean, data_median, data_skew, data_labels], axis=1)
            features = pd.concat([features, data_out], axis=0)

features.columns=['Min X', 'Min Y', 'Min Z', 'Min Abs', 'Max X', 'Max Y', 'Max Z', 'Max Abs',
                 'Mean X', 'Mean Y', 'Mean Z', 'Mean Abs', 'Median X', 'Median Y', 'Median Z',
                'Median Abs', 'Skew X', 'Skew Y', 'Skew Z', 'Skew Abs', 'Label']

print(features)

X = features.iloc[:, 1:-1]
Y = features.iloc[:, -1]
flist = features.values.tolist()

train, test = train_test_split(flist, test_size=.10, random_state=5, shuffle=True)
i = 0
dataout.create_group('dataset')
GTest = dataout.create_group('dataset/Test')
GTrain = dataout.create_group('dataset/Train')

while i < len(test):
    GTest.create_dataset(name='test{}'.format(int(i/500)), data=test[i:i+499])
    i += 500

i = 0
while i < len(train):
    GTrain.create_dataset(name='train{}'.format(int(i/500)), data=train[i:i+499])
    i += 500
