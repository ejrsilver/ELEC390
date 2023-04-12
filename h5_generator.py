import h5py

ethan = h5py.File('ethan.h5', 'r')
jacob = h5py.File('Jacob_Schaffer.h5', 'r')
lauren = h5py.File('lauren_data.h5', 'r')
traintest = h5py.File('train_test.h5', 'r')
dataout = h5py.File('data.h5', 'w')

dataout.create_group('Ethan')
GEJ = dataout.create_group('Ethan/Jump')
GEW = dataout.create_group('Ethan/Walk')
dataout.create_group('Lauren')
GLJ = dataout.create_group('Lauren/Jump')
GLW = dataout.create_group('Lauren/Walk')
dataout.create_group('Jacob')
GJJ = dataout.create_group('Jacob/Jump')
GJW = dataout.create_group('Jacob/Walk')
dataout.create_group('dataset')
GTrain = dataout.create_group('dataset/Train')
GTest = dataout.create_group('dataset/Test')

# Add Ethan Data
for dataset in ethan['Jump'].keys():
    GEJ.create_dataset(name=dataset, data=ethan['Jump'][dataset])
for dataset in ethan['Walk'].keys():
    GEW.create_dataset(name=dataset, data=ethan['Walk'][dataset])

# Add Lauren Data
for dataset in lauren.keys():
    if 'jumping' in dataset:
        GLJ.create_dataset(name=dataset, data=lauren[dataset])
    if 'walking' in dataset:
        GLW.create_dataset(name=dataset, data=lauren[dataset])

# Add Jacob Data
for dataset in jacob['Jumping'].keys():
    GJJ.create_dataset(name=dataset, data=jacob['Jumping'][dataset])
for dataset in jacob['Walking'].keys():
    GJW.create_dataset(name=dataset, data=jacob['Walking'][dataset])

# Add Train Data
for dataset in traintest['dataset']['Train'].keys():
    GTrain.create_dataset(name=dataset, data=traintest['dataset']['Train'][dataset])

# Add Test Data
for dataset in traintest['dataset']['Test'].keys():
    GTest.create_dataset(name=dataset, data=traintest['dataset']['Test'][dataset])