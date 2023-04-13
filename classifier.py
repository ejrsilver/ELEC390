import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.decomposition import PCA
import h5py as h5


def Classify(model, scaler, df):

    df = df.dropna()
    myarray = np.array(df)

    for i in range(0, len(myarray)):
        if i == 0:
            myarray[i][1] = myarray[i][1]
            myarray[i][2] = myarray[i][2]
            myarray[i][3] = myarray[i][3]
        else:
            myarray[i][1] = (myarray[i][1] + myarray[i - 1][1] + myarray[i - 2][1] + myarray[i - 3][1]) / 4
            myarray[i][2] = (myarray[i][2] + myarray[i - 1][2] + myarray[i - 2][2] + myarray[i - 3][2]) / 4
            myarray[i][3] = (myarray[i][3] + myarray[i - 1][3] + myarray[i - 2][3] + myarray[i - 3][3]) / 4

    # Average total acceleration
    avgAccel = 0
    for i in range(0, len(myarray)):
        avgAccel += abs(myarray[i][4] / 500)

    # max total acceleration
    maxAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][4]) > maxAccel:
            maxAccel = abs(myarray[i][4])

    # Average X acceleration
    avgXAccel = 0
    for i in range(0, len(myarray)):
        avgXAccel += myarray[i][1] / 500

    # Average Y acceleration
    avgYAccel = 0
    for i in range(0, len(myarray)):
        avgYAccel += myarray[i][2] / 500

    # Average Z acceleration
    avgZAccel = 0
    for i in range(0, len(myarray)):
        avgZAccel += abs(myarray[i][3] / 500)
    # average time above avg Z acceleration

    avgZTime = 0
    timer = False
    times = 0
    totalTime = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][3]) >= avgZAccel:
            if timer == False:
                timer = True
                times += 1
                totalTime += 1
            else:
                totalTime += 1
        else:
            timer = False

    avgZTime = totalTime / times

    # max Z acceleration
    maxZAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][3]) > maxZAccel:
            maxZAccel = abs(myarray[i][3])

    # max X acceleration
    maxXAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][1]) > maxXAccel:
            maxXAccel = abs(myarray[i][1])

    # max Y acceleration
    maxYAccel = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][2]) > maxYAccel:
            maxYAccel = abs(myarray[i][2])

    # average time above avg acceleration

    avgTime = 0
    timer = False
    times = 0
    totalTime = 0
    for i in range(0, len(myarray)):
        if abs(myarray[i][4]) >= abs(avgAccel):
            if timer == False:
                timer = True
                times += 1
                totalTime += 1
            else:
                totalTime += 1
        else:
            timer = False

    avgTime = totalTime / times

    info = [[avgXAccel, maxXAccel, avgYAccel, maxYAccel, avgZAccel, maxZAccel, avgZTime, avgAccel, maxAccel, avgTime]]
    df = pd.DataFrame(info)
    Features = df
    df.columns = ["avgXAccel", "maxXAccel", "avgYAccel", "maxYAccel", "avgZAccel", "maxZAccel", "avgTimeAboveAvgZ",
                    "avgAccel", "maxAccel", "avgTimeAboveAvg"]

    df = scaler.transform(df)
    pred = pd.DataFrame(model.predict(df))
    final = pd.concat([Features, pred], axis=1)
    final.columns = ["avgXAccel", "maxXAccel", "avgYAccel", "maxYAccel", "avgZAccel", "maxZAccel", "avgTimeAboveAvgZ",
                  "avgAccel", "maxAccel", "avgTimeAboveAvg", 'label']
    return final





def train():
    with h5.File('./project_files.h5', 'r') as hdf:
        Train = pd.DataFrame(hdf.get('/dataset/train'))
        Test = pd.DataFrame(hdf.get('/dataset/test'))

    Train = Train.dropna()
    Test = Test.dropna()

    Train.columns = ["avgXAccel", "maxXAccel", "avgYAccel", "maxYAccel", "avgZAccel", "maxZAccel", "avgTimeAboveAvgZ",
                     "avgAccel", "maxAccel", "avgTimeAboveAvg", "label"]
    Test.columns = ["avgXAccel", "maxXAccel", "avgYAccel", "maxYAccel", "avgZAccel", "maxZAccel", "avgTimeAboveAvgZ",
                    "avgAccel", "maxAccel", "avgTimeAboveAvg", "label"]

    Y_train = Train['label']
    Y_test = Test['label']

    X_train = Train.iloc[:, :-1]
    X_test = Test.iloc[:, :-1]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Now, similar to the code presented in the lecture slides, train and test your model.

    l_reg = LogisticRegression(max_iter=10000)
    clf = make_pipeline(StandardScaler(), l_reg)

    clf.fit(X_train, Y_train)

    return clf, scaler



