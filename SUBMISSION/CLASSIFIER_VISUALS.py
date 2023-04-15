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
import numpy as np
import pandas as pd

# Step 2. Read the dataset as a Pandas data frame
with h5.File('./project_files.h5', 'r') as hdf:
    Train = pd.DataFrame(hdf.get('/dataset/train'))
    Test = pd.DataFrame(hdf.get('/dataset/test'))

Train = Train.dropna()
Test = Test.dropna()

Train.columns = ["avgXAccel", "maxXAccel",  "avgYAccel", "maxYAccel", "avgZAccel" ,"maxZAccel", "avgTimeAboveAvgZ", "avgAccel" , "maxAccel", "avgTimeAboveAvg", "label"]
Test.columns = ["avgXAccel", "maxXAccel",  "avgYAccel", "maxYAccel", "avgZAccel" ,"maxZAccel", "avgTimeAboveAvgZ", "avgAccel" , "maxAccel", "avgTimeAboveAvg", "label"]

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

y_pred = clf.predict(X_test)
y_clef_prob = clf.predict_proba(X_test)
print("y_pred is: ", y_pred)
print("y_clf_prob is: ", y_clef_prob)

acc = accuracy_score(Y_test, y_pred)
print("Accuracy is: ", acc)
recall = recall_score(Y_test, y_pred)
print(recall)

cm = confusion_matrix(Y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

fpr, tpr, _ = roc_curve(Y_test, y_clef_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

auc = roc_auc_score(Y_test, y_clef_prob[:, 1])
print("AUC is: ", auc)
