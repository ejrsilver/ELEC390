import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.inspection import DecisionBoundaryDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, recall_score, confusion_matrix, ConfusionMatrixDisplay, roc_curve, \
    RocCurveDisplay, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

infilename = 'data.h5'
datain = h5py.File(infilename, 'r')

# Import training data
train = []
for dataset in datain['dataset']['Train'].keys():
    train += np.array(datain['dataset']['Train'][dataset]).tolist()

train_dataset = pd.DataFrame(np.array(train))
train_dataset.columns=['Min X', 'Min Y', 'Min Z', 'Min Abs', 'Max X', 'Max Y', 'Max Z', 'Max Abs',
                 'Mean X', 'Mean Y', 'Mean Z', 'Mean Abs', 'Median X', 'Median Y', 'Median Z',
                'Median Abs', 'Skew X', 'Skew Y', 'Skew Z', 'Skew Abs', 'Label']

# Import testing data
test = []
for dataset in datain['dataset']['Test'].keys():
    test += np.array(datain['dataset']['Test'][dataset]).tolist()

test_dataset = pd.DataFrame(np.array(test))
test_dataset.columns=['Min X', 'Min Y', 'Min Z', 'Min Abs', 'Max X', 'Max Y', 'Max Z', 'Max Abs',
                 'Mean X', 'Mean Y', 'Mean Z', 'Mean Abs', 'Median X', 'Median Y', 'Median Z',
                'Median Abs', 'Skew X', 'Skew Y', 'Skew Z', 'Skew Abs', 'Label']

# Creating the classifier
train_X = train_dataset.iloc[:, :-1]
train_Y = train_dataset.iloc[:, -1].astype('int32')
test_X = test_dataset.iloc[:, :-1]
test_Y = test_dataset.iloc[:, -1].astype('int32')

sc = StandardScaler()
l_reg = LogisticRegression(max_iter=10000)
clf = make_pipeline(StandardScaler(), l_reg)
pca = PCA(n_components=5)
train_X = pca.fit_transform(train_X)
test_X = pca.fit_transform(test_X)
clf.fit(train_X, train_Y)
Y_pred = clf.predict(test_X)

Y_clf_prob = clf.predict_proba(test_X)
acc = accuracy_score(test_Y, Y_pred)
recall = recall_score(test_Y, Y_pred)
cm = confusion_matrix(test_Y, Y_pred)

cm_display = ConfusionMatrixDisplay(cm).plot()
plt.show()

fpr, tpr, _ = roc_curve(test_Y, Y_clf_prob[:, 1], pos_label=clf.classes_[1])
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr).plot()
plt.show()

auc = roc_auc_score(test_Y, Y_clf_prob[:, 1])
print('y_pred is: ', Y_pred)
print('Y_clf_prob is: ', Y_clf_prob)
print('accuracy is: ', acc)
print('recall is: ', recall)
print('the AUC is: ', auc)

# pca_pipe = make_pipeline(StandardScaler(), PCA(n_components=2))
# X_train_pca = pca_pipe.fit_transform(train_X)
# X_test_pca = pca_pipe.fit_transform(test_X)
#
# clf2 = make_pipeline(StandardScaler(), l_reg)
# clf2.fit(X_train_pca, train_Y)
#
# Y_pred_pca = clf2.predict(X_test_pca)
# disp = DecisionBoundaryDisplay.from_estimator(clf2, X_train_pca, response_method="predict", xlabel='X1', ylabel='X2', alpha=0.5,)
#
# disp.ax_.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=train_Y)
# plt.show()