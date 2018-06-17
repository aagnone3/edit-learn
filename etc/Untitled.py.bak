import os
from os import path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from scipy import stats
from sklearn.preprocessing import RobustScaler, Imputer
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.model_selection import StratifiedShuffleSplit, ShuffleSplit, GridSearchCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.metrics import (mean_squared_error, precision_recall_fscore_support, confusion_matrix,
                             precision_score, make_scorer, accuracy_score)

from feature_selection import SelectThreshold


all_nans_ind = 533
X = pd.read_csv('data.features')
y = pd.read_csv('data.labels')
X.values[:, all_nans_ind] = 0
print(X.shape, y.shape)

col = "crs:Temperature"
vals = y[col].values
mode = stats.mstats.mode(vals)[0][0]
#_ = plt.hist(vals, bins=50)
#plt.figure()
#_ = plt.hist(vals[vals != mode], bins=50)

#y['crs:Exposure2012'].hist()


# remove rejected photos
def mask_rejected_photos(x):
    return np.array(
        (x['crs:Exposure2012'] < -4) | (x['crs:Saturation'] < -99.0),
        dtype=bool
    )
mask = mask_rejected_photos(y)
print("\nRemoving {}/{} photos as rejected.".format(mask.sum(), len(mask)))
X = X.loc[~mask]
vals = vals[~mask]
print("Data shapes now {}, {}".format(X.shape, vals.shape))

y_std = RobustScaler().fit_transform(Imputer(strategy="median").fit_transform(vals.reshape([-1, 1])))
y_std = y_std.reshape([-1])
mask = (y_std < y_std.mean() - 3*y_std.std()) | (y_std > y_std.mean() + 3*y_std.std())
X = X.loc[~mask, :]
vals = vals[~mask]
print("\nFiltering {}/{} as outliers.".format(mask.sum(), len(mask)))
print("Data shapes now {}, {}".format(X.shape, vals.shape))

# imputing missing data with strategy "most_frequent"
strategy = "median"
print("Imputing missing data with strategy {}".format(strategy))
X.values[:] = Imputer(strategy=strategy, axis=0).fit_transform(X.values)

# X = X.loc[:, [col for col in X.columns if col[0] != "F"]]
#print(X.shape)

splitter = ShuffleSplit(n_splits=1, test_size=0.2)
train_ind, test_ind = next(splitter.split(X.values, vals))
X_train, y_train, X_test, y_test = X.values[train_ind], vals[train_ind], X.values[test_ind], vals[test_ind]
print("After split", X_train.shape, y_train.shape, X_test.shape, y_test.shape)


# mutual information
#mi = mutual_info_regression(X_test, y_test)
#mi /= mi.max()
#print(mi.shape)
#
#inds = np.argsort(mi)[::-1]
#print(mi[inds[:10]])
#print(X.columns[inds[:10]])
#MIN_MI = 0.2
#n_feats = np.sum(mi > 0.2)
#print("Using {} features with MI > {}".format(n_feats, MIN_MI))



#mode = stats.mstats.mode(vals)[0][0]
#_ = plt.hist(vals, bins=20)
#plt.figure()
#sns.distplot(vals)

# it's empirically seen that there are 5-6 modes of the distribution of Temperature
# use VQ (GMM-clustering-based) to assign all Temperature values to one of the 6 modes
# then, train an N-way classifier on the modes
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import LabelEncoder
gmm = GMM(
    n_components=5,
    covariance_type='diag',
    means_init=np.array([7200, 6430, 5177, 3774, 2500]).reshape([-1, 1])
).fit(vals.reshape([-1, 1]))
print(gmm.means_)
distances = np.abs(gmm.means_.reshape([1, -1]) - vals.reshape([-1, 1]))
inds = np.argmin(distances, axis=1)
y_vq = LabelEncoder().fit_transform(gmm.means_.astype(int)[inds].reshape([-1, ]))

splitter = ShuffleSplit(n_splits=1, test_size=0.2)
train_ind, test_ind = next(splitter.split(X.values, vals))
X_train, y_train, X_test, y_test = X.values[train_ind], y_vq[train_ind], X.values[test_ind], y_vq[test_ind]
print("After split", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

def nmi(X, y):
    mi = mutual_info_regression(X_train, y_train)
    return mi / mi.max()

#inds = np.argsort(mi)[::-1]
#print(mi[inds[:10]])
#print(X.columns[inds[:10]])
#MIN_MI = 0.2
#n_feats = np.sum(mi > 0.2)
#feat_cols_to_use = inds[:n_feats]
#print("Using {} features with MI > {}".format(n_feats, MIN_MI))


#feat_cols_to_use = inds[:n_feats]
params = {
    "n_estimators": [250],
    #"learning_rate": [1e-2, 0.05],
    "max_depth": [3, 4],
    #"subsample": [0.9, 0.85],
    "min_samples_split": [2, 1e-3, 1e-2],
    "min_samples_leaf": [1, 1e-3, 1e-2],
    #"min_impurity_split": [1e-5, 1e-3, 1e-1],
    "max_features": [None, 'auto'],
}
clf = Pipeline([
    ("impute", Imputer(strategy="most_frequent")),
    ("select", SelectThreshold(nmi, thresh=0.2)),
    ("clf", GridSearchCV(
        GradientBoostingClassifier(
        ), params, n_jobs=-1, verbose=1, return_train_score=True)
    )
])

clf = clf.fit(X_train, y_train)
joblib.dump(clf, 'temperature_model.pkl')
print(clf.named_steps['clf'].best_estimator_)

train_losses = accuracy_score(y_train, clf.predict(X_train))
test_losses = accuracy_score(y_test, clf.predict(X_test))
print("Train metric: {}".format(train_losses))
print("Test metric: {}".format(test_losses))
#important_feature_inds = np.argsort(clf.named_steps['clf'].best_estimator_.feature_importances_)[::-1]
#X.columns[feat_cols_to_use[:10]]

#confusion_matrix(y_test, clf.predict(X_test[:, feat_cols_to_use]))
#old_cols = X.columns

#X2 = pd.read_csv('predict/mills.features').set_index('fn', drop=True)
#X2.shape
#
#print(np.setdiff1d(old_cols, X2.columns))
#a = [i for i, e in enumerate(old_cols) if e == 'exif:MeteringMode_1']
#b = [i for i, e in enumerate(old_cols) if e == 'exif:MeteringMode_2']
#print(a, b)
#np.sort(feat_cols_to_use)
#
#y_pred = clf.predict(X2.values[:, feat_cols_to_use])
#temp_preds = gmm.means_[y_pred]
