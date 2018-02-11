"""
TODO description
"""
import os
from argparse import ArgumentParser
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor, GradientBoostingClassifier, IsolationForest
from sklearn.model_selection import StratifiedShuffleSplit, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score

from edit_learn.extract.xmp import parse_target_types

sns.set(color_codes=True)
FN_DESIRED_FIELDS = os.path.join("..", "extract", "res", "desired_fields")
CLF_AND_SCORER = {
    "categorical": (RandomForestClassifier(), "Acc", accuracy_score),
    "numerical": (GradientBoostingRegressor(), "MSE", mean_squared_error)
}


def select_features(df, target_types):
    """select_features

    :param df:
    """
    print("{} samples".format(df.shape[0]))
    embedding_cols = ["F{}".format(i + 1) for i in range(512)]
    exif_cols = [col for col in df.columns if "exif" in col]
    feature_cols = embedding_cols + exif_cols
    features = df.loc[:, feature_cols]

    print("{} embedding features".format(len(embedding_cols)))
    print("{} exif data features".format(len(exif_cols)))
    print("{} features".format(features.shape[1]))

    # transform exif features using one-hot or standardization
    #for col in exif_cols:
    #    target_type = target_types[col]
    #    if target_type == "binary":
    #        new_features = binarizer.fit(features[col])

    null_mask = features.isnull().sum() > 0
    print("Removing {} features with more than 0 null values.".format(len(null_mask[null_mask])))
    features = features.loc[:, ~null_mask]

    print("Finalized {} features".format(features.shape[1]))
    return features


def select_labels(df, target_types):
    """select_labels

    :param df:
    :param target_types:
    """
    label_cols = [col for col in df.columns if "crs" in col]
    labels = df.loc[:, label_cols]
    print("Started with {} labels".format(labels.shape[1]))

    no_tone_curve_cols = [col for col in labels.columns if "ToneCurve" not in col]
    print("\nRemoving {} tone curve labels.".format(len(no_tone_curve_cols)))
    labels = labels.loc[:, labels.columns.isin(no_tone_curve_cols)]

    MIN_VAR = 0.001
    low_var = RobustScaler().fit_transform(labels).var(axis=0) < MIN_VAR
    print("\nRemoving {} labels with small variance.".format(len(low_var[low_var])))
    print(labels.columns[low_var])
    labels = labels.loc[:, ~low_var]

    # consider converting numerical to categorical for outputs with a small number of unique values
    CATEG_NUMER_THRESH = 10
    n_convert = 0
    print("Considering labels to convert to categorical.")
    labels_data = defaultdict(lambda: dict())
    for col in labels.columns:
        n_unique = len(labels[col].unique())
        # TODO don't just use a threshold number of unique values, choose based on the modes of the distribution
        if n_unique < CATEG_NUMER_THRESH:
            n_convert += 1
            labels_data[col]["type"] = "categorical"
            print("\t{}: {}".format(col, n_unique))
        else:
            labels_data[col]["type"] = target_types[to_lookup(col)]

        # for categorical labels, use a label encoder and store the encoder. either way, store the data
        if labels_data[col]["type"] == "categorical":
            encoder = LabelEncoder()
            encoder = encoder.fit(labels[col])
            labels_data[col]["data"] = encoder.transform(labels[col])
            labels_data[col]["encoder"] = encoder
        else:
            labels_data[col]["data"] = labels[col]

    print("Converted {n_classes} labels to categorical.".format(n_classes=n_convert))
    print("\nFinalized {} labels".format(len(labels_data)))
    print(Counter(
        [d["type"] for d in labels_data.values()]
    ))

    return labels_data


def to_lookup(tok):
    """to_lookup

    :param tok:
    """
    pos = tok.find("_")
    if pos > -1:
        return tok[:pos]
    return tok


def build_model(X, y, target_type):
    """build_model

    :param X:
    :param y:
    :param target_type:
    """
    inner_clf, metric_name, metric_func = CLF_AND_SCORER[target_type]
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)

    params = {
        "n_estimators": [10],
        "max_features": [None],
        "max_depth": [2]
    }
    # TODO add scorer to the grid search
    clf = Pipeline([
        ("std", RobustScaler()),
        ("clf", GridSearchCV(inner_clf, params, n_jobs=-1, verbose=3, return_train_score=True))
    ])
    grid_search = clf.named_steps["clf"]
    clf.fit(X_train, y_train)
    y_pred_train = clf.predict(X_train)
    print("Train {metric_name}: {metric_value}".format(
        metric_name=metric_name, metric_value=metric_func(y_train, y_pred_train))
    )

    y_pred = clf.predict(X_test)
    print("Test {metric_name}: {metric_value}".format(
        metric_name=metric_name, metric_value=metric_func(y_test, y_pred))
    )
    sns.jointplot(x=y_test, y=y_pred, kind="reg")
    desired_columns = ["mean_test_score", "mean_train_score"] +\
        list(filter(lambda c: "param_" in c, grid_search.cv_results_.keys())) +\
        list(filter(lambda c: "rank" in c, grid_search.cv_results_.keys()))
    results = pd.DataFrame.from_dict(grid_search.cv_results_).loc[:, desired_columns].sort_values("rank_test_score")
    return results


def train_models(input_fn):
    df = pd.read_csv(input_fn)
    target_types = parse_target_types().replace("binary", "categorical")

    # prep features
    features = select_features(df, target_types)

    # prep labels
    labels_data = select_labels(df, target_types)
    results = [
        build_model(features, labels_dict["data"], labels_dict["type"])
        for label, labels_dict in labels_data.items()
        if labels_dict["type"] == 'categorical'
    ]


def parse_args():
    """parse_args
    """
    parser = ArgumentParser()
    parser.add_argument('-i', '--input-file', dest='input_file',
                        help='Input data to train on.')
    args = parser.parse_args()
    return args


def cli():
    if __name__ == '__main__':
        args = parse_args()
        train_models(args.input_file)
cli()
