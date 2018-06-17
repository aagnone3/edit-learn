"""
TODO description
"""
import os
import re
import pickle
from os import path
import logging
from argparse import ArgumentParser
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
# from matplotlib import pyplot as plt
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder, Imputer
from sklearn.ensemble import (
    # RandomForestRegressor,
    #RandomForestClassifier,
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    # IsolationForest
)
from sklearn.mixture import GaussianMixture as GMM
from sklearn.cluster import KMeans
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_array, check_X_y
from sklearn.model_selection import ShuffleSplit, train_test_split, GridSearchCV
from sklearn.metrics import r2_score, mean_squared_error, f1_score, accuracy_score, make_scorer
from sklearn.feature_selection import mutual_info_regression

from ielearn.extract.xmp import parse_target_types
from ielearn.predict.feature_selection import SelectThreshold

sns.set(color_codes=True)
logger = logging.getLogger("IMG-EDIT-LEARN")
logging.basicConfig(level=logging.INFO)

MODELS_FN = "models.pkl"


class UniformPredictor(BaseEstimator, ClassifierMixin):

    def __init__(self, uniform_prediction=0, **kwargs):
        self.uniform_prediction = uniform_prediction

    def fit(self, X, y):
        X, y = check_X_y(X, y)
        self.uniform_prediction = pd.value_counts(y).idxmax()
        return self

    def predict(self, X):
        X = check_array(X)
        return self.uniform_prediction * np.ones((X.shape[0], 1))


FN_DESIRED_FIELDS = os.path.join("..", "extract", "res", "desired_fields")
SPECIAL_CHARS = re.compile("\W+")
CLF_AND_SCORER = {
    "categorical": (GradientBoostingClassifier(), "Acc", accuracy_score),
    "categorical_uniform": (UniformPredictor(), "Acc", accuracy_score),
    "numerical": (GradientBoostingRegressor(), "MSE", mean_squared_error),
    "numerical_uniform": (UniformPredictor(), "MSE", mean_squared_error)
}



def select_features(df, target_types):
    """select_features

    :param df:
    """
    logger.info("{} samples".format(df.shape[0]))
    embedding_cols = ["F{}".format(i + 1) for i in range(512)]
    exif_cols = [col for col in df.columns if "exif" in col]
    feature_cols = embedding_cols + exif_cols
    features = df.loc[:, feature_cols]

    logger.info("{} embedding features".format(len(embedding_cols)))
    logger.info("{} exif data features".format(len(exif_cols)))
    logger.info("{} features".format(features.shape[1]))

    # transform exif features using one-hot or standardization
    # for col in exif_cols:
    #     target_type = target_types[col]
    #     if target_type == "binary":
    #         new_features = binarizer.fit(features[col])

    #null_mask = features.isnull().sum() > 0
    #logger.info("Removing {} features with more than 0 null values: {}".format(
    #    len(null_mask[null_mask]),
    #    features.columns[null_mask]))
    #features = features.loc[:, ~null_mask]

    logger.info("Finalized {} features.".format(features.shape[1]))
    return features


def detect_label_outliers(y, encoder=None):
    # remove outliers
    # TODO per-class outlier removal also

    # if an encoder is provided, use it to invert the encoding before computing stats.
    if encoder:
        y = encoder.inverse_transform(y)

    # perform outlier removal on mean/var-normalized samples (after filling missing data).
    y_std = RobustScaler().fit_transform(
        Imputer(strategy="median").fit_transform(
            y.reshape([-1, 1])
        )
    ).reshape([-1])
    mask = (y_std < y_std.mean() - 3*y_std.std()) | (y_std > y_std.mean() + 3*y_std.std())
    logger.info("Filtering {}/{} as outliers.".format(mask.sum(), len(mask)))
    return mask


def select_labels(df, target_types):
    """select_labels

    :param df:
    :param target_types:
    """

    def to_lookup(tok):
        """to_lookup

        :param tok:
        """
        pos = tok.find("_")
        if pos > -1:
            return tok[:pos]
        return tok

    label_cols = [col for col in df.columns if "crs" in col]
    labels = df.loc[:, label_cols]
    logger.info("Started with {} labels".format(labels.shape[1]))

    no_tone_curve_cols = [col for col in labels.columns if "ToneCurve" not in col]
    logger.info("\nRemoving {} tone curve labels.".format(len(no_tone_curve_cols)))
    labels = labels.loc[:, labels.columns.isin(no_tone_curve_cols)]

    # note that these are missing labels, not missing features. so we need semi-supervised
    # learning, not data imputation.
    # TODO make sure this is ok
    num_missing_labels = labels.isnull().sum()
    incomplete_label_counts = dict(num_missing_labels[num_missing_labels > 0])
    if len(incomplete_label_counts) > 0:
        logger.warning("Missing labels")
        for label, n_missing in incomplete_label_counts.items():
            logger.warning("  - {label}: {num} missing.".format(label=label, num=n_missing))
            to_impute = labels[label].values.reshape((-1, 1))
            label_imputer = Imputer(strategy="most_frequent").fit(to_impute)
            labels.loc[:, label] = label_imputer.transform(to_impute)
        logger.error("Zero-filling potentially a ton of labels. Verify after prototyping.")

    # remove labels with negligible variance
    MIN_VAR = 0.001
    low_var = RobustScaler().fit_transform(labels).var(axis=0) < MIN_VAR
    logger.info("Removing {}/{} labels with small variance.".format(low_var.sum(), len(low_var)))
    logger.info(labels.columns[low_var])
    labels = labels.loc[:, ~low_var]

    # converting regression -> classification for targets with a small number of unique values
    CATEG_NUMER_THRESH = 10
    # maximum prior of any class for a target label to be selected for modeling. above this threshold, you're
    # better off predicting that class.
    MAX_SINGLE_CLASS_PRIOR = 0.99
    # minimum class prior for the class to remain intact (and not be ignored)
    MIN_SINGLE_CLASS_PRIOR = 0.001
    n_convert = 0
    n_dominant_prior = 0
    labels_data = defaultdict(lambda: dict())
    new_categ_classes = []
    for col in labels.columns:
        # convert a numberical label to categorical if there are only a "few" unique values
        # if the resulting categorical distribution is dominated by one class, don't perform any modeling
        n_unique = len(labels[col].unique())
        few_n_unique = n_unique < CATEG_NUMER_THRESH

        label_counts = labels[col].value_counts().sort_values(ascending=False)
        label_counts /= label_counts.sum()
        if label_counts.values[0] >= MAX_SINGLE_CLASS_PRIOR:
            logger.warning("Target label {} will not be modeled, as its main class {} has a prior of {:.5f}, which "
                           "is higher than the maximum allowed of {}.".format(
                               col, label_counts.index[0], label_counts.values[0], MAX_SINGLE_CLASS_PRIOR))
            # TODO this needs to turn into a uniform predictor, not avoiding modeling altogether.
            labels_data[col]["data"] = np.full_like(y, label_counts.index[0])
            labels_data[col]["type"] = "categorical_uniform"
            n_dominant_prior += 1
        else:
            # TODO don't just use a threshold number of unique values, choose based on the modes of the distribution
            # TODO generalize Temperature model VQ to other models
            y = labels[col].values
            if col == "crs:Temperature":
                # TODO Quadratic Discriminant Analysis

                outlier_mask = detect_label_outliers(y)
                y = y[~outlier_mask]

                #clus = KMeans(n_clusters=5).fit(y.reshape([-1, 1]))
                #modes = clus.cluster_centers_
                clus = GMM(
                    n_components=5,
                    covariance_type='diag',
                    reg_covar=1e-3,
                    means_init=np.array([7200, 6430, 5177, 3774, 2500]).reshape([-1, 1])
                ).fit(y.reshape([-1, 1]))
                modes = clus.means_

                distances = np.abs(modes.reshape([1, -1]) - y.reshape([-1, 1]))
                inds = np.argmin(distances, axis=1)
                y_vq = modes.astype(int)[inds].reshape([-1,])
                logger.info("Fitted Temperature Modes: {}".format(modes))
                encoder = LabelEncoder().fit(y_vq)
                labels_data[col]["data"] = encoder.transform(y_vq)
                labels_data[col]["type"] = "categorical"
                labels_data[col]["encoder"] = encoder
                labels_data[col]["sample_mask"] = ~outlier_mask
                n_convert += 1
                new_categ_classes.append(col)
            else:
                if few_n_unique:
                    classes_insuff_prior = label_counts.index[label_counts.values < MIN_SINGLE_CLASS_PRIOR]
                    use_mask = ~np.isin(y, classes_insuff_prior)
                    labels_data[col]["sample_mask"] = use_mask
                    y = y[use_mask]
                    if len(classes_insuff_prior) > 0:
                        logger.info("Targets {} ({}/{}) have priors under the configured minimum of {}. This led to a "
                                    "removal of {}/{} samples.".format(
                                        classes_insuff_prior, len(classes_insuff_prior), len(label_counts),
                                        MIN_SINGLE_CLASS_PRIOR, np.sum(~use_mask), len(use_mask)))

                    n_convert += 1
                    new_categ_classes.append(col)
                    labels_data[col]["type"] = "categorical"
                    logger.info("Class {} converted to categorical with {} target classes.".format(col, n_unique))
                else:
                    labels_data[col]["type"] = target_types.loc[to_lookup(col), 'prediction_type']

                # for categorical labels, use a label encoder and store the encoder. either way, store the data
                if labels_data[col]["type"] == "categorical":
                    encoder = LabelEncoder().fit(labels[col])
                    labels_data[col]["data"] = encoder.transform(y)
                    labels_data[col]["encoder"] = encoder
                else:
                    labels_data[col]["data"] = y

    msg = "{}/{} labels were not modeled due to a dominant target class or value.\n".format(
        n_dominant_prior, len(labels.columns))
    msg += "Converted {}/{} labels to categorical.\n".format(n_convert, len(labels.columns))
    msg += "Finalized {} labels.\n".format(len(labels_data))
    logger.info(msg)
    logger.info((Counter(
        [d["type"] for d in list(labels_data.values())]
    )))

    return labels_data


def nmi(X, y):
    """
    Normalized mutual information between X and y.
    :param X:
    :param y:
    """
    mi = mutual_info_regression(X, y)
    return mi / mi.max()


def build_uniform_predictor(label, X, model_info, inner_clf, metric_name, metric_func):
    y, target_type = model_info["data"], model_info["type"]
    sample_mask = model_info.get('sample_mask', np.ones(len(y), dtype=bool))
    X = X.loc[sample_mask]
    clf = Pipeline([
        ("impute", Imputer(strategy="most_frequent" if target_type == "categorical" else "mean")),
        ("predict", inner_clf)
    ])
    clf.fit(X, y)
    return clf, None


def get_train_test_split(X, y, is_categorical=False, log_counts=False):
    """

    :param X:
    :param y:
    """
    label_counts = pd.Series(Counter(y)).sort_values(ascending=False)
    if is_categorical and label_counts.values[1:].sum() > 5:
        # we want to stratify the labels, but minority classes will likely only contain a few samples.
        # so we'll stratify by the top few labels only.
        logger.info("Stratifying the train-test split.")
        label_ratios = (label_counts / label_counts.sum())
        top_labels = label_ratios.index[:1]
        y_strat = pd.Series(y).isin(top_labels).values

        stratified_splitter = ShuffleSplit(n_splits=1, train_size=0.8)
        train_ind, test_ind = next(stratified_splitter.split(X, y_strat))
        X_train, X_test, y_train, y_test = X.iloc[train_ind], X.iloc[test_ind], y[train_ind], y[test_ind]
    else:
        logger.info("Not stratifying the train-test split.")
        splitter = ShuffleSplit(n_splits=1, train_size=0.8)
        train_ind, test_ind = next(splitter.split(X, y))
        X_train, y_train, X_test, y_test = X.iloc[train_ind], y[train_ind], X.iloc[test_ind], y[test_ind]

    logger.info("X_train: {}".format(X_train.shape))
    logger.info("X_test: {}".format(X_test.shape))
    logger.info("y_train: {}".format(y_train.shape))
    logger.info("y_test: {}".format(y_test.shape))
    if is_categorical:
        logger.info("y class counts: {}".format(Counter(y)))
        logger.info("y_train class counts: {}".format(Counter(y_train)))
        logger.info("y_test class counts: {}".format(Counter(y_test)))

    return X_train, y_train, X_test, y_test


def build_predictor(label, X, model_info, inner_clf, metric_name, metric_func):
    """build_model

    :param X:
    :param y:
    :param target_type:
    """
    y, target_type, encoder = model_info["data"], model_info["type"], model_info.get("encoder")
    sample_mask = model_info.get('sample_mask', np.ones(len(y), dtype=bool))
    logger.info("Sample mask: {}".format(Counter(sample_mask)))
    X = X.loc[sample_mask]

    # if the majority class isn't totally dominant, stratify the train-test split.
    X_train, y_train, X_test, y_test = get_train_test_split(X, y, is_categorical="categorical" in target_type)

    # parameter fitting
    params = {
        "n_estimators": [250],
        "learning_rate": [1e-2, 0.05],
        "max_depth": [3, 4],
        "subsample": [0.9],
        "min_samples_split": [2, 1e-2],
        "min_samples_leaf": [1, 1e-2],
        # "max_features": [None, 'auto'],
    }
    clf = Pipeline([
        ("impute", Imputer(strategy="most_frequent" if target_type == "categorical" else "mean")),
        ("select", SelectThreshold(nmi, thresh=0.2)),
        ("standardize", RobustScaler()),
        ("predict", GridSearchCV(inner_clf, params,
                                 scoring=make_scorer(metric_func), n_jobs=-1, verbose=0, return_train_score=True))
    ])
    grid_search = clf.named_steps["predict"]
    clf.fit(X_train, y_train)

    # performance on the training data
    y_pred_train = clf.predict(X_train)
    logger.info(("({label}) Performance on Train ({metric_name}): {metric_value}".format(
        label=label, metric_name=metric_name, metric_value=metric_func(y_train, y_pred_train))
    ))

    # performance on the evaluation data
    y_pred = clf.predict(X_test)
    logger.info(("({label}) Performance on Test ({metric_name}): {metric_value}".format(
        label=label, metric_name=metric_name, metric_value=metric_func(y_test, y_pred))
    ))

    # grab some desired data from the grid search to store to disk
    desired_columns = ["mean_test_score", "mean_train_score"] +\
        list([c for c in list(grid_search.cv_results_.keys()) if "param_" in c]) +\
        list([c for c in list(grid_search.cv_results_.keys()) if "rank" in c])
    results = pd.DataFrame.from_dict(grid_search.cv_results_).loc[:, desired_columns].sort_values("rank_test_score")
    return clf, results


def build_model(label, X, model_info, inner_clf, metric_name, metric_func):
    """build_model

    :param X:
    :param y:
    :param target_type:
    """
    FUNC_MAPPING = {
        "categorical": build_predictor,
        "categorical_uniform": build_uniform_predictor,
        "numerical": build_predictor,
        "numerical_uniform": build_uniform_predictor
    }

    y, target_type, encoder = model_info["data"], model_info["type"], model_info.get("encoder")
    func = FUNC_MAPPING[target_type]
    return func(label, X, model_info, inner_clf, metric_name, metric_func)


def ensure_dir(dir_name):
    if not path.exists(dir_name):
        os.makedirs(dir_name)


def train_models(features_fn, labels_fn, target_label):
    features_df = pd.read_csv(features_fn)
    labels_df = pd.read_csv(labels_fn)
    target_types = parse_target_types().replace("binary", "categorical")

    # prep features
    features = select_features(features_df, target_types)

    # prep labels
    labels_data = select_labels(labels_df, target_types)

    # save a list of the feature names for prediction-time
    data = {
        "feature_names": features.columns,
        "models": {}
    }

    for label, train_info in labels_data.items():
        if target_label is None or label == target_label:
            logger.info("Training {model_type} model for {label}".format(model_type=train_info["type"], label=label))

            target_type = train_info["type"]
            inner_clf, metric_name, metric_func = CLF_AND_SCORER[target_type]
            clf, results = build_model(label, features, train_info, inner_clf, metric_name, metric_func)
            data["models"][label] = {
                "encoder": train_info.get("encoder"),
                "predictor": clf,
                "stats": results
            }

    return data


def persist_models(model_data, fn):
    # only if there are models to persist
    n_models = len(model_data["models"])
    if n_models > 0:
        # if the models file already exists, update the data structure with the passed models and re-write to disk.
        # else, persist the passed data structure.
        if path.exists(fn):
            logger.info("Updating {} with the {} new finalized models.".format(fn, n_models))
            existing_model_data = joblib.load(fn)
            for name, data in model_data["models"].items():
                existing_model_data['models'][name] = data
            joblib.dump(existing_model_data, fn)
        else:
            logger.info("Persisting the {} finalized models to {}.".format(n_models, fn))
            joblib.dump(model_data, fn)

    else:
        logger.warning("Not updating models in {} since no models were passed to persist.".format(fn))


def cli():
    if __name__ == '__main__':

        def parse_args():
            """parse_args
            """
            parser = ArgumentParser()
            parser.add_argument(dest='features_fn', help='Path to csv data with features.')
            parser.add_argument(dest='labels_fn', help='Path to csv data with labels.')
            parser.add_argument('--target_label', dest='target_label', default=None, required=False,
                                help='A single target label to model.')
            parser.add_argument('--models_fn', dest='models_fn', default=MODELS_FN, required=False,
                                help='An existing models pickle file to use.')
            args = parser.parse_args()
            return args

        args = parse_args()
        model_data = train_models(args.features_fn, args.labels_fn, args.target_label)
        persist_models(model_data, args.models_fn)

cli()
