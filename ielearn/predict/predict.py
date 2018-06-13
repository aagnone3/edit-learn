import logging
import pickle
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from os import path
from sklearn.externals import joblib
from ielearn.extract import extract
from ielearn.extract.xmp import update_xmp_files
from ielearn.predict.train import nmi, UniformPredictor
from ielearn.util import (
    xmp_nef_pairs,
    mask_rejected_photos
)

logger = logging.getLogger("IMG-EDIT-LEARN")
DIR = path.dirname(__file__)
MODELS_FN = path.join(DIR, "models.pkl")


def predict(xmp_fns, nef_fns, models_fn):
    # load the models from disk
    model_data = joblib.load(models_fn)
    feature_names = model_data["feature_names"]
    models = model_data["models"]

    # extract image embeddings, exif features, and xmp labels
    features, labels = extract.extract(xmp_fns, nef_fns)

    # identify out any features that weren't present for modeling
    unsupp_features = np.setdiff1d(features.columns, feature_names)
    if len(unsupp_features) > 0:
        logger.warning("Detected {} features that were not present for modeling"
                       " These values therefore cannot not be used during prediction: {}".format(
                           len(unsupp_features), unsupp_features))
        features.drop(unsupp_features, axis=1, inplace=True)

    # filter out photos marked as rejected in Adobe Lightroom
    rejected_mask = mask_rejected_photos(labels)
    n_rejected = int(rejected_mask.sum())
    if n_rejected > 0:
        logger.info("Filtering {}/{} files as rejected".format(n_rejected, len(rejected_mask)))
    labels = labels.loc[~rejected_mask]
    features = features.loc[~rejected_mask]

    # identify out any modeled features that are not present now (will be missing XMP data)
    missing_features = np.setdiff1d(feature_names, features.columns)
    if len(missing_features) > 0:
        logger.warning("Detected {} features not present in probe image data"
                       " These values will be imputed: {}".format(len(missing_features), missing_features))
        # fill in the missing data columns as nan in the correct order (set by the model_data dict).
        # these values will then be imputed by the individual model pipelines.
        nans = np.full_like(np.empty((len(features), len(missing_features))), np.nan)
        nan_df = pd.DataFrame(nans, columns=missing_features, index=features.index)
        features = pd.concat((features, nan_df), axis=1).loc[:, feature_names]

    # identify target XMP settings that don't have models
    label_keys = labels.keys()
    unmodeled_xmp = np.setdiff1d(list(label_keys), list(models.keys()))
    if len(unmodeled_xmp):
        logger.warning("No predictor available for {}/{} XMP settings."
                       " These values will not be changed: {}".format(
                           len(unmodeled_xmp), len(label_keys), unmodeled_xmp))

    # use image embedding and xmp data to predict xmp edits
    new_values = pd.DataFrame(np.nan, index=labels.index, columns=labels.columns)
    for label in labels.columns:
        if label in models:
            prediction = models[label]["predictor"].predict(features)

            # if an encoder was used with this model, invert the encodind
            encoder = models[label].get('encoder')
            if encoder:
                prediction = encoder.inverse_transform(prediction)
            new_values.loc[:, label] = prediction

    return new_values


def parse_args():
    """parse_args"""
    parser = ArgumentParser()
    parser.add_argument(dest="input_fn",
                        help="Path to a file which contains a list of NEF and XMP files to parse (one per line).")
    parser.add_argument('--models_fn', dest='models_fn', default=MODELS_FN, required=False,
                        help='An existing models pickle file to use.')
    return parser.parse_args()


def cli():
    if __name__ == "__main__":
        args = parse_args()
        xmp_fns, nef_fns = xmp_nef_pairs(args.input_fn)
        update_xmp_files(predict(xmp_fns, nef_fns, args.models_fn))
cli()
