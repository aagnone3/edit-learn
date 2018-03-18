"""
Extract neural embeddings for all of the files specified in the passed file name list.
"""
import os
import logging
import multiprocessing as mp
import pickle

import numpy as np
import pandas as pd
from numpy.linalg import norm
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

INPUT_SHAPE = (224, 224, 3)
OUTPUT_DIMENSION = 512
MODEL = VGG16(
    weights='imagenet',
    input_shape=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), pooling='max',
    include_top=False
)


def extract_embedding(img_path):
    """
    Use a pre-trained model to extract the embedding of the picture located at img_path.
    """
    img = img_to_array(load_img(img_path, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1])))
    img = preprocess_input(np.expand_dims(img, axis=0))
    feat = MODEL.predict(img)
    norm_feat = feat[0] / norm(feat[0])
    return norm_feat


def run_extraction(fns):
    """run_extraction
    Extract features and index the images.

    :param fns:
    """

    feats = []
    n_fns = len(fns)
    for i, img_path in enumerate(fns):
        logger.info("Extracting embedding from image %i/%i.", i, n_fns)
        feats.append(extract_embedding(img_path))
    df = pd.DataFrame(
        feats,
        columns=["F{}".format(i+1) for i in range(OUTPUT_DIMENSION)]
    )
    df['fn'] = fns
    return df
