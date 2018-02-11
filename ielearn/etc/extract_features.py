"""
Extract neural embeddings for all of the files specified in the passed file name list.
"""
import os
import argparse
import logging
import multiprocessing as mp
import cPickle as pickle

import numpy as np
from numpy.linalg import norm
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
INPUT_SHAPE = (224, 224, 3)
MODEL = VGG16(
    weights='imagenet',
    INPUT_SHAPE=(INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]), pooling='max',
    include_top=False
)


def extract_feat(img_path):
    """
    Use a pre-trained model to extract the embedding of the picture located at img_path.
    """
    img = img_to_array(load_img(img_path, target_size=(INPUT_SHAPE[0], INPUT_SHAPE[1])))
    img = preprocess_input(np.expand_dims(img, axis=0))
    feat = model.predict(img)
    norm_feat = feat[0] / norm(feat[0])
    return norm_feat


def parse_args():
    """
    Parse arguments from the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("-directory", required=True,
                        help="Name of directory which contains images to be indexed")
    parser.add_argument("-index", required=True,
                        help="Name of index file")
    return ap.parse_args()


def main():
    """
    Extract features and index the images.
    """
    args = parse_args()
    with open(args.fn) as fp:
        img_list = [line.strip("\n") for line in fp.readlines()]

    feats = []
    for i, img_path in enumerate(img_list):
        logger.info("Extracting embedding from image %i/%i.", i, len(img_list))
        feats.append(extract_feat(img_path))
    feats = np.array(feats)

    logger.info("writing extracted embeddings to disk.")
    with open(args.index, 'wb') as fp:
        pickle.dump({'features': feats, 'names': img_list}, fp)


if __name__ == "__main__":
    main()
