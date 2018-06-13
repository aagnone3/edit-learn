"""
Extract a data set from NEF and XMP files.
"""
from os import path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from itertools import filterfalse
from argparse import ArgumentParser

from ielearn.util import (
    imap_unordered_bar,
    get_lines,
    fn_has_ext,
    xmp_nef_pairs,
    base_fn_add,
    raise_after_logging,
    remove_extension,
    mask_rejected_photos
)
from ielearn.extract import (
    embedding,
    xmp
)

logger = logging.getLogger("IMG-EDIT-LEARN")
logging.basicConfig(level=logging.INFO)


def is_exif_descriptor(d):
    return d.startswith("exif:")


def extract(xmp_fns, nef_fns):
    """extract"""
    if len(xmp_fns) != len(nef_fns):
        raise ValueError("# of XMP and NEF file names differ!")

    # parse the xmp files
    xmp_df = xmp.run_extraction(xmp_fns)

    # extract embeddings from the images
    embedding_df = embedding.run_extraction(nef_fns)

    # filter out photos marked as rejected in Adobe Lightroom
    rejected_mask = mask_rejected_photos(xmp_df)
    n_rejected = int(rejected_mask.sum())
    if n_rejected > 0:
        logger.info("Filtering {}/{} files as rejected".format(n_rejected, len(rejected_mask)))
    xmp_df = xmp_df.loc[~rejected_mask]
    embedding_df = embedding_df.loc[~rejected_mask]

    # segregate features and labels
    exif_cols = list(filter(is_exif_descriptor, xmp_df.columns))
    non_exif_cols = list(filterfalse(is_exif_descriptor, xmp_df.columns))
    features = pd.concat((
        embedding_df,
        xmp_df.loc[:, exif_cols]
    ), axis=1).set_index("fn")
    labels = xmp_df.loc[:, non_exif_cols].set_index("fn")
    return features, labels

    # merge the two DataFrames by their file name (with extension removed)
    # merge_col = 'fn_trunc'
    # xmp_df[merge_col] = xmp_df['fn'].map(remove_extension)
    # embedding_df[merge_col] = embedding_df['fn'].map(remove_extension)
    # del embedding_df['fn']
    # main_df = xmp_df.merge(embedding_df, how='inner', on=merge_col)
    # del main_df[merge_col]
    # return main_df


def parse_args():
    """parse_args"""
    parser = ArgumentParser()
    parser.add_argument(dest="input_fn",
                        help="Path to a file which contains a list of NEF and XMP files to parse (one per line).")
    parser.add_argument(dest="base_fn",
                        help="Base path to where the parsed data sets should be written to.")
    return parser.parse_args()


def cli():
    if __name__ == "__main__":
        args = parse_args()
        xmp_fns, nef_fns = xmp_nef_pairs(args.input_fn)
        features, labels = extract(xmp_fns, nef_fns)
        features.to_csv(base_fn_add(args.base_fn, ".features"))
        labels.to_csv(base_fn_add(args.base_fn, ".labels"))
cli()
