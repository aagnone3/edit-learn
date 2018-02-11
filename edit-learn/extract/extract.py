"""
Extract a data set from NEF and XMP files.
"""
import os
import logging
from argparse import ArgumentParser
import numpy as np
import pandas as pd

from edit_learn.extract import (
    embedding,
    xmp
)

logger = logging.getLogger("EXTRACT")
logging.basicConfig(level=logging.INFO)


def get_lines(fn):
    """get_lines

    :param fn:
    """
    if not os.path.exists(fn) or not os.path.isfile(fn):
        raise IOError("Invalid path given: {}".format(fn))
    with open(fn) as fp:
        lines = fp.read().splitlines()
    return lines


def remove_extension(fn):
    """remove_extension

    :param fn:
    """
    return os.path.splitext(fn)[0]


def parse_args():
    """parse_args"""
    parser = ArgumentParser()
    parser.add_argument("-i", "--nef_files", dest="nef_list_fn",
                        help="Path to a file which contains a list of NEF files to parse (one per line).")
    parser.add_argument("-x", "--xmp_files", dest="xmp_list_fn",
                        help="Path to a file which contains a list of XMP files to parse (one per line).")
    parser.add_argument("-o", "--out_file", dest="out_fn",
                        help="Path to where the parsed data set should be written to.")
    return parser.parse_args()


def extract():
    """extract"""
    # parse command-line arguments
    args = parse_args()

    # parse the passed file lists
    xmp_fns = get_lines(args.xmp_list_fn)
    nef_fns = get_lines(args.nef_list_fn)

    # detect a difference in the number of XMP and NEF files
    n_xmp = len(xmp_fns)
    n_nef = len(nef_fns)
    if n_xmp != n_nef:
        logger.warning("A different number of XMP and NEF files were parsed."
                       "# XMP: {}, # NEF: {}".format(n_xmp, n_nef))

    # parse the xmp files
    logger.info("Extracting XMP and EXIF data from the XMP data files.")
    xmp_df = xmp.run_extraction(xmp_fns)

    # extract embeddings from the images
    logger.info("Extracting neural embeddings from the NEF images.")
    embedding_df = embedding.run_extraction(nef_fns)

    # merge the two DataFrames by their file name (with extension removed)
    merge_col = 'fn_trunc'
    xmp_df[merge_col] = xmp_df['fn'].map(remove_extension)
    embedding_df[merge_col] = embedding_df['fn'].map(remove_extension)
    del embedding_df['fn']
    main_df = xmp_df.merge(embedding_df, how='inner', on=merge_col)
    del main_df[merge_col]
    main_df.to_csv(args.out_fn, index=False)


if __name__ == "__main__":
    extract()
