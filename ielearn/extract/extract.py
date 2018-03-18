"""
Extract a data set from NEF and XMP files.
"""
import os
from os import path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser

from ielearn.util import (
    imap_unordered_bar,
    get_lines,
    file_parts,
    fn_has_ext,
    raise_after_logging,
    remove_extension
)
from ielearn.extract import (
    embedding,
    xmp
)

logger = logging.getLogger("EXTRACT")
logging.basicConfig(level=logging.INFO)


def xmp_nef_pairs(fns):
    # parse out the xmp and nef files
    xmp_fns = {
        path.splitext(fn)[0]: fn
        for fn in filter(partial(fn_has_ext, "xmp"), fns)
    }
    nef_fns = {
        path.splitext(fn)[0]: fn
        for fn in filter(partial(fn_has_ext, "nef"), fns)
    }

    # detect issues with the numbers of XMP and NEF files
    n_xmp = len(xmp_fns)
    n_nef = len(nef_fns)
    if n_xmp == 0:
        raise_after_logging(IOError, "No XMP files were found in {}".format(input_fn))
    if n_nef == 0:
        raise_after_logging(IOError, "No NEF files were found in {}".format(input_fn))
    if n_xmp != n_nef:
        logger.warning("A different number of XMP and NEF files were parsed. "
                       "# XMP: {}, # NEF: {}. "
                       "Only detected {{XMP, NEF}} pairs will be used.".format(n_xmp, n_nef))

    nef_fns_final = []
    xmp_fns_final = []
    for base_name, xmp_fn in xmp_fns.items():
        nef_fn = nef_fns.get(base_name, None)
        if nef_fn is not None:
            nef_fns_final.append(nef_fn)
            xmp_fns_final.append(xmp_fn)

    logger.info("Detected {} pairs of {{XMP, NEF}} files.".format(len(xmp_fns_final)))
    return xmp_fns_final, nef_fns_final


def extract(input_fn, output_fn):
    """extract"""
    # parse the passed file lists
    fns = sorted(get_lines(input_fn))

    # explicitly pair the XMP and NEF files
    xmp_fns, nef_fns = xmp_nef_pairs(fns)

    # parse the xmp files
    logger.info("Extracting XMP and EXIF data from the XMP data files.")
    xmp_df = xmp.run_extraction(xmp_fns[:100])

    # extract embeddings from the images
    logger.info("Extracting neural embeddings from the NEF images.")
    embedding_df = embedding.run_extraction(nef_fns[:100])

    # merge the two DataFrames by their file name (with extension removed)
    merge_col = 'fn_trunc'
    xmp_df[merge_col] = xmp_df['fn'].map(remove_extension)
    embedding_df[merge_col] = embedding_df['fn'].map(remove_extension)
    del embedding_df['fn']
    main_df = xmp_df.merge(embedding_df, how='inner', on=merge_col)
    del main_df[merge_col]
    main_df.to_csv(output_fn, index=False)


def parse_args():
    """parse_args"""
    parser = ArgumentParser()
    parser.add_argument(dest="input_fn",
                        help="Path to a file which contains a list of NEF and XMP files to parse (one per line).")
    parser.add_argument(dest="output_fn",
                        help="Path to where the parsed data set should be written to.")
    return parser.parse_args()


def cli():
    if __name__ == "__main__":
        args = parse_args()
        extract(args.input_fn, args.output_fn)
cli()
