"""
Utility functions
"""
import os
from os import path
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from functools import partial
from multiprocessing import Pool, cpu_count
from argparse import ArgumentParser

logger = logging.getLogger("IMG-EDIT-LEARN")
logging.basicConfig(level=logging.INFO)


def num_cores():
    n_cores = cpu_count()
    return max(1, n_cores - 1)


def imap_unordered_bar(func, args, n_proc=2):
    p = Pool(n_proc)
    ret = []
    with tqdm(total=len(args)) as pbar:
        for i, res in tqdm(enumerate(p.imap_unordered(func, args))):
            pbar.update()
            ret.append(res)
    pbar.close()
    p.close()
    p.join()
    return ret


def get_lines(fn):
    """get_lines

    :param fn:
    """
    if not os.path.exists(fn) or not os.path.isfile(fn):
        raise IOError("Invalid path given: {}".format(fn))
    with open(fn) as fp:
        lines = fp.read().splitlines()
    return lines


def file_parts(fn):
    base_name, ext = path.splitext(path.basename(fn))
    return path.dirname(fn), base_name, ext


def base_fn_add(fn, tok):
    base_dir, base_name, ext = file_parts(fn)
    return "{}{}{}{}".format(base_dir, base_name, tok, ext)


def fn_has_ext(ext_query, fn):
    _, _, ext = file_parts(fn)
    return ext[1:].lower() == ext_query.lower()


def raise_after_logging(exc, msg):
    logger.error(msg)
    raise exc(msg)


def remove_extension(fn):
    """remove_extension

    :param fn:
    """
    return os.path.splitext(fn)[0]


def xmp_nef_pairs(input_fn):
    # parse the passed file lists
    fns = sorted(get_lines(input_fn))

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
        raise_after_logging(IOError, "No XMP files were found.")
    if n_nef == 0:
        raise_after_logging(IOError, "No NEF files were found.")
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


def mask_rejected_photos(xmp_df):
    # Adobe Lightroom does not store information in the xmp file on whether the photo was flagged as 'rejected',
    # so here it is assumed that a value of < -3 in the crs:Exposure2012 field or a value less than -99.0 in the
    # crs:Saturation field indicates 'rejected'
    return np.array(
        (xmp_df["crs:Exposure2012"] == -3) | (xmp_df["crs:Saturation"] < -99.0),
        dtype=bool
    )
