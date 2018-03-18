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
from multiprocessing import Pool
from argparse import ArgumentParser

logger = logging.getLogger("EXTRACT")
logging.basicConfig(level=logging.INFO)


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
