"""
XMP File processing (analysis and synthesis).
"""
import os
import logging
import numpy as np
import pandas as pd
from tempfile import NamedTemporaryFile
from tqdm import tqdm
from functools import partial
from ielearn.util import imap_unordered_bar
from libxmp.utils import file_to_dict
from libxmp.consts import (
    XMP_NS_EXIF_Aux,
    XMP_NS_Photoshop,
    XMP_NS_EXIF,
    XMP_NS_XMP,
    XMP_NS_DC,
    XMP_NS_XMP_MM,
    XMP_NS_CameraRaw,
    XMP_NS_TIFF
)

XMP_PROPERTIES = (
    XMP_NS_EXIF,
    XMP_NS_EXIF_Aux,
    XMP_NS_Photoshop,
    XMP_NS_XMP,
    XMP_NS_DC,
    XMP_NS_XMP_MM,
    XMP_NS_CameraRaw,
    XMP_NS_TIFF
)
PROPERTIES = (
    XMP_NS_EXIF,
    XMP_NS_TIFF
)
FN_TYPE_MAP = os.path.join(os.path.dirname(__file__), "res", "type_map.csv")
XMP_XPACKET_HEADER="<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
XMP_XPACKET_FOOTER="<?xpacket end=\"w\"?>"

logger = logging.getLogger("EXTRACT")
logging.basicConfig(level=logging.INFO)


def parse_target_types():
    temp = pd.read_csv(FN_TYPE_MAP, index_col=0, names=['dtype'])
    return pd.Series(index=temp.index, data=temp['dtype'])


def parse_floats(s):
    """parse_floats

    :param s:
    """
    if isinstance(s, str):
        if "/" in s:
            # parse a ratio to its float value
            num, den = s.split("/")
            return [float(num) / float(den)]
        elif "," in s:
            # parse a csv variable into multiple new columns
            return [float(el) for el in s.split(",")]
        else:
            # parse to float directly
            return [float(s)]
    else:
        # parse to float directly
        return [float(s)]


def convert_types(df, type_map):
    """convert_types

    :param df:
    """

    data = []
    data_fields = []
    logging.info("Converting data types for the parsed XMP data.")
    for column in tqdm(df.columns):
        dtype = type_map.get(column, None)
        if not dtype:
            raise TypeError("Unexpected type {} for property {}".format(dtype, column))

        if dtype == "categorical":
            values = pd.get_dummies(df[column]).values.tolist()
            data.extend(list(zip(*values)))
            data_fields.extend(["{}_{}".format(column, i) for i in range(len(values[0]))])
        elif dtype == "binary":
            data.append(df[column].fillna(0).replace({"True": 1, "False": 0}).astype(int).values.tolist())
            data_fields.append(column)
        else:
            # dtype == "numerical"
            values = df[column].replace('', np.nan).apply(parse_floats).values.tolist()
            lengths = np.array([len(val) if isinstance(val, list) else 1 for val in values])
            target_len = np.max(lengths)
            if np.any(lengths > 1):
                for i, val in enumerate(values):
                    if lengths[i] < target_len:
                        values[i] = [None] * target_len
            values = list(zip(*values))
            data.extend(values)
            data_fields.extend(["{}_{}".format(column, i) for i in range(target_len)])

        if len(data) != len(data_fields):
            raise RuntimeError("The number of data data_fields and the number of data column names is different.")

    return data_fields, data


def parse_xmp_data(fn):
    with open(fn, 'r') as fp:
        header = fp.readline()
    if "xpacket" in header:
        # the file is already in a compatible format for the XMP parser.
        return file_to_dict(fn)

    # need to wrap the file with a header and footer that allows
    # the XMP parser to parse the file into a dict.
    # we will only transform the data in a temporary file, leaving
    # the original file untouched.
    with NamedTemporaryFile(mode='w', delete=False) as fp,\
            open(fn, 'r') as raw_fp:
        temp_fn = fp.name
        fp.write(XMP_XPACKET_HEADER + "\n")
        for line in raw_fp:
            fp.write("{line}\n".format(line=line))
        fp.write(XMP_XPACKET_FOOTER + "\n")
    xmp_data = file_to_dict(temp_fn)
    os.remove(temp_fn)
    return xmp_data


def xmp_to_vec(fn, type_map):
    """xmp_to_vec

    :param fn:
    :param type_map:
    """
    # read in the core data of interest from the XMP file.
    xmp_data = parse_xmp_data(fn)
    xmp_data = file_to_dict(fn)
    df = pd.DataFrame(
        [
            tup[:2]
            for _, data in list(xmp_data.items())
            for tup in data
        ],
        columns=["field", "value"]
    )

    # filter down to the desired properties only.
    df = df.loc[df['field'].isin(type_map.index)]

    # return a mapping from the desired properties to their values.
    return {
        field: value
        for field, value in zip(df["field"].values, df["value"].values)
    }


def xmp_extract(fns, type_map):
    """xmp_extract

    :param fns:
    :param type_map:
    """
    logger.info("Extracting raw XMP data.")
    func = partial(xmp_to_vec, type_map=type_map)
    xmp_data = imap_unordered_bar(func, fns, n_proc=2)
    xmp_data = pd.DataFrame(xmp_data)

    # convert the data types
    data_fields, data = convert_types(xmp_data, type_map)
    df = pd.DataFrame(data).transpose()
    df.columns = data_fields
    df['fn'] = fns

    return df


def run_extraction(fns):
    """run_extraction

    :param fns:
    """
    # parse the map of prediction target dtypes
    type_map = parse_target_types()

    # parse the labels from each xmp file
    df = xmp_extract(fns, type_map)

    return df
