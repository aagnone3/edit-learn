import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
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
FN_DESIRED_FIELDS = os.path.join(os.path.dirname(__file__), "res", "desired_labels")
logger = logging.getLogger("EXTRACT")
logging.basicConfig(level=logging.INFO)


def parse_target_types():
    temp = pd.read_csv(FN_DESIRED_FIELDS, index_col=0, names=['dtype'])
    return pd.Series(index=temp.index, data=temp['dtype'])

def xmp_extract(fns, type_map):
    """xmp_extract

    :param fns:
    :param type_map:
    """
    data = []
    logger.info("Extracting raw XMP data.")
    for fn in tqdm(fns):
        c, d = xmp_to_vec(fn, type_map)
        data.append(pd.DataFrame(d.reshape((1, -1)), columns=c))
    return pd.concat(data)


def xmp_to_vec(fn, type_map):
    """xmp_to_vec

    :param fn:
    :param type_map:
    """
    # read in the core data of interest from the XMP file.
    xmp_data = file_to_dict(fn)
    df = pd.DataFrame([tup[:2] for _, data in xmp_data.items() for tup in data], columns=["field", "value"])

    # filter down to the desired properties only.
    df = df.loc[df['field'].isin(type_map.index)]

    return df["field"].values, df["value"].values


def convert_types(df, type_map):
    """convert_types

    :param df:
    """

    def str_to_float(s):
        """str_to_float

        :param s:
        """
        if isinstance(s, (str, unicode)):
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

    converted = []
    columns = []
    logging.info("Converting data types for the parsed XMP data.")
    for column in tqdm(df.columns):
        dtype = type_map[column]
        if dtype == "categorical":
            values = pd.get_dummies(df[column]).values.tolist()
            converted.extend(zip(*values))
            columns.extend(["{}_{}".format(column, i) for i in xrange(len(values[0]))])
        elif dtype == "binary":
            converted.append(df[column].fillna(0).replace({"True": 1, "False": 0}).astype(int).values.tolist())
            columns.append(column)
        elif dtype == "numerical":
            values = df[column].replace('', np.nan).apply(str_to_float).values.tolist()
            lengths = np.array([len(val) if isinstance(val, list) else 1 for val in values])
            target_len = np.max(lengths)
            columns.extend(["{}_{}".format(column, i) for i in xrange(target_len)])
            if np.any(lengths > 1):
                for i, val in enumerate(values):
                    if lengths[i] < target_len:
                        values[i] = [None] * target_len
            values = zip(*values)
            converted.extend(values)
        else:
            raise TypeError("Unexpected type {} for property {}".format(dtype, column))

        if len(converted) != len(columns):
            raise RuntimeError("The number of data columns and the number of data column names is different.")

    return columns, converted


def run_extraction(fns):
    """run_extraction

    :param fns:
    """
    # parse the map of prediction target dtypes
    type_map = parse_target_types()

    # parse the labels from each xmp file
    df = xmp_extract(fns, type_map)

    # convert the data types
    columns, data = convert_types(df, type_map)
    df = pd.DataFrame(data).transpose()
    df.columns = columns
    df['fn'] = fns
    return df

