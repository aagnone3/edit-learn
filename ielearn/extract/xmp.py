"""
XMP File processing (analysis and synthesis).
"""
import os
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from shutil import copyfile
from multiprocessing import cpu_count, Pool
from tempfile import NamedTemporaryFile
from functools import partial
from subprocess import check_call
from ielearn.util import (
    imap_unordered_bar,
    num_cores
)
from libxmp import XMPFiles
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
XMP_XPACKET_HEADER = "<?xpacket begin=\"\" id=\"W5M0MpCehiHzreSzNTczkc9d\"?>"
XMP_XPACKET_FOOTER = "<?xpacket end=\"w\"?>"
XMP_NS = {
    "crs": XMP_NS_CameraRaw,
    "exif": XMP_NS_EXIF
}

logger = logging.getLogger("IMG-EDIT-LEARN")
logging.basicConfig(level=logging.INFO)


def lightroom_compatible(name, value, type_map):

    # some properties expect an explicit sign symbol
    sign = type_map.loc[name, 'sign']
    prefix = ""
    if sign == "signed":
        prefix = "\+" if value > 0 else ""

    # parse based on the data type
    dtype = type_map.loc[name, 'dtype']
    if dtype == 'bool':
        if value == 1:
            rv = "True"
        elif value == 0:
            rv = "False"
        else:
            # fail-safe against a non-binary prediction
            logger.error("Non-binary output {} for expected-binary field {}".format(value, name))
    elif dtype == 'int':
        rv = "{}{}".format(prefix, int(value))
    elif dtype == 'float':
        return "{}{:.2f}".format(prefix, float(value))
    else:
        logger.error("Unplanned dtype {}.".format(dtype))
    return rv


def update_single_xmp_file(update_fn_vals, type_map):
    fn = update_fn_vals.name
    logger.info(fn)

    # TODO only for dev
    temp_fn = fn + ".bak"
    copyfile(fn, temp_fn)

    xmp_meta = file_to_meta(fn)
    for name, value in update_fn_vals.iteritems():
        cmd = "sed -i 's/{name}=\".*\"/{name}=\"{value}\"/g' \"{fn}\"".format(
            name=name,
            value=lightroom_compatible(name, value, type_map),
            fn=fn
        )
        check_call(cmd, shell=True)


def update_xmp_file_batch(update_vals, type_map):
    update_vals.apply(update_single_xmp_file, args=tuple([type_map]), axis=1)


def update_xmp_files(update_vals):
    # make updates to the XMP settings
    type_map = parse_target_types()
    update_vals.dropna(axis=1, inplace=True)
    logger.info("# files to update: {}".format(len(update_vals)))
    n_cores = num_cores()
    df_parts = np.array_split(update_vals, n_cores)
    with Pool(n_cores) as pool:
        pool.map(partial(update_xmp_file_batch, type_map=type_map), df_parts)


def file_to_meta(fn):
    xmp_files = XMPFiles()

    with open(fn, 'r') as fp:
        header = fp.readline()
    if "xpacket" in header:
        # the file is already in a compatible format for the XMP parser.
        xmp_files.open_file(fn, open_read=True)
        xmp_obj = xmp_files.get_xmp()
    else:
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
        xmp_files.open_file(temp_fn, open_read=True, open_onlyxmp=True)
        xmp_obj = xmp_files.get_xmp()
        os.remove(temp_fn)

    return xmp_obj


# def update_xmp_file(fn, xmp_obj):
#     # make changes here
#     xmp_obj.get_property(XMP_NS_CameraRaw, "RawFileName")
#     xmp_obj.set_property(XMP_NS_CameraRaw, "RawFileName", "hey")
#     xmp_obj.get_property(XMP_NS_CameraRaw, "RawFileName")
#     xmp_obj.set_array_item(XMP_NS_CameraRaw, "ToneCurve", 1, "99, 99")
#
#     # dump the updated object to the file
#     with open(fn, 'w') as fp:
#         fp.write(str(xmp_obj))


# def file_to_meta(fn):
#     with open(fn, 'r') as fp:
#         header = fp.readline()
#     if "xpacket" in header:
#         # the file is already in a compatible format for the XMP parser.
#         return file_to_dict(fn)
#
#     # need to wrap the file with a header and footer that allows
#     # the XMP parser to parse the file into a dict.
#     # we will only transform the data in a temporary file, leaving
#     # the original file untouched.
#     with NamedTemporaryFile(mode='w', delete=False) as fp,\
#             open(fn, 'r') as raw_fp:
#         temp_fn = fp.name
#         fp.write(XMP_XPACKET_HEADER + "\n")
#         for line in raw_fp:
#             fp.write("{line}\n".format(line=line))
#         fp.write(XMP_XPACKET_FOOTER + "\n")
#
#     xmp_files = XMPFiles()
#     xmp_files.open_file(temp_fn, open_read=True, open_onlyxmp=True)
#     xmp_obj = xmp_files.get_xmp()
#     os.remove(temp_fn)
#     return xmp_obj


def parse_target_types():
    return pd.read_csv(
        FN_TYPE_MAP,
        index_col=0,
        names=['prediction_type', 'sign', 'dtype']
    )


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
        dtype = type_map.loc[column, 'prediction_type']
        if not dtype:
            raise TypeError("Unexpected type {} for property {}.".format(dtype, column))

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
            if target_len > 1:
                for i, val in enumerate(values):
                    if lengths[i] < target_len:
                        values[i] = [None] * target_len
                data_fields.extend(["{}_{}".format(column, i) for i in range(target_len)])
            else:
                data_fields.append(column)

            data.extend(list(zip(*values)))

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
    xmp_to_vec(fns[0], type_map=type_map)
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
    logger.info("Extracting XMP and EXIF data from {} XMP data files.".format(len(fns)))

    # parse the map of prediction target dtypes
    type_map = parse_target_types()

    # parse the labels from each xmp file
    df = xmp_extract(fns, type_map)

    return df
