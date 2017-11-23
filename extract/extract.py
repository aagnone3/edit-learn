import os
import pdb
import numpy as np
import pandas as pd
from argparse import ArgumentParser
from libxmp.utils import file_to_dict, XMPFiles
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

META_NAMES = []
ALL_PROPERTIES = [XMP_NS_EXIF, XMP_NS_EXIF_Aux, XMP_NS_Photoshop, XMP_NS_XMP, XMP_NS_DC, XMP_NS_XMP_MM, XMP_NS_CameraRaw, XMP_NS_TIFF]
PROPERTIES = [XMP_NS_EXIF, XMP_NS_TIFF]
#PROPERTIES = [XMP_NS_EXIF, XMP_NS_EXIF_Aux, XMP_NS_CameraRaw, XMP_NS_TIFF]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("-f", "--file", dest="fn", help="Path to a file which contains a list of XMP files to parse (one per line).")
    return parser.parse_args()


def xmp_extract(fns):
    data = []
    for fn in fns:
        data.append([])
        df.loc[os.path.basename(fn)] = xmp_to_vec(fn)
    return df


def xmp_to_vec(fn):
    xmp_data = file_to_dict(fn)
    df = pd.DataFrame([tup[:2] for _, data in xmp_data.items() for tup in data], columns=["property", "value"])
    data = []
    for elem in df["property"]:
        property_details = elem.split(":")
        data.append((property_details[0], ":".join(property_details[1:])))
    df["group"], df["subgroup"] = zip(*data)
    desired_fields = pd.read_csv("desired_fields").values
    new_df = df.loc[df["property"].isin(desired_fields)]
    pdb.set_trace()

    clean_data = [
        [
            elem[1]
            for elem in data[prop]
        ]
        for prop in PROPERTIES
    ]
    vec = np.array(clean_data)
    return vec


def xmp_to_df(fn):
    data = file_to_dict(fn)
    print(data[XMP_NS_EXIF])

    clean_data = [
        [
            [prop, elem[0], elem[1]]
            for elem in data[prop]
        ]
        for prop in PROPERTIES
    ]
    df = pd.DataFrame(np.vstack(clean_data), columns=["PropertyType", "Property", "Value"])
    return df


def main():
    args = parse_args()
    with open(args.fn) as fp:
        fns = fp.read().splitlines()
    #df = xmp_to_df(fns[0])
    df = xmp_extract(fns)
    pdb.set_trace()


if __name__ == "__main__":
    main()
