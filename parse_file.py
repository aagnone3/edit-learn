import os
import pdb
import numpy as np
import pandas as pd
from glob import glob
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


def parse_args():
	parser = ArgumentParser()
	parser.add_argument("-f", "--file", dest="fn", help="Path of file to parse.")
	return parser.parse_args()


def xmp_to_df(fn):
	data = file_to_dict(fn)
	
	properties = [XMP_NS_EXIF, XMP_NS_EXIF_Aux, XMP_NS_Photoshop, XMP_NS_XMP, XMP_NS_DC, XMP_NS_XMP_MM, XMP_NS_CameraRaw, XMP_NS_TIFF]
	clean_data = [
		[
			[prop, elem[0], elem[1]]
			for elem in data[prop]
		]
		for prop in properties
	]
	df = pd.DataFrame(np.vstack(clean_data), columns=["PropertyType", "Property", "Value"])
	return df


def xmp_to_vec(fn):
	data = file_to_dict(fn)
	
	properties = [XMP_NS_EXIF, XMP_NS_EXIF_Aux, XMP_NS_Photoshop, XMP_NS_XMP, XMP_NS_DC, XMP_NS_XMP_MM, XMP_NS_CameraRaw, XMP_NS_TIFF]
	clean_data = [
		[
			elem[1]
			for elem in data[prop]
		]
		for prop in properties
	]
	vec = np.array(clean_data)
	return vec


def main():
	args = parse_args()
	df = xmp_to_df(args.fn)
	vec = xmp_to_vec(args.fn)
	pdb.set_trace()


if __name__ == "__main__":
	main()

