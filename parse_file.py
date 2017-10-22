import os
import pdb
from glob import glob
from argparse import ArgumentParser
from libxmp.utils import file_to_dict, XMPFiles


def parse_args():
	parser = ArgumentParser()
	parser.add_argument("-f", "--file", dest="fn", help="Path of file to parse.")
	return parser.parse_args()


def main():
	args = parse_args()
	data = file_to_dict(args.fn)
	if len(data) == 0:
		print("Empty dictionary returned :(")
		exit(-1)
	print("w00t!!!")


if __name__ == "__main__":
	main()

