import os
import pdb
from glob import glob
from libxmp.utils import file_to_dict, XMPFiles

fns = glob(os.path.join("xmp_files", "*.xmp"))
for fn in fns:
	xmp = file_to_dict(fn)
	xmp2 = XMPFiles(file_path=fn)
	print(xmp)
	pdb.set_trace()

