from glob import glob
from os import path
from pkg_resources import parse_version
from setuptools import setup, find_packages

ROOT_DIR = path.dirname(__file__)

# Read requirements
fn = path.join(ROOT_DIR, 'requirements.txt')
with open(fn, 'r') as fh:
    requirements = [str(x).strip() for x in fh.readlines()]

# Read from and write to the version file
fn = path.join(ROOT_DIR, "ielearn", "_version.py")
with open(fn, 'r+') as fh:
    version_found = False
    while not version_found:
        vpos = fh.tell()
        line = fh.readline()
        if line == '':
            # reached EOF without finding the version
            # End of file
            raise ValueError("Could not find __version__ in %s." % fn)
        elif line.startswith('__version__'):
            exec(line)
            version_found = True

# Get list of data files
data_files = ['README.rst']  # + glob('conf/*')

setup(
    name="img-edit-learn",
    version=__version__,
    description="Machine Learning for Personalized Image Editing",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python :: 2.7",
        "Topic :: Scientific/Engineering"
    ],
    url="https://github.com/aagnone3/img-edit-learn",
    author="Anthony Agnone",
    author_email="anthonyagnone@gmail.com",
    packages=find_packages(exclude=['*.test', 'test']),
    install_requires=requirements,
    zip_safe=False,
    data_files=[('share/aagnone/ielearn', data_files)]
)
