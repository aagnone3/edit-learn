from os import path
from setuptools import setup, find_packages

MODULE_NAME = "ielearn"
VERSION_FN = path.join(MODULE_NAME, "_version.py")
REQUIREMENTS_FN = path.join(path.dirname(__file__), 'requirements.txt')

# Read requirements
with open(REQUIREMENTS_FN, 'r') as fh:
    requirements = [str(x).strip() for x in fh.readlines()]

# Read from and write to the version file
with open(VERSION_FN) as fp:
    version = float(fp.readlines()[0].strip().split('=')[1].replace(' ', ''))

# Get list of data files
data_files = ['README.rst']

setup(
    name="img-edit-learn",
    version=version,
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
