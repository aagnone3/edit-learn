#!/bin/bash
set -e

# update pip
pip install --upgrade pip

# install package dependencies
#pip install -r requirements.txt

# install test dependencies
pip install pytest pytest-cov codecov sphinx

echo $PWD
ls
pip install .
