#!/bin/bash
set -e

# install test dependencies
pip install --upgrade pip
pip install pytest pytest-cov codecov sphinx

# install libexempi
# TODO in a Docker container
sudo apt update
sudo apt install libexempi3

# install the package
pip install .
