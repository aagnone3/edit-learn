#! /bin/bash

# vars
PYTEST_OPTIONS=-xvs
TEST_DIR=test

# add the package's root directory to the PYTHONPATH
export PYTHONPATH=${TRAVIS_BUILD_DIR}

# invocation
py.test ${PYTEST_OPTIONS} ${TEST_DIR}

