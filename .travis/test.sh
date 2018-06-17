#!/bin/bash
set -e

PYTEST_OPTIONS=(--cov=${MODULE} -xvs --pyargs ${MODULE})
COVERAGE_FILE=${TEST_DIR}/.coverage

# ensure a clean temporary test directory
function ensure_clean_test_dir() {
    [[ -d ${TEST_DIR} ]] && rm -rf ${TEST_DIR}
    mkdir -p ${TEST_DIR}
}

function do_test() {
    export PYTHONPATH=${PYTHONPATH}:${TRAVIS_BUILD_DIR}
    echo "PYTHONPATH: ${PYTHONPATH}"
    ls
    env
    cd ${TEST_DIR}
    py.test "${PYTEST_OPTIONS[@]}" ${TRAVIS_BUILD_DIR}
    [[ -f ${COVERAGE_FILE} ]] && cp ${COVERAGE_FILE} ${TRAVIS_BUILD_DIR} || echo "Warning: code coverage file not available."
    cd ${OLDPWD}
}

ensure_clean_test_dir
do_test
