#!/bin/bash
set -e

PYTEST_OPTIONS=(--cov=${MODULE} -xvs --pyargs ${MODULE})
COVERAGE_FILE=${TEST_DIR}/.coverage

py.test "${PYTEST_OPTIONS[@]}" .
cp ${COVERAGE_FILE} ${TRAVIS_BUILD_DIR}
