.DEFAULT_GOAL := help
PWD ?= $(PWD)

MODULE_NAME = $(shell grep "MODULE_NAME =" setup.py | head -n1 | cut -d= -f2 | sed 's/["\s ]//g')
VERSION_FN = ${MODULE_NAME}/version.py

help:
	@echo "Personal container."
	@echo ""
	@echo "Targets:"
	@echo "  help          Print this help message"
	@echo "  build         Build the package"
	@echo "  pypi          Upload the the main PyPi server"
	@echo "  pypi_test     Upload the the test PyPi server"
	@echo "  bump_version  Bump the module version"

.PHONY: build
build:
	python setup.py sdist bdist_wheel

.PHONY: bump_version
bump_version:
	bin/bump_version ${VERSION_FN} --minor

.PHONY: pypi_test
pypi_test: build
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

.PHONY: pypi
pypi: build bump_version
	twine upload dist/*

.PHONY: test
test:
	PYTHONPATH=${PYTHONPATH}:${PWD}/${MODULE_NAME} py.test -v
