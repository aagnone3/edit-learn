.. -*- mode: rst -*-

|Travis|_ |PyPi|_ |TestStatus|_ |PythonVersion|_

.. |Travis| image:: https://travis-ci.org/aagnone3/img-edit-learn.svg?branch=master
.. _Travis: https://travis-ci.org/aagnone3/img-edit-learn

.. |PyPi| image:: https://badge.fury.io/py/img-edit-learn.svg
.. _PyPi: https://badge.fury.io/py/img-edit-learn

.. |TestStatus| image:: https://travis-ci.org/aagnone3/img-edit-learn.svg
.. _TestStatus: https://travis-ci.org/aagnone3/img-edit-learn.svg

.. |PythonVersion| image:: https://img.shields.io/pypi/pyversions/img-edit-learn.svg
.. _PythonVersion: https://img.shields.io/pypi/pyversions/img-edit-learn.svg


img-edit-learn
================

img-edit-learn is a python package offering a machine learning pipeline to
automate Adobe Lightroom edits to images, based on past edits. At a high-level,
the tool pairs image embeddings with image edits to learn edits based on the embeddings.

Currently, the embeddings are extracted using the VGG16_ convolutional neural network (CNN),
and the edits are parsed from standard Adobe Lighroom XMP files.

.. _VGG16: https://keras.io/applications/#vgg16

Documentation
-------------

Coming soon.

Dependencies
~~~~~~~~~~~~

img-edit-learn is tested to work under Python 2.7.
See the requirements via the following command::
  cat requirements.txt

Installation
~~~~~~~~~~~~

img-edit-learn is currently available on the PyPi's repository and you can
install it via `pip`::

  pip install -U img-edit-learn

If you prefer, you can clone it and run the setup.py file. Use the following
commands to get a copy from GitHub and install all dependencies::

  git clone https://github.com/scikit-learn-contrib/img-edit-learn.git
  cd img-edit-learn
  pip install .

Or install using pip and GitHub::

  pip install -U git+https://github.com/aagnone/img-edit-learn.git

Testing
~~~~~~~

TODO

About
-----

TODO
