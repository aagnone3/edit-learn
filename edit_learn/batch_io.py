"""
A collection of utility functions for file i/o.
"""
import os
from abc import ABCMeta

import pandas as pd
import numpy as np


def extension(fn):
    """extension

    :param fn:
    """
    return os.path.splitext(fn)[1]


class Factory(object):
    """
    Abstract factory class to dynamically return a specific internal instance.
    Subclasses of this class should define the internal classes supported.
    """
    __metaclass__ = ABCMeta

    # defined by subclasses
    _classes = {}

    @staticmethod
    def __new__(cls, *args, **kwargs):
        """
        Return an instantiated instance of the correct type.
        :param name:
            Name of the class to create.
        :param args:
            Positional arguments to pass to the class constructor.
        :param kwargs:
            Keyword arguments to pass to the class constructor.
        :return:
            Initialized instance.
        """
        # check the mapping from name to class pointer.
        class_name = args[0]
        algo = cls._classes.get(class_name)

        # if a bad key is given, raise an error.
        if not algo:
            raise ValueError("The requested class {} does not exist."
                             "Supported classes are {}".format(class_name, cls._classes.keys()))

        # if all is well, initialize the instance and return it.
        return algo(*args[1:], **kwargs)


class BatchFileHandler(object):

    DEFAULT_BATCH_SIZE = 5000
    JOIN_FUNCS = {
        ".pkl": pd.concat,
        ".csv": pd.concat
    }
    SUPPORTED_EXTENSIONS = (".pkl", ".csv")

    def __init__(self, file_ext, batch_size):
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        if file_ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError("Invalid file type specified: {}".format(file_ext))
        self.file_ext = file_ext
        self.join_func = BatchFileHandler.JOIN_FUNCS[self.file_ext]

    def verify_path(self, fn):
        """verify_path

        :param fn:
        """
        if not os.path.isfile(fn) or not os.path.exists(fn):
            raise IOError("Invalid path given for read: {}".format(fn))

    def __enter__(self):
        """__enter__"""
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        """__exit__

        :param exc_type:
        :param exc_value:
        :param exc_tb:
        """
        if exc_value:
            raise exc_value


class BatchFileReader(BatchFileHandler):
    """
    Reads files into a DataFrame in a batchd fashion.
    """
    READ_FUNCS = {
        ".pkl": pd.read_pickle,
        ".csv": pd.read_csv
    }

    def __init__(self, fns, batch_size=None):
        super(BatchFileReader, self).__init__(extension(fns[0]), batch_size)
        for fn in fns:
            self.verify_path(fn)
        self.fns = fns
        self.read_func = self.READ_FUNCS[self.file_ext]

    def read(self):
        """read"""
        return self.join_func([
            self.read_func(fn)
            for fn in fns
        ])


class BatchFileWriter(BatchFileHandler):
    """
    Writes a DataFrame to files in a batchd fashion.
    """
    WRITE_FUNCS = {
        ".pkl": pd.DataFrame.to_pickle,
        ".csv": pd.DataFrame.to_csv
    }

    def __init__(self, fn, batch_size=None):
        """__init__

        :param df:
        :param fn:
        :param batch_size:
        """
        super(BatchFileWriter, self).__init__(extension(fn), batch_size)
        self.fn = fn
        self.write_func = self.WRITE_FUNCS[self.file_ext]

    def write(self, df):
        """write

        :param df:
        """
        self.write_func(df, self.fn)


class BatchFileIO(Factory):
    """
    Factory to return a batch reader/writer.
    """
    _classes = {
        "r": BatchFileReader,
        "w": BatchFileWriter
    }


def demo():
    """demo"""
    # batch read
    from glob import glob
    data = BatchFileIO('r', glob("*.pkl")).read()
    print data.shape

    # batch write
    BatchFileIO('w', 'new_file.pkl').write(data)
    print os.path.exists('new_file.pkl')


if __name__ == '__main__':
    demo()
