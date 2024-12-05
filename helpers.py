# Original code from https://github.com/bellaanderssen/thesis23/blob/main/helpers.py

import os
import sys

from weka.core.converters import Loader, Saver
import weka.core.jvm as jvm
from weka.filters import Filter


class JVM:
    def __init__(self, max_heap_size="-Xmx20g"):
        max_heap_size="20g"
        self._max_heap_size = max_heap_size

    def __enter__(self):
        jvm.start(max_heap_size=self._max_heap_size, packages=True)

    def __exit__(self, exc_type, exc_value, traceback):
        jvm.stop()


def assert_file_exists(filepath):
    filepath = os.path.expanduser(filepath)
    if not os.path.isfile(filepath):
        print("File does not exist: {}".format(filepath))
        sys.exit(1)


def assert_dir_exists(dirpath):
    dirpath = os.path.expanduser(dirpath)
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)


def assert_dir_contains_config(dirpath):
    dirpath = os.path.expanduser(dirpath)
    assert_dir_exists(dirpath)
    config_path = os.path.join(dirpath, 'config.ini')
    assert_file_exists(config_path)
    return config_path

def load_arff(filepath):
    loader = Loader(classname="weka.core.converters.ArffLoader")
    filepath = os.path.expanduser(filepath)
    data = loader.load_file(filepath)
    data.class_is_last()
    return data


def load_csv(filepath):
    filepath = os.path.expanduser(filepath)
    loader = Loader(classname="weka.core.converters.CSVLoader")
    data = loader.load_file(filepath)
    data.class_is_last()
    return data


def save_csv(data, filepath):
    filepath = os.path.expanduser(filepath)
    saver = Saver(classname="weka.core.converters.CSVSaver")
    saver.save_file(data, filepath)


def data_to_nominal(data, filter_options="-R first-last"):
    num_to_nominal = Filter(
        classname="weka.filters.unsupervised.attribute.NumericToNominal",
        options=filter_options.split(),
    )
    num_to_nominal.inputformat(data)
    return num_to_nominal.filter(data)


def fill_na(data, fill_value='NA'):
    naFilter = Filter(
        classname="weka.filters.unsupervised.attribute.ReplaceMissingValues"
    )
    naFilter.inputformat(data)
    return naFilter.filter(data)