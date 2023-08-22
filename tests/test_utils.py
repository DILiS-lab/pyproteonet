import os
from pathlib import Path
import urllib.request
import zipfile

from pyproteonet.data import Dataset
from pyproteonet.io import datasets

TESTDATA_DIR = Path(os.path.dirname(__file__)) / 'testdata'


def load_maxlfq_benchmark(path: Path = TESTDATA_DIR / 'maxlfq_benchmark')->Dataset:
    return datasets.load_maxlfq_benchmark(path=path)
        