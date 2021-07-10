import sys

from polars import *
from polars import __version__

print(
    "WARNING!\npy-polars was renamed to polars, please install polars!\nhttps://pypi.org/project/polars/",
    file=sys.stderr,
)
