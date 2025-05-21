import os
import sys
from functools import partial


def verbose() -> bool:
    return os.getenv("POLARS_VERBOSE") == "1"


def eprint(*a, **kw):
    return print(*a, file=sys.stderr, **kw)
