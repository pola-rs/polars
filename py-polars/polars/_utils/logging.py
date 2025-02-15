import os
import sys
from functools import partial


def verbose() -> bool:
    return os.getenv("POLARS_VERBOSE") == "1"


eprint = partial(print, file=sys.stderr)
