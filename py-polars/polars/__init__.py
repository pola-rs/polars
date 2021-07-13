# flake8: noqa
from .datatypes import *
from .eager import *
from .functions import *
from .lazy import *

# during docs building the binary code is not yet available
try:
    from .polars import version

    __version__ = version()
except ImportError:
    pass
