# flake8: noqa
from .datatypes import *
from .frame import DataFrame, StringCache, wrap_df
from .functions import *
from .lazy import *
from .series import Series, wrap_s

# during docs building the binary code is not yet available
try:
    from .frame import version

    __version__ = version()
except ImportError:
    pass

__pdoc__ = {"ffi": False}
