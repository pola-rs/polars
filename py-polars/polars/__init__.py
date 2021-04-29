# flake8: noqa
"""
.. include:: ./index.md
"""

from .series import Series, wrap_s
from .frame import DataFrame, wrap_df, StringCache
from .functions import *
from .lazy import *
from .datatypes import *

# during docs building the binary code is not yet available
try:
    from .frame import version

    __version__ = version()
except ImportError:
    pass

__pdoc__ = {"ffi": False}
