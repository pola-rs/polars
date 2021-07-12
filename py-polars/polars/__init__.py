# flake8: noqa
from .datatypes import *
from .frame import *
from .functions import *
from .lazy import *
from .series import *

# during docs building the binary code is not yet available
try:
    from .frame import version

    __version__ = version()
except ImportError:
    pass

__pdoc__ = {"ffi": False}
