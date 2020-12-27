from .series import Series, wrap_s
from .frame import DataFrame, version
from .functions import *
from .lazy import *

__version__ = version()

__pdoc__ = {"ffi": False}
