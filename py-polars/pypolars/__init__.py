from .series import Series, wrap_s
from .frame import DataFrame
from .pandas import *

# needed for side effects
from pypolars.lazy import *

__pdoc__ = {"ffi": False}
