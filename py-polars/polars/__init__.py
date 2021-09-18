# flake8: noqa
import warnings

try:
    from polars.polars import version
except ImportError as e:
    version = lambda: ""
    # this is only useful for documentation
    warnings.warn("polars binary missing!")

# mypy needs these imported explicitly
from polars.eager.frame import DataFrame, wrap_df
from polars.eager.series import Series, wrap_s
from polars.lazy.expr import Expr, wrap_expr
from polars.lazy.frame import LazyFrame, wrap_ldf

from . import cfg, convert, datatypes, eager, functions, io, lazy, string_cache
from .cfg import *
from .convert import *
from .datatypes import *
from .eager import *
from .functions import *
from .io import *

# explicit imports make mypy happy
from .lazy import *
from .lazy import col, lit
from .lazy import to_list as list
from .string_cache import *

__all__ = (
    convert.__all__
    + datatypes.__all__
    + eager.__all__
    + functions.__all__
    + io.__all__
    + lazy.__all__
    + string_cache.__all__
    + cfg.__all__
)

__version__ = version()
