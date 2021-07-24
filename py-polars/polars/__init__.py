# flake8: noqa

# mypy needs these imported explicitly
from polars.eager.frame import DataFrame, wrap_df
from polars.eager.series import Series, wrap_s
from polars.lazy.expr import Expr, wrap_expr
from polars.lazy.frame import LazyFrame, wrap_ldf

from . import convert, datatypes, eager, functions, io, lazy, string_cache
from .convert import *
from .datatypes import *
from .eager import *
from .functions import *
from .io import *
from .lazy import *
from .string_cache import *

# during docs building the binary code is not yet available
try:
    from .polars import version

    __version__ = version()
except ImportError:
    import warnings

    warnings.warn("Polars binary files missing!", UserWarning, stacklevel=2)

__all__ = (
    convert.__all__
    + datatypes.__all__
    + eager.__all__
    + functions.__all__
    + io.__all__
    + lazy.__all__
    + string_cache.__all__
)
