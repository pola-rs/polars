from __future__ import annotations

import inspect
import re
import sys
from importlib import import_module
from importlib.util import find_spec
from types import ModuleType
from typing import TYPE_CHECKING, Any

_FSSPEC_AVAILABLE = True
_NUMPY_AVAILABLE = True
_PANDAS_AVAILABLE = True
_PYARROW_AVAILABLE = True
_ZONEINFO_AVAILABLE = True
_HYPOTHESIS_AVAILABLE = True


class _LazyModule(ModuleType):
    """
    Module that can act both as a lazy-loader and as a proxy.

    Notes
    -----
    We do NOT register this module with `sys.modules` so as not to cause
    confusion in the global environment. This way we have a valid proxy
    module for our own use, but it lives _exclusively_ within polars.

    """

    _mod_pfx: dict[str, str] = {
        "numpy": "np.",
        "pandas": "pd.",
        "pyarrow": "pa.",
    }

    def __init__(
        self,
        module_name: str,
        module_available: bool,
    ) -> None:
        """
        Initialise lazy-loading proxy module.

        Parameters
        ----------
        module_name : str
            the name of the module to lazy-load (if available).

        module_available : bool
            indicate if the referenced module is actually available (we will proxy it
            in both cases, but raise a helpful error when invoked if it doesn't exist).

        """
        self._module_available = module_available
        self._module_name = module_name
        self._globals = globals()
        super().__init__(module_name)

    def _import(self) -> ModuleType:
        # import the referenced module, replacing the proxy in this module's globals
        module = import_module(self.__name__)
        self._globals[self._module_name] = module
        self.__dict__.update(module.__dict__)
        return module

    def __getattr__(self, attr: Any) -> Any:
        # have "hasattr('__wrapped__')" return False without triggering import
        # (it's for decorators, not modules, but keeps "make doctest" happy)
        if attr == "__wrapped__":
            raise AttributeError(
                f"{self._module_name!r} object has no attribute {attr!r}"
            )

        # accessing the proxy module's attributes triggers import of the real thing
        if self._module_available:
            # import the module and return the requested attribute
            module = self._import()
            return getattr(module, attr)
        else:
            # user has not installed the proxied module
            if re.match(r"^__\w+__$", attr):
                # allow some minimal introspection on private module
                # attrs to avoid unnecessary error-handling elsewhere
                return None

            # all other attribute access raises a helpful exception
            pfx = self._mod_pfx.get(self._module_name, "")
            raise ModuleNotFoundError(
                f"{pfx}{attr} requires '{self._module_name}' module to be installed"
            ) from None


def _lazy_import(module_name: str) -> tuple[ModuleType, bool]:
    """
    Lazy import the given module; avoids up-front import costs.

    Parameters
    ----------
    module_name : str
        name of the module to import, eg: "pyarrow".

    Notes
    -----
    If the requested module is not available (eg: has not been installed), a proxy
    module is created in its place, which raises an exception on any attribute
    access. This allows for import and use as normal, without requiring explicit
    guard conditions - if the module is never used, no exception occurs; if it
    is, then a helpful exception is raised.

    Returns
    -------
    tuple[Module, bool]: a lazy-loading module and a boolean indicating if the
    requested/underlying module exists (if not, the returned module is a proxy).

    """
    # check if module is LOADED
    if module_name in sys.modules:
        return sys.modules[module_name], True

    # check if module is AVAILABLE
    try:
        module_spec = find_spec(module_name)
        module_available = not (module_spec is None or module_spec.loader is None)
    except ModuleNotFoundError:
        module_available = False

    # create lazy/proxy module that imports the real one on first use
    # (or raises an explanatory ModuleNotFoundError if not available)
    return (
        _LazyModule(
            module_name=module_name,
            module_available=module_available,
        ),
        module_available,
    )


if TYPE_CHECKING:
    import fsspec
    import hypothesis
    import numpy
    import pandas
    import pyarrow

    if sys.version_info >= (3, 9):
        import zoneinfo
    else:
        from backports import zoneinfo
else:
    fsspec, _FSSPEC_AVAILABLE = _lazy_import("fsspec")
    numpy, _NUMPY_AVAILABLE = _lazy_import("numpy")
    pandas, _PANDAS_AVAILABLE = _lazy_import("pandas")
    pyarrow, _PYARROW_AVAILABLE = _lazy_import("pyarrow")
    hypothesis, _HYPOTHESIS_AVAILABLE = _lazy_import("hypothesis")
    zoneinfo, _ZONEINFO_AVAILABLE = (
        _lazy_import("zoneinfo")
        if sys.version_info >= (3, 9)
        else _lazy_import("backports.zoneinfo")
    )


def _NUMPY_TYPE(obj: Any) -> bool:
    return _NUMPY_AVAILABLE and any(
        "numpy." in str(o)
        for o in (obj if inspect.isclass(obj) else obj.__class__).mro()
    )


def _PANDAS_TYPE(obj: Any) -> bool:
    return _PANDAS_AVAILABLE and any(
        "pandas." in str(o)
        for o in (obj if inspect.isclass(obj) else obj.__class__).mro()
    )


def _PYARROW_TYPE(obj: Any) -> bool:
    return _PYARROW_AVAILABLE and any(
        "pyarrow." in str(o)
        for o in (obj if inspect.isclass(obj) else obj.__class__).mro()
    )


__all__ = [
    "fsspec",
    "numpy",
    "pandas",
    "pyarrow",
    "zoneinfo",
    "_LazyModule",
    "_FSSPEC_AVAILABLE",
    "_NUMPY_AVAILABLE",
    "_NUMPY_TYPE",
    "_PANDAS_AVAILABLE",
    "_PANDAS_TYPE",
    "_PYARROW_AVAILABLE",
    "_PYARROW_TYPE",
    "_ZONEINFO_AVAILABLE",
    "_HYPOTHESIS_AVAILABLE",
]
