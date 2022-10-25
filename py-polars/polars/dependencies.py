from __future__ import annotations

import re
import sys
from importlib.machinery import ModuleSpec
from importlib.util import LazyLoader, find_spec, module_from_spec
from types import ModuleType
from typing import TYPE_CHECKING, Any

_mod_pfx = {
    "numpy": "np.",
    "pandas": "pd.",
    "pyarrow": "pa.",
}

_FSSPEC_AVAILABLE = True
_NUMPY_AVAILABLE = True
_PANDAS_AVAILABLE = True
_PYARROW_AVAILABLE = True
_ZONEINFO_AVAILABLE = True
_HYPOTHESIS_AVAILABLE = True


def _proxy_module(module_name: str, register: bool = True) -> ModuleType:
    """
    Create a module that raises a helpful/explanatory exception on attribute access.

    Parameters
    ----------
    module_name : str
        the name of the new/proxy module.

    register : bool
        indicate if the module should be registered with ``sys.modules``.

    """
    # module-level getattr for the proxy
    def __getattr__(*args: Any, **kwargs: Any) -> None:
        attr = args[0]
        # handle some introspection issues on private module attrs
        if re.match(r"^__\w+__$", attr):
            return None

        # other attribute access raises exception
        pfx = _mod_pfx.get(module_name, "")
        raise ModuleNotFoundError(
            f"{pfx}{attr} requires '{module_name}' module to be installed"
        ) from None

    # create module that raises an exception on attribute access.
    proxy_module = module_from_spec(ModuleSpec(module_name, None))
    for name, obj in (("__getattr__", __getattr__),):
        setattr(proxy_module, name, obj)

    # add proxy into sys.modules under the target module's name
    if register:
        sys.modules[module_name] = proxy_module
    return proxy_module


def lazy_import(module_name: str) -> tuple[ModuleType, bool]:
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
        spec = find_spec(module_name)
        if spec is None or spec.loader is None:
            spec = None
    except ModuleNotFoundError:
        spec = None

    # if NOT available, return proxy module that raises on attribute access
    if spec is None:
        return _proxy_module(module_name), False
    else:
        # handle modules that have old-style loaders (ref: #5326)
        if not hasattr(spec.loader, "exec_module"):
            if hasattr(spec.loader, "load_module"):
                spec.loader.exec_module = (  # type: ignore[assignment, union-attr]
                    # wrap deprecated 'load_module' for use with 'exec_module'
                    lambda module: spec.loader.load_module(module.__name__)  # type: ignore[union-attr] # noqa: E501
                )
            if not hasattr(spec.loader, "create_module"):
                spec.loader.create_module = (  # type: ignore[assignment, union-attr]
                    # note: returning 'None' implies use of the standard machinery
                    lambda spec: None
                )

        # module IS available, but not yet imported into the environment; create
        # a lazy loader that proxies (then replaces) the module in sys.modules
        loader = LazyLoader(spec.loader)  # type: ignore[arg-type]
        spec.loader = loader
        module = module_from_spec(spec)
        sys.modules[module_name] = module
        loader.exec_module(module)
        return module, True


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
    fsspec, _FSSPEC_AVAILABLE = lazy_import("fsspec")
    numpy, _NUMPY_AVAILABLE = lazy_import("numpy")
    pandas, _PANDAS_AVAILABLE = lazy_import("pandas")
    pyarrow, _PYARROW_AVAILABLE = lazy_import("pyarrow")
    hypothesis, _HYPOTHESIS_AVAILABLE = lazy_import("hypothesis")
    zoneinfo, _ZONEINFO_AVAILABLE = (
        lazy_import("zoneinfo")
        if sys.version_info >= (3, 9)
        else lazy_import("backports.zoneinfo")
    )


def _NUMPY_TYPE(obj: Any) -> bool:
    return _NUMPY_AVAILABLE and "numpy." in str(type(obj))


def _PANDAS_TYPE(obj: Any) -> bool:
    return _PANDAS_AVAILABLE and "pandas." in str(type(obj))


def _PYARROW_TYPE(obj: Any) -> bool:
    return _PYARROW_AVAILABLE and "pyarrow." in str(type(obj))


__all__ = [
    "fsspec",
    "numpy",
    "pandas",
    "pyarrow",
    "zoneinfo",
    "_proxy_module",
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
