# This module represents the Rust API functions exposed to Python through PyO3. We do a
# bit of trickery here to allow overwriting it with other function pointers.

import builtins
import os
import sys


def pllts() -> None:
    import _polars_lts_cpu._polars_lts_cpu as plr

    sys.modules[__name__] = plr


def pl64() -> None:
    import _polars64._polars64 as plr

    sys.modules[__name__] = plr


def pl32() -> None:
    import _polars32._polars32 as plr

    sys.modules[__name__] = plr


if hasattr(builtins, "__POLARS_PLR"):
    sys.modules[__name__] = builtins.__POLARS_PLR
else:
    # PyO3 registers a module called `polars` in the polars namespace that contains all
    # the functions that we want to have exposed.

    _FORCE = os.environ.get("POLARS_FORCE_PKG")
    _PREFER = os.environ.get("POLARS_PREFER_PKG")

    if (_FORCE is None and _PREFER is None) or _PREFER == "lts":
        try:
            pllts()
        except ImportError as _:
            try:
                pl64()
            except ImportError as _:
                pl32()
    elif _FORCE is not None:
        if _FORCE == "lts":
            pllts()
        elif _FORCE == "64":
            pl64()
        elif _FORCE == "32":
            pl32()
        else:
            msg = f"Invalid value for `POLARS_FORCE_PKG` variable: '{_FORCE}'"
            raise ValueError(msg)
    else:
        if _PREFER == "64":
            try:
                pl64()
            except ImportError as _:
                try:
                    pllts()
                except ImportError as _:
                    pl32()
        elif _PREFER == "32":
            try:
                pl32()
            except ImportError as _:
                try:
                    pllts()
                except ImportError as _:
                    pl64()
        else:
            msg = f"Invalid value for `POLARS_PREFER_PKG` variable: '{_PREFER}'"
            raise ValueError(msg)
