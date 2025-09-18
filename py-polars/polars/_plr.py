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
    # Each of the Polars variants registers a `_polars...` package that we can import
    # the PLR from.

    _force = os.environ.get("POLARS_FORCE_PKG")
    _prefer = os.environ.get("POLARS_PREFER_PKG")

    pkgs = {"lts": pllts, "64": pl64, "32": pl32}

    if _force is not None:
        try:
            pkgs[_force]()
        except KeyError:
            msg = f"Invalid value for `POLARS_FORCE_PKG` variable: '{_force}'"
            raise ValueError(msg) from None
    else:
        preference = [pllts, pl64, pl32]
        if _prefer is not None:
            try:
                preference.insert(0, pkgs[_prefer])
            except KeyError:
                msg = f"Invalid value for `POLARS_PREFER_PKG` variable: '{_prefer}'"
                raise ValueError(msg) from None

        for pkg in preference:
            try:
                pkg()
                break
            except ImportError:
                pass
