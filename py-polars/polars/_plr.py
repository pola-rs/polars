# This module represents the Rust API functions exposed to Python through PyO3. We do a
# bit of trickery here to allow overwriting it with other function pointers.

import builtins
import sys

if hasattr(builtins, "__POLARS_PLR"):
    sys.modules[__name__] = builtins.__POLARS_PLR
else:
    # PyO3 registers a module called `polars` in the polars namespace that contains all
    # the functions that we want to have exposed.
    import polars.polars as plr

    sys.modules[__name__] = plr
