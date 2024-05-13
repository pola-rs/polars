import os

from polars.testing.parametric import load_profile

load_profile(
    profile=os.environ.get("POLARS_HYPOTHESIS_PROFILE", "fast"),  # type: ignore[arg-type]
)
