from __future__ import annotations

import os
import re
import sys

from hypothesis import settings

from polars.type_aliases import ParametricProfileNames


def load_profile(
    profile: ParametricProfileNames | int = "fast", set_environment: bool = False
) -> None:
    """
    Load a named (or custom) hypothesis profile for use with the parametric tests.

    Parameters
    ----------
    profile : {str, int}, optional
        Name of the profile to load; one of "fast", "balanced", "expensive", or
        the integer number of iterations to run (which will create and register
        a custom profile with that value).

    set_environment : bool, default False
        If True, also set the environment variable ``POLARS_HYPOTHESIS_PROFILE``
        to the given profile name/value.

    Examples
    --------
    >>> # load a custom profile that will run with 1500 iterations
    >>> from polars.testing.parametric.profiles import load_profile
    >>> load_profile(1500)

    """
    common_settings = {"print_blob": True, "deadline": None}
    profile_name = str(profile)

    # Register standard/named profiles
    # --------------------------------
    for name, iterations in (
        ("fast", 100),
        ("balanced", 1_000),
        ("expensive", 10_000),
    ):
        settings.register_profile(
            name=f"polars.{name}",
            max_examples=iterations,
            **common_settings,  # type: ignore[arg-type]
        )

    # Register a custom profile with 'n' iterations.
    # ----------------------------------------------
    # (Set the ideal number to balance time-vs-coverage for your machine).
    if profile_name.isdigit() or re.match(r"polars\.custom\.[\d_]+$", profile_name):
        n_iterations = int(profile_name.replace("polars.custom.", ""))
        profile_name = f"polars.custom.{profile_name}"
        settings.register_profile(
            name=profile_name,
            max_examples=n_iterations,
            **common_settings,  # type: ignore[arg-type]
        )

    # Load the chosen profile
    profile_name = f"polars.{profile_name.replace('polars.', '')}"
    settings.load_profile(profile_name)

    if set_environment:
        set_profile(profile_name)  # type: ignore[arg-type]


def set_profile(profile: ParametricProfileNames | int) -> None:
    """
    Set the env var ``POLARS_HYPOTHESIS_PROFILE`` to the given profile name/value.

    Parameters
    ----------
    profile : {str, int}, optional
        Name of the profile to load; one of "fast", "balanced", "expensive", or
        the integer number of iterations to run (which will create and register
        a custom profile with that value).

    Examples
    --------
    >>> # prefer the 'balanced' profile for running parametric tests
    >>> from polars.testing.parametric.profiles import set_profile
    >>> set_profile("balanced")

    """
    profile_name = str(profile).split(".")[-1]
    if profile_name.replace("_", "").isdigit():
        profile_name = str(int(profile_name))

    elif sys.version_info >= (3, 8):
        from typing import get_args

        valid_profile_names = get_args(ParametricProfileNames)
        if profile_name not in valid_profile_names:
            raise ValueError(
                f"Invalid profile name {profile_name!r}; expected one of {valid_profile_names}!r"
            )

    os.environ["POLARS_HYPOTHESIS_PROFILE"] = profile_name
