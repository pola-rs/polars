try:
    import polars._plr as plr

    _POLARS_VERSION = plr.__version__
    _POLARS_BUILD_COMMIT = plr._BUILD_COMMIT
except ImportError:
    # This is only useful for documentation
    import warnings

    warnings.warn("Polars binary is missing!", stacklevel=2)
    _POLARS_VERSION = ""
    _POLARS_BUILD_COMMIT = ""


def get_polars_version() -> str:
    """
    Return the version of the Python Polars package as a string.

    If the Polars binary is missing, returns an empty string.
    """
    return _POLARS_VERSION


def get_polars_build_commit() -> str:
    """
    Return the commit SHA that Polars was built from.

    This is only baked in by the release CI workflow; local and source builds
    report `<unknown>`. If the Polars binary is missing, returns an empty string.
    """
    return _POLARS_BUILD_COMMIT
