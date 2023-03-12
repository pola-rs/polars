try:
    from polars.polars import get_polars_version as _get_polars_version

    polars_version_string = _get_polars_version()
except ImportError:
    # this is only useful for documentation
    import warnings

    warnings.warn("polars binary missing!", stacklevel=2)
    polars_version_string = ""


def get_polars_version() -> str:
    """
    Return the version of the Python Polars package as a string.

    If the Polars binary is missing, returns an empty string.
    """
    return polars_version_string
