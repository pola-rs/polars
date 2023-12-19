"""
Determine whether Polars can be run on the current CPU.

This must be done in pure Python, before the Polars binary is imported.
"""
# Set to True during build process if Polars was built with a limited instruction set
_LTS_CPU = False


# TODO: Implement this.
def __verify_cpu_support() -> None:
    """Raise an informative error if Polars cannot be run on the current CPU."""
    ...
