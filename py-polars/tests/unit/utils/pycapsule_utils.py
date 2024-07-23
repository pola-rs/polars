from typing import Any


class PyCapsuleStreamHolder:
    """
    Hold the Arrow C Stream pycapsule.

    A class that exposes _only_ the Arrow C Stream interface via Arrow PyCapsules.
    This ensures that pyarrow is seeing _only_ the `__arrow_c_stream__` dunder, and
    that nothing else (e.g. the dataframe or array interface) is actually being
    used.

    This is used by tests across multiple files.
    """

    arrow_obj: Any

    def __init__(self, arrow_obj: object) -> None:
        self.arrow_obj = arrow_obj

    def __arrow_c_stream__(self, requested_schema: object = None) -> object:
        return self.arrow_obj.__arrow_c_stream__(requested_schema)
