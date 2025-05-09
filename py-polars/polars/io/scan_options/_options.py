from typing import Literal
from polars.io.scan_options.cast_options import ScanCastOptions


class ScanOptions:
    def __init__(
        self,
        *,
        cast_options: ScanCastOptions | None,
        extra_columns: Literal["ignore", "raise"],
    ) -> None:
        self.cast_options = cast_options
        self.extra_columns = extra_columns
