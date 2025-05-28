from __future__ import annotations

from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from polars.io.scan_options.cast_options import ScanCastOptions


class ScanOptions:
    """Holds options for UnifiedScanArgs."""

    def __init__(
        self,
        *,
        cast_options: ScanCastOptions | None,
        extra_columns: Literal["ignore", "raise"],
        missing_columns: Literal["insert", "raise"] = "raise",
    ) -> None:
        self.cast_options = cast_options
        self.extra_columns = extra_columns
        self.missing_columns = missing_columns
