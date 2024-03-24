from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable

if TYPE_CHECKING:
    import pyarrow as pa


class ODBCCursorProxy:
    """Cursor proxy for ODBC connections (requires `arrow-odbc`)."""

    def __init__(self, connection_string: str) -> None:
        self.connection_string = connection_string
        self.execute_options: dict[str, Any] = {}
        self.query: str | None = None

    def close(self) -> None:
        """Close the cursor (n/a: nothing to close)."""

    def execute(self, query: str, **execute_options: Any) -> None:
        """Execute a query (n/a: just store query for the fetch* methods)."""
        self.execute_options = execute_options
        self.query = query

    def fetch_arrow_table(
        self, batch_size: int = 10_000, *, fetch_all: bool = False
    ) -> pa.Table:
        """Fetch all results as a pyarrow Table."""
        from pyarrow import Table

        return Table.from_batches(
            self.fetch_record_batches(batch_size=batch_size, fetch_all=True)
        )

    def fetch_record_batches(
        self, batch_size: int = 10_000, *, fetch_all: bool = False
    ) -> Iterable[pa.RecordBatch]:
        """Fetch results as an iterable of RecordBatches."""
        from arrow_odbc import read_arrow_batches_from_odbc
        from pyarrow import RecordBatch

        n_batches = 0
        batch_reader = read_arrow_batches_from_odbc(
            query=self.query,
            batch_size=batch_size,
            connection_string=self.connection_string,
            **self.execute_options,
        )
        for batch in batch_reader:
            yield batch
            n_batches += 1

        if n_batches == 0 and fetch_all:
            # empty result set; return empty batch with accurate schema
            yield RecordBatch.from_pylist([], schema=batch_reader.schema)

    # note: internally arrow-odbc always reads batches
    fetchall = fetch_arrow_table
    fetchmany = fetch_record_batches
