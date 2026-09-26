from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypeAlias

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from polars._typing import SchemaDict
    from polars.dataframe.frame import DataFrame
    from polars.lazyframe.resolver import FilterExpr

FilterSupport: TypeAlias = Literal["partial", "full"]


class FileReaderBuilder(abc.ABC):
    """
    File reader builder.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    def reader_capabilities(self) -> ReaderCapabilities:
        """Get the capabilities of the reader."""
        return ReaderCapabilities()

    @abc.abstractmethod
    def build_file_reader(
        self,
        source: str | bytes,
        multi_scan_context: dict[Any, Any],
    ) -> FileReader:
        """
        Build a file reader for the given source.

        Note: This can be called concurrently.

        Parameters
        ----------
        source
            Source to scan from.
        multi_scan_context
            Shared context for this scan.
        """

    def explain_properties(self) -> dict[str, str]:
        """Properties to display in IR explain."""
        return {}


class FileReader(abc.ABC):
    """
    File reader implementation.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.
    """

    # Rust: FileReader::schema
    @abc.abstractmethod
    def schema(self) -> SchemaDict:
        """Get the schema of the columns contained in the file."""

    # Rust: FileReader::initialize
    def fetch_metadata(self) -> None:
        """
        Fetch file metadata.

        This is guaranteed to be called before any other methods on the FileReader.
        FileReaders for formats that have relatively small metadata sections can
        implement metadata prefetch logic here.
        """
        return None

    # Rust: FileReader::prepare_read
    def initialize_serial(self) -> None:
        """
        Initializer called serially, in order for each reader when reading from a files list.

        This may not be called e.g. on early-stop.
        """  # noqa: W505
        return None

    # Rust: FileReader::begin_read
    @abc.abstractmethod
    def collect_batches(
        self,
        *,
        columns: Sequence[str],
        row_index: tuple[str, int] | None,
        pre_slice: tuple[int, int] | None,
        filters: Sequence[FilterExpr],
        num_pipelines: int,
    ) -> Iterator[DataFrame]:
        """
        Create an iterator that yields DataFrame chunks of the file with the requested parameters.

        Expected apply order for requested operations: row_index, pre_slice, filters.

        Parameters
        ----------
        columns
            Columns to read.

            Note, this can be an empty list. In this case, the reader should still
            yield DataFrames with the correct height.
        row_index
            Requested row index. Not given if ReaderCapabilities.row_index was False.
        pre_slice
            Requested pre-slice. Not given if ReaderCapabilities.pre_slice was False.
        filters
            Requested filters. Not given if ReaderCapabilities.supported_filter was None.
        num_pipelines
            Recommended number of pipelines to use.
        """  # noqa: W505

    @abc.abstractmethod
    def n_rows_in_file(self, limit: int | None = None) -> int:
        """
        Return the number of rows in the file.

        A `limit` may be passed, in this case the reader may stop counting if
        the number of rows counted reaches `limit`.
        """

    def on_drop(self) -> None:
        """Called when this reader is dropped."""
        return None


@dataclass
class ReaderCapabilities:
    """
    Reader capabilities.

    Determines which parameters will be passed to the reader.

    .. warning::
        This functionality is considered **unstable**. It may be changed
        at any point without it being considered a breaking change.

    Parameters
    ----------
    row_index
        The reader supports applying a row index.
    pre_slice
        The reader supports slicing the data.
    negative_pre_slice
        The reader supports slicing the data with a negative offset (offset
        relative to end of file).
    supported_filter
        The reader supports applying filters:

        * "partial" - Filters are not fully applied; polars will evaluate / filter
          rows returned by the reader.
        * "full" - Filters are fully applied; polars will not evaluate the filter
          on rows returned by the reader.
    """

    row_index: bool = False
    pre_slice: bool = False
    negative_pre_slice: bool = False
    supported_filter: FilterSupport | None = None


def send_dfs(
    dfs_iter: Iterator[DataFrame],
    tx: Any,  # ExternalPythonReaderDataFrameTx
) -> None:
    for df in dfs_iter:
        if not tx.send_df(df):
            break
