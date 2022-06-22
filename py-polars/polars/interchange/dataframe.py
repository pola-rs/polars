from __future__ import annotations

from typing import Any, Iterable, Optional, Sequence

import polars as pl
from polars.interchange._utils import _ix_chunk_to_polars_df
from polars.interchange.buffer import _PYARROW_AVAILABLE
from polars.interchange.column import _IXColumn


# TODO:
#  - private implementation public interface or just re-use public.
#  - protocol might default allow_copy to true, but we may not need to
#  - privately.
class IXDataFrame:
    """Polars DataFrame object that only implements the DataFrame API.

    A data frame class, with only the methods required by the interchange
    protocol defined.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string.
    Columns may be accessed by name or by position.

    This could be a public data frame class, or an object with the methods and
    attributes defined on this DataFrame class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.
    """

    def __init__(
        self, df: pl.DataFrame, nan_as_null: bool = False, allow_copy: bool = True
    ) -> None:
        """
        Produces a dictionary object following the dataframe protocol specification.

        ``nan_as_null`` is a keyword intended for the consumer to tell the
        producer to overwrite null values in the data with ``NaN`` (or ``NaT``).
        It is intended for cases where the consumer does not support the bit
        mask or byte mask that is the producer's native representation.

        ``allow_copy`` is a keyword that defines whether or not the library is
        allowed to make a copy of the data. For example, copying data would be
        necessary if a library supports strided buffers, given that this protocol
        specifies contiguous buffers.
        """
        # may not need the check
        if not isinstance(df, pl.DataFrame):
            raise ValueError("`df` must of a Polars DataFrame")

        self._df = df
        self._nan_as_null = nan_as_null
        self._allow_copy = allow_copy

    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> "IXDataFrame":
        return IXDataFrame(self._df, nan_as_null, nan_as_null)

    @property
    def metadata(self) -> dict[str, Any]:
        """
        The metadata for the data frame, as a dictionary with string keys. The
        contents of `metadata` may be anything, they are meant for a library
        to store information that it needs to, e.g., roundtrip losslessly or
        for two implementations to share data that is not (yet) part of the
        interchange protocol specification. For avoiding collisions with other
        entries, please add name the keys with the name of the library
        followed by a period and the desired name, e.g, ``package.indexcol``.
        """
        # polars doesn't need this metadata
        return dict()

    def num_columns(self) -> int:
        """
        Return the number of columns in the DataFrame.
        """
        return len(self._df.columns)

    def num_rows(self) -> int:
        # TODO: not happy with Optional, but need to flag it may be expensive
        #       why include it if it may be None - what do we expect consumers
        #       to do here?
        """
        Return the number of rows in the DataFrame, if available.
        """
        return len(self._df)

    def num_chunks(self) -> int:
        """
        Return the number of chunks the DataFrame consists of.
        """
        return self._df.n_chunks()

    def column_names(self) -> Iterable[str]:
        """
        Return an iterator yielding the column names.
        """
        return self._df.columns

    def get_column(self, i: int) -> _IXColumn:
        """
        Return the column at the indicated position.
        """
        return _IXColumn(self._df.select_at_idx(i), allow_copy=self._allow_copy)

    def get_column_by_name(self, name: str) -> _IXColumn:
        """
        Return the column whose name is the indicated name.
        """
        return _IXColumn(self._df[name], allow_copy=self._allow_copy)

    def get_columns(self) -> Iterable[_IXColumn]:
        """
        Return an iterator yielding the columns.
        """
        return [
            _IXColumn(self._df[name], allow_copy=self._allow_copy)
            for name in self._df.columns
        ]

    def select_columns(self, indices: Sequence[int]) -> "IXDataFrame":
        """
        Create a new DataFrame by selecting a subset of columns by index.
        """
        # TODO: validate don't need abc
        if not isinstance(indices, Sequence):
            raise ValueError("`indicies` is not a Sequence")

        # use list for slicing
        if not isinstance(indices, list):
            indices = list(indices)

        return IXDataFrame(self._df[:, indices], allow_copy=self._allow_copy)

    def select_columns_by_name(self, names: Sequence[str]) -> "IXDataFrame":
        """
        Create a new DataFrame by selecting a subset of columns by name.
        """
        # TODO: validate don't need abc
        if not isinstance(names, Sequence):
            raise ValueError("`names` is not a Sequence")

        # use list for slicing
        if not isinstance(names, list):
            names = list(names)

        return IXDataFrame(self._df[:, names], allow_copy=self._allow_copy)

    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable[IXDataFrame]:
        """
        Return an iterator yielding the chunks.

        By default (None), yields the chunks that the data is stored as by the
        producer. If given, ``n_chunks`` must be a multiple of
        ``self.num_chunks()``, meaning the producer must subdivide each chunk
        before yielding it.
        """
        # TODO: can columns be chunked independently?
        ...


def _from_dataframe(df: IXDataFrame, allow_copy: bool = False) -> pl.DataFrame:
    # if for some reason num_chunks is not 1 or more
    if df.num_chunks() < 1:
        ...

    # pull interchange df by chunk
    dataframes = []
    for chunk in df.get_chunks():
        dataframes.append(_ix_chunk_to_polars_df(chunk))

    return pl.concat(dataframes)


def from_dataframe(df: IXDataFrame, allow_copy: bool = False) -> pl.DataFrame:
    """Construct a Polars DataFrame from a DataFrame Interchange
    Protocol compliant DataFrame.

    Parameters
    ----------
    df : InterchangeableDataFrame
        DataFrame containing ``__dataframe__`` attribute indicating
        protocol compliance.
    allow_copy : bool, optional
        Allow copies for interchange requiring them, by default False.

    Returns
    -------
    pl.DataFrame
        Polars DataFrame

    Raises
    ------
    ValueError (TODO: Unsupported error)
        A DataFrame does not support the protocol if the attribute
        ``__dataframe__`` is not found.
    """
    if not _PYARROW_AVAILABLE:
        raise ImportError(  # pragma: no cover
            "'pyarrow' is required when using from_dataframe()."
        )

    if isinstance(df, pl.DataFrame):
        return df

    # make sure that the ``__dataframe__`` API is implemented
    if not hasattr(df, "__dataframe__"):
        raise ValueError("`df` does not support __dataframe__")

    # get interchange DataFrame object
    ix_df = df.__dataframe__(allow_copy=allow_copy)

    # TODO: extend/vstack? protocol may not support this behavior
    if ix_df.num_chunks() > 1 and not allow_copy:
        raise RuntimeError("Copies are required if dataframe contains multiple chunks")

    # convert DataFrame object to Polars DataFrame
    return _from_dataframe(df=ix_df, allow_copy=allow_copy)
