from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from polars import functions as F
from polars.series.utils import expr_dispatch
from polars.utils._wrap import wrap_s
from polars.utils.deprecation import (
    deprecate_renamed_function,
    deprecate_renamed_parameter,
)

if TYPE_CHECKING:
    from datetime import date, datetime, time

    from polars import Expr, Series
    from polars.polars import PySeries
    from polars.type_aliases import (
        IntoExpr,
        IntoExprColumn,
        NullBehavior,
        ToStructStrategy,
    )


@expr_dispatch
class ListNameSpace:
    """A namespace for :class:`List` `Series`."""

    _accessor = "list"

    def __init__(self, series: Series):
        self._s: PySeries = series._s

    def all(self) -> Series:
        """
        Evaluate whether all :class:`Boolean` values in each list are `true`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series(
        ...     [[True, True], [False, True], [False, False], [None], [], None],
        ...     dtype=pl.List(pl.Boolean),
        ... )
        >>> s.list.all()
        shape: (6,)
        Series: '' [bool]
        [
            true
            false
            false
            true
            true
            null
        ]

        """

    def any(self) -> Series:
        """
        Evaluate whether any :class:`Boolean` value in each list is `true`.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        Examples
        --------
        >>> s = pl.Series(
        ...     [[True, True], [False, True], [False, False], [None], [], None],
        ...     dtype=pl.List(pl.Boolean),
        ... )
        >>> s.list.any()
        shape: (6,)
        Series: '' [bool]
        [
            true
            true
            false
            false
            false
            null
        ]

        """

    def len(self) -> Series:
        """
        Get the number of elements in each list, including `null` elements.

        Returns
        -------
        Series
            A :class:`UInt32` `Series`.

        Examples
        --------
        >>> s = pl.Series([[1, 2, None], [5]])
        >>> s.list.len()
        shape: (2,)
        Series: '' [u32]
        [
            3
            1
        ]

        """

    def drop_nulls(self) -> Series:
        """
        Remove all `null` values in each list.

        The original order of the remaining list elements is preserved.

        Examples
        --------
        >>> s = pl.Series("values", [[None, 1, None, 2], [None], [3, 4]])
        >>> s.list.drop_nulls()
        shape: (3,)
        Series: 'values' [list[i64]]
        [
            [1, 2]
            []
            [3, 4]
        ]

        """

    def sample(
        self,
        n: int | IntoExprColumn | None = None,
        *,
        fraction: float | IntoExprColumn | None = None,
        with_replacement: bool = False,
        shuffle: bool = False,
        seed: int | None = None,
    ) -> Series:
        """
        Randomly sample elements from each list.

        Parameters
        ----------
        n
            The number of elements to return. Cannot be used with `fraction`. Defaults
            to `1` if `fraction` is `None`.
        fraction
            The fraction of elements to return. Cannot be used with `n`.
        with_replacement
            Whether to allow elements to be sampled more than once.
        shuffle
            Whether to shuffle the order of the sampled elements. If `shuffle=False`
            (the default), the order will be neither stable nor fully random.
        seed
            The seed for the random number generator. If `seed=None` (the default), a
            random seed is generated anew for each `sample` operation. Set to an integer
            (e.g. `seed=0`) for fully reproducible results.

        Warnings
        --------
        `sample(fraction=1)` returns the expression as-is! To properly shuffle the
        values, add `shuffle=True`.

        Examples
        --------
        >>> s = pl.Series("values", [[1, 2, 3], [4, 5]])
        >>> s.list.sample(n=pl.Series("n", [2, 1]), seed=1)
        shape: (2,)
        Series: 'values' [list[i64]]
        [
            [2, 1]
            [5]
        ]

        """

    def sum(self) -> Series:
        """
        Get the sum of the elements in each list.

        Examples
        --------
        >>> s = pl.Series("values", [[1], [2, 3]])
        >>> s.list.sum()
        shape: (2,)
        Series: 'values' [i64]
        [
                1
                5
        ]
        """

    def max(self) -> Series:
        """
        Get the maximum value of the elements in each list.

        Examples
        --------
        >>> s = pl.Series("values", [[1], [2, 3]])
        >>> s.list.max()
        shape: (2,)
        Series: 'values' [i64]
        [
                1
                3
        ]
        """

    def min(self) -> Series:
        """
        Get the minimum value of the elements in each list.

        Examples
        --------
        >>> s = pl.Series("values", [[1], [2, 3]])
        >>> s.list.min()
        shape: (2,)
        Series: 'values' [i64]
        [
                1
                2
        ]
        """

    def mean(self) -> Series:
        """
        Get the mean of the elements in each list.

        Examples
        --------
        >>> s = pl.Series("values", [[1], [2, 3]])
        >>> s.list.mean()
        shape: (2,)
        Series: 'values' [f64]
        [
                1.0
                2.5
        ]
        """

    def sort(self, *, descending: bool = False) -> Series:
        """
        Sort each list.

        Parameters
        ----------
        descending
            Whether to sort in descending instead of ascending order.

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [9, 1, 2]])
        >>> s.list.sort()
        shape: (2,)
        Series: 'a' [list[i64]]
        [
                [1, 2, 3]
                [1, 2, 9]
        ]
        >>> s.list.sort(descending=True)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
                [3, 2, 1]
                [9, 2, 1]
        ]

        """

    def reverse(self) -> Series:
        """
        Reverse the order of the elements in each list.

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [9, 1, 2]])
        >>> s.list.reverse()
        shape: (2,)
        Series: 'a' [list[i64]]
        [
                [1, 2, 3]
                [2, 1, 9]
        ]
        """

    def unique(self, *, maintain_order: bool = False) -> Series:
        """
        Get the unique values that appear in each list, removing duplicates.

        Parameters
        ----------
        maintain_order
            Whether to keep the unique elements in the same order as in the input data.
            This is slower.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 1, 2]])
        >>> s.list.unique()
        shape: (1,)
        Series: 'a' [list[i64]]
        [
                [1, 2]
        ]

        """

    def concat(self, other: list[Series] | Series | list[Any]) -> Series:
        """
        Concatenate the list elements in two or more :class:`List` `Series`.

        Parameters
        ----------
        other
            The other `Series`.

        Examples
        --------
        >>> a = pl.Series("a", [["a"], ["x"]])
        >>> b = pl.Series("b", [["b", "c"], ["y", "z"]])
        >>> a.list.concat(b)
        shape: (2,)
        Series: '' [list[str]]
        [
                ["a", "b", "c"]
                ["x", "y", "z"]
        ]

        """

    def get(self, index: int | Series | list[int]) -> Series:
        """
        Get a single element from each list by index.

        For instance, `list.get(0)` would return the first item of each list,
        and `list.get(-1)` would return the last item.

        If an index is out of bounds, the resulting element will be `null`.

        Parameters
        ----------
        index
            The index of the element to return from each list.

        See Also
        --------
        gather : Get multiple list elements by index.

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [], [1, 2]])
        >>> s.list.get(0)
        shape: (3,)
        Series: 'a' [i64]
        [
                3
                null
                1
        ]

        """

    def gather(
        self,
        indices: Series | list[int] | list[list[int]],
        *,
        null_on_oob: bool = False,
    ) -> Series:
        """
        Get multiple elements from each list by index.

        The indices may be defined in a single Python list, by lists in another
        :class:`List` `Series`, or by a Python list of Python lists.

        Parameters
        ----------
        indices
            The indices of the elements to return from each list.
        null_on_oob
            Whether to set elements for out-of-bounds indices to `null`,
            rather than raising an error. The latter is much faster.

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [], [1, 2, 3, 4, 5]])
        >>> s.list.gather([0, 4], null_on_oob=True)
        shape: (3,)
        Series: 'a' [list[i64]]
        [
                [3, null]
                [null, null]
                [1, 5]
        ]

        """

    def __getitem__(self, item: int) -> Series:
        return self.get(item)

    def join(self, separator: IntoExpr) -> Series:
        """
        Join all string items in a list and place a separator between them.

        Raises an error if the inner dtype of the :class:`List` `Series` is not
        :class:`String`.

        Parameters
        ----------
        separator
            A string to separate the items with.

        Returns
        -------
        Series
            A :class:`String` `Series`.

        Examples
        --------
        >>> s = pl.Series([["foo", "bar"], ["hello", "world"]])
        >>> s.list.join(separator="-")
        shape: (2,)
        Series: '' [str]
        [
            "foo-bar"
            "hello-world"
        ]

        """

    def first(self) -> Series:
        """
        Get the first element of each list.

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [], [1, 2]])
        >>> s.list.first()
        shape: (3,)
        Series: 'a' [i64]
        [
                3
                null
                1
        ]
        """

    def last(self) -> Series:
        """
        Get the last element of each list.

        Examples
        --------
        >>> s = pl.Series("a", [[3, 2, 1], [], [1, 2]])
        >>> s.list.first()
        shape: (3,)
        Series: 'a' [i64]
        [
                1
                null
                2
        ]
        """

    def contains(self, item: float | str | bool | int | date | datetime) -> Series:
        """
        Check if each list contains the given item.

        Parameters
        ----------
        item
            The item that will be checked for membership.

        Returns
        -------
        Series
            A :class:`Boolean` `Series`.

        >>> s = pl.Series("a", [[3, 2, 1], [], [1, 2]])
        >>> s.list.contains(1)
        shape: (3,)
        Series: 'a' [bool]
        [
                true
                false
                true
        ]

        """

    def arg_min(self) -> Series:
        """
        Get the index of the minimum value in each list.

        Returns
        -------
        Series
            A :class:`UInt32` or :class:`UInt64` `Series` (depending on whether polars
            is compiled in `bigidx` mode).

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [2, 1]])
        >>> s.list.arg_min()
        shape: (2,)
        Series: 'a' [u32]
        [
                0
                1
        ]

        """

    def arg_max(self) -> Series:
        """
        Get the index of the maximum value in each list.

        Returns
        -------
        Series
            A :class:`UInt32` or :class:`UInt64` `Series` (depending on whether polars
            is compiled in `bigidx` mode).

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2], [2, 1]])
        >>> s.list.arg_max()
        shape: (2,)
        Series: 'a' [u32]
        [
                1
                0
        ]

        """

    def diff(self, n: int = 1, null_behavior: NullBehavior = "ignore") -> Series:
        """
        Get the first discrete difference between shifted elements of each list.

        Parameters
        ----------
        n
            The number of elements to shift by when calculating the difference.
        null_behavior : {'ignore', 'drop'}
            How to handle `null` values.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.list.diff()
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [null, 1, … 1]
            [null, -8, -1]
        ]

        >>> s.list.diff(n=2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [null, null, … 2]
            [null, null, -9]
        ]

        >>> s.list.diff(n=2, null_behavior="drop")
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [2, 2]
            [-9]
        ]

        """

    @deprecate_renamed_parameter("periods", "n", version="0.19.11")
    def shift(self, n: int | IntoExprColumn = 1) -> Series:
        """
        Shift list elements by the given number of indices.

        Parameters
        ----------
        n
            The number of indices to shift forward by. If negative, elements are shifted
            backward instead.

        Notes
        -----
        This method is similar to the `LAG` operation in SQL when the value for `n`
        is positive. With a negative value for `n`, it is similar to `LEAD`.

        Examples
        --------
        By default, list elements are shifted forward by one index:

        >>> s = pl.Series([[1, 2, 3], [4, 5]])
        >>> s.list.shift()
        shape: (2,)
        Series: '' [list[i64]]
        [
                [null, 1, 2]
                [null, 4]
        ]

        Pass a negative value to shift backwards instead:

        >>> s.list.shift(-2)
        shape: (2,)
        Series: '' [list[i64]]
        [
                [3, null, null]
                [null, null]
        ]

        """

    def slice(self, offset: int | Expr, length: int | Expr | None = None) -> Series:
        """
        Get a contiguous set of elements from each list.

        Parameters
        ----------
        offset
            The start index. Negative indexing is supported.
        length
            The length of the slice. If `length=None`, all elements starting from the
            `offset` will be selected.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.list.slice(1, 2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [2, 3]
            [2, 1]
        ]

        """

    def head(self, n: int | Expr = 5) -> Series:
        """
        Get the first `n` elements of each list.

        Parameters
        ----------
        n
            The number of elements to return. Negative values are not supported.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.list.head(2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [1, 2]
            [10, 2]
        ]

        """

    def tail(self, n: int | Expr = 5) -> Series:
        """
        Get the last `n` elements of each list.

        Parameters
        ----------
        n
            The number of elements to return. Negative values are not supported.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3, 4], [10, 2, 1]])
        >>> s.list.tail(2)
        shape: (2,)
        Series: 'a' [list[i64]]
        [
            [3, 4]
            [2, 1]
        ]

        """

    def explode(self) -> Series:
        """
        Put every element of every list on its own row.

        Returns
        -------
        Series
            A `Series` with the same data type as the inner data type of the list
            elements.

        See Also
        --------
        Series.explode : Explode a :class:`List` `Series`.
        Series.str.explode : Explode a :class:`String` `Series`.
        Series.reshape : Reshape a `Series` to a flat `Series` or a :class:`List`
                         `Series`.

        Examples
        --------
        >>> s = pl.Series("a", [[1, 2, 3], [4, 5, 6]])
        >>> s.list.explode()
        shape: (6,)
        Series: 'a' [i64]
        [
            1
            2
            3
            4
            5
            6
        ]

        """

    def count_matches(
        self, element: float | str | bool | int | date | datetime | time | Expr
    ) -> Expr:
        """
        Count the number of occurrences of `element` in each list.

        Parameters
        ----------
        element
            A scalar value.

        Examples
        --------
        >>> s = pl.Series("a", [[0], [1], [1, 2, 3, 2], [1, 2, 1], [4, 4]])
        >>> number_of_twos = s.list.count_matches(2)
        >>> number_of_twos
        shape: (5,)
        Series: 'a' [u32]
        [
                0
                0
                2
                1
                0
        ]

        """

    def to_array(self, width: int) -> Series:
        """
        Convert a :class:`List` `Series` to an :class:`Array` `Series`.

        Parameters
        ----------
        width
            The width of the resulting :class:`Array` `Series`.

        Returns
        -------
        Series
            An :class:`Array` `Series`.

        Examples
        --------
        >>> s = pl.Series([[1, 2], [3, 4]], dtype=pl.List(pl.Int8))
        >>> s.list.to_array(2)
        shape: (2,)
        Series: '' [array[i8, 2]]
        [
                [1, 2]
                [3, 4]
        ]

        """

    def to_struct(
        self,
        n_field_strategy: ToStructStrategy = "first_non_null",
        fields: Callable[[int], str] | Sequence[str] | None = None,
    ) -> Series:
        """
        Convert a :class:`List` `Series` to a :class:`Struct` `Series`.

        Parameters
        ----------
        n_field_strategy : {'first_non_null', 'max_width'}
            Whether to set the number of :class:`Struct` fields to:

            * `"first_non_null"`: the length of the first non zero-length list.
            * `"max_width"`: the maximum length of all lists.
        fields
            If the name and number of the desired fields is known in advance
            a list of field names can be given, which will be assigned by index.
            Otherwise, to dynamically assign field names, a custom function can be
            used; if neither are set, fields will be `field_0, field_1 .. field_n`.

        Notes
        -----
        For performance reasons, the length of the first non-`null` list is used to
        determine the number of output fields. If the list can be of different lengths,
        then `n_field_strategy="max_width"` must be used to obtain the expected result.

        Examples
        --------
        Convert :class:`List` to :class:`Struct` with default field name assignment:

        >>> s1 = pl.Series("n", [[0, 1, 2], [0, 1]])
        >>> s2 = s1.list.to_struct()
        >>> s2
        shape: (2,)
        Series: 'n' [struct[3]]
        [
            {0,1,2}
            {0,1,null}
        ]
        >>> s2.struct.fields
        ['field_0', 'field_1', 'field_2']

        Convert :class:`List` to :class:`Struct` with field name assignment by
        function/index:

        >>> s3 = s1.list.to_struct(fields=lambda idx: f"n{idx:02}")
        >>> s3.struct.fields
        ['n00', 'n01', 'n02']

        Convert :class:`List` to :class:`Struct` with field name assignment by index
        from a list of names:

        >>> s1.list.to_struct(fields=["one", "two", "three"]).struct.unnest()
        shape: (2, 3)
        ┌─────┬─────┬───────┐
        │ one ┆ two ┆ three │
        │ --- ┆ --- ┆ ---   │
        │ i64 ┆ i64 ┆ i64   │
        ╞═════╪═════╪═══════╡
        │ 0   ┆ 1   ┆ 2     │
        │ 0   ┆ 1   ┆ null  │
        └─────┴─────┴───────┘

        """
        s = wrap_s(self._s)
        return (
            s.to_frame()
            .select(
                F.col(s.name).list.to_struct(
                    # note: in eager mode, 'upper_bound' is always zero, as (unlike
                    # in lazy mode) there is no need to determine/track the schema.
                    n_field_strategy,
                    fields,
                    upper_bound=0,
                )
            )
            .to_series()
        )

    def eval(self, expr: Expr, *, parallel: bool = False) -> Series:
        """
        Evaluate any polars expression across each list's elements.

        Use :func:`polars.element()` to refer to the list element, similar to how
        you usually would use `pl.col()` to refer to a column in an expression.

        Parameters
        ----------
        expr
            The expression to evaluate.
        parallel
            Whether to execute the computation in parallel.

            .. note::
                This option should likely not be enabled in an aggregation context,
                as the computation is already parallelized per group.

        Examples
        --------
        >>> df = pl.DataFrame({"a": [1, 8, 3], "b": [4, 5, 2]})
        >>> df.with_columns(
        ...     pl.concat_list(["a", "b"]).list.eval(pl.element().rank()).alias("rank")
        ... )
        shape: (3, 3)
        ┌─────┬─────┬────────────┐
        │ a   ┆ b   ┆ rank       │
        │ --- ┆ --- ┆ ---        │
        │ i64 ┆ i64 ┆ list[f64]  │
        ╞═════╪═════╪════════════╡
        │ 1   ┆ 4   ┆ [1.0, 2.0] │
        │ 8   ┆ 5   ┆ [2.0, 1.0] │
        │ 3   ┆ 2   ┆ [2.0, 1.0] │
        └─────┴─────┴────────────┘

        """

    def set_union(self, other: Series) -> Series:
        """
        Compute set unions between elements in two :class:`List` `Series`.

        Parameters
        ----------
        other
            The other `Series`.

        Examples
        --------
        >>> a = pl.Series([[1, 2, 3], [], [None, 3], [5, 6, 7]])
        >>> b = pl.Series([[2, 3, 4], [3], [3, 4, None], [6, 8]])
        >>> a.list.set_union(b)  # doctest: +IGNORE_RESULT
        shape: (4,)
        Series: '' [list[i64]]
        [
                [1, 2, 3, 4]
                [3]
                [null, 3, 4]
                [5, 6, 7, 8]
        ]

        """

    def set_difference(self, other: Series) -> Series:
        """
        Compute set differences between elements in two :class:`List` `Series`.

        Parameters
        ----------
        other
            The other `Series`.

        See Also
        --------
        polars.Series.list.diff: Get the first discrete difference between shifted
                                 elements of each list.

        Examples
        --------
        >>> a = pl.Series([[1, 2, 3], [], [None, 3], [5, 6, 7]])
        >>> b = pl.Series([[2, 3, 4], [3], [3, 4, None], [6, 8]])
        >>> a.list.set_difference(b)
        shape: (4,)
        Series: '' [list[i64]]
        [
                [1]
                []
                []
                [5, 7]
        ]

        """

    def set_intersection(self, other: Series) -> Series:
        """
        Compute set intersections between elements in two :class:`List` `Series`.

        Parameters
        ----------
        other
            The other `Series`.

        Examples
        --------
        >>> a = pl.Series([[1, 2, 3], [], [None, 3], [5, 6, 7]])
        >>> b = pl.Series([[2, 3, 4], [3], [3, 4, None], [6, 8]])
        >>> a.list.set_intersection(b)
        shape: (4,)
        Series: '' [list[i64]]
        [
                [2, 3]
                []
                [null, 3]
                [6]
        ]

        """

    def set_symmetric_difference(self, other: Series) -> Series:
        """
        Compute set symmetric differences between elements in two :class:`List` `Series`.

        Parameters
        ----------
        other
            The other `Series`.

        Examples
        --------
        >>> a = pl.Series([[1, 2, 3], [], [None, 3], [5, 6, 7]])
        >>> b = pl.Series([[2, 3, 4], [3], [3, 4, None], [6, 8]])
        >>> a.list.set_symmetric_difference(b)
        shape: (4,)
        Series: '' [list[i64]]
        [
                [1, 4]
                [3]
                [4]
                [5, 7, 8]
        ]

        """  # noqa: W505

    @deprecate_renamed_function("count_matches", version="0.19.3")
    def count_match(
        self, element: float | str | bool | int | date | datetime | time | Expr
    ) -> Expr:
        """
        Count how often the value produced by `element` occurs.

        .. deprecated:: 0.19.3
            This method has been renamed to :func:`count_matches`.

        Parameters
        ----------
        element
            An expression that produces a single value

        """

    @deprecate_renamed_function("len", version="0.19.8")
    def lengths(self) -> Series:
        """
        Return the number of elements in each list.

        .. deprecated:: 0.19.8
            This method has been renamed to :func:`len`.

        """

    @deprecate_renamed_function("gather", version="0.19.14")
    @deprecate_renamed_parameter("index", "indices", version="0.19.14")
    def take(
        self,
        indices: Series | list[int] | list[list[int]],
        *,
        null_on_oob: bool = False,
    ) -> Series:
        """
        Take sublists by multiple indices.

        .. deprecated:: 0.19.14
            This method has been renamed to :func:`gather`.

        Parameters
        ----------
        indices
            Indices to return per sublist
        null_on_oob
            Behavior if an index is out of bounds:
            `True` -> set to `null`
            `False` -> raise an error
            Note that defaulting to raising an error is much cheaper.
        """
