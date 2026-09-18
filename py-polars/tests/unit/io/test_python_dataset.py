from __future__ import annotations

from datetime import date, timedelta
from typing import TYPE_CHECKING, Any

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

import polars as pl
from polars._plr import PyLazyFrame
from polars._utils.wrap import wrap_ldf
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch


class CapturingDataset:
    """Minimal dataset provider that records the predicate Polars lowers for it."""

    def __init__(self, df: pl.DataFrame) -> None:
        self.df = df
        self.arrow_schema = df.to_arrow().schema
        self.pyarrow_predicate: str | None = None

    def schema(self) -> pa.Schema:
        return self.arrow_schema

    def to_dataset_scan(
        self, *, pyarrow_predicate: str | None = None, **_kwargs: Any
    ) -> tuple[pl.LazyFrame, str]:
        self.pyarrow_predicate = pyarrow_predicate

        def impl(*_args: Any, **_kwargs: Any) -> tuple[Iterator[pl.DataFrame], bool]:
            # Return everything: the engine re-applies the predicate itself.
            return iter([self.df]), False

        lf = pl.LazyFrame._scan_python_function(
            self.arrow_schema, impl, pyarrow=True, is_pure=True
        )

        return lf, "v1"


def lowered_predicate(
    df: pl.DataFrame, predicate: pl.Expr, *, partial: bool = False
) -> str | None:
    """
    Return the predicate string handed to a dataset provider for `predicate`.

    Also asserts that the scan produces correct results, and - when something was
    lowered - that the string builds a PyArrow expression selecting the same rows
    (a superset of them, if only part of the predicate was lowered).
    """
    dataset = CapturingDataset(df)
    lf = wrap_ldf(PyLazyFrame.new_from_dataset_object(dataset))

    expected = df.filter(predicate)
    assert_frame_equal(lf.filter(predicate).collect(), expected)

    pyarrow_predicate = dataset.pyarrow_predicate

    if pyarrow_predicate is not None:
        from polars._utils.convert import (
            to_py_date,
            to_py_datetime,
            to_py_time,
            to_py_timedelta,
        )

        # The provider protocol says this is valid Python building a PyArrow
        # expression - the same environment `scan_delta` evaluates it in.
        expr = eval(
            pyarrow_predicate,
            {
                "pa": pa,
                "to_py_date": to_py_date,
                "to_py_datetime": to_py_datetime,
                "to_py_time": to_py_time,
                "to_py_timedelta": to_py_timedelta,
            },
        )
        filtered = pl.from_arrow(ds.dataset(df.to_arrow()).to_table(filter=expr))
        assert isinstance(filtered, pl.DataFrame)

        if partial:
            assert_frame_equal(filtered.filter(predicate), expected)
        else:
            assert_frame_equal(filtered, expected)

    return pyarrow_predicate


@pytest.fixture
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5, 6],
            "cat": ["alpha", "beta", "gamma", "delta", "alpha", "beta"],
            "val": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
        }
    )


def test_dataset_provider_predicate_comparison(df: pl.DataFrame) -> None:
    assert lowered_predicate(df, pl.col("id") > 3) == "(pa.compute.field('id') > 3)"


def test_dataset_provider_predicate_is_in(df: pl.DataFrame) -> None:
    assert (
        lowered_predicate(df, pl.col("cat").is_in(["alpha", "beta"]))
        == '(pa.compute.field(\'cat\')).isin(["alpha","beta"])'
    )
    assert (
        lowered_predicate(df, pl.col("id").is_in([1, 3, 5]))
        == "(pa.compute.field('id')).isin([1,3,5])"
    )


def test_dataset_provider_predicate_is_in_empty(df: pl.DataFrame) -> None:
    assert lowered_predicate(df, pl.col("id").is_in([])) == "pa.compute.scalar(False)"


def test_dataset_provider_predicate_is_in_nulls_equal(df: pl.DataFrame) -> None:
    df = df.with_columns(pl.when(pl.col("id") > 3).then(pl.col("cat")).alias("cat"))

    # Nulls are dropped from the haystack unless they are to compare equal.
    assert (
        lowered_predicate(df, pl.col("cat").is_in(["alpha", None]))
        == "(pa.compute.field('cat')).isin([\"alpha\"])"
    )
    assert (
        lowered_predicate(df, pl.col("cat").is_in(["alpha", None], nulls_equal=True))
        == "(pa.compute.field('cat')).isin([\"alpha\",None])"
    )


def test_dataset_provider_predicate_is_in_boolean() -> None:
    df = pl.DataFrame({"flag": [True, False, True]})
    assert (
        lowered_predicate(df, pl.col("flag").is_in([True]))
        == "(pa.compute.field('flag')).isin([True])"
    )


def test_dataset_provider_predicate_is_in_date() -> None:
    df = pl.DataFrame({"d": [date(2020, 1, 1), date(2020, 1, 2), date(2020, 1, 3)]})
    assert (
        lowered_predicate(df, pl.col("d").is_in([date(2020, 1, 1), date(2020, 1, 3)]))
        == "(pa.compute.field('d')).isin([to_py_date(18262),to_py_date(18264)])"
    )


def test_dataset_provider_predicate_is_in_not_lowered() -> None:
    df = pl.DataFrame(
        {
            "id": [1, 2, 3],
            "bin": [b"a", b"b", b"c"],
            "st": [{"a": 1}, {"a": 2}, {"a": 3}],
            "ids": [[1, 2], [3], [1]],
        }
    )
    # Values we cannot safely write out as Python source text.
    assert lowered_predicate(df, pl.col("bin").is_in([b"a", b"c"])) is None
    assert lowered_predicate(df, pl.col("st").is_in([{"a": 1}])) is None
    # A haystack that is not a literal at all.
    assert lowered_predicate(df, pl.col("id").is_in(pl.col("ids"))) is None

    arrays = pl.DataFrame({"a": [[1, 2], [3, 4]]}, schema={"a": pl.Array(pl.Int64, 2)})
    haystack = pl.Series([[[1, 2]]], dtype=pl.List(pl.Array(pl.Int64, 2)))
    assert lowered_predicate(arrays, pl.col("a").is_in(pl.lit(haystack))) is None


def test_dataset_provider_predicate_series_literal_not_lowered(
    df: pl.DataFrame,
) -> None:
    # A `Series` literal is only meaningful as an `is_in` haystack.
    assert lowered_predicate(df, pl.col("id") == pl.lit(pl.Series("s", [3]))) is None


def test_dataset_provider_predicate_is_between(df: pl.DataFrame) -> None:
    assert (
        lowered_predicate(df, pl.col("id").is_between(2, 4))
        == "((pa.compute.field('id') >= 2) & (pa.compute.field('id') <= 4))"
    )
    assert (
        lowered_predicate(df, pl.col("id").is_between(2, 4, closed="none"))
        == "((pa.compute.field('id') > 2) & (pa.compute.field('id') < 4))"
    )


def test_dataset_provider_predicate_nested_conjunction(df: pl.DataFrame) -> None:
    # Top-level conjuncts are lowered one by one, but an `&` nested under an `|`
    # has to be written out as an operator of its own.
    assert (
        lowered_predicate(
            df, ((pl.col("id") > 3) & (pl.col("cat") == "alpha")) | (pl.col("id") < 2)
        )
        == "(((pa.compute.field('id') > 3) & (pa.compute.field('cat') == 'alpha'))"
        " | (pa.compute.field('id') < 2))"
    )


def test_dataset_provider_predicate_eq_missing(df: pl.DataFrame) -> None:
    df = df.with_columns(pl.when(pl.col("id") > 3).then(pl.col("cat")).alias("cat"))

    # `==v` and `!=v` are not Python operators. A null is not equal to a non-null
    # literal, where the plain comparison would evaluate to null.
    assert lowered_predicate(df, pl.col("cat").eq_missing("alpha")) == (
        "((pa.compute.field('cat') == 'alpha') & ~(pa.compute.field('cat')).is_null())"
    )
    assert lowered_predicate(df, pl.col("cat").ne_missing("alpha")) == (
        "((pa.compute.field('cat') != 'alpha') | (pa.compute.field('cat')).is_null())"
    )
    # The column may be on either side.
    assert lowered_predicate(df, pl.lit("alpha").eq_missing(pl.col("cat"))) == (
        "((pa.compute.field('cat') == 'alpha') & ~(pa.compute.field('cat')).is_null())"
    )
    # Against a null literal the optimizer has already rewritten it to `is_null`.
    assert (
        lowered_predicate(df, pl.col("cat").eq_missing(None))
        == "(pa.compute.field('cat')).is_null()"
    )
    assert (
        lowered_predicate(df, pl.col("cat").ne_missing(None))
        == "~(pa.compute.field('cat')).is_null()"
    )
    # Two columns are not lowered: the operands are written out twice.
    assert lowered_predicate(df, pl.col("cat").eq_missing(pl.col("cat"))) is None


def test_dataset_provider_predicate_xor(df: pl.DataFrame) -> None:
    # PyArrow expressions have no `^` operator.
    assert lowered_predicate(df, (pl.col("id") > 3) ^ (pl.col("val") > 1.0)) == (
        "(((pa.compute.field('id') > 3) | (pa.compute.field('val') > 1))"
        " & ~((pa.compute.field('id') > 3) & (pa.compute.field('val') > 1)))"
    )
    # `^` on integers is a bitwise operation, which the rewrite does not hold for.
    assert lowered_predicate(df, (pl.col("id") ^ 1) > 2) is None
    # A bare column is not known to be boolean without consulting the schema,
    # not even under a `~`, which is also the bitwise negation.
    flags = pl.DataFrame({"f": [True, False, True], "g": [False, False, True]})
    assert lowered_predicate(flags, pl.col("f") ^ pl.col("g")) is None
    assert lowered_predicate(flags, ~pl.col("f") ^ pl.col("g").is_null()) is None


def test_dataset_provider_predicate_xor_operands(df: pl.DataFrame) -> None:
    # Operands that are known to be boolean without a schema: a nested `&` over
    # comparisons, a boolean literal, and a boolean function.
    assert (
        lowered_predicate(
            df, ((pl.col("id") > 3) & (pl.col("val") > 1.0)) ^ (pl.col("id") < 2)
        )
        is not None
    )
    assert lowered_predicate(df, (pl.col("id") > 3) ^ pl.lit(True)) is not None
    assert (
        lowered_predicate(df, pl.col("cat").is_null() ^ (pl.col("id") > 3)) is not None
    )


def test_dataset_provider_predicate_arithmetic(df: pl.DataFrame) -> None:
    # Only `Float64` arithmetic lowers: PyArrow's checked kernels raise on
    # integer overflow where Polars wraps.
    assert (
        lowered_predicate(df, pl.col("val") * 2 > 1.0)
        == "((pa.compute.field('val') * 2) > 1)"
    )
    assert (
        lowered_predicate(df, pl.col("val") + 1.0 <= 1.5)
        == "((pa.compute.field('val') + 1) <= 1.5)"
    )
    # PyArrow expressions have no reflected arithmetic, so a literal on the left
    # has to become an expression of its own.
    assert (
        lowered_predicate(df, 2.0 - pl.col("val") > 1.0)
        == "((pa.compute.scalar(2) - pa.compute.field('val')) > 1)"
    )


def test_dataset_provider_predicate_arithmetic_not_lowered(
    df: pl.DataFrame,
) -> None:
    # Integer arithmetic would raise on overflow in PyArrow where Polars wraps.
    assert lowered_predicate(df, pl.col("id") * pl.col("id") > 4) is None
    assert lowered_predicate(df, pl.col("id") + 1 > 0) is None
    # `/` never lowers: PyArrow raises on division by zero where Polars yields
    # `inf`/`NaN`, follows the operand types where Polars always float-divides,
    # and its safe casts lose precision (`Float32`) or error (large integers).
    assert lowered_predicate(df, pl.col("id") / 4 > 1.0) is None
    assert lowered_predicate(df, pl.col("val") / 2.0 > 1.0) is None
    # `Float32` arithmetic is not `Float64`: rounding can differ.
    floats = pl.DataFrame({"x": [1.0]}, schema={"x": pl.Float32})
    assert lowered_predicate(floats, pl.col("x") * 2 > 1.0) is None
    # Temporal arithmetic has no PyArrow equivalent with Polars' semantics
    # (`Date + Duration` truncates to a `Date` in Polars).
    temporal = pl.DataFrame(
        {
            "x": [date(2024, 1, 1)],
            "d": [timedelta(hours=1)],
            "y": [date(2024, 1, 1)],
        }
    )
    assert (
        lowered_predicate(temporal, (pl.col("x") + pl.col("d")) == pl.col("y")) is None
    )


def test_dataset_provider_predicate_not_lowered(df: pl.DataFrame) -> None:
    # `%` has no PyArrow expression form.
    assert lowered_predicate(df, pl.col("id") % 2 == 0) is None
    # Neither does a cast.
    assert lowered_predicate(df, pl.col("id").cast(pl.String) == "1") is None


def test_dataset_provider_predicate_partial(df: pl.DataFrame) -> None:
    # Unconvertible conjuncts are dropped, the rest is still lowered.
    assert (
        lowered_predicate(
            df, (pl.col("id") > 3) & (pl.col("id") % 2 == 0), partial=True
        )
        == "(pa.compute.field('id') > 3)"
    )


class ParquetDataset:
    """Dataset provider that expands to a Parquet scan, the way Iceberg does."""

    def __init__(self, paths: list[Path], schema: pl.Schema | None = None) -> None:
        self.paths = paths
        self.pl_schema = (
            schema if schema is not None else pl.scan_parquet(paths[0]).collect_schema()
        )
        self.arrow_schema = self.pl_schema.to_arrow()

    def schema(self) -> pa.Schema:
        return self.arrow_schema

    def to_dataset_scan(self, **_kwargs: Any) -> tuple[pl.LazyFrame, str]:
        # The manifest knows the sizes, so the expansion does not have to list them.
        sizes = [p.stat().st_size for p in self.paths]
        lf = pl.scan_parquet(self.paths, schema=self.pl_schema, _source_sizes=sizes)
        return lf, "v1"


def test_python_dataset_resolves_heavy_footers(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    # A dataset is expanded into a Parquet scan by the optimizer, after the point where
    # a globbed scan would have resolved its footers. Without the resolve in
    # `expand_datasets` the distributed planner has no row groups to cut the heavy file
    # between, however much of the table it holds.
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    paths = []
    for i, n in enumerate([1, 1, 4000]):
        path = tmp_path / f"{i}.parquet"
        pl.DataFrame({"x": range(n)}).write_parquet(path, row_group_size=500)
        paths.append(path)

    dataset = ParquetDataset(paths)
    lf = wrap_ldf(PyLazyFrame.new_from_dataset_object(dataset, resolve_heavy_sources=4))

    assert lf.collect().height == 4002

    captured = capfd.readouterr().err
    assert "parquet resolve: pinned 1 / 1 heavy sources" in captured


def test_python_dataset_leaves_footers_alone_without_the_flag(
    tmp_path: Path,
    capfd: pytest.CaptureFixture[str],
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")

    paths = []
    for i, n in enumerate([1, 4000]):
        path = tmp_path / f"{i}.parquet"
        pl.DataFrame({"x": range(n)}).write_parquet(path, row_group_size=500)
        paths.append(path)

    lf = wrap_ldf(PyLazyFrame.new_from_dataset_object(ParquetDataset(paths)))

    assert lf.collect().height == 4001
    assert "heavy sources" not in capfd.readouterr().err


def test_python_dataset_with_no_sources_resolves_nothing() -> None:
    # An empty table, or a predicate that eliminated every file: there is not even a
    # source 0 whose footer could be read.
    lf = wrap_ldf(
        PyLazyFrame.new_from_dataset_object(
            ParquetDataset([], pl.Schema({"x": pl.Int64})), resolve_heavy_sources=4
        )
    )

    assert lf.collect().height == 0
