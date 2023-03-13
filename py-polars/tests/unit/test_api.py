from __future__ import annotations

from typing import no_type_check

import polars as pl
from polars.testing import assert_frame_equal


def test_custom_df_namespace() -> None:
    @pl.api.register_dataframe_namespace("split")
    class SplitFrame:
        def __init__(self, df: pl.DataFrame):
            self._df = df

        def by_first_letter_of_column_names(self) -> list[pl.DataFrame]:
            return [
                self._df.select([col for col in self._df.columns if col[0] == f])
                for f in sorted({col[0] for col in self._df.columns})
            ]

        def by_first_letter_of_column_values(self, col: str) -> list[pl.DataFrame]:
            return [
                self._df.filter(pl.col(col).str.starts_with(c))
                for c in sorted(set(df.select(pl.col(col).str.slice(0, 1)).to_series()))
            ]

    df = pl.DataFrame(
        data=[["xx", 2, 3, 4], ["xy", 4, 5, 6], ["yy", 5, 6, 7], ["yz", 6, 7, 8]],
        schema=["a1", "a2", "b1", "b2"],
        orient="row",
    )

    dfs = df.split.by_first_letter_of_column_names()  # type: ignore[attr-defined]
    assert [d.rows() for d in dfs] == [
        [("xx", 2), ("xy", 4), ("yy", 5), ("yz", 6)],
        [(3, 4), (5, 6), (6, 7), (7, 8)],
    ]
    dfs = df.split.by_first_letter_of_column_values("a1")  # type: ignore[attr-defined]
    assert [d.rows() for d in dfs] == [
        [("xx", 2, 3, 4), ("xy", 4, 5, 6)],
        [("yy", 5, 6, 7), ("yz", 6, 7, 8)],
    ]


@no_type_check
def test_custom_expr_namespace() -> None:
    @pl.api.register_expr_namespace("power")
    class PowersOfN:
        def __init__(self, expr: pl.Expr):
            self._expr = expr

        def next(self, p: int) -> pl.Expr:
            return (p ** (self._expr.log(p).ceil()).cast(pl.Int64)).cast(pl.Int64)

        def previous(self, p: int) -> pl.Expr:
            return (p ** (self._expr.log(p).floor()).cast(pl.Int64)).cast(pl.Int64)

        def nearest(self, p: int) -> pl.Expr:
            return (p ** (self._expr.log(p)).round(0).cast(pl.Int64)).cast(pl.Int64)

    df = pl.DataFrame([1.4, 24.3, 55.0, 64.001], schema=["n"])
    assert df.select(
        [
            pl.col("n"),
            pl.col("n").power.next(p=2).alias("next_pow2"),
            pl.col("n").power.previous(p=2).alias("prev_pow2"),
            pl.col("n").power.nearest(p=2).alias("nearest_pow2"),
        ]
    ).rows() == [
        (1.4, 2, 1, 1),
        (24.3, 32, 16, 32),
        (55.0, 64, 32, 64),
        (64.001, 128, 64, 64),
    ]


@no_type_check
def test_custom_lazy_namespace() -> None:
    @pl.api.register_lazyframe_namespace("split")
    class SplitFrame:
        def __init__(self, ldf: pl.LazyFrame):
            self._ldf = ldf

        def by_column_dtypes(self) -> list[pl.LazyFrame]:
            return [
                self._ldf.select(pl.col(tp)) for tp in dict.fromkeys(self._ldf.dtypes)
            ]

    ldf = pl.DataFrame(
        data=[["xx", 2, 3, 4], ["xy", 4, 5, 6], ["yy", 5, 6, 7], ["yz", 6, 7, 8]],
        schema=["a1", "a2", "b1", "b2"],
        orient="row",
    ).lazy()

    df1, df2 = (d.collect() for d in ldf.split.by_column_dtypes())
    assert_frame_equal(
        df1, pl.DataFrame([("xx",), ("xy",), ("yy",), ("yz",)], schema=["a1"])
    )
    assert_frame_equal(
        df2,
        pl.DataFrame(
            [(2, 3, 4), (4, 5, 6), (5, 6, 7), (6, 7, 8)], schema=["a2", "b1", "b2"]
        ),
    )


def test_custom_series_namespace() -> None:
    @pl.api.register_series_namespace("math")
    class CustomMath:
        def __init__(self, s: pl.Series):
            self._s = s

        def square(self) -> pl.Series:
            return self._s * self._s

    s = pl.Series("n", [1.5, 31.0, 42.0, 64.5])
    assert s.math.square().to_list() == [  # type: ignore[attr-defined]
        2.25,
        961.0,
        1764.0,
        4160.25,
    ]


def test_class_namespaces_are_registered() -> None:
    # confirm that existing (and new) namespaces
    # have been added to that class's "_accessors" attr
    for pcls in (pl.Expr, pl.DataFrame, pl.LazyFrame, pl.Series):
        namespaces: set[str] = getattr(pcls, "_accessors", set())
        for name in dir(pcls):
            if not name.startswith("_"):
                attr = getattr(pcls, name)
                if isinstance(attr, property):
                    try:
                        obj = attr.fget(pcls)  # type: ignore[misc]
                    except Exception:
                        continue

                    if obj.__class__.__name__.endswith("NameSpace"):
                        ns = obj._accessor
                        assert (
                            ns in namespaces
                        ), f"{ns!r} should be registered in {pcls.__name__}._accessors"
