from __future__ import annotations

from datetime import date, datetime, time, timezone
from typing import TYPE_CHECKING

import pyarrow as pa
import pyarrow.dataset as ds
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch


def helper_dataset_test(
    file_path: Path,
    query: Callable[[pl.LazyFrame], pl.LazyFrame],
    batch_size: int | None = None,
    n_expected: int | None = None,
    check_predicate_pushdown: bool = False,
) -> None:
    dset = ds.dataset(file_path, format="ipc")
    q = pl.scan_ipc(file_path).pipe(query)

    expected = q.collect()
    out = pl.scan_pyarrow_dataset(dset, batch_size=batch_size).pipe(query).collect()
    assert_frame_equal(out, expected)
    if n_expected is not None:
        assert len(out) == n_expected

    if check_predicate_pushdown:
        assert "FILTER" not in q.explain()


@pytest.mark.write_disk
def test_pyarrow_dataset_source(df: pl.DataFrame, tmp_path: Path) -> None:
    file_path = tmp_path / "small.ipc"
    df.write_ipc(file_path, compat_level=pl.CompatLevel.oldest())

    helper_dataset_test(
        file_path,
        lambda lf: lf.filter("bools").select("bools", "floats", "date"),
        n_expected=1,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(~pl.col("bools")).select("bools", "floats", "date"),
        n_expected=2,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int_nulls").is_null()).select(
            "bools", "floats", "date"
        ),
        n_expected=1,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int_nulls").is_not_null()).select(
            "bools", "floats", "date"
        ),
        n_expected=2,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(
            pl.col("int_nulls").is_not_null() == pl.col("bools")
        ).select("bools", "floats", "date"),
        n_expected=0,
        check_predicate_pushdown=True,
    )
    # this equality on a column with nulls fails as pyarrow has different
    # handling kleene logic. We leave it for now and document it in the function.
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int") == 10).select(
            "bools", "floats", "int_nulls"
        ),
        n_expected=0,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int") != 10).select(
            "bools", "floats", "int_nulls"
        ),
        n_expected=3,
        check_predicate_pushdown=True,
    )

    for closed, n_expected in zip(
        ["both", "left", "right", "none"], [3, 2, 2, 1], strict=True
    ):
        helper_dataset_test(
            file_path,
            lambda lf, closed=closed: lf.filter(  # type: ignore[misc]
                pl.col("int").is_between(1, 3, closed=closed)
            ).select("bools", "floats", "date"),
            n_expected=n_expected,
            check_predicate_pushdown=True,
        )
    # this predicate is not supported by pyarrow
    # check if we still do it on our side
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("floats").sum().over("date") == 10).select(
            "bools", "floats", "date"
        ),
        n_expected=0,
    )
    # temporal types
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("date") < date(1972, 1, 1)).select(
            "bools", "floats", "date"
        ),
        n_expected=1,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(
            pl.col("datetime") > datetime(1970, 1, 1, second=13)
        ).select("bools", "floats", "date"),
        n_expected=1,
        check_predicate_pushdown=True,
    )
    # not yet supported in pyarrow
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("time") >= time(microsecond=100)).select(
            "bools", "time", "date"
        ),
        n_expected=3,
        check_predicate_pushdown=True,
    )
    # pushdown is_in
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int").is_in([1, 3, 20])).select(
            "bools", "floats", "date"
        ),
        n_expected=2,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(
            pl.col("date").is_in([date(1973, 8, 17), date(1973, 5, 19)])
        ).select("bools", "floats", "date"),
        n_expected=2,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(
            pl.col("datetime").is_in(
                [
                    datetime(1970, 1, 1, 0, 0, 12, 341234),
                    datetime(1970, 1, 1, 0, 0, 13, 241324),
                ]
            )
        ).select("bools", "floats", "date"),
        n_expected=2,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("int").is_in(list(range(120)))).select(
            "bools", "floats", "date"
        ),
        n_expected=3,
        check_predicate_pushdown=True,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("cat").is_in([])).select("bools", "floats", "date"),
        n_expected=0,
    )
    helper_dataset_test(
        file_path,
        lambda lf: lf.select(pl.exclude("enum")),
        batch_size=2,
        n_expected=3,
    )

    # direct filter
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.Series([True, False, True])).select(
            "bools", "floats", "date"
        ),
        n_expected=2,
    )

    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("bools") & pl.col("int").is_in([1, 2])).select(
            "bools", "floats"
        ),
        n_expected=1,
        check_predicate_pushdown=True,
    )


def test_pyarrow_dataset_partial_predicate_pushdown(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

    df = pl.DataFrame({"a": [1, 2, 3], "b": [10.0, 20.0, 30.0]})
    file_path = tmp_path / "0"
    df.write_parquet(file_path)
    dset = ds.dataset(file_path, format="parquet")

    # col("a") > 1 is convertible; col("a") * col("b") > 25 is not (arithmetic
    # on two columns cannot be expressed as a pyarrow compute expression).
    # The optimizer pushes both terms into the scan's SELECTION, so our
    # MintermIter-based partial conversion should push the convertible part.
    q = pl.scan_pyarrow_dataset(dset).filter(
        (pl.col("a") > 1) & (pl.col("a") * pl.col("b") > 25)
    )

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    # partial predicate was pushed to pyarrow
    binop_pred = "converted pyarrow predicate: <pyarrow.compute.Expression (a > 1)>"
    assert binop_pred in capture

    resid_pred = (
        'residual predicate: Some([([(col("a").cast(Float64)) * (col("b"))]) > (25.0)])'
    )
    assert resid_pred in capture
    # Verify: correctness
    expected = (
        df.lazy().filter((pl.col("a") > 1) & (pl.col("a") * pl.col("b") > 25)).collect()
    )
    assert_frame_equal(result, expected)


def test_pyarrow_dataset_is_in_predicate_pushdown(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

    df = pl.DataFrame({"id": [1, 2, 3, 4, 5], "val": [10, 20, 30, 40, 50]})
    dset = ds.dataset(df.to_arrow(compat_level=pl.CompatLevel.oldest()))

    pred = pl.col("id").is_in([1, 3])
    expected = pl.DataFrame({"id": [1, 3], "val": [10, 30]})
    q = pl.scan_pyarrow_dataset(dset).filter(pred)

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    isin_pred = 'predicate node: col("id").is_in([[1, 3]])'
    assert isin_pred in capture

    resid_pred = "residual predicate: None"
    assert resid_pred in capture

    assert_frame_equal(result, expected)
    assert_frame_equal(df.filter(pred), expected)

    pred = pl.col("id").is_in(range(1, 4))
    expected = pl.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
    q = pl.scan_pyarrow_dataset(dset).filter(pred)

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "0")

    isin_pred = 'predicate node: col("id").is_in([[1, 2, 3]])'
    assert isin_pred in capture

    resid_pred = "residual predicate: None"
    assert resid_pred in capture

    assert_frame_equal(result, expected)
    assert_frame_equal(df.filter(pred), expected)


def test_pyarrow_dataset_is_in_predicate_pushdown_nulls_equality(
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

    df = pl.DataFrame({"id": [1, 2, 3, 4, None], "val": [10, 20, 30, 40, 50]})
    dset = ds.dataset(df.to_arrow(compat_level=pl.CompatLevel.oldest()))

    pred = pl.col("id").is_in([1, None, 3])
    expected = pl.DataFrame({"id": [1, 3], "val": [10, 30]})
    q = pl.scan_pyarrow_dataset(dset).filter(pred)

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    assert (
        "converted pyarrow predicate: <pyarrow.compute.Expression is_in(id" in capture
    )
    assert "1," in capture
    assert "3" in capture

    resid_pred = "residual predicate: None"
    assert "residual predicate: None" in capture

    assert_frame_equal(result, expected)
    assert_frame_equal(df.filter(pred), expected)

    pred = pl.col("id").is_in([1, None, 3], nulls_equal=True)
    expected = pl.DataFrame({"id": [1, 3, None], "val": [10, 30, 50]})
    q = pl.scan_pyarrow_dataset(dset).filter(pred)

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    assert "<pyarrow.compute.Expression is_in(id" in capture
    assert "null," in capture
    assert "residual predicate: None" in capture

    assert_frame_equal(result, expected)
    assert_frame_equal(df.filter(pred), expected)

    pred = pl.col("id").is_in(pl.lit(None, dtype=pl.List(pl.Int64)))
    expected = df.clear()
    q = pl.scan_pyarrow_dataset(dset).filter(pred)

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    assert "converted pyarrow predicate: <pyarrow.compute.Expression false>" in capture
    assert "residual predicate: None" in capture

    assert_frame_equal(q.collect(), expected)
    assert_frame_equal(df.filter(pred), expected)

    pred = pl.col("id").is_in(pl.Series([None], dtype=pl.List(pl.Int64)))
    expected = df.clear()
    q = pl.scan_pyarrow_dataset(dset).filter(pred)

    capfd.readouterr()
    result = q.collect()
    capture = capfd.readouterr().err

    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "0")

    assert "converted pyarrow predicate: <pyarrow.compute.Expression false>" in capture
    assert "residual predicate: None" in capture

    assert_frame_equal(q.collect(), expected)
    assert_frame_equal(df.filter(pred), expected)


def test_pyarrow_dataset_comm_subplan_elim(tmp_path: Path) -> None:
    df0 = pl.DataFrame({"a": [1, 2, 3]})

    df1 = pl.DataFrame({"a": [1, 2]})

    file_path_0 = tmp_path / "0.parquet"
    file_path_1 = tmp_path / "1.parquet"

    df0.write_parquet(file_path_0)
    df1.write_parquet(file_path_1)

    ds0 = ds.dataset(file_path_0, format="parquet")
    ds1 = ds.dataset(file_path_1, format="parquet")

    lf0 = pl.scan_pyarrow_dataset(ds0)
    lf1 = pl.scan_pyarrow_dataset(ds1)

    assert_frame_equal(
        lf0.join(lf1, on="a", how="inner").collect(),
        pl.DataFrame({"a": [1, 2]}),
        check_row_order=False,
    )


def test_pyarrow_dataset_predicate_verbose_log(
    tmp_path: Path,
    plmonkeypatch: PlMonkeyPatch,
    capfd: pytest.CaptureFixture[str],
) -> None:
    plmonkeypatch.setenv("POLARS_VERBOSE_SENSITIVE", "1")

    df = pl.DataFrame({"a": [1, 2, 3]})
    file_path_0 = tmp_path / "0"

    df.write_parquet(file_path_0)
    dset = ds.dataset(file_path_0, format="parquet")

    q = pl.scan_pyarrow_dataset(dset).filter(pl.col("a") < 3)

    capfd.readouterr()
    assert_frame_equal(q.collect(), pl.DataFrame({"a": [1, 2]}))
    capture = capfd.readouterr().err

    assert (
        "[SENSITIVE]: python_scan_predicate: "
        'predicate node: [(col("a")) < (3)], '
        "converted pyarrow predicate: <pyarrow.compute.Expression (a < 3)>, "
        "residual predicate: None"
    ) in capture

    q = pl.scan_pyarrow_dataset(dset).filter(pl.col("a").cast(pl.String) < "3")

    capfd.readouterr()
    assert_frame_equal(q.collect(), pl.DataFrame({"a": [1, 2]}))
    capture = capfd.readouterr().err

    assert (
        "[SENSITIVE]: python_scan_predicate: "
        'predicate node: [(col("a").strict_cast(String)) < ("3")], '
        "converted pyarrow predicate: <conversion failed>, "
        'residual predicate: Some([(col("a").strict_cast(String)) < ("3")])'
    ) in capture


@pytest.mark.write_disk
def test_pyarrow_dataset_python_scan(tmp_path: Path) -> None:
    df = pl.DataFrame({"x": [0, 1, 2, 3]})
    file_path = tmp_path / "0.parquet"
    df.write_parquet(file_path)

    dataset = ds.dataset(file_path)
    lf = pl.scan_pyarrow_dataset(dataset)
    out = lf.collect(engine="streaming")

    assert_frame_equal(df, out)


def test_pyarrow_dataset_allow_pyarrow_filter_false() -> None:
    df = pl.DataFrame({"item": ["foo", "bar", "baz"], "price": [10.0, 20.0, 30.0]})
    dataset = ds.dataset(df.to_arrow(compat_level=pl.CompatLevel.oldest()))

    # basic scan without filter
    result = pl.scan_pyarrow_dataset(dataset, allow_pyarrow_filter=False).collect()
    assert_frame_equal(result, df)

    # with filter (predicate should be applied by Polars, not PyArrow)
    result = (
        pl.scan_pyarrow_dataset(dataset, allow_pyarrow_filter=False)
        .filter(pl.col("price") > 15)
        .collect()
    )

    expected = pl.DataFrame({"item": ["bar", "baz"], "price": [20.0, 30.0]})
    assert_frame_equal(result, expected)

    # check user-specified `batch_size` doesn't error (ref: #25316)
    result = (
        pl.scan_pyarrow_dataset(dataset, allow_pyarrow_filter=False, batch_size=1000)
        .filter(pl.col("price") > 15)
        .collect()
    )
    assert_frame_equal(result, expected)

    # check `allow_pyarrow_filter=True` still works
    result = (
        pl.scan_pyarrow_dataset(dataset, allow_pyarrow_filter=True)
        .filter(pl.col("price") > 15)
        .collect()
    )
    assert_frame_equal(result, expected)


def test_scan_pyarrow_dataset_filter_with_timezone_26029() -> None:
    table = pa.table(
        {
            "valid_from": [
                datetime(2025, 8, 26, 10, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 26, 11, 0, 0, tzinfo=timezone.utc),
            ],
            "valid_to": [
                datetime(2025, 8, 26, 12, 0, 0, tzinfo=timezone.utc),
                datetime(2025, 8, 26, 13, 0, 0, tzinfo=timezone.utc),
            ],
            "value": [1, 2],
        }
    )
    dataset = ds.dataset(table)

    lower_bound_time = datetime(2025, 8, 26, 11, 30, 0, tzinfo=timezone.utc)
    lf = pl.scan_pyarrow_dataset(dataset).filter(
        (pl.col("valid_from") <= lower_bound_time)
        & (pl.col("valid_to") > lower_bound_time)
    )

    assert_frame_equal(lf.collect(), pl.DataFrame(table))


def test_scan_pyarrow_dataset_filter_slice_order() -> None:
    table = pa.table(
        {
            "index": [0, 1, 2],
            "year": [2025, 2026, 2026],
            "month": [0, 0, 0],
        }
    )
    dataset = ds.dataset(table)

    q = pl.scan_pyarrow_dataset(dataset).head(2).filter(pl.col("year") == 2026)

    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"index": 1, "year": 2026, "month": 0}),
    )

    import pyarrow.compute as pc

    import polars.io.pyarrow_dataset.anonymous_scan

    # Test post-filter in engine: this tests the correct result.
    # Head is applied in scan_pyarrow, filter is applied in engine.
    assert_frame_equal(
        pl.scan_pyarrow_dataset(dataset).head(2).filter(pl.col.year == 2026).collect(),
        pl.DataFrame({"index": 1, "year": 2026, "month": 0}),
    )

    year_eq_2026 = pc.field("year") == 2026

    # Test post-filter in engine: this tests that the filter is not applied pyarrow.
    assert_frame_equal(
        polars.io.pyarrow_dataset.anonymous_scan._scan_pyarrow_dataset_impl(
            dataset,
            n_rows=2,
            predicate=year_eq_2026,
            with_columns=None,
        )[0].__next__(),
        pl.DataFrame({"index": [0, 1], "year": [2025, 2026], "month": [0, 0]}),
    )

    # Head is applied in _scan_pyarrow
    assert_frame_equal(
        pl.concat(
            polars.io.pyarrow_dataset.anonymous_scan._scan_pyarrow_dataset_impl(
                dataset,
                n_rows=1,
                predicate=None,
                with_columns=None,
                allow_pyarrow_filter=False,
            )[0]
        ),
        pl.DataFrame({"index": 0, "year": 2025, "month": 0}),
    )

    assert not polars.io.pyarrow_dataset.anonymous_scan._scan_pyarrow_dataset_impl(
        dataset,
        n_rows=0,
        predicate=year_eq_2026,
        with_columns=None,
        allow_pyarrow_filter=False,
    )[1]


@pytest.mark.write_disk
def test_arrow_predicate_conversions(tmp_path: Path) -> None:
    """Test that various arrow predicates are correctly converted and pushed down."""
    # Create test data with various data types
    df = pl.DataFrame(
        {
            "id": [1, 2, 3, 4, 5],
            "value": [10, 20, 30, 40, 50],
            "name": ["a", "b", "c", "d", "e"],
            "is_active": [True, False, True, False, True],
        }
    )

    file_path = tmp_path / "test_predicates.ipc"
    # array_filter can't handle string_view
    df.write_ipc(file_path, compat_level=pl.CompatLevel.oldest())

    # Test simple equality comparison
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("id") == 2),
        n_expected=1,
        check_predicate_pushdown=True,
    )

    # Test greater than comparison
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("value") > 25),
        n_expected=3,
        check_predicate_pushdown=True,
    )

    # Test less than or equal comparison
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("value") <= 20),
        n_expected=2,
        check_predicate_pushdown=True,
    )

    # Test boolean column filter
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(pl.col("is_active")),
        n_expected=3,
        check_predicate_pushdown=True,
    )

    # Test NOT filter
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter(~pl.col("is_active")),
        n_expected=2,
        check_predicate_pushdown=True,
    )

    # Test AND logic
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter((pl.col("id") > 2) & (pl.col("value") < 45)),
        n_expected=2,
        check_predicate_pushdown=True,
    )

    # Test OR logic
    helper_dataset_test(
        file_path,
        lambda lf: lf.filter((pl.col("id") == 1) | (pl.col("id") == 5)),
        n_expected=2,
        check_predicate_pushdown=True,
    )

    # Test is_null
    df_with_nulls = pl.DataFrame(
        {
            "id": [1, 2, None, 4],
            "value": [10, None, 30, 40],
        }
    )
    file_path_nulls = tmp_path / "test_nulls.ipc"
    df_with_nulls.write_ipc(file_path_nulls)

    helper_dataset_test(
        file_path_nulls,
        lambda lf: lf.filter(pl.col("id").is_null()),
        n_expected=1,
        check_predicate_pushdown=True,
    )

    helper_dataset_test(
        file_path_nulls,
        lambda lf: lf.filter(pl.col("id").is_not_null()),
        n_expected=3,
        check_predicate_pushdown=True,
    )


def test_pyarrow_dataset_streaming_source() -> None:
    df = pl.DataFrame({"item": ["foo", "bar", "baz"], "price": [10.0, 20.0, 30.0]})
    dataset = pl.scan_pyarrow_dataset(
        ds.dataset(df.to_arrow(compat_level=pl.CompatLevel.oldest()))
    )
    assert "streaming-python-scan" in dataset.select(pl.all()).show_graph(
        engine="streaming", plan_stage="physical", raw_output=True
    )


def test_pyarrow_dataset_residual_predicate() -> None:
    df = pl.DataFrame({"item": ["doo", None, "baz", None], "price": [1, 2, 3, 4]})
    dataset = pl.scan_pyarrow_dataset(
        ds.dataset(df.to_arrow(compat_level=pl.CompatLevel.oldest()))
    )

    # The first expression is not convertible to pyarrow
    assert dataset.filter(
        pl.col("item").str.head(2).is_in(["do"]) & (pl.col("price") <= 2)
    ).collect().to_dict(as_series=False) == {"item": ["doo"], "price": [1]}
