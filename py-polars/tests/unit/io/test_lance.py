import pickle
import random
from pathlib import Path
from typing import Any

import lance
import lancedb
import pyarrow as pa
import pytest

import polars as pl
from polars._typing import EngineType
from polars.testing.asserts.frame import assert_frame_equal


def test_scan_from_lance(tmp_path: Path) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": 1})

    ds = lance.write_dataset(df, ds_path)

    q = pl.scan_lance(ds)
    assert_frame_equal(q.collect(), df)

    del ds
    assert_frame_equal(pickle.loads(pickle.dumps(q)).collect(), df)


def test_scan_from_lance_no_version_scans_latest(tmp_path: Path) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": 1})

    ds = lance.write_dataset(df, ds_path)

    q = pl.scan_lance(ds)
    assert_frame_equal(q.collect(), df)

    lance.write_dataset(df, ds_path, mode="append")

    assert_frame_equal(
        pickle.loads(pickle.dumps(q)).collect(),
        pl.DataFrame({"a": [1, 1]}),
    )


def test_scan_from_lance_versioned(tmp_path: Path) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": 1})

    ds = lance.write_dataset(df, ds_path)

    version = ds.version

    q = pl.scan_lance(ds, version=version)
    assert_frame_equal(q.collect(), df)

    lance.write_dataset(df, ds_path, mode="append")

    assert_frame_equal(pickle.loads(pickle.dumps(q)).collect(), df)


@pytest.mark.write_disk
def test_scan_from_lancedb(tmp_path: Path) -> None:
    db = lancedb.connect(tmp_path / "database.lance")

    df = pl.DataFrame({"a": 1})
    tbl = db.create_table("table", schema=df.schema.to_arrow())

    q = pl.scan_lance(tbl)
    assert q.collect().shape == (0, 1)
    assert pl.scan_lance(tbl).collect().shape == (0, 1)

    tbl.add(df)
    q = pl.scan_lance(tbl)

    assert_frame_equal(q.collect(), df)
    assert_frame_equal(pl.scan_lance(tbl).collect(), df)

    assert pl.scan_lance(tbl).select(pl.len()).collect().item() == 1


@pytest.mark.slow
def test_scan_from_lance_slice_parametric(tmp_path: Path) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": range(100)})

    ds = lance.write_dataset(df, ds_path, max_rows_per_file=5)

    slice_offset = 0
    slice_len = 0

    try:
        for slice_offset in random.sample(range(-500, 500), 20):
            for slice_len in random.sample(range(500), 5):
                q = pl.scan_lance(ds).slice(slice_offset, slice_len)
                assert_frame_equal(q.collect(), df.slice(slice_offset, slice_len))

                q = pl.scan_lance(ds).with_row_index().slice(slice_offset, slice_len)
                assert_frame_equal(
                    q.collect(),
                    df.lazy().with_row_index().slice(slice_offset, slice_len).collect(),
                )
    except Exception as exc:
        msg = f"slice = {(slice_offset, slice_len)}, {exc = }"
        raise type(exc)(msg) from exc


def test_scan_lance_filter_pushdown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capfd: pytest.CaptureFixture[str]
) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": range(100), "b": [str(x) for x in range(100)]})

    # Note: Lance does not support filters on `large_string`.
    ds = lance.write_dataset(
        df.to_arrow().cast(pa.schema({"a": pa.int64(), "b": pa.string()})),
        ds_path,
        max_rows_per_file=7,
    )

    monkeypatch.setenv("POLARS_VERBOSE", "1")

    for predicate, n_pushed, n_filters in [
        (pl.col("a") > 50, 1, 1),
        ((pl.col("a") > 50) & (pl.col("a") < 70), 2, 2),
        (pl.col("a").is_in([3, 10, 99]), 1, 1),
        (pl.col("b") == "3", 1, 1),
        # Not supported by lance
        (pl.col("b").str.starts_with("1"), 0, 1),
        ((pl.col("a") > 50) & pl.col("b").str.starts_with("6"), 1, 2),
    ]:
        capfd.readouterr()

        q = pl.scan_lance(ds).filter(predicate)

        assert_frame_equal(q.collect(), df.filter(predicate))
        err = capfd.readouterr().err
        assert f"filters pushed to lance: {n_pushed} / {n_filters}" in err, err

        # Operations applied after the filter
        assert_frame_equal(q.head(3).collect(), df.filter(predicate).head(3))
        assert_frame_equal(q.tail(3).collect(), df.filter(predicate).tail(3))
        assert_frame_equal(
            q.with_row_index().collect(),
            df.filter(predicate).with_row_index(),
        )
        assert_frame_equal(
            q.with_row_index().slice(2, 5).collect(),
            df.filter(predicate).with_row_index().slice(2, 5),
        )
        assert_frame_equal(q.select("a").collect(), df.filter(predicate).select("a"))
        assert q.select(pl.len()).collect().item() == df.filter(predicate).height
        assert (
            q.select(pl.len()).collect(engine="streaming").item()
            == df.filter(predicate).height
        )


def test_scan_lance_filter_pushdown_explain(tmp_path: Path) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": range(10)})

    ds = lance.write_dataset(df, ds_path)

    plan = pl.scan_lance(ds).filter(pl.col("a") > 5).explain()
    assert "filter:" in plan


@pytest.mark.slow
def test_scan_lance_filter_slice_parametric(tmp_path: Path) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": range(100)})

    ds = lance.write_dataset(df, ds_path, max_rows_per_file=5)

    predicate = pl.col("a").is_in(range(0, 100, 3)) | (pl.col("a") > 80)
    expect = df.filter(predicate)

    slice_offset = 0
    slice_len = 0

    try:
        for slice_offset in random.sample(range(-100, 100), 20):
            for slice_len in random.sample(range(100), 5):
                q = pl.scan_lance(ds).filter(predicate).slice(slice_offset, slice_len)
                assert_frame_equal(q.collect(), expect.slice(slice_offset, slice_len))

                q = (
                    pl.scan_lance(ds)
                    .filter(predicate)
                    .with_row_index()
                    .slice(slice_offset, slice_len)
                )
                assert_frame_equal(
                    q.collect(),
                    expect.with_row_index().slice(slice_offset, slice_len),
                )
    except Exception as exc:
        msg = f"slice = {(slice_offset, slice_len)}, {exc = }"
        raise type(exc)(msg) from exc


def test_scan_lance_multiple_scans_same_dataset(tmp_path: Path) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": range(20), "b": range(20, 40)})

    ds = lance.write_dataset(df, ds_path, max_rows_per_file=3)

    lf = pl.scan_lance(ds)

    q = pl.concat(
        [
            lf.filter(pl.col("a") > 15),
            lf.filter(pl.col("a") < 3),
            lf.filter(pl.col("b") < 22).select("a"),
        ],
        how="diagonal",
    )

    expect = pl.concat(
        [
            df.filter(pl.col("a") > 15),
            df.filter(pl.col("a") < 3),
            df.filter(pl.col("b") < 22).select("a"),
        ],
        how="diagonal",
    )

    assert_frame_equal(q.collect(), expect)
    assert_frame_equal(q.collect(engine="streaming"), expect)


def test_scan_lance_resolver_limit_before_filter(tmp_path: Path) -> None:
    import pyarrow.compute as pc

    from polars.io.cloud._utils import NoPickleOption
    from polars.io.lance._scan_resolver import LanceScanResolver
    from polars.lazyframe.resolver import FilterExpr

    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": range(100)})

    ds = lance.write_dataset(df, ds_path, max_rows_per_file=7)

    resolver = LanceScanResolver(
        dataset_=NoPickleOption(ds),
        dataset_uri_=None,
        version=None,
        storage_options=None,
        credential_provider_builder=None,
    )

    for limit in [10, 60]:
        lf, props = resolver.resolve_lazyframe(  # type: ignore[misc]
            projection=None,
            limit=limit,
            filters=[
                FilterExpr(
                    expr=pl.col("a") > 50,
                    pyarrow_str=None,
                    _pyarrow_expr=pc.field("a") > 50,
                )
            ],
            filter_columns=["a"],
            filter_drop_columns_idx=None,
            existing_resolved_version_key=None,
        )

        assert lf is not None
        assert set(props.applied_filters) == {0}
        assert_frame_equal(lf.collect(), df.head(limit).filter(pl.col("a") > 50))


@pytest.mark.parametrize("engine", ["in-memory", "streaming"])
def test_scan_lance_filter_pushdown_skips_files(
    tmp_path: Path, engine: EngineType
) -> None:
    ds_path = tmp_path / "dataset.lance"

    df = pl.DataFrame({"a": range(30), "b": [str(x) for x in range(30)]})

    lance.write_dataset(
        df.to_arrow().cast(pa.schema({"a": pa.int64(), "b": pa.string()})),
        ds_path,
        max_rows_per_file=10,
    )

    # Lance only skips fragments based on the filter if there is an index.
    lance.dataset(ds_path).create_scalar_index(column="a", index_type="BTREE")

    ds = lance.dataset(ds_path)

    # Delete the data files of the last fragment (rows where `a >= 20`).
    for data_file in ds.get_fragments()[-1].data_files():
        (ds_path / "data" / data_file.path).unlink()

    def assert_raises_not_found(q: pl.LazyFrame, **kw: Any) -> None:
        with pytest.raises(pa.ArrowInvalid, match="Not found"):
            q.collect(engine=engine, **kw)

    # Baseline: Scanning reads the deleted files.
    assert_raises_not_found(pl.scan_lance(ds))

    for predicate in [
        pl.col("a") < 5,
        pl.col("a").is_in([1, 12]),
        # Only the lance-supported part of the predicate is pushed.
        (pl.col("a") < 5) & pl.col("b").str.starts_with("1"),
    ]:
        q = pl.scan_lance(ds).filter(predicate)
        expect = df.filter(predicate)

        assert_frame_equal(q.collect(engine=engine), expect)
        assert_frame_equal(
            q.with_row_index().collect(engine=engine), expect.with_row_index()
        )
        assert_frame_equal(q.tail(2).collect(engine=engine), expect.tail(2))
        assert q.select(pl.len()).collect(engine=engine).item() == expect.height

        # Deleted files are read without predicate pushdown.
        assert_raises_not_found(
            q, optimizations=pl.QueryOptFlags(predicate_pushdown=False)
        )

    # Filter matches rows in the deleted files.
    assert_raises_not_found(pl.scan_lance(ds).filter(pl.col("a") > 25))

    # Filter not supported by lance.
    assert_raises_not_found(pl.scan_lance(ds).filter(pl.col("b").str.starts_with("1")))
