import pickle
import random
from pathlib import Path

import lance
import lancedb
import pytest

import polars as pl
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
