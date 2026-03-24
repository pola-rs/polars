from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any

import pytest
from hypothesis import given
from hypothesis.strategies import booleans

import polars as pl
from polars.testing import assert_frame_equal
from polars.testing.parametric.strategies import column, dataframes

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import UniqueKeepStrategy
    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


@pytest.mark.write_disk
@pytest.mark.slow
def test_streaming_out_of_core_unique(
    io_files_path: Path, tmp_path: Path, plmonkeypatch: PlMonkeyPatch, capfd: Any
) -> None:
    morsel_size = os.environ.get("POLARS_IDEAL_MORSEL_SIZE")
    if morsel_size is not None and int(morsel_size) < 1000:
        pytest.skip("test is too slow for small morsel sizes")

    tmp_path.mkdir(exist_ok=True)
    plmonkeypatch.setenv("POLARS_TEMP_DIR", str(tmp_path))
    plmonkeypatch.setenv("POLARS_FORCE_OOC", "1")
    plmonkeypatch.setenv("POLARS_VERBOSE", "1")
    plmonkeypatch.setenv("POLARS_STREAMING_GROUPBY_SPILL_SIZE", "256")
    df = pl.read_csv(io_files_path / "foods*.csv")
    # this creates 10M rows
    q = df.lazy()
    q = q.join(q, how="cross").select(df.columns).head(10_000)

    # uses out-of-core unique
    df1 = q.join(q.head(1000), how="cross").unique().collect(engine="streaming")
    # this ensures the cross join gives equal result but uses the in-memory unique
    df2 = q.join(q.head(1000), how="cross").collect(engine="streaming").unique()
    assert df1.shape == df2.shape

    # TODO: Re-enable this check when this issue is fixed: https://github.com/pola-rs/polars/issues/10466
    _ = capfd.readouterr().err
    # assert "OOC group_by started" in err


def test_streaming_unique() -> None:
    df = pl.DataFrame({"a": [1, 2, 2, 2], "b": [3, 4, 4, 4], "c": [5, 6, 7, 7]})
    q = df.lazy().unique(subset=["a", "c"], maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))

    q = df.lazy().unique(subset=["b", "c"], maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))

    q = df.lazy().unique(subset=None, maintain_order=False).sort(["a", "b", "c"])
    assert_frame_equal(q.collect(engine="streaming"), q.collect(engine="in-memory"))


def test_streaming_unique_list_of_struct_with_decimal_26505() -> None:
    df = pl.DataFrame(
        {"a": [[{"f0": None, "f1": b"x"}]]},
        schema={
            "a": pl.List(
                pl.Struct(
                    [pl.Field("f0", pl.Decimal(10, 2)), pl.Field("f1", pl.Binary())]
                )
            )
        },
    )
    result = df.lazy().unique(maintain_order=True).collect(engine="streaming")
    assert_frame_equal(result, df)


@given(
    df=dataframes(cols=[column("key")]), descending=booleans(), nulls_last=booleans()
)
@pytest.mark.parametrize("maintain_order", [False, True])
@pytest.mark.parametrize("keep", ["any", "first"])
def test_sorted_streaming_unique_vs_in_memory(
    df: pl.DataFrame,
    descending: bool,
    nulls_last: bool,
    maintain_order: bool,
    keep: UniqueKeepStrategy,
) -> None:
    df = df.sort("key", descending=descending, nulls_last=nulls_last)
    lf = (
        df.lazy()
        .set_sorted("key", descending=descending, nulls_last=nulls_last)
        .unique("key", keep=keep, maintain_order=maintain_order)
    )
    dot = lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert isinstance(dot, str)
    assert "sorted-unique" in dot

    assert_frame_equal(
        lf.collect(engine="streaming"),
        lf.collect(engine="in-memory"),
        check_row_order=maintain_order,
    )


@given(
    df=dataframes(cols=[column("key1"), column("key2")]),
    descending=booleans(),
    nulls_last=booleans(),
)
@pytest.mark.parametrize("maintain_order", [False, True])
@pytest.mark.parametrize("keep", ["any", "first"])
def test_sorted_streaming_unique_vs_in_memory_multikey(
    df: pl.DataFrame,
    descending: bool,
    nulls_last: bool,
    maintain_order: bool,
    keep: UniqueKeepStrategy,
) -> None:
    df = df.sort(["key1", "key2"], descending=descending, nulls_last=nulls_last)
    lf = (
        df.lazy()
        .set_sorted(["key1", "key2"], descending=descending, nulls_last=nulls_last)
        .unique(["key1", "key2"], keep=keep, maintain_order=maintain_order)
    )
    dot = lf.show_graph(engine="streaming", plan_stage="physical", raw_output=True)
    assert isinstance(dot, str)
    assert "sorted-unique" in dot

    assert_frame_equal(
        lf.collect(engine="streaming"),
        lf.collect(engine="in-memory"),
        check_row_order=maintain_order,
    )
