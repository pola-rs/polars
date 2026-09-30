from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from decimal import Decimal
from typing import TYPE_CHECKING

import numpy as np
import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from tests.conftest import PlMonkeyPatch

pytestmark = pytest.mark.xdist_group("streaming")


def assert_df_sorted_by(
    df: pl.DataFrame,
    sort_df: pl.DataFrame,
    cols: list[str],
    descending: list[bool] | None = None,
) -> None:
    if descending is None:
        descending = [False] * len(cols)

    # Is sorted by the key columns?
    keycols = sort_df[cols]
    equal = keycols.head(-1) == keycols.tail(-1)

    # Tuple inequality.
    # a0 < b0 || (a0 == b0 && (a1 < b1 || (a1 == b1 && ...))
    # Evaluating in reverse is easiest.
    ordered = equal[cols[-1]]
    for c, desc in zip(cols[::-1], descending[::-1], strict=True):
        ordered &= equal[c]
        if desc:
            ordered |= keycols[c].head(-1) > keycols[c].tail(-1)
        else:
            ordered |= keycols[c].head(-1) < keycols[c].tail(-1)

    assert ordered.all()

    # Do all the rows still exist?
    assert Counter(df.rows()) == Counter(sort_df.rows())


def test_streaming_sort_multiple_columns_logical_types() -> None:
    data = {
        "foo": [3, 2, 1],
        "bar": ["a", "b", "c"],
        "baz": [
            datetime(2023, 5, 1, 15, 45),
            datetime(2023, 5, 1, 13, 45),
            datetime(2023, 5, 1, 14, 45),
        ],
    }

    result = pl.LazyFrame(data).sort("foo", "baz").collect(engine="streaming")

    expected = pl.DataFrame(
        {
            "foo": [1, 2, 3],
            "bar": ["c", "b", "a"],
            "baz": [
                datetime(2023, 5, 1, 14, 45),
                datetime(2023, 5, 1, 13, 45),
                datetime(2023, 5, 1, 15, 45),
            ],
        }
    )
    assert_frame_equal(result, expected)


def test_streaming_sort() -> None:
    assert (
        pl.Series(np.random.randint(0, 100, 100))
        .to_frame("s")
        .lazy()
        .sort("s")
        .collect(engine="streaming")["s"]
        .is_sorted()
    )


def test_streaming_sort_multiple_columns(str_ints_df: pl.DataFrame) -> None:
    df = str_ints_df
    out = df.lazy().sort(["strs", "vals"]).collect(engine="streaming")
    assert_frame_equal(out, out.sort(["strs", "vals"]))


def test_streaming_sort_sorted_flag() -> None:
    # empty
    q = pl.LazyFrame(
        schema={
            "store_id": pl.UInt16,
            "item_id": pl.UInt32,
            "timestamp": pl.Datetime,
        }
    ).sort("timestamp")

    assert q.collect(engine="streaming")["timestamp"].flags["SORTED_ASC"]


@pytest.mark.parametrize(
    ("sort_by"),
    [
        ["fats_g", "category"],
        ["fats_g", "category", "calories"],
        ["fats_g", "category", "calories", "sugars_g"],
    ],
)
def test_streaming_sort_varying_order_and_dtypes(
    io_files_path: Path, sort_by: list[str]
) -> None:
    q = pl.scan_parquet(io_files_path / "foods*.parquet")
    df = q.collect()
    assert_df_sorted_by(df, q.sort(sort_by).collect(engine="streaming"), sort_by)
    assert_df_sorted_by(df, q.sort(sort_by).collect(engine="in-memory"), sort_by)


def test_streaming_sort_fixed_reverse() -> None:
    df = pl.DataFrame(
        {
            "a": [1, 1, 2, 1, 2, 4, 1, 7],
            "b": [1, 2, 2, 1, 2, 4, 8, 7],
        }
    )
    descending = [True, False]
    q = df.lazy().sort(by=["a", "b"], descending=descending)

    assert_df_sorted_by(
        df, q.collect(engine="streaming"), ["a", "b"], descending=descending
    )
    assert_df_sorted_by(
        df, q.collect(engine="in-memory"), ["a", "b"], descending=descending
    )


def test_reverse_variable_sort_13573() -> None:
    df = pl.DataFrame(
        {
            "a": ["one", "two", "three"],
            "b": ["four", "five", "six"],
        }
    ).lazy()
    assert df.sort("a", "b", descending=[True, False]).collect(
        engine="streaming"
    ).to_dict(as_series=False) == {
        "a": ["two", "three", "one"],
        "b": ["five", "six", "four"],
    }


def test_nulls_last_streaming_sort() -> None:
    assert pl.LazyFrame({"x": [1, None]}).sort("x", nulls_last=True).collect(
        engine="streaming"
    ).to_dict(as_series=False) == {"x": [1, None]}


@pytest.mark.parametrize("descending", [True, False])
@pytest.mark.parametrize("nulls_last", [True, False])
def test_sort_descending_nulls_last(descending: bool, nulls_last: bool) -> None:
    df = pl.DataFrame({"x": [1, 3, None, 2, None], "y": [1, 3, 0, 2, 0]})

    null_sentinel = 100 if descending ^ nulls_last else -100
    ref_x = [1, 3, None, 2, None]
    ref_x.sort(key=lambda k: null_sentinel if k is None else k, reverse=descending)
    ref_y = [1, 3, 0, 2, 0]
    ref_y.sort(key=lambda k: null_sentinel if k == 0 else k, reverse=descending)

    assert_frame_equal(
        df.lazy()
        .sort("x", descending=descending, nulls_last=nulls_last)
        .collect(engine="streaming"),
        pl.DataFrame({"x": ref_x, "y": ref_y}),
    )

    assert_frame_equal(
        df.lazy()
        .sort(["x", "y"], descending=descending, nulls_last=nulls_last)
        .collect(engine="streaming"),
        pl.DataFrame({"x": ref_x, "y": ref_y}),
    )


N_ROWS = 517

PARTITION_ENV = {
    "POLARS_SORT_PARTITION_THRESHOLD_BYTES": "1",
    "POLARS_SORT_TARGET_BUCKET_BYTES": "1024",
    "POLARS_SORT_BUILDER_MEMORY_BYTES": "65536",
    "POLARS_SORT_SAMPLE_ROWS": "8",
    "POLARS_IDEAL_MORSEL_SIZE": "7",
}

# The memory manager only sees allocations once a thread's drift exceeds the
# drift threshold, so small frames never trigger a spill unless it is zero.
SPILL_ENV = {
    "POLARS_OOC_MEMORY_BUDGET_MB": "0",
    "POLARS_OOC_SPILL_MIN_BYTES": "1",
    "POLARS_OOC_DRIFT_THRESHOLD": "0",
}


def _apply_sort_mode(mode: str, plmonkeypatch: PlMonkeyPatch, tmp_path: Path) -> None:
    if mode == "in_memory":
        return
    for name, value in PARTITION_ENV.items():
        plmonkeypatch.setenv(name, value)
    if mode == "spilled":
        tmp_path.mkdir(exist_ok=True)
        for name, value in SPILL_ENV.items():
            plmonkeypatch.setenv(name, value)
        plmonkeypatch.setenv("POLARS_OOC_SPILL_DIR", str(tmp_path))


@pytest.fixture(
    params=[
        "in_memory",
        "partitioned",
        pytest.param("spilled", marks=[pytest.mark.write_disk, pytest.mark.slow]),
    ]
)
def sort_mode(
    request: pytest.FixtureRequest, plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> str:
    mode = str(request.param)
    _apply_sort_mode(mode, plmonkeypatch, tmp_path)
    return mode


def assert_sort_matches_in_memory(
    lf: pl.LazyFrame,
    keys: list[str],
    *,
    exact: bool = False,
    sliced: bool = False,
    optimizations: pl.QueryOptFlags | None = None,
) -> pl.DataFrame:
    """Compare a streaming sort against the in-memory engine.

    A stable sort is fully determined, so the frames must be equal (`exact`). An
    unstable sort only fixes the order of the key columns; which of a group of
    tied rows ends up where is free, and a slice can therefore keep different
    rows.
    """
    opts = pl.QueryOptFlags() if optimizations is None else optimizations
    expected = lf.collect(engine="in-memory", optimizations=opts)
    actual = lf.collect(engine="streaming", optimizations=opts)

    if exact:
        assert_frame_equal(actual, expected)
        return actual

    if keys:
        assert_frame_equal(actual.select(keys), expected.select(keys))
    if not sliced:
        assert Counter(actual.rows()) == Counter(expected.rows())
    else:
        assert actual.height == expected.height
    return actual


def _payload(keys: pl.Series) -> pl.DataFrame:
    n = len(keys)
    return pl.DataFrame(
        {
            "k": keys,
            "idx": pl.Series(range(n), dtype=pl.Int64),
            "pad": pl.Series([f"payload-{i % 97}" for i in range(n)]),
        }
    )


def _direct_key_series() -> dict[str, pl.Series]:
    rng = np.random.default_rng(1234)
    n = N_ROWS
    raw = [int(v) for v in rng.integers(-40, 40, n)]
    ints: list[int | None] = [None if i % 37 == 0 else v for i, v in enumerate(raw)]
    uints = [None if v is None else abs(v) for v in ints]

    floats: list[float | None] = []
    for i, v in enumerate(ints):
        if v is None:
            floats.append(None)
        elif i % 53 == 0:
            floats.append(float("nan"))
        elif i % 101 == 0:
            floats.append(float("inf") if i % 202 == 0 else float("-inf"))
        elif i % 71 == 0:
            floats.append(-0.0)
        else:
            floats.append(v / 8)

    words = ["", "a", "zz", "Ångström", "日本語", "b" * 300, "aa", "ab"]
    strings = [None if v is None else words[abs(v) % len(words)] for v in ints]

    enum_cats = ["delta", "alpha", "charlie", "bravo"]
    enum_vals = [None if v is None else enum_cats[abs(v) % 4] for v in ints]

    decimals = [None if v is None else Decimal(v) / Decimal(1000) for v in ints]

    return {
        "i8": pl.Series("k", ints, dtype=pl.Int8),
        "i16": pl.Series("k", ints, dtype=pl.Int16),
        "i32": pl.Series("k", ints, dtype=pl.Int32),
        "i64": pl.Series("k", ints, dtype=pl.Int64),
        "i128": pl.Series(
            "k",
            [None if v is None else v * 10**30 for v in ints],
            dtype=pl.Int128,
        ),
        "u8": pl.Series("k", uints, dtype=pl.UInt8),
        "u16": pl.Series("k", uints, dtype=pl.UInt16),
        "u32": pl.Series("k", uints, dtype=pl.UInt32),
        "u64": pl.Series("k", uints, dtype=pl.UInt64),
        "f32": pl.Series("k", floats, dtype=pl.Float32),
        "f16": pl.Series("k", floats, dtype=pl.Float16),
        "f64": pl.Series("k", floats, dtype=pl.Float64),
        "date": pl.Series("k", ints, dtype=pl.Int32).cast(pl.Date),
        "datetime": pl.Series(
            "k", [None if v is None else v * 3_600_000_000 for v in ints]
        ).cast(pl.Datetime("us")),
        "duration": pl.Series(
            "k", [None if v is None else v * 1_000_000 for v in ints]
        ).cast(pl.Duration("us")),
        "time": pl.Series(
            "k", [None if v is None else abs(v) * 1_000_000_000 for v in ints]
        ).cast(pl.Time),
        "decimal": pl.Series("k", decimals, dtype=pl.Decimal(18, 3)),
        "enum": pl.Series("k", enum_vals, dtype=pl.Enum(enum_cats)),
        "string": pl.Series("k", strings, dtype=pl.String),
        "binary": pl.Series(
            "k",
            [None if s is None else s.encode() for s in strings],
            dtype=pl.Binary,
        ),
        "boolean": pl.Series(
            "k", [None if v is None else v % 2 == 0 for v in ints], dtype=pl.Boolean
        ),
    }


def _row_encoded_key_series() -> dict[str, pl.Series]:
    rng = np.random.default_rng(99)
    n = N_ROWS
    raw = [int(v) for v in rng.integers(0, 12, n)]
    ints: list[int | None] = [None if i % 41 == 0 else v for i, v in enumerate(raw)]
    cats = ["delta", "alpha", "charlie", "bravo"]

    return {
        "categorical": pl.Series(
            "k",
            [None if v is None else cats[v % 4] for v in ints],
            dtype=pl.Categorical,
        ),
        "struct": pl.Series(
            "k",
            [None if v is None else {"a": v % 3, "b": str(v)} for v in ints],
            dtype=pl.Struct({"a": pl.Int32, "b": pl.String}),
        ),
        "list": pl.Series(
            "k",
            [None if v is None else [v % 3] * (v % 4) for v in ints],
            dtype=pl.List(pl.Int32),
        ),
        "array": pl.Series(
            "k",
            [None if v is None else [v % 3, v % 5] for v in ints],
            dtype=pl.Array(pl.Int32, 2),
        ),
        "null": pl.Series("k", [None] * n, dtype=pl.Null),
    }


DIRECT_KEYS = _direct_key_series()
ROW_ENCODED_KEYS = _row_encoded_key_series()


@pytest.mark.parametrize("key_dtype", list(DIRECT_KEYS))
@pytest.mark.parametrize("descending", [False, True])
def test_sort_direct_key_dtypes(
    sort_mode: str, key_dtype: str, descending: bool
) -> None:
    lf = _payload(DIRECT_KEYS[key_dtype]).lazy()
    assert_sort_matches_in_memory(
        lf.sort("k", descending=descending, maintain_order=True),
        ["k"],
        exact=True,
    )


def test_sort_enum_follows_category_order(sort_mode: str) -> None:
    cats = ["delta", "alpha", "charlie", "bravo"]
    keys = pl.Series("k", [cats[i % 4] for i in range(N_ROWS)], dtype=pl.Enum(cats))
    out = _payload(keys).lazy().sort("k").collect(engine="streaming")
    assert out["k"].unique(maintain_order=True).to_list() == cats


@pytest.mark.parametrize("key_dtype", list(ROW_ENCODED_KEYS))
@pytest.mark.parametrize("descending", [False, True])
def test_sort_row_encoded_key_dtypes(
    sort_mode: str, key_dtype: str, descending: bool
) -> None:
    lf = _payload(ROW_ENCODED_KEYS[key_dtype]).lazy()
    assert_sort_matches_in_memory(
        lf.sort("k", descending=descending, maintain_order=True),
        ["k"],
        exact=True,
    )


@pytest.mark.parametrize("descending", [False, True])
@pytest.mark.parametrize("nulls_last", [False, True])
@pytest.mark.parametrize("maintain_order", [False, True])
def test_sort_order_flag_combinations(
    sort_mode: str, descending: bool, nulls_last: bool, maintain_order: bool
) -> None:
    lf = _payload(DIRECT_KEYS["i32"]).lazy()
    assert_sort_matches_in_memory(
        lf.sort(
            "k",
            descending=descending,
            nulls_last=nulls_last,
            maintain_order=maintain_order,
        ),
        ["k"],
        exact=maintain_order,
    )


@pytest.mark.parametrize("descending", [[False, True], [True, False], [True, True]])
@pytest.mark.parametrize("nulls_last", [[False, True], [True, False]])
def test_sort_multi_key_mixed_flags(
    sort_mode: str, descending: list[bool], nulls_last: list[bool]
) -> None:
    keys = DIRECT_KEYS["i8"]
    lf = pl.DataFrame(
        {
            "a": keys,
            "b": DIRECT_KEYS["string"],
            "idx": pl.Series(range(len(keys)), dtype=pl.Int64),
        }
    ).lazy()
    assert_sort_matches_in_memory(
        lf.sort(
            ["a", "b"],
            descending=descending,
            nulls_last=nulls_last,
            maintain_order=True,
        ),
        ["a", "b"],
        exact=True,
    )


def test_sort_stability_with_duplicate_keys(sort_mode: str) -> None:
    keys = pl.Series("k", [i % 5 for i in range(N_ROWS)], dtype=pl.Int32)
    lf = _payload(keys).lazy()
    out = assert_sort_matches_in_memory(
        lf.sort("k", maintain_order=True), ["k"], exact=True
    )
    assert out["idx"].to_list() == sorted(range(N_ROWS), key=lambda i: (i % 5, i))


DEGENERATE_KEYS: dict[str, pl.Series] = {
    "constant": pl.Series("k", [7] * N_ROWS, dtype=pl.Int32),
    "all_null": pl.Series("k", [None] * N_ROWS, dtype=pl.Int32),
    "mostly_null": pl.Series(
        "k", [None if i % 8 else i % 11 for i in range(N_ROWS)], dtype=pl.Int32
    ),
    "dominant_middle": pl.Series(
        "k",
        [0 if i % 20 == 0 else (100 if i % 20 == 1 else 50) for i in range(N_ROWS)],
        dtype=pl.Int32,
    ),
    "skewed": pl.Series(
        "k", [0 if i % 3 else i * i for i in range(N_ROWS)], dtype=pl.Int64
    ),
    "already_sorted": pl.Series("k", list(range(N_ROWS)), dtype=pl.Int32),
    "reverse_sorted": pl.Series("k", list(range(N_ROWS))[::-1], dtype=pl.Int32),
    "periodic": pl.Series("k", [i % 7 for i in range(N_ROWS)], dtype=pl.Int32),
    "two_values": pl.Series("k", [i % 2 for i in range(N_ROWS)], dtype=pl.Int32),
}


@pytest.mark.parametrize("pattern", list(DEGENERATE_KEYS))
@pytest.mark.parametrize("nulls_last", [False, True])
def test_sort_degenerate_key_patterns(
    sort_mode: str, pattern: str, nulls_last: bool
) -> None:
    lf = _payload(DEGENERATE_KEYS[pattern]).lazy()
    assert_sort_matches_in_memory(
        lf.sort("k", nulls_last=nulls_last, maintain_order=True),
        ["k"],
        exact=True,
    )


@pytest.mark.parametrize(
    ("offset", "length"),
    [
        (0, 5),
        (0, N_ROWS),
        (3, 10),
        (200, 1),
        (400, 200),
        (N_ROWS - 1, 10),
        (N_ROWS, 10),
        (-5, 5),
        (-N_ROWS, 10),
        (-200, 150),
    ],
)
@pytest.mark.parametrize("maintain_order", [False, True])
def test_sort_slice(
    sort_mode: str, offset: int, length: int, maintain_order: bool
) -> None:
    lf = _payload(DEGENERATE_KEYS["periodic"]).lazy()
    assert_sort_matches_in_memory(
        lf.sort("k", maintain_order=maintain_order).slice(offset, length),
        ["k"],
        exact=maintain_order,
        sliced=True,
    )


@pytest.mark.parametrize("nulls_last", [False, True])
def test_sort_slice_inside_null_bucket(sort_mode: str, nulls_last: bool) -> None:
    lf = _payload(DEGENERATE_KEYS["mostly_null"]).lazy()
    offset = 0 if not nulls_last else 300
    assert_sort_matches_in_memory(
        lf.sort("k", nulls_last=nulls_last, maintain_order=True).slice(offset, 50),
        ["k"],
        exact=True,
        sliced=True,
    )


def test_sort_head_and_tail(sort_mode: str) -> None:
    lf = _payload(DIRECT_KEYS["i32"]).lazy()
    assert_sort_matches_in_memory(
        lf.sort("k", maintain_order=True).head(9),
        ["k"],
        exact=True,
        sliced=True,
    )
    assert_sort_matches_in_memory(
        lf.sort("k", maintain_order=True).tail(9),
        ["k"],
        exact=True,
        sliced=True,
    )


def test_sort_head_stops_mid_bucket(sort_mode: str) -> None:
    lf = _payload(DEGENERATE_KEYS["periodic"]).lazy()
    assert_sort_matches_in_memory(
        lf.sort("k", maintain_order=True).head(9),
        ["k"],
        exact=True,
        sliced=True,
        optimizations=pl.QueryOptFlags(slice_pushdown=False),
    )


@pytest.mark.parametrize("k", [1, 10, 300])
def test_sort_top_k(sort_mode: str, k: int) -> None:
    lf = _payload(DIRECT_KEYS["i32"]).lazy()
    assert_sort_matches_in_memory(lf.top_k(k, by="k"), ["k"], sliced=True)
    assert_sort_matches_in_memory(lf.bottom_k(k, by="k"), ["k"], sliced=True)


@pytest.mark.parametrize("height", [0, 1, 2, 3])
def test_sort_tiny_inputs(sort_mode: str, height: int) -> None:
    keys = pl.Series("k", list(range(height))[::-1], dtype=pl.Int32)
    lf = _payload(keys).lazy()
    assert_sort_matches_in_memory(lf.sort("k", maintain_order=True), ["k"], exact=True)


def test_sort_object_key_raises() -> None:
    lf = pl.LazyFrame({"o": pl.Series([object(), object()], dtype=pl.Object)})
    with pytest.raises(pl.exceptions.InvalidOperationError, match="does not support"):
        lf.sort("o").collect(engine="streaming")


def test_sort_fewer_rows_than_buckets(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path
) -> None:
    _apply_sort_mode("partitioned", plmonkeypatch, tmp_path)
    plmonkeypatch.setenv("POLARS_SORT_MAX_BUCKETS", "64")
    plmonkeypatch.setenv("POLARS_SORT_TARGET_BUCKET_BYTES", "1")
    keys = pl.Series("k", [5, 1, 3, None, 2], dtype=pl.Int32)
    lf = _payload(keys).lazy()
    assert_sort_matches_in_memory(lf.sort("k", maintain_order=True), ["k"], exact=True)


@pytest.mark.parametrize("max_buckets", ["2", "4", "256"])
def test_sort_bucket_counts(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path, max_buckets: str
) -> None:
    _apply_sort_mode("partitioned", plmonkeypatch, tmp_path)
    plmonkeypatch.setenv("POLARS_SORT_MAX_BUCKETS", max_buckets)
    plmonkeypatch.setenv("POLARS_SORT_SAMPLE_ROWS", "512")
    lf = _payload(DIRECT_KEYS["i32"]).lazy()
    assert_sort_matches_in_memory(lf.sort("k", maintain_order=True), ["k"], exact=True)


def test_sort_key_only_frame(sort_mode: str) -> None:
    lf = pl.DataFrame({"k": DIRECT_KEYS["i32"]}).lazy()
    assert_sort_matches_in_memory(lf.sort("k", maintain_order=True), ["k"], exact=True)


def test_sort_expression_sort_and_sort_by(sort_mode: str) -> None:
    df = _payload(DIRECT_KEYS["i32"])
    assert_sort_matches_in_memory(df.lazy().select(pl.col("k").sort()), ["k"])
    assert_sort_matches_in_memory(
        df.lazy().select(pl.col("idx").sort_by(["k", "idx"], nulls_last=True)),
        ["idx"],
    )
    assert_sort_matches_in_memory(
        df.lazy().select(pl.col("pad").sort_by("idx", descending=True)),
        [],
        exact=True,
    )
    assert_sort_matches_in_memory(
        df.lazy().sort(pl.col("k").abs(), maintain_order=True),
        [],
        exact=True,
    )


def test_sort_group_by_and_over(sort_mode: str) -> None:
    df = _payload(DEGENERATE_KEYS["periodic"])
    assert_sort_matches_in_memory(
        df.lazy()
        .group_by("k", maintain_order=True)
        .agg(pl.col("idx").sum(), pl.col("pad").first()),
        ["k"],
        exact=True,
    )
    assert_sort_matches_in_memory(
        df.lazy().select(
            "idx", pl.col("idx").cum_sum().over("k", order_by="idx").alias("c")
        ),
        ["idx"],
        exact=True,
    )


def test_sort_feeds_sorted_group_by(sort_mode: str) -> None:
    df = _payload(DEGENERATE_KEYS["periodic"])
    assert_sort_matches_in_memory(
        df.lazy().sort("k").group_by("k", maintain_order=True).agg(pl.col("idx").sum()),
        ["k"],
        exact=True,
    )


# Spilling is left out: the in-memory sink drops the sorted flag of every frame
# it spills, independently of where that frame came from.
@pytest.mark.parametrize("mode", ["in_memory", "partitioned"])
def test_sort_sets_sorted_flag(
    plmonkeypatch: PlMonkeyPatch, tmp_path: Path, mode: str
) -> None:
    _apply_sort_mode(mode, plmonkeypatch, tmp_path)
    df = _payload(DIRECT_KEYS["i32"])
    out = df.lazy().sort("k").collect(engine="streaming")
    assert out["k"].flags["SORTED_ASC"]
    out = df.lazy().sort("k", descending=True).collect(engine="streaming")
    assert out["k"].flags["SORTED_DESC"]


def test_sort_feeds_merge_sorted(sort_mode: str) -> None:
    left = _payload(DEGENERATE_KEYS["periodic"]).lazy().sort("k")
    right = _payload(DEGENERATE_KEYS["two_values"]).lazy().sort("k")
    assert_sort_matches_in_memory(left.merge_sorted(right, key="k"), ["k"])


def test_sort_resume_over_phase_boundary_join(
    sort_mode: str, plmonkeypatch: PlMonkeyPatch
) -> None:
    plmonkeypatch.setenv("POLARS_JOIN_SAMPLE_LIMIT", "16")
    df = _payload(DEGENERATE_KEYS["periodic"])
    other = pl.DataFrame(
        {
            "k": pl.Series([0, 1, 2, 3, 4, 5, 6], dtype=pl.Int32),
            "extra": list("abcdefg"),
        }
    ).lazy()
    assert_sort_matches_in_memory(
        df.lazy().sort("k", maintain_order=True).join(other, on="k", how="inner"),
        [],
    )


def test_sort_resume_over_phase_boundary_horizontal_concat(sort_mode: str) -> None:
    sorted_lf = (
        _payload(DEGENERATE_KEYS["periodic"]).lazy().sort("k", maintain_order=True)
    )
    other = pl.DataFrame({"extra": pl.Series(range(N_ROWS), dtype=pl.Int64)}).lazy()
    assert_sort_matches_in_memory(
        pl.concat([sorted_lf, other], how="horizontal"),
        ["k"],
        exact=True,
    )


def test_sort_resume_over_phase_boundary_multiplexer(sort_mode: str) -> None:
    sorted_lf = (
        _payload(DEGENERATE_KEYS["periodic"])
        .lazy()
        .sort("k", maintain_order=True)
        .with_row_index("rn")
    )
    assert_sort_matches_in_memory(
        sorted_lf.join(sorted_lf.select("rn", pl.col("idx").alias("idx2")), on="rn"),
        [],
    )
