"""Bounded semantic feasibility and seeded coverage probe (no future fast path)."""
# This one-off probe retains heterogeneous data and the executed helper structure.
# ruff: noqa: ANN001, ANN003, ANN201, C408, EM101

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import platform
import random
import sys
from datetime import date
from decimal import Decimal
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

import polars as pl
from polars.testing import assert_frame_equal, assert_series_equal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tests.unit.io.test_struct_field_pruning_iceberg import (
    OLD_VERSIONS,
    _f,
    _scan,
    _write_file,
)

NO_PD = pl.QueryOptFlags(predicate_pushdown=False, projection_pushdown=False)
RECORDS = []


def record(kind, **data):
    RECORDS.append(dict(kind=kind, **data))
    print(json.dumps(RECORDS[-1]), flush=True)


def check(q):
    expected = q.collect(optimizations=NO_PD)
    for engine in ("streaming", "in-memory"):
        assert_frame_equal(q.collect(engine=engine), expected)
    return expected


def field(expr, names):
    for name in names:
        expr = expr.struct.field(name)
    return expr


def mapping_probe(out):
    for i, (names, reverse) in enumerate([({}, False), *OLD_VERSIONS]):
        path = out / f"mapped-{i}.parquet"
        _write_file(path, i * 10, names, reverse)
        root = _scan([path])
        physical = pl.scan_parquet(path)
        for ids, logical in [([2, 3], ["a"]), ([2, 5, 6], ["c", "x"])]:
            physical_names = [
                names.get(k, {2: "s", 3: "a", 5: "c", 6: "x"}[k]) for k in ids
            ]
            candidate = physical.select(
                field(pl.col(physical_names[0]), physical_names[1:]).alias("v")
            ).collect()["v"]
            expected = root.select(field(pl.col("s"), logical).alias("v")).collect(
                optimizations=NO_PD
            )["v"]
            assert_series_equal(candidate, expected)
            check(
                root.filter(field(pl.col("s"), logical).is_not_null()).select("id", "s")
            )
            record(
                "mapping",
                file=i,
                reverse=reverse,
                renamed=names,
                logical=logical,
                values=expected.to_list(),
            )

    schema = pa.schema(
        [_f("s", pa.struct([_f("c", pa.struct([_f("marker", pa.int32(), 4)]), 3)]), 2)]
    )
    values = [None, {"c": None}, {"c": {"marker": None}}, {"c": {"marker": 1}}]
    path = out / "defaults.parquet"
    pq.write_table(pa.table({"s": values}, schema=schema), path)
    target = pa.schema(
        [
            _f(
                "s",
                pa.struct(
                    [
                        _f(
                            "c",
                            pa.struct(
                                [_f("marker", pa.int64(), 4), _f("v", pa.int64(), 5)]
                            ),
                            3,
                        )
                    ]
                ),
                2,
            )
        ]
    )
    target_pl = pl.from_arrow(target.empty_table()).schema
    for default in (None, 9):
        lf = pl.scan_parquet(
            path,
            schema=target_pl,
            _column_mapping=("iceberg-column-mapping", target),
            _default_values=(
                "iceberg",
                ({}, {5: pl.Series([default], dtype=pl.Int64)}),
            ),
            cast_options=pl.ScanCastOptions(
                integer_cast="upcast", missing_struct_fields="insert"
            ),
        )
        leaf = field(pl.col("s"), ["c", "v"])
        expected = check(lf.select(leaf.alias("v")))["v"].to_list()
        validity = [s is not None and s["c"] is not None for s in values]
        candidate = [default if valid else None for valid in validity]
        assert candidate == expected
        naive = [default] * len(values)
        if default is not None:
            assert naive != expected
        for predicate in (leaf == 9, leaf.is_null()):
            check(lf.filter(predicate))
        record(
            "missing_default",
            default=default,
            expected=expected,
            naive=naive,
            ancestor_masked=candidate,
            naive_wrong=naive != expected,
            ancestor_validity=validity,
        )
    raw = (
        pl.scan_parquet(path)
        .select(field(pl.col("s"), ["c", "marker"]).alias("v"))
        .collect()["v"]
    )
    upcast = raw.cast(pl.Int64)
    expected = lf.select(field(pl.col("s"), ["c", "marker"]).alias("v")).collect(
        optimizations=NO_PD
    )["v"]
    assert_series_equal(upcast, expected)
    record(
        "upcast",
        input_dtype=str(raw.dtype),
        output_dtype=str(upcast.dtype),
        values=upcast.to_list(),
    )

    # Strict scalar casts require null propagation BEFORE cast: hidden child values
    # can be invalid. These are controlled Arrow buffers, not a new decoder path.
    raw_strings = pa.array(["bad-hidden", "12", "bad-visible"])
    parent = pa.StructArray.from_arrays(
        [raw_strings], names=["v"], mask=pa.array([True, False, False])
    )
    frame = pl.from_arrow(pa.table({"s": parent}))
    masked = frame.select(pl.col("s").struct.field("v")).to_series()
    for label, series in (
        ("raw_child", pl.from_arrow(raw_strings)),
        ("ancestor_masked", masked),
    ):
        try:
            series.head(2).cast(pl.Int64, strict=True)
            outcome = "success"
        except pl.exceptions.InvalidOperationError:
            outcome = "InvalidOperationError"
        assert outcome == (
            "success" if label == "ancestor_masked" else "InvalidOperationError"
        )
        record("strict_cast_hidden", input=label, outcome=outcome)
    for series in (pl.from_arrow(raw_strings), masked):
        try:
            series.cast(pl.Int64, strict=True)
        except pl.exceptions.InvalidOperationError:
            pass
        else:
            raise AssertionError("visible invalid cast must error")
    record(
        "strict_cast_visible",
        outcome="both error; no general scan-transform eligibility proved",
    )


def randomized_probe(out, cases):
    rng = random.Random(20261112)
    specs = [
        ("int32", pa.int32(), lambda i: i - 20),
        ("float", pa.float64(), lambda i: float("nan") if i % 11 == 0 else i / 3),
        ("string", pa.string(), lambda i: f"v{i % 7}"),
        ("binary", pa.binary(), lambda i: bytes([i % 255])),
        ("bool", pa.bool_(), lambda i: bool(i % 2)),
        ("decimal", pa.decimal128(10, 2), lambda i: Decimal(i).scaleb(-2)),
        ("date", pa.date32(), lambda i: date(2020, 1, i % 28 + 1)),
        ("float16", pa.float16(), lambda i: float(i)),
    ]
    for case in range(cases):
        depth = case % 6 + 1
        label, dtype, value = specs[case % len(specs)]
        vals = [None if rng.random() < 0.2 else value(i) for i in range(97)]
        dtype = pa.struct([pa.field("v", dtype), pa.field("sibling", pa.int64())])
        vals = [
            None if rng.random() < 0.2 else {"v": v, "sibling": i}
            for i, v in enumerate(vals)
        ]
        names = []
        for level in range(depth - 1):
            name = f"n{level}"
            names.insert(0, name)
            dtype = pa.struct([pa.field(name, dtype)])
            vals = [None if rng.random() < 0.2 else {name: v} for v in vals]
        path = out / f"random-{case}.parquet"
        dictionary = bool(case % 2)
        version = "1.0" if case % 3 else "2.0"
        pq.write_table(
            pa.table({"s": pa.array(vals, type=dtype), "i": range(97)}),
            path,
            use_dictionary=dictionary,
            data_page_version=version,
            row_group_size=23,
            data_page_size=64,
            write_batch_size=16,
        )
        leaf = field(pl.col("s"), [*names, "v"])
        for parallel in ("auto", "prefiltered"):
            lf = pl.scan_parquet(path, use_statistics=False, parallel=parallel)
            for predicate in (leaf.is_null(), leaf.is_not_null(), leaf == leaf):
                check(lf.filter(predicate).select("s", "i"))
                check(lf.filter(predicate).select("i"))
            # End-to-end slice correctness; this does NOT prove the flat decoder
            # receives Filter::Range, whose direct unit coverage remains a gap.
            check(lf.slice(11, 43).filter(leaf.is_not_null()).select("s", "i"))
        record(
            "random_tree",
            case=case,
            depth=depth,
            dtype=label,
            dictionary=dictionary,
            page_version=version,
            rows=97,
            queries=14,
            engine_checks=28,
        )


def categorical_hive_probe(out):
    categorical = pl.Series("v", ["a", "b", None, "a"], dtype=pl.Categorical)
    frame = (
        categorical.to_frame()
        .with_row_index("i")
        .select(
            s=pl.when(pl.col("i") == 3).then(None).otherwise(pl.struct("v")),
            i=pl.col("i"),
        )
    )
    path = out / "categorical.parquet"
    frame.write_parquet(path, row_group_size=2)
    leaf = pl.col("s").struct.field("v")
    for parallel in ("auto", "prefiltered"):
        lf = pl.scan_parquet(path, use_statistics=False, parallel=parallel)
        for pred in (leaf == "a", leaf.is_null()):
            result = check(lf.filter(pred).select(leaf, "s", "i"))
            assert result.schema["v"] == pl.Categorical
    record(
        "categorical", queries=4, engine_checks=8, dtype="Categorical", exact_dtype=True
    )
    directory = out / "hive"
    for part in (1, 2):
        dest = directory / f"p={part}"
        dest.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"s": [None, {"v": 2}, {"v": 20}], "i": [0, 1, 2]}).write_parquet(
            dest / "data.parquet"
        )
    for parallel in ("auto", "prefiltered"):
        lf = pl.scan_parquet(
            directory, hive_partitioning=True, use_statistics=False, parallel=parallel
        )
        result = check(
            lf.filter((leaf > 10) & (pl.col("p") == 2)).select("s", "i", "p")
        )
        assert result["p"].to_list() == [2]
    record("hive", queries=2, engine_checks=4, predicate="s.v > 10 AND p == 2")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--out", type=Path, default=Path("target/nested-followup/mapping")
    )
    parser.add_argument("--cases", type=int, default=48)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runtime = Path(
        importlib.import_module("_polars_runtime_32._polars_runtime").__file__
    )
    metadata = dict(
        machine=platform.node(),
        platform=platform.platform(),
        revision=os.environ.get("PROBE_SOURCE_REVISION", "not supplied"),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_sha256=hashlib.file_digest(runtime.open("rb"), "sha256").hexdigest(),
        command=sys.argv,
        threads=pl.thread_pool_size(),
        seed=20261112,
        probe_env={
            k: v for k, v in os.environ.items() if k.startswith("POLARS_NESTED_PROBE")
        },
        caveat="semantic reference probes only; native mapped paths still use root fallback",
    )
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    mapping_probe(args.out)
    randomized_probe(args.out, args.cases)
    categorical_hive_probe(args.out)
    (args.out / "results.json").write_text(
        json.dumps(RECORDS, indent=2, default=str) + "\n"
    )
    print(f"PASS: {args.cases} seeded trees; {len(RECORDS)} result records", flush=True)


if __name__ == "__main__":
    main()
