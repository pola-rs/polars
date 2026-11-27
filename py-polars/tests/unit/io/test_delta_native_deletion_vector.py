from __future__ import annotations

import struct
import zlib
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError
from polars.io.delta._deletion_vectors import _decode_z85, _native_deletion_vectors
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path

    from polars._typing import DeletionFiles


def _native(vectors: dict[int, tuple[str | bytes, int, int, int]]) -> DeletionFiles:
    return ("delta-deletion-vector-native", vectors)


@pytest.fixture
def plugin_decoder() -> None:
    """Native descriptors are decoded by the polars-delta plugin."""
    pytest.importorskip("polars_delta")
    from polars.io.delta._plugin import dv_decoder_available

    assert dv_decoder_available()


def bitmap(rows: list[int]) -> bytes:
    assert rows == sorted(set(rows))
    if not rows:
        return struct.pack("<IQ", 1681511377, 0)
    assert max(rows) < 65536
    return struct.pack(
        "<IQIIIHHI", 1681511377, 1, 0, 12346, 1, 0, len(rows) - 1, 16
    ) + struct.pack(f"<{len(rows)}H", *rows)


def frame(rows: list[int]) -> bytes:
    body = bitmap(rows)
    return struct.pack(">I", len(body)) + body + struct.pack(">I", zlib.crc32(body))


def actions(dv: dict[str, object] | None) -> pl.DataFrame:
    return pl.DataFrame(
        {"path": ["a%20b.parquet"], "deletion_vector": [dv]},
        schema={
            "path": pl.String,
            "deletion_vector": pl.Struct(
                {
                    "storageType": pl.String,
                    "pathOrInlineDv": pl.String,
                    "offset": pl.Int32,
                    "sizeInBytes": pl.Int32,
                    "cardinality": pl.Int64,
                }
            ),
        },
    )


def test_native_descriptor_capability_and_validation() -> None:
    paths = ["/table/a b.parquet"]
    assert _native_deletion_vectors(pl.DataFrame(), paths, "/table/") is None
    assert _native_deletion_vectors(actions(None), paths, "/table/") == (
        "delta-deletion-vector-native",
        {},
    )
    dv: dict[str, object] = {
        "storageType": "u",
        "pathOrInlineDv": "ab^-aqEH.-t@S}K{vb[*k^",
        "offset": 4,
        "sizeInBytes": 40,
        "cardinality": 6,
    }
    result = _native_deletion_vectors(actions(dv), paths, "/table/")
    assert result == (
        "delta-deletion-vector-native",
        {
            0: (
                "/table/ab/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin",
                4,
                40,
                6,
            )
        },
    )
    with pytest.raises(ComputeError, match="schema"):
        _native_deletion_vectors(
            pl.DataFrame({"deletion_vector": [1]}), paths, "/table/"
        )
    with pytest.raises(ComputeError, match="absolute"):
        _native_deletion_vectors(
            actions({**dv, "storageType": "p", "pathOrInlineDv": "relative"}),
            paths,
            "/table/",
        )
    with pytest.raises(ComputeError, match="Z85"):
        _decode_z85("#####")


@pytest.mark.parametrize("storage", ["u", "p"])
@pytest.mark.parametrize("offset", [None, 0, 1])
def test_native_persisted_descriptor_offset(storage: str, offset: int | None) -> None:
    location = "^-aqEH.-t@S}K{vb[*k^" if storage == "u" else "/table/dv.bin"
    result = _native_deletion_vectors(
        actions(
            {
                "storageType": storage,
                "pathOrInlineDv": location,
                "offset": offset,
                "sizeInBytes": 44,
                "cardinality": 6,
            }
        ),
        ["/table/a b.parquet"],
        "/table/",
    )
    assert result is not None
    # PROTOCOL and Spark default an absent persisted offset to zero.
    assert isinstance(result[1], dict)
    assert result[1][0][1] == (0 if offset is None else offset)


def test_native_inline_descriptor() -> None:
    literal = "^Bg9^0rr910000000000iXQKl0rr91000f55c8Xg0@@D72lkbi5=-{L"
    result = _native_deletion_vectors(
        actions(
            {
                "storageType": "i",
                "pathOrInlineDv": literal,
                "offset": None,
                "sizeInBytes": 44,
                "cardinality": 6,
            }
        ),
        ["/table/a b.parquet"],
        "/table/",
    )
    assert result == (
        "delta-deletion-vector-native",
        {0: (_decode_z85(literal), 0, 44, 6)},
    )


@pytest.mark.parametrize("inline", [False, True])
def test_native_scan_deletions(
    tmp_path: Path, inline: bool, plugin_decoder: None
) -> None:
    path = tmp_path / "data.parquet"
    df = pl.DataFrame({"a": range(10)})
    df.write_parquet(path)
    body = bitmap([1, 3, 8])
    if inline:
        location: str | bytes = body
        offset = 0
    else:
        dv_path = tmp_path / "dv.bin"
        dv_path.write_bytes(b"\x01" + frame([1, 3, 8]))
        location, offset = str(dv_path), 1
    lf = pl.scan_parquet(
        path,
        _deletion_files=_native({0: (location, offset, len(body), 3)}),
    )
    expected = df.filter(~pl.col("a").is_in([1, 3, 8]))
    assert_frame_equal(lf.collect(), expected)
    assert_frame_equal(lf.slice(2, 3).collect(), expected.slice(2, 3))
    assert_frame_equal(lf.tail(3).collect(), expected.tail(3))
    assert lf.select(pl.len()).collect().item() == 7


def test_native_lazy_reads_and_source_remapping(
    tmp_path: Path, plugin_decoder: None
) -> None:
    paths = [tmp_path / f"part{i}.parquet" for i in range(2)]
    for i, path in enumerate(paths):
        pl.DataFrame({"a": [i * 10, i * 10 + 1]}).write_parquet(path)
    body = bitmap([0])
    lf = pl.scan_parquet(
        paths,
        # Table statistics prune before reader/DV initialization; Parquet
        # row-group statistics are evaluated after the DV has been loaded.
        _table_statistics=pl.DataFrame(
            {
                "len": pl.Series([2, 2], dtype=pl.UInt32),
                "a_min": [0, 10],
                "a_max": [1, 11],
                "a_nc": pl.Series([0, 0], dtype=pl.UInt32),
            }
        ),
        _deletion_files=_native(
            {
                0: (str(tmp_path / "missing.bin"), 1, len(body), 1),
                1: (body, 0, len(body), 1),
            }
        ),
    )
    assert lf.filter(pl.col("a") >= 10).collect()["a"].to_list() == [11]
    assert lf.filter(pl.col("a") >= 10).slice(0, 1).collect()["a"].to_list() == [11]
    assert lf.filter(pl.col("a") >= 10).tail(1).collect()["a"].to_list() == [11]


@pytest.mark.parametrize(
    ("data_path", "dv_path", "expected"),
    [
        ("file:///table/a%20b.parquet", "file:///table/dv%20x.bin", "/table/dv x.bin"),
        (
            "file:///table/a%20b.parquet",
            "file:///table/dv%2520x.bin",
            "/table/dv%20x.bin",
        ),
        (
            "file:///table/a%20b.parquet",
            "file:///elsewhere/dv%20x.bin",
            "/elsewhere/dv x.bin",
        ),
    ],
)
def test_native_absolute_local_uri_paths(
    data_path: str, dv_path: str, expected: str
) -> None:
    add_actions = actions(
        {
            "storageType": "p",
            "pathOrInlineDv": dv_path,
            "offset": 1,
            "sizeInBytes": 44,
            "cardinality": 6,
        }
    )
    add_actions = add_actions.with_columns(pl.lit(data_path).alias("path"))
    assert _native_deletion_vectors(add_actions, ["/table/a b.parquet"], "/table/") == (
        "delta-deletion-vector-native",
        {0: (expected, 1, 44, 6)},
    )


@pytest.mark.parametrize("root", ["/table/", "s3://bucket/table/"])
def test_native_relative_uuid_prefix_decodes_once(root: str) -> None:
    add_actions = actions(
        {
            "storageType": "u",
            "pathOrInlineDv": "a%2520b^-aqEH.-t@S}K{vb[*k^",
            "offset": 1,
            "sizeInBytes": 44,
            "cardinality": 6,
        }
    )
    data_path = root + ("a%20b.parquet" if root.startswith("s3:") else "a b.parquet")
    assert _native_deletion_vectors(add_actions, [data_path], root) == (
        "delta-deletion-vector-native",
        {
            0: (
                root + "a%20b/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin",
                1,
                44,
                6,
            )
        },
    )


@pytest.mark.parametrize(
    ("storage", "location", "key"),
    [
        (
            "u",
            "space%20prefix^-aqEH.-t@S}K{vb[*k^",
            "space prefix/deletion_vector_d2c639aa-8816-431a-aaf6-d3fe2512ff61.bin",
        ),
        ("p", "s3://bucket/table%20root/dv%20x%2520y.bin", "dv x%20y.bin"),
    ],
)
def test_native_descriptor_uses_raw_cloud_key(
    storage: str, location: str, key: str
) -> None:
    root = "s3://bucket/table%20root/"
    add_actions = actions(
        {
            "storageType": storage,
            "pathOrInlineDv": location,
            "offset": 1,
            "sizeInBytes": 44,
            "cardinality": 6,
        }
    )
    assert _native_deletion_vectors(add_actions, [root + "a%20b.parquet"], root) == (
        "delta-deletion-vector-native",
        {0: ("s3://bucket/table root/" + key, 1, 44, 6)},
    )


def _scan_dv(
    tmp_path: Path, dv_file: bytes, offset: int, size: int, n: int = 3
) -> pl.LazyFrame:
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": range(10)}).write_parquet(path)
    dv_path = tmp_path / "dv.bin"
    dv_path.write_bytes(dv_file)
    return pl.scan_parquet(
        path,
        _deletion_files=_native({0: (str(dv_path), offset, size, n)}),
    )


def test_native_file_vector_errors(tmp_path: Path, plugin_decoder: None) -> None:
    body = bitmap([1, 3, 8])
    good = frame([1, 3, 8])
    size = len(body)

    # Shared file: a second vector after the first, read at its own offset.
    second = frame([0, 2])
    lf = _scan_dv(
        tmp_path, b"\x01" + good + second, 1 + len(good), len(bitmap([0, 2])), 2
    )
    assert lf.collect()["a"].to_list() == [1, 3, 4, 5, 6, 7, 8, 9]

    # Short read: the frame extends past the end of the file (never a panic).
    with pytest.raises(ComputeError, match="shorter than the vector"):
        _scan_dv(tmp_path, b"\x01" + good[:-3], 1, size).collect()
    with pytest.raises(ComputeError, match="shorter than the vector"):
        _scan_dv(tmp_path, b"\x01", 1, size).collect()
    with pytest.raises(ComputeError, match="shorter than the vector"):
        _scan_dv(tmp_path, b"\x01" + good, 10_000, size).collect()
    # CRC mismatch.
    bad_crc = bytearray(good)
    bad_crc[-1] ^= 1
    with pytest.raises(ComputeError, match="CRC"):
        _scan_dv(tmp_path, b"\x01" + bytes(bad_crc), 1, size).collect()
    # Size prefix disagrees with the descriptor.
    with pytest.raises(ComputeError, match="size mismatch"):
        _scan_dv(tmp_path, b"\x01" + good, 1, size - 1).collect()
    bad_prefix = struct.pack(">I", size + 1) + good[4:]
    with pytest.raises(ComputeError, match="size mismatch"):
        _scan_dv(tmp_path, b"\x01" + bad_prefix, 1, size).collect()
    # Unsupported version byte.
    with pytest.raises(ComputeError, match="version"):
        _scan_dv(tmp_path, b"\x02" + good, 1, size).collect()


def test_native_vector_without_decoder_names_package(tmp_path: Path) -> None:
    import subprocess
    import sys

    body = bitmap([1])
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": range(3)}).write_parquet(path)
    code = (
        "import sys, polars as pl\n"
        "sys.modules['polars_delta'] = None\n"
        f"lf = pl.scan_parquet({str(path)!r}, _deletion_files=("
        f"'delta-deletion-vector-native', {{0: ({body!r}, 0, {len(body)}, 1)}}))\n"
        "try:\n"
        "    lf.collect()\n"
        "except Exception as e:\n"
        "    print(type(e).__name__, e)\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.startswith("ComputeError")
    assert "polars-delta" in out


def test_native_bad_vector_does_not_fail_its_file_mates(
    tmp_path: Path, plugin_decoder: None
) -> None:
    # Two vectors share a file, and the second one reaches past its end. Only the
    # scan of the second one may fail.
    good = frame([1, 3, 8])
    size = len(bitmap([1, 3, 8]))
    paths = [tmp_path / f"part{i}.parquet" for i in range(2)]
    for i, path in enumerate(paths):
        pl.DataFrame({"a": range(i * 10, i * 10 + 10)}).write_parquet(path)
    dv_path = tmp_path / "dv.bin"
    dv_path.write_bytes(b"\x01" + good)
    lf = pl.scan_parquet(
        paths,
        _deletion_files=_native(
            {
                0: (str(dv_path), 1, size, 3),
                1: (str(dv_path), 1 + len(good), size, 3),
            }
        ),
    )
    assert lf.head(2).collect()["a"].to_list() == [0, 2]
    with pytest.raises((ComputeError, OSError), match=r"shorter than the vector|range"):
        lf.collect()


def test_native_vector_in_fresh_process_loads_plugin(tmp_path: Path) -> None:
    import subprocess
    import sys

    pytest.importorskip("polars_delta")
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": range(5)}).write_parquet(path)
    body = bitmap([1, 3])
    code = (
        "import polars as pl\n"
        f"lf = pl.scan_parquet({str(path)!r}, _deletion_files=("
        f"'delta-deletion-vector-native', {{0: ({body!r}, 0, {len(body)}, 2)}}))\n"
        "print(lf.collect()['a'].to_list())\n"
    )
    out = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=True
    ).stdout
    assert out.strip() == "[0, 2, 4]"


def _fresh_process(code: str, *, timeout: float = 60) -> str:
    import subprocess
    import sys

    return subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=True,
        timeout=timeout,
    ).stdout.strip()


def test_native_vectors_starting_together_load_the_plugin_once(tmp_path: Path) -> None:
    pytest.importorskip("polars_delta")
    paths = []
    for i in range(16):
        paths.append(tmp_path / f"p{i}.parquet")
        pl.DataFrame({"a": range(5)}).write_parquet(paths[-1])
    # One vector per file, all different, so each file's scan loads on its own.
    vectors = {i: (bitmap([i % 5]), 0, len(bitmap([i % 5])), 1) for i in range(16)}
    code = (
        "import time\n"
        "import polars as pl\n"
        "import polars.io.delta._plugin as plugin\n"
        "calls = []\n"
        "load = plugin.dv_decoder_load_error\n"
        "def slow():\n"
        "    calls.append(1)\n"
        "    time.sleep(0.5)\n"
        "    return load()\n"
        "plugin.dv_decoder_load_error = slow\n"
        f"lf = pl.scan_parquet({[str(p) for p in paths]!r}, _deletion_files=("
        f"'delta-deletion-vector-native', {vectors!r}))\n"
        "print(lf.collect().height, len(calls))\n"
    )
    assert _fresh_process(code) == "64 1"


def test_native_vector_scan_while_interpreter_exits(tmp_path: Path) -> None:
    # A scan collected from daemon threads while the interpreter shuts down reaches the
    # plugin loader from a Rust thread: it must neither hang nor crash the process. The
    # window is narrow (about 2% of runs with eight threads before the loader waited for
    # the `atexit` hook), so this is a probabilistic guard.
    pytest.importorskip("polars_delta")
    path = tmp_path / "data.parquet"
    pl.DataFrame({"a": range(5)}).write_parquet(path)
    body = bitmap([1, 3])
    code = (
        "import threading\n"
        "import polars as pl\n"
        f"lf = pl.scan_parquet({str(path)!r}, _deletion_files=("
        f"'delta-deletion-vector-native', {{0: ({body!r}, 0, {len(body)}, 2)}}))\n"
        "def run():\n"
        "    while True:\n"
        "        try:\n"
        "            lf.collect()\n"
        "        except pl.exceptions.PolarsError:\n"
        "            pass\n"
        "for _ in range(8):\n"
        "    threading.Thread(target=run, daemon=True).start()\n"
    )
    for _ in range(30):
        _fresh_process(code, timeout=30)
