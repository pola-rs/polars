from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.exceptions import ComputeError, DuplicateError
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("as_path", [False, True])
def test_read_json_glob(tmp_path: Path, as_path: bool) -> None:
    second = tmp_path / "b.json"
    first = tmp_path / "a.json"
    second.write_text('[{"value": 2.5}]')
    first.write_text('[{"value": 1}, {"value": 2}]')
    pattern = tmp_path / "*.json"

    result = pl.read_json(pattern if as_path else str(pattern))
    expected = pl.DataFrame({"value": [1.0, 2.0, 2.5]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("path_column", ["source", ""])
def test_read_json_glob_include_file_paths(tmp_path: Path, path_column: str) -> None:
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text('[{"value": 1}, {"value": 2}]')
    second.write_text('[{"value": 3}]')

    result = pl.read_json(tmp_path / "*.json", include_file_paths=path_column)
    expected = pl.DataFrame(
        {
            "value": [1, 2, 3],
            path_column: [str(first), str(first), str(second)],
        }
    )
    assert_frame_equal(result, expected)


def test_read_json_glob_schema(tmp_path: Path) -> None:
    for i in range(2):
        (tmp_path / f"{i}.json").write_text(
            json.dumps([{"a": i, "b": None if i == 0 else i + 1}])
        )

    result = pl.read_json(
        tmp_path / "*.json",
        schema={"a": pl.Int8, "b": pl.Int16},
        schema_overrides={"b": pl.Float64},
    )
    expected = pl.DataFrame(
        {"a": [0, 1], "b": [None, 2.0]}, schema={"a": pl.Int8, "b": pl.Float64}
    )
    assert_frame_equal(result, expected)


def test_read_json_glob_sparse_and_empty_files(tmp_path: Path) -> None:
    records = [[], [{"a": 1, "b": True}], [{"b": False, "a": 2.5}], [{"c": "x"}], []]
    paths = [tmp_path / f"{i}.json" for i in range(len(records))]
    for path, data in zip(paths, records, strict=True):
        path.write_text(json.dumps(data))

    path_column = "^source.*$"
    result = pl.read_json(tmp_path / "*.json", include_file_paths=path_column)
    expected = pl.DataFrame(
        {
            "a": [1.0, 2.5, None],
            "b": [True, False, None],
            "c": [None, None, "x"],
            path_column: [str(path) for path in paths[1:4]],
        }
    )
    assert_frame_equal(result, expected)


def test_read_json_glob_infer_schema_length(tmp_path: Path) -> None:
    for i in range(2):
        (tmp_path / f"{i}.json").write_text(
            json.dumps([{"a": i}, {"a": i, "b": i + 1}])
        )

    with pytest.raises(ComputeError, match="extra field in struct data: b"):
        pl.read_json(tmp_path / "*.json", infer_schema_length=1)

    result = pl.read_json(tmp_path / "*.json", infer_schema_length=None)
    expected = pl.DataFrame({"a": [0, 0, 1, 1], "b": [None, 1, None, 2]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("as_path", [False, True])
def test_read_json_glob_literal_brackets(tmp_path: Path, as_path: bool) -> None:
    path = tmp_path / "data[1].json"
    path.write_text('[{"value": 1}]')
    (tmp_path / "data1.json").write_text('[{"value": 2}]')

    result = pl.read_json(path if as_path else str(path), include_file_paths="source")
    expected = pl.DataFrame({"value": [1], "source": [str(path)]})
    assert_frame_equal(result, expected)


def test_read_json_glob_no_matches(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pl.read_json(tmp_path / "*.json")


def test_read_json_glob_invalid_file(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text('[{"value": 1}]')
    (tmp_path / "b.json").write_text("invalid json")

    with pytest.raises(ComputeError):
        pl.read_json(tmp_path / "*.json")


@pytest.mark.parametrize("source_kind", ["bytes", "string_io", "bytes_io"])
def test_read_json_include_file_paths_in_memory(source_kind: str) -> None:
    payload = '[{"value": 1}, {"value": 2}]'
    source: bytes | io.StringIO | io.BytesIO
    if source_kind == "bytes":
        source = payload.encode()
    elif source_kind == "string_io":
        source = io.StringIO(payload)
    else:
        source = io.BytesIO(payload.encode())

    result = pl.read_json(source, include_file_paths="source")
    expected = pl.DataFrame({"value": [1, 2], "source": ["in-mem", "in-mem"]})
    assert_frame_equal(result, expected)


@pytest.mark.parametrize("path_column", ["source", ""])
def test_read_json_include_file_paths_collision(path_column: str) -> None:
    with pytest.raises(DuplicateError):
        pl.read_json(
            json.dumps([{path_column: "original"}]).encode(),
            include_file_paths=path_column,
        )


@pytest.mark.parametrize(("payload", "height"), [(b"[]", 0), (b"[{}, {}]", 2)])
def test_read_json_include_file_paths_empty(payload: bytes, height: int) -> None:
    result = pl.read_json(payload, include_file_paths="source")
    expected = pl.DataFrame(
        {"source": ["in-mem"] * height}, schema={"source": pl.String}
    )
    assert_frame_equal(result, expected)
