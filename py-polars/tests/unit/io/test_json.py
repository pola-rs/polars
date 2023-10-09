from __future__ import annotations

import io
import json
from typing import TYPE_CHECKING

import pytest

import polars as pl
from polars.testing import assert_frame_equal

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize("buf", [io.BytesIO(), io.StringIO()])
def test_to_from_buffer(df: pl.DataFrame, buf: io.IOBase) -> None:
    df.write_json(buf)
    buf.seek(0)
    read_df = pl.read_json(buf)
    assert_frame_equal(df, read_df, categorical_as_str=True)


@pytest.mark.write_disk()
def test_to_from_file(df: pl.DataFrame, tmp_path: Path) -> None:
    tmp_path.mkdir(exist_ok=True)

    file_path = tmp_path / "small.json"
    df.write_json(file_path)
    out = pl.read_json(file_path)

    assert_frame_equal(df, out, categorical_as_str=True)


def test_to_from_buffer_arraywise_schema() -> None:
    buf = io.StringIO(
        """
    [
        {"a": 5, "b": "foo", "c": null},
        {"a": 11.4, "b": null, "c": true, "d": 8},
        {"a": -25.8, "b": "bar", "c": false}
    ]"""
    )

    read_df = pl.read_json(buf, schema={"b": pl.Utf8, "e": pl.Int16})

    assert_frame_equal(
        read_df,
        pl.DataFrame(
            {
                "b": pl.Series(["foo", None, "bar"], dtype=pl.Utf8),
                "e": pl.Series([None, None, None], dtype=pl.Int16),
            }
        ),
    )


def test_to_from_buffer_arraywise_schema_override() -> None:
    buf = io.StringIO(
        """
    [
        {"a": 5, "b": "foo", "c": null},
        {"a": 11.4, "b": null, "c": true, "d": 8},
        {"a": -25.8, "b": "bar", "c": false}
    ]"""
    )

    read_df = pl.read_json(buf, schema_overrides={"c": pl.Int64, "d": pl.Float64})

    assert_frame_equal(
        read_df,
        pl.DataFrame(
            {
                "a": pl.Series([5, 11.4, -25.8], dtype=pl.Float64),
                "b": pl.Series(["foo", None, "bar"], dtype=pl.Utf8),
                "c": pl.Series([None, 1, 0], dtype=pl.Int64),
                "d": pl.Series([None, 8, None], dtype=pl.Float64),
            }
        ),
        check_column_order=False,
    )


def test_write_json_to_string() -> None:
    # Tests if it runs if no arg given
    df = pl.DataFrame({"a": [1, 2, 3]})
    expected_str = '{"columns":[{"name":"a","datatype":"Int64","bit_settings":"","values":[1,2,3]}]}'
    assert df.write_json() == expected_str


def test_write_json(df: pl.DataFrame) -> None:
    # Text-based conversion loses time info
    df = df.select(pl.all().exclude(["cat", "time"]))
    s = df.write_json()
    f = io.BytesIO()
    f.write(s.encode())
    f.seek(0)
    out = pl.read_json(f)
    assert_frame_equal(out, df)

    file = io.BytesIO()
    df.write_json(file)
    file.seek(0)
    out = pl.read_json(file)
    assert_frame_equal(out, df)


def test_write_json_row_oriented() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    out = df.write_json(row_oriented=True)
    assert out == '[{"a":1,"b":"a"},{"a":2,"b":"b"},{"a":3,"b":null}]'

    # Test round trip
    f = io.BytesIO()
    f.write(out.encode())
    f.seek(0)
    result = pl.read_json(f)
    assert_frame_equal(result, df)


def test_write_ndjson() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    out = df.write_ndjson()
    assert out == '{"a":1,"b":"a"}\n{"a":2,"b":"b"}\n{"a":3,"b":null}\n'

    # Test round trip
    f = io.BytesIO()
    f.write(out.encode())
    f.seek(0)
    result = pl.read_ndjson(f)
    assert_frame_equal(result, df)


def test_write_ndjson_with_trailing_newline() -> None:
    input = """{"Column1":"Value1"}\n"""
    df = pl.read_ndjson(io.StringIO(input))

    expected = pl.DataFrame({"Column1": ["Value1"]})
    assert_frame_equal(df, expected)


def test_read_ndjson_empty_array() -> None:
    assert pl.read_ndjson(io.StringIO("""{"foo": {"bar": []}}""")).to_dict(False) == {
        "foo": [{"": None}]
    }


def test_ndjson_nested_null() -> None:
    json_payload = """{"foo":{"bar":[{}]}}"""
    df = pl.read_ndjson(io.StringIO(json_payload))

    # 'bar' represents an empty list of structs; check the schema is correct (eg: picks
    # up that it IS a list of structs), but confirm that list is empty (ref: #11301)
    assert df.schema == {"foo": pl.Struct([pl.Field("bar", pl.List(pl.Struct([])))])}
    assert df.to_dict(False) == {"foo": [{"bar": []}]}


def test_ndjson_nested_utf8_int() -> None:
    ndjson = """{"Accumulables":[{"Value":32395888},{"Value":"539454"}]}"""
    assert pl.read_ndjson(io.StringIO(ndjson)).to_dict(False) == {
        "Accumulables": [[{"Value": "32395888"}, {"Value": "539454"}]]
    }


def test_write_json_categoricals() -> None:
    data = {"column": ["test1", "test2", "test3", "test4"]}
    df = pl.DataFrame(data).with_columns(pl.col("column").cast(pl.Categorical))

    assert (
        df.write_json(row_oriented=True, file=None)
        == '[{"column":"test1"},{"column":"test2"},{"column":"test3"},{"column":"test4"}]'
    )


def test_json_supertype_infer() -> None:
    json_string = """[
{"c":[{"b": [], "a": "1"}]},
{"c":[{"b":[]}]},
{"c":[{"b":["1"], "a": "1"}]}]
"""
    python_infer = pl.from_records(json.loads(json_string))
    polars_infer = pl.read_json(io.StringIO(json_string))
    assert_frame_equal(python_infer, polars_infer)


def test_json_sliced_list_serialization() -> None:
    data = {"col1": [0, 2], "col2": [[3, 4, 5], [6, 7, 8]]}
    df = pl.DataFrame(data)
    f = io.BytesIO()
    sliced_df = df[1, :]
    sliced_df.write_ndjson(f)
    assert f.getvalue() == b'{"col1":2,"col2":[6,7,8]}\n'


def test_json_deserialize_empty_list_10458() -> None:
    schema = {"LIST_OF_STRINGS": pl.List(pl.Utf8)}
    serialized_schema = pl.DataFrame(schema=schema).write_json()
    df = pl.read_json(io.StringIO(serialized_schema))
    assert df.schema == schema


def test_json_deserialize_9687() -> None:
    response = {
        "volume": [0.0, 0.0, 0.0],
        "open": [1263.0, 1263.0, 1263.0],
        "close": [1263.0, 1263.0, 1263.0],
        "high": [1263.0, 1263.0, 1263.0],
        "low": [1263.0, 1263.0, 1263.0],
    }

    result = pl.read_json(json.dumps(response).encode())

    assert result.to_dict(False) == {k: [v] for k, v in response.items()}


def test_ndjson_ignore_errors() -> None:
    # this schema is inconsistent as "value" is string and object
    jsonl = r"""{"Type":"insert","Key":[1],"SeqNo":1,"Timestamp":1,"Fields":[{"Name":"added_id","Value":2},{"Name":"body","Value":{"a": 1}}]}
    {"Type":"insert","Key":[1],"SeqNo":1,"Timestamp":1,"Fields":[{"Name":"added_id","Value":2},{"Name":"body","Value":{"a": 1}}]}"""

    buf = io.BytesIO(jsonl.encode())

    # check if we can replace with nulls
    assert pl.read_ndjson(buf, ignore_errors=True).to_dict(False) == {
        "Type": ["insert", "insert"],
        "Key": [[1], [1]],
        "SeqNo": [1, 1],
        "Timestamp": [1, 1],
        "Fields": [
            [{"Name": "added_id", "Value": "2"}, {"Name": "body", "Value": None}],
            [{"Name": "added_id", "Value": "2"}, {"Name": "body", "Value": None}],
        ],
    }

    schema = {
        "Fields": pl.List(
            pl.Struct([pl.Field("Name", pl.Utf8), pl.Field("Value", pl.Int64)])
        )
    }
    # schema argument only parses Fields
    assert pl.read_ndjson(buf, schema=schema, ignore_errors=True).to_dict(False) == {
        "Fields": [
            [{"Name": "added_id", "Value": 2}, {"Name": "body", "Value": None}],
            [{"Name": "added_id", "Value": 2}, {"Name": "body", "Value": None}],
        ]
    }

    # schema_overrides argument does schema inference, but overrides Fields
    assert pl.read_ndjson(buf, schema_overrides=schema, ignore_errors=True).to_dict(
        False
    ) == {
        "Type": ["insert", "insert"],
        "Key": [[1], [1]],
        "SeqNo": [1, 1],
        "Timestamp": [1, 1],
        "Fields": [
            [{"Name": "added_id", "Value": 2}, {"Name": "body", "Value": None}],
            [{"Name": "added_id", "Value": 2}, {"Name": "body", "Value": None}],
        ],
    }


def test_write_json_duration() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series(
                [91762939, 91762890, 6020836], dtype=pl.Duration(time_unit="ms")
            )
        }
    )
    assert (
        df.write_json(row_oriented=True)
        == '[{"a":"P1DT5362.939S"},{"a":"P1DT5362.890S"},{"a":"PT6020.836S"}]'
    )
