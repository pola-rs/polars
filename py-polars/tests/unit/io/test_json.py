from __future__ import annotations

import io
import json
import typing
from collections import OrderedDict
from decimal import Decimal as D
from io import BytesIO
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import pytest

import polars as pl
from polars.testing import assert_frame_equal


def test_write_json() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": ["a", "b", None]})
    out = df.write_json()
    assert out == '[{"a":1,"b":"a"},{"a":2,"b":"b"},{"a":3,"b":null}]'

    # Test round trip
    f = io.BytesIO()
    f.write(out.encode())
    f.seek(0)
    result = pl.read_json(f)
    assert_frame_equal(result, df)


def test_write_json_categoricals() -> None:
    data = {"column": ["test1", "test2", "test3", "test4"]}
    df = pl.DataFrame(data).with_columns(pl.col("column").cast(pl.Categorical))
    expected = (
        '[{"column":"test1"},{"column":"test2"},{"column":"test3"},{"column":"test4"}]'
    )
    assert df.write_json() == expected


def test_write_json_duration() -> None:
    df = pl.DataFrame(
        {
            "a": pl.Series(
                [91762939, 91762890, 6020836], dtype=pl.Duration(time_unit="ms")
            )
        }
    )

    # we don't guarantee a format, just round-circling
    value = df.write_json()
    expected = '[{"a":"PT91762.939S"},{"a":"PT91762.89S"},{"a":"PT6020.836S"}]'
    assert value == expected


def test_write_json_decimal() -> None:
    df = pl.DataFrame({"a": pl.Series([D("1.00"), D("2.00"), None])})

    # we don't guarantee a format, just round-circling
    value = df.write_json()
    assert value == """[{"a":"1.00"},{"a":"2.00"},{"a":null}]"""


def test_json_infer_schema_length_11148() -> None:
    response = [{"col1": 1}] * 2 + [{"col1": 1, "col2": 2}] * 1
    result = pl.read_json(json.dumps(response).encode(), infer_schema_length=2)
    with pytest.raises(AssertionError):
        assert set(result.columns) == {"col1", "col2"}

    response = [{"col1": 1}] * 2 + [{"col1": 1, "col2": 2}] * 1
    result = pl.read_json(json.dumps(response).encode(), infer_schema_length=3)
    assert set(result.columns) == {"col1", "col2"}


def test_to_from_buffer_arraywise_schema() -> None:
    buf = io.StringIO(
        """
    [
        {"a": 5, "b": "foo", "c": null},
        {"a": 11.4, "b": null, "c": true, "d": 8},
        {"a": -25.8, "b": "bar", "c": false}
    ]"""
    )

    read_df = pl.read_json(buf, schema={"b": pl.String, "e": pl.Int16})

    assert_frame_equal(
        read_df,
        pl.DataFrame(
            {
                "b": pl.Series(["foo", None, "bar"], dtype=pl.String),
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
                "b": pl.Series(["foo", None, "bar"], dtype=pl.String),
                "c": pl.Series([None, 1, 0], dtype=pl.Int64),
                "d": pl.Series([None, 8, None], dtype=pl.Float64),
            }
        ),
        check_column_order=False,
    )


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
    assert pl.read_ndjson(io.StringIO("""{"foo": {"bar": []}}""")).to_dict(
        as_series=False
    ) == {"foo": [{"bar": []}]}


def test_ndjson_nested_null() -> None:
    json_payload = """{"foo":{"bar":[{}]}}"""
    df = pl.read_ndjson(io.StringIO(json_payload))

    # 'bar' represents an empty list of structs; check the schema is correct (eg: picks
    # up that it IS a list of structs), but confirm that list is empty (ref: #11301)
    # We don't support empty structs yet. So Null is closest.
    assert df.schema == {
        "foo": pl.Struct([pl.Field("bar", pl.List(pl.Struct({"": pl.Null})))])
    }
    assert df.to_dict(as_series=False) == {"foo": [{"bar": []}]}


def test_ndjson_nested_string_int() -> None:
    ndjson = """{"Accumulables":[{"Value":32395888},{"Value":"539454"}]}"""
    assert pl.read_ndjson(io.StringIO(ndjson)).to_dict(as_series=False) == {
        "Accumulables": [[{"Value": "32395888"}, {"Value": "539454"}]]
    }


def test_json_supertype_infer() -> None:
    json_string = """[
{"c":[{"b": [], "a": "1"}]},
{"c":[{"b":[]}]},
{"c":[{"b":["1"], "a": "1"}]}]
"""
    python_infer = pl.from_records(json.loads(json_string))
    polars_infer = pl.read_json(io.StringIO(json_string))
    assert_frame_equal(python_infer, polars_infer)


def test_ndjson_sliced_list_serialization() -> None:
    data = {"col1": [0, 2], "col2": [[3, 4, 5], [6, 7, 8]]}
    df = pl.DataFrame(data)
    f = io.BytesIO()
    sliced_df = df[1, :]
    sliced_df.write_ndjson(f)
    assert f.getvalue() == b'{"col1":2,"col2":[6,7,8]}\n'


def test_json_deserialize_9687() -> None:
    response = {
        "volume": [0.0, 0.0, 0.0],
        "open": [1263.0, 1263.0, 1263.0],
        "close": [1263.0, 1263.0, 1263.0],
        "high": [1263.0, 1263.0, 1263.0],
        "low": [1263.0, 1263.0, 1263.0],
    }

    result = pl.read_json(json.dumps(response).encode())

    assert result.to_dict(as_series=False) == {k: [v] for k, v in response.items()}


def test_ndjson_ignore_errors() -> None:
    # this schema is inconsistent as "value" is string and object
    jsonl = r"""{"Type":"insert","Key":[1],"SeqNo":1,"Timestamp":1,"Fields":[{"Name":"added_id","Value":2},{"Name":"body","Value":{"a": 1}}]}
    {"Type":"insert","Key":[1],"SeqNo":1,"Timestamp":1,"Fields":[{"Name":"added_id","Value":2},{"Name":"body","Value":{"a": 1}}]}"""

    buf = io.BytesIO(jsonl.encode())

    # check if we can replace with nulls
    assert pl.read_ndjson(buf, ignore_errors=True).to_dict(as_series=False) == {
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
            pl.Struct([pl.Field("Name", pl.String), pl.Field("Value", pl.Int64)])
        )
    }
    # schema argument only parses Fields
    assert pl.read_ndjson(buf, schema=schema, ignore_errors=True).to_dict(
        as_series=False
    ) == {
        "Fields": [
            [{"Name": "added_id", "Value": 2}, {"Name": "body", "Value": None}],
            [{"Name": "added_id", "Value": 2}, {"Name": "body", "Value": None}],
        ]
    }

    # schema_overrides argument does schema inference, but overrides Fields
    result = pl.read_ndjson(buf, schema_overrides=schema, ignore_errors=True)
    expected = {
        "Type": ["insert", "insert"],
        "Key": [[1], [1]],
        "SeqNo": [1, 1],
        "Timestamp": [1, 1],
        "Fields": [
            [{"Name": "added_id", "Value": 2}, {"Name": "body", "Value": None}],
            [{"Name": "added_id", "Value": 2}, {"Name": "body", "Value": None}],
        ],
    }
    assert result.to_dict(as_series=False) == expected


def test_json_null_infer() -> None:
    json = BytesIO(
        bytes(
            """
    [
      {
        "a": 1,
        "b": null
      }
    ]
    """,
            "UTF-8",
        )
    )

    assert pl.read_json(json).schema == OrderedDict({"a": pl.Int64, "b": pl.Null})


def test_ndjson_null_buffer() -> None:
    data = io.BytesIO(
        b"""\
    {"id": 1, "zero_column": 0, "empty_array_column": [], "empty_object_column": {}, "null_column": null}
    {"id": 2, "zero_column": 0, "empty_array_column": [], "empty_object_column": {}, "null_column": null}
    {"id": 3, "zero_column": 0, "empty_array_column": [], "empty_object_column": {}, "null_column": null}
    {"id": 4, "zero_column": 0, "empty_array_column": [], "empty_object_column": {}, "null_column": null}
    """
    )

    assert pl.read_ndjson(data).schema == OrderedDict(
        [
            ("id", pl.Int64),
            ("zero_column", pl.Int64),
            ("empty_array_column", pl.List(pl.Null)),
            ("empty_object_column", pl.Struct([pl.Field("", pl.Null)])),
            ("null_column", pl.Null),
        ]
    )


def test_ndjson_null_inference_13183() -> None:
    assert pl.read_ndjson(
        b"""
    {"map": "a", "start_time": 0.795, "end_time": 1.495}
    {"map": "a", "start_time": 1.6239999999999999, "end_time": 2.0540000000000003}
    {"map": "c", "start_time": 2.184, "end_time": 2.645}
    {"map": "a", "start_time": null, "end_time": null}
    """.strip()
    ).to_dict(as_series=False) == {
        "map": ["a", "a", "c", "a"],
        "start_time": [0.795, 1.6239999999999999, 2.184, None],
        "end_time": [1.495, 2.0540000000000003, 2.645, None],
    }


@pytest.mark.write_disk()
@typing.no_type_check
def test_json_wrong_input_handle_textio(tmp_path: Path) -> None:
    # this shouldn't be passed, but still we test if we can handle it gracefully
    df = pl.DataFrame(
        {
            "x": [1, 2, 3],
            "y": ["a", "b", "c"],
        }
    )
    file_path = tmp_path / "test.ndjson"
    df.write_ndjson(file_path)
    with open(file_path) as f:  # noqa: PTH123
        assert_frame_equal(pl.read_ndjson(f), df)


def test_json_normalize() -> None:
    data = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mark", "family": "Regner"}},
        {"id": 2, "name": "Faye Raker"},
    ]

    assert pl.json_normalize(data, max_level=0).to_dict(as_series=False) == {
        "id": [1, None, 2],
        "name": [
            '{"first": "Coleen", "last": "Volk"}',
            '{"given": "Mark", "family": "Regner"}',
            "Faye Raker",
        ],
    }

    assert pl.json_normalize(data, max_level=1).to_dict(as_series=False) == {
        "id": [1, None, 2],
        "name.first": ["Coleen", None, None],
        "name.last": ["Volk", None, None],
        "name.given": [None, "Mark", None],
        "name.family": [None, "Regner", None],
        "name": [None, None, "Faye Raker"],
    }

    data = [
        {
            "id": 1,
            "name": "Cole Volk",
            "fitness": {"height": 130, "weight": 60},
        },
        {"name": "Mark Reg", "fitness": {"height": 130, "weight": 60}},
        {
            "id": 2,
            "name": "Faye Raker",
            "fitness": {"height": 130, "weight": 60},
        },
    ]
    assert pl.json_normalize(data, max_level=0).to_dict(as_series=False) == {
        "id": [1, None, 2],
        "name": ["Cole Volk", "Mark Reg", "Faye Raker"],
        "fitness": [
            '{"height": 130, "weight": 60}',
            '{"height": 130, "weight": 60}',
            '{"height": 130, "weight": 60}',
        ],
    }
    assert pl.json_normalize(data, max_level=1).to_dict(as_series=False) == {
        "id": [1, None, 2],
        "name": ["Cole Volk", "Mark Reg", "Faye Raker"],
        "fitness.height": [130, 130, 130],
        "fitness.weight": [60, 60, 60],
    }


def test_empty_json() -> None:
    df = pl.read_json(io.StringIO("{}"))
    assert df.shape == (0, 0)
    assert isinstance(df, pl.DataFrame)

    df = pl.read_json(b'{"j":{}}')
    assert df.dtypes == [pl.Struct([])]
    assert df.shape == (0, 1)
