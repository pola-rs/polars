import datetime
import math

import polars as pl


def test_list_json_encode() -> None:
    df = pl.DataFrame(
        {
            "a": [[1, None, 3], [4, 5, 6], None],
            "b": [
                [
                    {
                        "foo": 1,
                    },
                    {"foo": 2},
                    {
                        "foo": 3,
                    },
                ],
                [
                    {
                        "foo": 3,
                    },
                    {"foo": 4},
                    {
                        "foo": 5,
                    },
                ],
                [
                    {
                        "foo": 6,
                    },
                    {"foo": 7},
                    {
                        "foo": 8,
                    },
                ],
            ],
            "c": [[True, False, True], [False, True, False], [True, True, False]],
            "d": [[1.6, 2.7, 3.8], [math.inf, -math.inf, 6.1], [7.2, 8.3, math.nan]],
        }
    )
    df = df.with_columns(
        pl.col("a").json_encode(),
        pl.col("b").json_encode(),
        pl.col("c").json_encode(),
        pl.col("d").json_encode(),
    )
    assert df.schema == {
        "a": pl.String,
        "b": pl.String,
        "c": pl.String,
        "d": pl.String,
    }
    assert df.to_dict(as_series=False) == {
        "a": ["[1,null,3]", "[4,5,6]", "null"],
        "b": [
            """[{"foo":1},{"foo":2},{"foo":3}]""",
            """[{"foo":3},{"foo":4},{"foo":5}]""",
            """[{"foo":6},{"foo":7},{"foo":8}]""",
        ],
        "c": ["[true,false,true]", "[false,true,false]", "[true,true,false]"],
        "d": ["[1.6,2.7,3.8]", "[null,null,6.1]", "[7.2,8.3,null]"],
    }


def test_struct_json_encode() -> None:
    assert pl.DataFrame(
        {"a": [{"a": [1, 2], "b": [45]}, {"a": [9, 1, 3], "b": None}]}
    ).with_columns(pl.col("a").json_encode().alias("encoded")).to_dict(
        as_series=False
    ) == {
        "a": [{"a": [1, 2], "b": [45]}, {"a": [9, 1, 3], "b": None}],
        "encoded": ['{"a":[1,2],"b":[45]}', '{"a":[9,1,3],"b":null}'],
    }


def test_struct_json_encode_logical_type() -> None:
    df = pl.DataFrame(
        {
            "a": [
                {
                    "a": [datetime.date(1997, 1, 1)],
                    "b": [datetime.datetime(2000, 1, 29, 10, 30)],
                    "c": [datetime.timedelta(1, 25)],
                }
            ]
        }
    ).select(pl.col("a").json_encode().alias("encoded"))
    assert df.to_dict(as_series=False) == {
        "encoded": ['{"a":["1997-01-01"],"b":["2000-01-29 10:30:00"],"c":["PT86425S"]}']
    }


def test_json_encode_in_struct_name_space() -> None:
    df = pl.DataFrame(
        {
            "a": [
                {
                    "a": 1,
                    "b": 2,
                },
                {
                    "a": 3,
                    "b": 4,
                },
            ]
        }
    ).with_columns(pl.col("a").json_encode().alias("encoded"))
    assert df.to_dict(as_series=False) == {
        "encoded": ['{"a":1,"b":2}', '{"a":3,"b":4}'],
    }
