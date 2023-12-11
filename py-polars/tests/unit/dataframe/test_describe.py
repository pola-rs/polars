from datetime import date

import polars as pl
from polars.testing import assert_frame_equal


def test_df_describe() -> None:
    df = pl.DataFrame(
        {
            "a": [1.0, 2.8, 3.0],
            "b": [4, 5, None],
            "c": [True, False, True],
            "d": [None, "b", "c"],
            "e": ["usd", "eur", None],
            "f": [date(2020, 1, 1), date(2021, 1, 1), date(2022, 1, 1)],
        },
        schema_overrides={"e": pl.Categorical},
    )

    result = df.describe()

    expected = pl.DataFrame(
        {
            "describe": [
                "count",
                "null_count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ],
            "a": [3.0, 0.0, 2.2666667, 1.101514, 1.0, 1.0, 2.8, 3.0, 3.0],
            "b": [3.0, 1.0, 4.5, 0.7071067811865476, 4.0, 4.0, 5.0, 5.0, 5.0],
            "c": [
                3.0,
                0.0,
                0.6666666666666666,
                0.5773502588272095,
                0.0,
                None,
                None,
                None,
                1.0,
            ],
            "d": ["3", "1", None, None, "b", None, None, None, "c"],
            "e": ["3", "1", None, None, None, None, None, None, None],
            "f": ["3", "0", None, None, "2020-01-01", None, None, None, "2022-01-01"],
        }
    )
    assert_frame_equal(result, expected)


def test_df_describe_struct() -> None:
    df = pl.DataFrame(
        {
            "numeric": [1, 2, 1, None],
            "struct": [{"x": 1, "y": 2}, {"x": 3, "y": 4}, {"x": 1, "y": 2}, None],
            "list": [[1, 2], [3, 4], [1, 2], None],
        }
    )

    result = df.describe()

    assert result.to_dict(as_series=False) == {
        "describe": [
            "count",
            "null_count",
            "mean",
            "std",
            "min",
            "25%",
            "50%",
            "75%",
            "max",
        ],
        "numeric": [
            4.0,
            1.0,
            1.3333333333333333,
            0.5773502691896257,
            1.0,
            1.0,
            1.0,
            2.0,
            2.0,
        ],
        "struct": ["4", "1", None, None, None, None, None, None, None],
        "list": ["4", "1", None, None, None, None, None, None, None],
    }

    for pcts in (None, []):  # type:ignore[var-annotated]
        assert df.describe(percentiles=pcts).rows() == [
            ("count", 4.0, "4", "4"),
            ("null_count", 1.0, "1", "1"),
            ("mean", 1.3333333333333333, None, None),
            ("std", 0.5773502691896257, None, None),
            ("min", 1.0, None, None),
            ("max", 2.0, None, None),
        ]

    result = df.describe(percentiles=(0.2, 0.4, 0.5, 0.6, 0.8))
    expected = pl.DataFrame(
        [
            ("count", 4.0, "4", "4"),
            ("null_count", 1.0, "1", "1"),
            ("mean", 1.3333333333333333, None, None),
            ("std", 0.5773502691896257, None, None),
            ("min", 1.0, None, None),
            ("20%", 1.0, None, None),
            ("40%", 1.0, None, None),
            ("50%", 1.0, None, None),
            ("60%", 1.0, None, None),
            ("80%", 2.0, None, None),
            ("max", 2.0, None, None),
        ],
        schema={
            "describe": pl.Utf8,
            "numeric": pl.Float64,
            "struct": pl.Utf8,
            "list": pl.Utf8,
        },
    )
    assert_frame_equal(result, expected)
