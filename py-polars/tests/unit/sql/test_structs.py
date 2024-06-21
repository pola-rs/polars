from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import StructFieldNotFoundError
from polars.testing import assert_frame_equal


@pytest.fixture()
def df_struct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [100, 200, 300, 400],
            "name": ["Alice", "Bob", "David", "Zoe"],
            "age": [32, 27, 19, 45],
            "other": [{"n": 1.5}, {"n": None}, {"n": -0.5}, {"n": 2.0}],
        }
    ).select(pl.struct(pl.all()).alias("json_msg"))


def test_struct_field_selection(df_struct: pl.DataFrame) -> None:
    res = df_struct.sql(
        """
        SELECT
          -- validate table alias resolution
          frame.json_msg.id AS ID,
          self.json_msg.name AS NAME,
          json_msg.age AS AGE
        FROM
          self AS frame
        WHERE
          json_msg.age > 20 AND
          json_msg.other.n IS NOT NULL  -- note: nested struct field
        ORDER BY
          json_msg.name DESC
        """
    )
    expected = pl.DataFrame(
        {"ID": [400, 100], "NAME": ["Zoe", "Alice"], "AGE": [45, 32]}
    )
    assert_frame_equal(expected, res)


@pytest.mark.parametrize(
    ("fields", "excluding"),
    [
        ("json_msg.*", ""),
        ("self.json_msg.*", ""),
        ("json_msg.other.*", ""),
        ("self.json_msg.other.*", ""),
    ],
)
def test_struct_field_wildcard_selection(
    fields: str,
    excluding: str,
    df_struct: pl.DataFrame,
) -> None:
    query = f"SELECT {fields} {excluding} FROM df_struct ORDER BY json_msg.id"
    print(query)
    res = pl.sql(query).collect()

    expected = df_struct.unnest("json_msg")
    if fields.endswith(".other.*"):
        expected = expected["other"].struct.unnest()
    if excluding:
        expected = expected.drop(excluding.split(","))

    assert_frame_equal(expected, res)


@pytest.mark.parametrize(
    "invalid_column",
    [
        "json_msg.invalid_column",
        "json_msg.other.invalid_column",
        "self.json_msg.other.invalid_column",
    ],
)
def test_struct_indexing_errors(invalid_column: str, df_struct: pl.DataFrame) -> None:
    with pytest.raises(StructFieldNotFoundError, match="invalid_column"):
        df_struct.sql(f"SELECT {invalid_column} FROM self")
