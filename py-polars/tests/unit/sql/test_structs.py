from __future__ import annotations

import pytest

import polars as pl
from polars.exceptions import SQLSyntaxError, StructFieldNotFoundError
from polars.testing import assert_frame_equal


@pytest.fixture()
def df_struct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [200, 300, 400],
            "name": ["Bob", "David", "Zoe"],
            "age": [45, 19, 45],
            "other": [{"n": 1.5}, {"n": None}, {"n": -0.5}],
        }
    ).select(pl.struct(pl.all()).alias("json_msg"))


@pytest.mark.parametrize(
    "order_by",
    [
        "ORDER BY json_msg.id DESC",
        "ORDER BY 2 DESC",
        "",
    ],
)
def test_struct_field_selection(order_by: str, df_struct: pl.DataFrame) -> None:
    res = df_struct.sql(
        f"""
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
        {order_by}
        """
    )
    if not order_by:
        res = res.sort(by="ID", descending=True)

    expected = pl.DataFrame({"ID": [400, 200], "NAME": ["Zoe", "Bob"], "AGE": [45, 45]})
    assert_frame_equal(expected, res)


def test_struct_field_group_by(df_struct: pl.DataFrame) -> None:
    res = pl.sql(
        """
        SELECT
          COUNT(json_msg.age) AS n,
          ARRAY_AGG(json_msg.name) AS names
        FROM df_struct
        GROUP BY json_msg.age
        ORDER BY 1 DESC
        """
    ).collect()

    expected = pl.DataFrame(
        data={"n": [2, 1], "names": [["Bob", "Zoe"], ["David"]]},
        schema_overrides={"n": pl.UInt32},
    )
    assert_frame_equal(expected, res)


def test_struct_field_group_by_errors(df_struct: pl.DataFrame) -> None:
    with pytest.raises(
        SQLSyntaxError,
        match="'name' should participate in the GROUP BY clause or an aggregate function",
    ):
        pl.sql(
            """
            SELECT
              json_msg.name,
              SUM(json_msg.age) AS sum_age
            FROM df_struct
            GROUP BY json_msg.age
            """
        ).collect()


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("nested #> '{c,1}'", 2),
        ("nested #> '{c,-1}'", 1),
        ("nested #>> '{c,0}'", "3"),
        ("nested -> '0' -> 0", "baz"),
        ("nested -> 'c' -> -1", 1),
        ("nested -> 'c' ->> 2", "1"),
    ],
)
def test_struct_field_operator_access(expr: str, expected: int | str) -> None:
    df = pl.DataFrame(
        {
            "nested": {
                "0": ["baz"],
                "b": ["foo", "bar"],
                "c": [3, 2, 1],
            },
        },
    )
    assert df.sql(f"SELECT {expr} FROM self").item() == expected


@pytest.mark.parametrize(
    ("fields", "excluding", "rename"),
    [
        ("json_msg.*", "age", {}),
        ("json_msg.*", "name", {"other": "misc"}),
        ("self.json_msg.*", "(age,other)", {"name": "ident"}),
        ("json_msg.other.*", "", {"n": "num"}),
        ("self.json_msg.other.*", "", {}),
        ("self.json_msg.other.*", "n", {}),
    ],
)
def test_struct_field_selection_wildcards(
    fields: str,
    excluding: str,
    rename: dict[str, str],
    df_struct: pl.DataFrame,
) -> None:
    exclude_cols = f"EXCLUDE {excluding}" if excluding else ""
    rename_cols = (
        f"RENAME ({','.join(f'{k} AS {v}' for k, v in rename.items())})"
        if rename
        else ""
    )
    res = df_struct.sql(
        f"""
        SELECT {fields} {exclude_cols} {rename_cols}
        FROM self ORDER BY json_msg.id
    """
    )

    expected = df_struct.unnest("json_msg")
    if fields.endswith(".other.*"):
        expected = expected["other"].struct.unnest()
    if excluding:
        expected = expected.drop(excluding.strip(")(").split(","))
    if rename:
        expected = expected.rename(rename)

    assert_frame_equal(expected, res)


@pytest.mark.parametrize(
    ("invalid_column", "error_type"),
    [
        ("json_msg.invalid_column", StructFieldNotFoundError),
        ("json_msg.other.invalid_column", StructFieldNotFoundError),
        ("self.json_msg.other.invalid_column", StructFieldNotFoundError),
        ("json_msg.other -> invalid_column", SQLSyntaxError),
        ("json_msg -> DATE '2020-09-11'", SQLSyntaxError),
    ],
)
def test_struct_field_selection_errors(
    invalid_column: str,
    error_type: type[Exception],
    df_struct: pl.DataFrame,
) -> None:
    error_msg = (
        "invalid json/struct path-extract"
        if ("->" in invalid_column)
        else "invalid_column"
    )
    with pytest.raises(error_type, match=error_msg):
        df_struct.sql(f"SELECT {invalid_column} FROM self")
