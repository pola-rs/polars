from __future__ import annotations

import pytest

import polars as pl
import polars.selectors as cs
from polars.exceptions import (
    SQLInterfaceError,
    SQLSyntaxError,
    StructFieldNotFoundError,
)
from polars.testing import assert_frame_equal
from tests.unit.sql import assert_sql_matches


@pytest.fixture
def df_struct() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "id": [200, 300, 400],
            "name": ["Bob", "David", "Zoe"],
            "age": [45, 19, 45],
            "other": [{"n": 1.5}, {"n": None}, {"n": -0.5}],
        }
    ).select(pl.struct(pl.all()).alias("json_msg"))


def test_struct_field_nested_dot_notation_22107() -> None:
    # ensure dot-notation references the given name at the right level of nesting
    df = pl.DataFrame(
        {
            "id": ["012345", "987654"],
            "name": ["A Book", "Another Book"],
            "author": [
                {"id": "888888", "name": "Iain M. Banks"},
                {"id": "444444", "name": "Dan Abnett"},
            ],
        }
    )

    res = df.sql("SELECT id, author.id AS author_id FROM self ORDER BY id")
    assert res.to_dict(as_series=False) == {
        "id": ["012345", "987654"],
        "author_id": ["888888", "444444"],
    }

    for name in ("author.name", "self.author.name"):
        res = df.sql(f"SELECT {name} FROM self ORDER BY id")
        assert res.to_dict(as_series=False) == {"name": ["Iain M. Banks", "Dan Abnett"]}

    for name in ("name", "self.name"):
        res = df.sql(f"SELECT {name} FROM self ORDER BY self.id DESC")
        assert res.to_dict(as_series=False) == {"name": ["Another Book", "A Book"]}

    # expected errors
    with pytest.raises(
        SQLInterfaceError,
        match="no table or struct column named 'foo' found",
    ):
        df.sql("SELECT foo.id FROM self ORDER BY id")

    with pytest.raises(
        SQLInterfaceError,
        match="no column named 'foo' found",
    ):
        df.sql("SELECT self.foo FROM self ORDER BY id")


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
          frame.json_msg.name AS NAME,
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
        schema_overrides={"n": pl.Int64},
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


def test_json_string_operator_access() -> None:
    df = pl.DataFrame(
        {
            "j": [
                '{"a": {"b": "x", "c": [1, 2, 3]}, "k": "v", "n": null, "it\'s": 1}',
                '{"k": 7}',
                None,
            ]
        }
    )
    assert_sql_matches(
        df,
        query="""
        SELECT
          j ->> 'k' AS k,
          j -> 'a' ->> 'b' AS b,
          j -> 'a' -> 'c' ->> -1 AS c_last,
          j -> 'a' -> 'c' ->> 0 AS c_first,
          j ->> '$.a.c[1]' AS c_path,
          j ->> 'n' AS n,
          j ->> 'missing' AS missing,
          j ->> 'it''s' AS quoted,
          j ->> '$.a.c[#-1]' AS from_end,
          j -> 'a' -> 'c' ->> -3 AS c_neg,
          j -> 'a' -> 'c' ->> -4 AS c_out,
          j ->> '$.a.c[#-4]' AS from_end_out,
          '{"a''b''c": 7}' ->> 'a''b''c' AS quotes,
          '{"a.b": 1}' ->> '$."a.b"' AS dotted,
          '{"z": [true]}' ->> 'z' AS lit,
          (SELECT '{"s": 2}') ->> 's' AS sub
        FROM self
        """,
        compare_with="duckdb",
    )
    # `->` returns text, so a string value has no quotes.
    res = df.sql(
        """
        SELECT j -> 'k' AS k, j #> '{a,b}' AS b, j #>> '{a,c,2}' AS c FROM self
        """
    )
    assert res.rows() == [("v", "x", "3"), ("7", None, None), (None, None, None)]
    res = pl.sql(
        """
        SELECT
          '{"a,b": {"c": 1}}' #>> '{"a,b", c}' AS quoted,
          '{"a": 1}' #>> '{}' AS whole,
          '{"a ": 1, "a": 2}' #>> '{a\\ }' AS escaped_space,
          '{"NULL": 1}' #>> '{NULL}' AS null_step,
          '{"NULL": 1}' #>> '{"NULL"}' AS null_key,
          '{"a\\"b": 1}' ->> '$."a\\"b"' AS escaped_quote
        """,
        eager=True,
    )
    assert res.row(0) == ("1", '{"a":1}', "1", None, "1", "1")
    # A NULL path gives NULL, with the rows of the left side.
    assert df.sql("SELECT j ->> NULL AS a FROM self").rows() == [(None,)] * 3
    assert df.sql("SELECT MAX(j) #>> '{NULL}' AS a FROM self").rows() == [(None,)]
    with pytest.raises(SQLInterfaceError, match="unsupported JSON path"):
        df.sql("SELECT j ->> '$[*]' FROM self")
    assert df.sql("SELECT j #>> '{a,c,-4}' AS a FROM self").rows() == [(None,)] * 3
    with pytest.raises(SQLInterfaceError, match="unsupported JSON key"):
        df.sql(r"SELECT j ->> 'a\b' FROM self")


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
    stmt = f"""
SELECT {fields} {exclude_cols} {rename_cols}
FROM self ORDER BY json_msg.id
"""
    res = df_struct.sql(stmt)

    expected = df_struct.unnest("json_msg")
    if fields.endswith(".other.*"):
        expected = expected["other"].struct.unnest()
    if excluding:
        expected = expected.select(cs.exclude(excluding.strip(")(").split(",")))
    if rename:
        expected = expected.rename(rename)

    assert_frame_equal(res, expected)


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
