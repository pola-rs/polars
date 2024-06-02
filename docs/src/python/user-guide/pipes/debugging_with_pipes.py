# --8<-- [start:setup]
from inspect import currentframe
import polars as pl

bar_table = "docs/data/debugging_with_pipes_1.csv"
baz_table = "docs/data/debugging_with_pipes_2.csv"
# --8<-- [end:setup]

# --8<-- [start:pipeline1]
df = (
    pl.scan_csv(bar_table)
    .filter(pl.col("bar") > 0)
    .join(pl.scan_csv(baz_table), on="foo")
    .select("bar", "baz")
    .group_by("bar")
    .agg(pl.count("baz"))
    .collect()
)
# --8<-- [end:pipeline1]


# --8<-- [start:assert_schema]
def assert_schema(
    lf: pl.LazyFrame,
    schema: dict[str, pl.PolarsDataType],
) -> pl.LazyFrame:
    "Assert that the schema conforms to expectations."
    if lf.schema != schema:
        msg = (
            "Wrong LazyFrame schema:\n"
            f"• expected: '{schema}',\n"
            f"• observed: '{dict(lf.schema)}'."
        )
        raise AssertionError(msg)
    return lf
# --8<-- [end:assert_schema]


# --8<-- [start:print_expr]
def print_expr(
    lf: pl.LazyFrame,
    expr: pl.Expr,
) -> pl.LazyFrame:
    "Evaluate and print an expression."
    df = lf.collect()  # switch to eager mode
    print(f"[line {currentframe().f_back.f_lineno}]")
    print(df.select(expr))
    return df.lazy()  # proceed in lazy mode
# --8<-- [end:print_expr]

# --8<-- [start:pipeline2]
schema = {"bar": pl.Int64, "baz": pl.Int64}

expr = pl.col("bar").unique().count()

df = (
    pl.scan_csv(bar_table)
    .pipe(print_expr, expr)  # ⇐ PRINT
    .filter(pl.col("bar") > 0)
    .join(pl.scan_csv(baz_table), on="foo")
    .select("bar", "baz")
    .pipe(assert_schema, schema)  # ⇐ ASSERT
    .group_by("bar")
    .agg(pl.count("baz"))
    .collect()
)
# --8<-- [end:pipeline2]
