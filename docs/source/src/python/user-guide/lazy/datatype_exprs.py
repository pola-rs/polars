"""
# --8<-- [start:setup]
import polars as pl
import polars.selectors as cs

# --8<-- [end:setup]
# --8<-- [start:basic]
dtype_expr = pl.dtype_of("UserID")

# For debugging you can collect the output datatype in a specific context.
schema = pl.Schema({ 'UserID': pl.UInt64, 'Name': pl.String })
dtype_expr.collect_dtype(schema)
# --8<-- [end:basic]

# --8<-- [start:basic-manipulation]
dtype_expr.wrap_in_list().collect_dtype(schema)

dtype_expr.to_signed_integer().collect_dtype(schema)
# --8<-- [end:basic-manipulation]

# --8<-- [start:basic-inspect]
df = schema.to_frame()
df.select(
    userid_dtype_name = pl.dtype_of('UserID').display(),
    userid_is_signed  = pl.dtype_of('UserID').matches(cs.signed_integer()),
)
# --8<-- [end:basic-inspect]

# --8<-- [start:inspect]
def inspect(expr: pl.Expr) -> pl.Expr:
    def print_and_return(s: pl.Series) -> pl.Series:
        print(s)
        return s

    return expr.map_batches(
        print_and_return,

        # Clarify that the expression returns the same datatype as the input
        # datatype.
        return_dtype=pl.dtype_of(expr),
    )

df = pl.DataFrame({
    'UserID': [1, 2, 3, 4, 5],
    'Name': ["Alice", "Bob", "Charlie", "Diana", "Ethan"],
})
df.select(inspect(pl.col('Name')))
# --8<-- [end:inspect]

# --8<-- [start:cast]
df = pl.DataFrame({
    'UserID': [1, 2, 3, 4, 5],
    'Name': ["Alice", "Bob", "Charlie", "Diana", "Ethan"],
}).with_columns(
    pl.col('UserID').cast(pl.dtype_of('Name'))
)
# --8<-- [end:cast]

"""
