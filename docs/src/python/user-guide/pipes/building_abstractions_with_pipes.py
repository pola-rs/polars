# --8<-- [start:setup]
import polars as pl
# --8<-- [end:setup]


# --8<-- [start:hypotenuse]
def hypotenuse(df: pl.DataFrame, col_x: str, col_y: str, col_r: str) -> pl.DataFrame:
    "Apply the Pythagorean theorem."
    x_squared = pl.col(col_x).pow(2)
    y_squared = pl.col(col_y).pow(2)
    r_squared = x_squared + y_squared
    r = r_squared.sqrt()
    return df.with_columns(r.alias(col_r))
# --8<-- [end:hypotenuse]


# --8<-- [start:pipe]
df = pl.DataFrame(
    {
        "x": [1.1, 2.2, 3.3],
        "y": [3.1, 2.2, 1.3],
    }
).pipe(hypotenuse, "x", "y", "r")

print(df)
# --8<-- [end:pipe]
