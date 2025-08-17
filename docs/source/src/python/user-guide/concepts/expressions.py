# --8<-- [start:expression]
import polars as pl

pl.col("weight") / (pl.col("height") ** 2)
# --8<-- [end:expression]

# --8<-- [start:print-expr]
bmi_expr = pl.col("weight") / (pl.col("height") ** 2)
print(bmi_expr)
# --8<-- [end:print-expr]

# --8<-- [start:df]
from datetime import date

df = pl.DataFrame(
    {
        "name": ["Alice Archer", "Ben Brown", "Chloe Cooper", "Daniel Donovan"],
        "birthdate": [
            date(1997, 1, 10),
            date(1985, 2, 15),
            date(1983, 3, 22),
            date(1981, 4, 30),
        ],
        "weight": [57.9, 72.5, 53.6, 83.1],  # (kg)
        "height": [1.56, 1.77, 1.65, 1.75],  # (m)
    }
)

print(df)
# --8<-- [end:df]

# --8<-- [start:select-1]
result = df.select(
    bmi=bmi_expr,
    avg_bmi=bmi_expr.mean(),
    ideal_max_bmi=25,
)
print(result)
# --8<-- [end:select-1]

# --8<-- [start:select-2]
result = df.select(deviation=(bmi_expr - bmi_expr.mean()) / bmi_expr.std())
print(result)
# --8<-- [end:select-2]

# --8<-- [start:with_columns-1]
result = df.with_columns(
    bmi=bmi_expr,
    avg_bmi=bmi_expr.mean(),
    ideal_max_bmi=25,
)
print(result)
# --8<-- [end:with_columns-1]

# --8<-- [start:filter-1]
result = df.filter(
    pl.col("birthdate").is_between(date(1982, 12, 31), date(1996, 1, 1)),
    pl.col("height") > 1.7,
)
print(result)
# --8<-- [end:filter-1]

# --8<-- [start:group_by-1]
result = df.group_by(
    (pl.col("birthdate").dt.year() // 10 * 10).alias("decade"),
).agg(pl.col("name"))
print(result)
# --8<-- [end:group_by-1]

# --8<-- [start:group_by-2]
result = df.group_by(
    (pl.col("birthdate").dt.year() // 10 * 10).alias("decade"),
    (pl.col("height") < 1.7).alias("short?"),
).agg(pl.col("name"))
print(result)
# --8<-- [end:group_by-2]

# --8<-- [start:group_by-3]
result = df.group_by(
    (pl.col("birthdate").dt.year() // 10 * 10).alias("decade"),
    (pl.col("height") < 1.7).alias("short?"),
).agg(
    pl.len(),
    pl.col("height").max().alias("tallest"),
    pl.col("weight", "height").mean().name.prefix("avg_"),
)
print(result)
# --8<-- [end:group_by-3]

# --8<-- [start:expression-expansion-1]
expr = (pl.col(pl.Float64) * 1.1).name.suffix("*1.1")
result = df.select(expr)
print(result)
# --8<-- [end:expression-expansion-1]

# --8<-- [start:expression-expansion-2]
df2 = pl.DataFrame(
    {
        "ints": [1, 2, 3, 4],
        "letters": ["A", "B", "C", "D"],
    }
)
result = df2.select(expr)
print(result)
# --8<-- [end:expression-expansion-2]
