# --8<-- [start:df]
import polars as pl
import datetime as dt

df = pl.DataFrame(
    {
        "name": ["Alice Archer", "Ben Brown", "Chloe Cooper", "Daniel Donovan"],
        "birthdate": [
            dt.date(1997, 1, 10),
            dt.date(1985, 2, 15),
            dt.date(1983, 3, 22),
            dt.date(1981, 4, 30),
        ],
        "weight": [57.9, 72.5, 53.6, 83.1],  # (kg)
        "height": [1.56, 1.77, 1.65, 1.75],  # (m)
    }
)

print(df)
# --8<-- [end:df]

# --8<-- [start:csv]
df.write_csv("docs/assets/data/output.csv")
df_csv = pl.read_csv("docs/assets/data/output.csv", try_parse_dates=True)
print(df_csv)
# --8<-- [end:csv]

# --8<-- [start:select]
result = df.select(
    pl.col("name"),
    pl.col("birthdate").dt.year().alias("birth_year"),
    (pl.col("weight") / (pl.col("height") ** 2)).alias("bmi"),
)
print(result)
# --8<-- [end:select]

# --8<-- [start:expression-expansion]
result = df.select(
    pl.col("name"),
    (pl.col("weight", "height") * 0.95).round(2).name.suffix("-5%"),
)
print(result)
# --8<-- [end:expression-expansion]

# --8<-- [start:with_columns]
result = df.with_columns(
    birth_year=pl.col("birthdate").dt.year(),
    bmi=pl.col("weight") / (pl.col("height") ** 2),
)
print(result)
# --8<-- [end:with_columns]

# --8<-- [start:filter]
result = df.filter(pl.col("birthdate").dt.year() < 1990)
print(result)
# --8<-- [end:filter]

# --8<-- [start:filter-multiple]
result = df.filter(
    pl.col("birthdate").is_between(dt.date(1982, 12, 31), dt.date(1996, 1, 1)),
    pl.col("height") > 1.7,
)
print(result)
# --8<-- [end:filter-multiple]

# --8<-- [start:group_by]
result = df.group_by(
    (pl.col("birthdate").dt.year() // 10 * 10).alias("decade"),
    maintain_order=True,
).len()
print(result)
# --8<-- [end:group_by]

# --8<-- [start:group_by-agg]
result = df.group_by(
    (pl.col("birthdate").dt.year() // 10 * 10).alias("decade"),
    maintain_order=True,
).agg(
    pl.len().alias("sample_size"),
    pl.col("weight").mean().round(2).alias("avg_weight"),
    pl.col("height").max().alias("tallest"),
)
print(result)
# --8<-- [end:group_by-agg]

# --8<-- [start:complex]
result = (
    df.with_columns(
        (pl.col("birthdate").dt.year() // 10 * 10).alias("decade"),
        pl.col("name").str.split(by=" ").list.first(),
    )
    .select(
        pl.all().exclude("birthdate"),
    )
    .group_by(
        pl.col("decade"),
        maintain_order=True,
    )
    .agg(
        pl.col("name"),
        pl.col("weight", "height").mean().round(2).name.prefix("avg_"),
    )
)
print(result)
# --8<-- [end:complex]

# --8<-- [start:join]
df2 = pl.DataFrame(
    {
        "name": ["Ben Brown", "Daniel Donovan", "Alice Archer", "Chloe Cooper"],
        "parent": [True, False, False, False],
        "siblings": [1, 2, 3, 4],
    }
)

print(df.join(df2, on="name", how="left"))
# --8<-- [end:join]

# --8<-- [start:concat]
df3 = pl.DataFrame(
    {
        "name": ["Ethan Edwards", "Fiona Foster", "Grace Gibson", "Henry Harris"],
        "birthdate": [
            dt.date(1977, 5, 10),
            dt.date(1975, 6, 23),
            dt.date(1973, 7, 22),
            dt.date(1971, 8, 3),
        ],
        "weight": [67.9, 72.5, 57.6, 93.1],  # (kg)
        "height": [1.76, 1.6, 1.66, 1.8],  # (m)
    }
)

print(pl.concat([df, df3], how="vertical"))
# --8<-- [end:concat]
