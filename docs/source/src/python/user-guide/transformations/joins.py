# --8<-- [start:prep-data]
import pathlib
import requests


DATA = [
    (
        "https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/data/monopoly_props_groups.csv",
        "docs/assets/data/monopoly_props_groups.csv",
    ),
    (
        "https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/data/monopoly_props_prices.csv",
        "docs/assets/data/monopoly_props_prices.csv",
    ),
]


for url, dest in DATA:
    if pathlib.Path(dest).exists():
        continue
    with open(dest, "wb") as f:
        f.write(requests.get(url, timeout=10).content)
# --8<-- [end:prep-data]

# --8<-- [start:props_groups]
import polars as pl

props_groups = pl.read_csv("docs/assets/data/monopoly_props_groups.csv").head(5)
print(props_groups)
# --8<-- [end:props_groups]

# --8<-- [start:props_prices]
props_prices = pl.read_csv("docs/assets/data/monopoly_props_prices.csv").head(5)
print(props_prices)
# --8<-- [end:props_prices]

# --8<-- [start:equi-join]
result = props_groups.join(props_prices, on="property_name")
print(result)
# --8<-- [end:equi-join]

# --8<-- [start:props_groups2]
props_groups2 = props_groups.with_columns(
    pl.col("property_name").str.to_lowercase(),
)
print(props_groups2)
# --8<-- [end:props_groups2]

# --8<-- [start:props_prices2]
props_prices2 = props_prices.select(
    pl.col("property_name").alias("name"), pl.col("cost")
)
print(props_prices2)
# --8<-- [end:props_prices2]

# --8<-- [start:join-key-expression]
result = props_groups2.join(
    props_prices2,
    left_on="property_name",
    right_on=pl.col("name").str.to_lowercase(),
)
print(result)
# --8<-- [end:join-key-expression]

# --8<-- [start:inner-join]
result = props_groups.join(props_prices, on="property_name", how="inner")
print(result)
# --8<-- [end:inner-join]

# --8<-- [start:left-join]
result = props_groups.join(props_prices, on="property_name", how="left")
print(result)
# --8<-- [end:left-join]

# --8<-- [start:right-join]
result = props_groups.join(props_prices, on="property_name", how="right")
print(result)
# --8<-- [end:right-join]

# --8<-- [start:left-right-join-equals]
print(
    result.equals(
        props_prices.join(
            props_groups,
            on="property_name",
            how="left",
            # Reorder the columns to match the order from above.
        ).select(pl.col("group"), pl.col("property_name"), pl.col("cost"))
    )
)
# --8<-- [end:left-right-join-equals]

# --8<-- [start:full-join]
result = props_groups.join(props_prices, on="property_name", how="full")
print(result)
# --8<-- [end:full-join]

# --8<-- [start:full-join-coalesce]
result = props_groups.join(
    props_prices,
    on="property_name",
    how="full",
    coalesce=True,
)
print(result)
# --8<-- [end:full-join-coalesce]

# --8<-- [start:semi-join]
result = props_groups.join(props_prices, on="property_name", how="semi")
print(result)
# --8<-- [end:semi-join]

# --8<-- [start:anti-join]
result = props_groups.join(props_prices, on="property_name", how="anti")
print(result)
# --8<-- [end:anti-join]

# --8<-- [start:players]
players = pl.DataFrame(
    {
        "name": ["Alice", "Bob"],
        "cash": [78, 135],
    }
)
print(players)
# --8<-- [end:players]

# --8<-- [start:non-equi]
result = players.join_where(props_prices, pl.col("cash") > pl.col("cost"))
print(result)
# --8<-- [end:non-equi]

# --8<-- [start:df_trades]
from datetime import datetime

df_trades = pl.DataFrame(
    {
        "time": [
            datetime(2020, 1, 1, 9, 1, 0),
            datetime(2020, 1, 1, 9, 1, 0),
            datetime(2020, 1, 1, 9, 3, 0),
            datetime(2020, 1, 1, 9, 6, 0),
        ],
        "stock": ["A", "B", "B", "C"],
        "trade": [101, 299, 301, 500],
    }
)
print(df_trades)
# --8<-- [end:df_trades]

# --8<-- [start:df_quotes]
df_quotes = pl.DataFrame(
    {
        "time": [
            datetime(2020, 1, 1, 9, 0, 0),
            datetime(2020, 1, 1, 9, 2, 0),
            datetime(2020, 1, 1, 9, 4, 0),
            datetime(2020, 1, 1, 9, 6, 0),
        ],
        "stock": ["A", "B", "C", "A"],
        "quote": [100, 300, 501, 102],
    }
)

print(df_quotes)
# --8<-- [end:df_quotes]

# --8<-- [start:asof]
df_asof_join = df_trades.join_asof(df_quotes, on="time", by="stock")
print(df_asof_join)
# --8<-- [end:asof]

# --8<-- [start:asof-tolerance]
df_asof_tolerance_join = df_trades.join_asof(
    df_quotes, on="time", by="stock", tolerance="1m"
)
print(df_asof_tolerance_join)
# --8<-- [end:asof-tolerance]

# --8<-- [start:cartesian-product]
tokens = pl.DataFrame({"monopoly_token": ["hat", "shoe", "boat"]})

result = players.select(pl.col("name")).join(tokens, how="cross")
print(result)
# --8<-- [end:cartesian-product]
