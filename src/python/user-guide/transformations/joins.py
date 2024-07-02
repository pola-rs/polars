# --8<-- [start:setup]
import polars as pl
from datetime import datetime

# --8<-- [end:setup]

# --8<-- [start:innerdf]
df_customers = pl.DataFrame(
    {
        "customer_id": [1, 2, 3],
        "name": ["Alice", "Bob", "Charlie"],
    }
)
print(df_customers)
# --8<-- [end:innerdf]

# --8<-- [start:innerdf2]
df_orders = pl.DataFrame(
    {
        "order_id": ["a", "b", "c"],
        "customer_id": [1, 2, 2],
        "amount": [100, 200, 300],
    }
)
print(df_orders)
# --8<-- [end:innerdf2]


# --8<-- [start:inner]
df_inner_customer_join = df_customers.join(df_orders, on="customer_id", how="inner")
print(df_inner_customer_join)
# --8<-- [end:inner]

# --8<-- [start:left]
df_left_join = df_customers.join(df_orders, on="customer_id", how="left")
print(df_left_join)
# --8<-- [end:left]

# --8<-- [start:full]
df_outer_join = df_customers.join(df_orders, on="customer_id", how="full")
print(df_outer_join)
# --8<-- [end:full]

# --8<-- [start:full_coalesce]
df_outer_coalesce_join = df_customers.join(
    df_orders, on="customer_id", how="full", coalesce=True
)
print(df_outer_coalesce_join)
# --8<-- [end:full_coalesce]

# --8<-- [start:df3]
df_colors = pl.DataFrame(
    {
        "color": ["red", "blue", "green"],
    }
)
print(df_colors)
# --8<-- [end:df3]

# --8<-- [start:df4]
df_sizes = pl.DataFrame(
    {
        "size": ["S", "M", "L"],
    }
)
print(df_sizes)
# --8<-- [end:df4]

# --8<-- [start:cross]
df_cross_join = df_colors.join(df_sizes, how="cross")
print(df_cross_join)
# --8<-- [end:cross]

# --8<-- [start:df5]
df_cars = pl.DataFrame(
    {
        "id": ["a", "b", "c"],
        "make": ["ford", "toyota", "bmw"],
    }
)
print(df_cars)
# --8<-- [end:df5]

# --8<-- [start:df6]
df_repairs = pl.DataFrame(
    {
        "id": ["c", "c"],
        "cost": [100, 200],
    }
)
print(df_repairs)
# --8<-- [end:df6]

# --8<-- [start:inner2]
df_inner_join = df_cars.join(df_repairs, on="id", how="inner")
print(df_inner_join)
# --8<-- [end:inner2]

# --8<-- [start:semi]
df_semi_join = df_cars.join(df_repairs, on="id", how="semi")
print(df_semi_join)
# --8<-- [end:semi]

# --8<-- [start:anti]
df_anti_join = df_cars.join(df_repairs, on="id", how="anti")
print(df_anti_join)
# --8<-- [end:anti]

# --8<-- [start:df7]
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
# --8<-- [end:df7]

# --8<-- [start:df8]
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
# --8<-- [end:df8]

# --8<-- [start:asofpre]
df_trades = df_trades.sort("time")
df_quotes = df_quotes.sort("time")  # Set column as sorted
# --8<-- [end:asofpre]

# --8<-- [start:asof]
df_asof_join = df_trades.join_asof(df_quotes, on="time", by="stock")
print(df_asof_join)
# --8<-- [end:asof]

# --8<-- [start:asof2]
df_asof_tolerance_join = df_trades.join_asof(
    df_quotes, on="time", by="stock", tolerance="1m"
)
print(df_asof_tolerance_join)
# --8<-- [end:asof2]
