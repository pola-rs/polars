# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:scan_iris_csv]
print(pl.scan_csv("hf://datasets/nameexhaustion/polars-docs/iris.csv").collect())
# --8<-- [end:scan_iris_csv]

# --8<-- [start:scan_iris_ndjson]
print(pl.scan_ndjson("hf://datasets/nameexhaustion/polars-docs/iris.jsonl").collect())
# --8<-- [end:scan_iris_ndjson]

# --8<-- [start:scan_iris_repr]
print(
    """\
shape: (150, 5)
┌──────────────┬─────────────┬──────────────┬─────────────┬───────────┐
│ sepal_length ┆ sepal_width ┆ petal_length ┆ petal_width ┆ species   │
│ ---          ┆ ---         ┆ ---          ┆ ---         ┆ ---       │
│ f64          ┆ f64         ┆ f64          ┆ f64         ┆ str       │
╞══════════════╪═════════════╪══════════════╪═════════════╪═══════════╡
│ 5.1          ┆ 3.5         ┆ 1.4          ┆ 0.2         ┆ setosa    │
│ 4.9          ┆ 3.0         ┆ 1.4          ┆ 0.2         ┆ setosa    │
│ 4.7          ┆ 3.2         ┆ 1.3          ┆ 0.2         ┆ setosa    │
│ 4.6          ┆ 3.1         ┆ 1.5          ┆ 0.2         ┆ setosa    │
│ 5.0          ┆ 3.6         ┆ 1.4          ┆ 0.2         ┆ setosa    │
│ …            ┆ …           ┆ …            ┆ …           ┆ …         │
│ 6.7          ┆ 3.0         ┆ 5.2          ┆ 2.3         ┆ virginica │
│ 6.3          ┆ 2.5         ┆ 5.0          ┆ 1.9         ┆ virginica │
│ 6.5          ┆ 3.0         ┆ 5.2          ┆ 2.0         ┆ virginica │
│ 6.2          ┆ 3.4         ┆ 5.4          ┆ 2.3         ┆ virginica │
│ 5.9          ┆ 3.0         ┆ 5.1          ┆ 1.8         ┆ virginica │
└──────────────┴─────────────┴──────────────┴─────────────┴───────────┘
"""
)
# --8<-- [end:scan_iris_repr]

# --8<-- [start:scan_parquet_hive]
print(pl.scan_parquet("hf://datasets/nameexhaustion/polars-docs/hive_dates/").collect())
# --8<-- [end:scan_parquet_hive]

# --8<-- [start:scan_parquet_hive_repr]
print(
    """\
shape: (4, 3)
┌────────────┬────────────────────────────┬─────┐
│ date1      ┆ date2                      ┆ x   │
│ ---        ┆ ---                        ┆ --- │
│ date       ┆ datetime[μs]               ┆ i32 │
╞════════════╪════════════════════════════╪═════╡
│ 2024-01-01 ┆ 2023-01-01 00:00:00        ┆ 1   │
│ 2024-02-01 ┆ 2023-02-01 00:00:00        ┆ 2   │
│ 2024-03-01 ┆ null                       ┆ 3   │
│ null       ┆ 2023-03-01 01:01:01.000001 ┆ 4   │
└────────────┴────────────────────────────┴─────┘
"""
)
# --8<-- [end:scan_parquet_hive_repr]

# --8<-- [start:scan_ipc]
print(pl.scan_ipc("hf://spaces/nameexhaustion/polars-docs/orders.feather").collect())
# --8<-- [end:scan_ipc]

# --8<-- [start:scan_ipc_repr]
print(
    """\
shape: (10, 9)
┌────────────┬───────────┬───────────────┬──────────────┬───┬─────────────────┬─────────────────┬────────────────┬─────────────────────────┐
│ o_orderkey ┆ o_custkey ┆ o_orderstatus ┆ o_totalprice ┆ … ┆ o_orderpriority ┆ o_clerk         ┆ o_shippriority ┆ o_comment               │
│ ---        ┆ ---       ┆ ---           ┆ ---          ┆   ┆ ---             ┆ ---             ┆ ---            ┆ ---                     │
│ i64        ┆ i64       ┆ str           ┆ f64          ┆   ┆ str             ┆ str             ┆ i64            ┆ str                     │
╞════════════╪═══════════╪═══════════════╪══════════════╪═══╪═════════════════╪═════════════════╪════════════════╪═════════════════════════╡
│ 1          ┆ 36901     ┆ O             ┆ 173665.47    ┆ … ┆ 5-LOW           ┆ Clerk#000000951 ┆ 0              ┆ nstructions sleep       │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ furiously am…           │
│ 2          ┆ 78002     ┆ O             ┆ 46929.18     ┆ … ┆ 1-URGENT        ┆ Clerk#000000880 ┆ 0              ┆ foxes. pending accounts │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ at th…                  │
│ 3          ┆ 123314    ┆ F             ┆ 193846.25    ┆ … ┆ 5-LOW           ┆ Clerk#000000955 ┆ 0              ┆ sly final accounts      │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ boost. care…            │
│ 4          ┆ 136777    ┆ O             ┆ 32151.78     ┆ … ┆ 5-LOW           ┆ Clerk#000000124 ┆ 0              ┆ sits. slyly regular     │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ warthogs c…             │
│ 5          ┆ 44485     ┆ F             ┆ 144659.2     ┆ … ┆ 5-LOW           ┆ Clerk#000000925 ┆ 0              ┆ quickly. bold deposits  │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ sleep s…                │
│ 6          ┆ 55624     ┆ F             ┆ 58749.59     ┆ … ┆ 4-NOT SPECIFIED ┆ Clerk#000000058 ┆ 0              ┆ ggle. special, final    │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ requests …              │
│ 7          ┆ 39136     ┆ O             ┆ 252004.18    ┆ … ┆ 2-HIGH          ┆ Clerk#000000470 ┆ 0              ┆ ly special requests     │
│ 32         ┆ 130057    ┆ O             ┆ 208660.75    ┆ … ┆ 2-HIGH          ┆ Clerk#000000616 ┆ 0              ┆ ise blithely bold,      │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ regular req…            │
│ 33         ┆ 66958     ┆ F             ┆ 163243.98    ┆ … ┆ 3-MEDIUM        ┆ Clerk#000000409 ┆ 0              ┆ uriously. furiously     │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ final requ…             │
│ 34         ┆ 61001     ┆ O             ┆ 58949.67     ┆ … ┆ 3-MEDIUM        ┆ Clerk#000000223 ┆ 0              ┆ ly final packages.      │
│            ┆           ┆               ┆              ┆   ┆                 ┆                 ┆                ┆ fluffily fi…            │
└────────────┴───────────┴───────────────┴──────────────┴───┴─────────────────┴─────────────────┴────────────────┴─────────────────────────┘
"""
)
# --8<-- [end:scan_ipc_repr]
