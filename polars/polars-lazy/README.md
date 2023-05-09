# polars-lazy

`polars-lazy` is a lazy query engine for the Polars DataFrame library. It allows you to perform operations on DataFrames in a lazy manner, only executing them when necessary. This can lead to significant performance improvements for large datasets.

## Features

| Feature | Description |
| --------------- | ------------------------------------------------ | --- | --- |
| nightly | Enable nightly features |
| compile | Features required to compile |
| streaming | Enable streaming data processing |
| default | Enable default features |
| parquet | Enable Parquet file format support |
| async | Enable asynchronous data processing |
| ipc | Enable IPC (Inter-process communication) support |
| json | Enable JSON file format support |
| csv | Enable CSV file format support |
| temporal | temporal data types |
| fmt | formatting DataFrames |
| strings | string processing |
| dtype-\* | various data types (e.g., dtype-u8) |
| object | object data types |
| date_offset | date offsets |
| trigonometry | `trigonometry` operations |
| sign | `sign` operations |
| timezones | `timezone-aware` operations |
| list\_\* | various list operations (e.g., list_take) |
| true_div | true division |
| approx_unique | approximate unique operation |
| is_in | `is_in` operation |
| repeat_by | `repeat_by` operation |
| round_series | `round_series` operation |
| is_first | `is_first` operation |
| is_unique | `is_unique` operation |
| cross_join | `cross_join` operation |
| asof_join | `asof_join` operation |
| concat_str | `concat_str` operation |
| arange | `arange` operation |
| mode | `mode` operation |
| cum_agg | `cum_agg` operation |
| interpolate | `interpolate` operation |
| rolling_window | `rolling_window` operation |
| rank | `rank` operation |
| diff | `diff` operation |
| pct_change | `pct_change` operation |
| moment | `moment` operation |
| abs | `abs` operation |
| random | `random` operation |
| dynamic_groupby | `dynamic_groupby` operation |
| ewma | `ewma` operation |
| dot_diagram | `dot_diagram` operation |
| unique_counts | `unique_counts` operation |
| log | `log` operation |
| search_sorted | `search_sorted` operation |
| merge_sorted | `merge_sorted` operation |
| pivot | `pivot` operation |
| top_k | `top_k` operation |
| semi_anti_join | semi/anti join operations |
| cse | common subexpression elimination |
| propagate_nans | NaN propagation |
| coalesce | coalesce operation |
| regex | regular expressions | | n |
| python | PyO3 Python bindings |
| row_hash | row hashing |
| string\_\* | various string operations |
| arg_where | arg_where operation |
| search_sorted | search_sorted operation |
| merge_sorted | merge_sorted operation |
| meta | metadata processing |
| pivot | pivot operation |
| top_k | top_k operation |
| semi_anti_join | semi/anti join operations |
| cse | common subexpression elimination |
| propagate_nans | NaN propagation |
| coalesce | coalesce operation |
| regex | regular expressions |
| serde | Serde serialization |
| fused | fused operations |
| binary_encoding | binary encoding |
| bigidx | big indices |
| panic_on_schema | Enable panic on schema mismatch |
| private | Enables private APIs. <sup> [1](#footnote1)</sup> |

To enable a specific operation, add it to the `features` section of your `Cargo.toml` file:

```toml
[dependencies]
polars-lazy = { version = "0.29.0", features = ["csv", "parquet", "json", "is_in", "repeat_by", "round_series"] }
```

<sup><a name="footnote1">1</a></sup> Private APIs in `polars` are not intended for public use and may change without notice. Use at your own risk.
