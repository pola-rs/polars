# Parquet test fixtures

`floating_orders_nan_count.parquet` — vendored from
[apache/parquet-testing](https://github.com/apache/parquet-testing/blob/master/data/floating_orders_nan_count.parquet)
(Apache-2.0). FLOAT/DOUBLE/FLOAT16 columns in IEEE754 and TypeDefined orders across five row groups
(no-NaN, mixed-NaN, all-NaN, zero-min, zero-max), used to exercise `IEEE_754_TOTAL_ORDER`
column-order decoding (PARQUET-2249).
