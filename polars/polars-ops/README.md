# polars-ops

`polars-ops` is a submodule of Polars that provides more operations on Polars data structures.

## Features

| Feature           | Description                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------- |
| simd              | Enables SIMD support for some mathematical operations                                                         |
| nightly           | Enables the use of unstable and experimental features that are only available on the nightly version of Rust. |
| dtype-categorical | Enables the categorical data type                                                                             |
| dtype-date        | Enables the date data type                                                                                    |
| dtype-datetime    | Enables the datetime data type                                                                                |
| dtype-time        | Enables the time data type                                                                                    |
| dtype-duration    | Enables the duration data type                                                                                |
| dtype-struct      | Enables the struct data type                                                                                  |
| dtype-u8          | Enables the unsigned 8-bit integer data type                                                                  |
| dtype-u16         | Enables the unsigned 16-bit integer data type                                                                 |
| dtype-i8          | Enables the signed 8-bit integer data type                                                                    |
| dtype-i16         | Enables the signed 16-bit integer data type                                                                   |
| dtype-decimal     | Enables the decimal data type                                                                                 |
| object            | Enables the object data type                                                                                  |
| propagate_nans    | Enables NaN propagation                                                                                       |
| performant        | Enables performant operations                                                                                 |
| big_idx           | Enables big index support                                                                                     |
| round_series      | Enables the round series operation                                                                            |
| is_first          | Enables the is_first operation                                                                                |
| is_unique         | Enables the is_unique operation                                                                               |
| approx_unique     | Enables the approx_unique operation                                                                           |
| fused             | Enables fused operations                                                                                      |
| binary_encoding   | Enables binary encoding support                                                                               |
| string_encoding   | Enables string encoding support                                                                               |
| to_dummies        | Enables the to_dummies operation                                                                              |
| interpolate       | Enables the interpolate operation                                                                             |
| list_to_struct    | Enables the list_to_struct operation                                                                          |
| list_count        | Enables the list_count operation                                                                              |
| diff              | Enables the diff operation                                                                                    |
| strings           | Enables string operations                                                                                     |
| string_justify    | Enables the string justify operation                                                                          |
| string_from_radix | Enables the string from radix operation                                                                       |
| extract_jsonpath  | Enables the extract_jsonpath operation                                                                        |
| log               | Enables the log operation                                                                                     |
| hash              | Enables the hash operation                                                                                    |
| rolling_window    | Enables the rolling_window operation                                                                          |
| moment            | Enables the moment operation                                                                                  |
| search_sorted     | Enables the search_sorted operation                                                                           |
| merge_sorted      | Enables the merge_sorted operation                                                                            |
| top_k             | Enables the top_k operation                                                                                   |
| pivot             | Enables the pivot operation                                                                                   |
| cross_join        | Enables the cross_join operation                                                                              |
| chunked_ids       | Enables the chunked_ids operation                                                                             |
| asof_join         | Enables the asof_join operation                                                                               |
| semi_anti_join    | Enables the semi_anti_join operation                                                                          |
| list_take         | Enables the list_take operation                                                                               |
