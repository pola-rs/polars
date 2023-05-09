# polars-time

`polars-time` is a Rust crate that provides time-related code for the Polars dataframe library.

## Features

`polars-time` has the following features:

| Feature        | Description                                              |
| -------------- | -------------------------------------------------------- |
| dtype-date     | Enables `Date` data type                                 |
| dtype-datetime | Enables `Datetime` data type                             |
| dtype-time     | Enables `Time` data type.                                |
| dtype-duration | Enables `Duration` data type.                            |
| rolling_window | Enables support for rolling window operations in Polars. |
| fmt            | Enables pretty formatting of dataframes                  |
| timezones      | Enables timezone parsing via chrono-tz                   |
| default        | The default feature set for polars-time.                 |
| private        | Enables private APIs. <sup>[1](#footnote1)</sup>         |

<sup><a name="footnote1">1</a></sup> Private APIs in `polars` are not intended for public use and may change without notice. Use at your own risk.
