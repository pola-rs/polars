# Polars SQL

## CLI

You can build `polars-sql` with cargo.

`$ cargo run --release --features=cli,parquet,csv,ipc`

This will run the CLI loop:

```bash
Welcome to Polars CLI. Commands end with ; or \n
Type help or \? for help.
>> \?
List of all client commands:
dataframes          \dd         Show registered frames.
help                \?          Display this help.
register            \rd         Register new dataframe: \rd <name> <source>
quit                \q          Exit

>> register taxis /home/ritchie46/example/csv-benchmark/yellow_tripdata_2010-01.parquet
Added dataframe "taxis" from file /home/ritchie46/example/csv-benchmark/yellow_tripdata_2010-01.parquet
shape: (14863778, 18)
┌───────────┬─────────────────────┬─────────────────────┬─────────────────┬─────┬─────────┬────────────┬──────────────┬──────────────┐
│ vendor_id ┆ pickup_datetime     ┆ dropoff_datetime    ┆ passenger_count ┆ ... ┆ mta_tax ┆ tip_amount ┆ tolls_amount ┆ total_amount │
│ ---       ┆ ---                 ┆ ---                 ┆ ---             ┆     ┆ ---     ┆ ---        ┆ ---          ┆ ---          │
│ str       ┆ str                 ┆ str                 ┆ i64             ┆     ┆ f64     ┆ f64        ┆ f64          ┆ f64          │
╞═══════════╪═════════════════════╪═════════════════════╪═════════════════╪═════╪═════════╪════════════╪══════════════╪══════════════╡
│ VTS       ┆ 2010-01-26 07:41:00 ┆ 2010-01-26 07:45:00 ┆ 1               ┆ ... ┆ 0.5     ┆ 0.0        ┆ 0.0          ┆ 5.0          │
│ DDS       ┆ 2010-01-30 23:31:00 ┆ 2010-01-30 23:46:12 ┆ 1               ┆ ... ┆ 0.5     ┆ 0.0        ┆ 0.0          ┆ 16.3         │
│ DDS       ┆ 2010-01-18 20:22:20 ┆ 2010-01-18 20:38:12 ┆ 1               ┆ ... ┆ 0.5     ┆ 0.0        ┆ 0.0          ┆ 12.7         │
│ VTS       ┆ 2010-01-09 01:18:00 ┆ 2010-01-09 01:35:00 ┆ 2               ┆ ... ┆ 0.5     ┆ 0.0        ┆ 0.0          ┆ 14.3         │
│ ...       ┆ ...                 ┆ ...                 ┆ ...             ┆ ... ┆ ...     ┆ ...        ┆ ...          ┆ ...          │
│ VTS       ┆ 2010-01-09 12:52:00 ┆ 2010-01-09 13:15:00 ┆ 1               ┆ ... ┆ 0.5     ┆ 0.0        ┆ 0.0          ┆ 45.5         │
│ CMT       ┆ 2010-01-09 14:00:44 ┆ 2010-01-09 14:14:33 ┆ 1               ┆ ... ┆ 0.5     ┆ 0.0        ┆ 0.0          ┆ 8.6          │
│ CMT       ┆ 2010-01-09 09:52:23 ┆ 2010-01-09 09:59:41 ┆ 1               ┆ ... ┆ 0.5     ┆ 0.0        ┆ 0.0          ┆ 7.8          │
│ CMT       ┆ 2010-01-05 15:25:59 ┆ 2010-01-05 15:33:54 ┆ 1               ┆ ... ┆ 0.5     ┆ 0.0        ┆ 0.0          ┆ 6.6          │
└───────────┴─────────────────────┴─────────────────────┴─────────────────┴─────┴─────────┴────────────┴──────────────┴──────────────┘
14863778 rows in set (1.098 sec)
```

Or pipe your SQL command directly inline:

```bash
$ echo "SELECT MIN(Close) as low, MAX(Close) as high from '/home/ritchie46/example/BTUSD.csv' WHERE Date > '2014' AND Date < '2015'" | ./polars-sql
```
