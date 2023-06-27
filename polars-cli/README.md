# Polars CLI

## CLI

**Build from source**

```bash
cargo install --locked
```

```bash
> polars
Polars CLI v0.1.0
Type .help for help.
>> .help
┌─────────┬────────────────────┐
│ command ┆ description        │
╞═════════╪════════════════════╡
│ .help   ┆ Display this help │
│ .exit   ┆ Exit this program  │
└─────────┴────────────────────┘
shape: (2, 2)

>> select * FROM read_csv('../../examples/datasets/foods1.csv');
shape: (27, 4)
┌────────────┬──────────┬────────┬──────────┐
│ category   ┆ calories ┆ fats_g ┆ sugars_g │
│ ---        ┆ ---      ┆ ---    ┆ ---      │
│ str        ┆ i64      ┆ f64    ┆ i64      │
╞════════════╪══════════╪════════╪══════════╡
│ vegetables ┆ 45       ┆ 0.5    ┆ 2        │
│ seafood    ┆ 150      ┆ 5.0    ┆ 0        │
│ meat       ┆ 100      ┆ 5.0    ┆ 0        │
│ fruit      ┆ 60       ┆ 0.0    ┆ 11       │
│ …          ┆ …        ┆ …      ┆ …        │
│ fruit      ┆ 130      ┆ 0.0    ┆ 25       │
│ meat       ┆ 100      ┆ 7.0    ┆ 0        │
│ vegetables ┆ 30       ┆ 0.0    ┆ 5        │
│ fruit      ┆ 50       ┆ 0.0    ┆ 11       │
└────────────┴──────────┴────────┴──────────┘
```

Or pipe your SQL command directly inline:

```bash
$ echo "SELECT category FROM read_csv('../../examples/datasets/foods1.csv')" | polars

shape: (27, 1)
┌────────────┐
│ category   │
│ ---        │
│ str        │
╞════════════╡
│ vegetables │
│ seafood    │
│ meat       │
│ fruit      │
│ …          │
│ fruit      │
│ meat       │
│ vegetables │
│ fruit      │
└────────────┘
```

## Features

| Feature   | Description                                               |
| --------- | --------------------------------------------------------- |
| default   | The default feature set that includes all other features. |
| highlight | Provides syntax highlighting                              |
| parquet   | Enables reading and writing of Apache Parquet files.      |
| json      | Enables reading and writing of JSON files.                |
| ipc       | Enables reading and writing of IPC/Apache Arrow files     |
