# Polars CLI

## CLI

**Build from source**

```bash
cargo install --locked
```

```bash
> polars
v0.28.0
Type help or \? for help.
>> \?
List of all client commands:
.help                \?          Display this help.
.exit                \q          Exit

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
