### Description
Filtering `scan_iceberg` on a **nested struct field** whose name contains a special character (e.g. `!`) raises `StructFieldNotFoundError`, even though the field genuinely exists.

Scoped precisely: a top-level column with the same special character in its name works fine (`pl.col("age!")`, no struct involved) - the crash is specific to `struct.field("age!")` on a nested field, which routes through a `col("<stat>").struct.field_by_name(...)` lookup during skip-batches evaluation.

```python
import polars as pl
from pyiceberg.catalog.sql import SqlCatalog
from pyiceberg.schema import Schema
from pyiceberg.types import NestedField, LongType, StructType

catalog = SqlCatalog(
    "bug", uri="sqlite:////tmp/repro/c.sqlite", warehouse="file:///tmp/repro/warehouse"
)
catalog.create_namespace("ns")
schema = Schema(
    NestedField(1, "id", LongType()),
    NestedField(2, "mydict", StructType(NestedField(3, "age!", LongType())), required=False),
)
tbl = catalog.create_table(("ns", "t"), schema)

pl.DataFrame(
    {"id": [1, 2], "mydict": [{"age!": 17}, {"age!": 42}]},
    schema={"id": pl.Int64, "mydict": pl.Struct({"age!": pl.Int64})},
).write_iceberg(tbl, mode="append")

pl.scan_iceberg(tbl).filter(
    pl.col("mydict").struct.field("age!") == 17
).select("id").collect()
```

```
polars.exceptions.StructFieldNotFoundError: age!

This error occurred in the following expression:
	col("mydict_nc").struct.field_by_name(age!)()
while evaluating this larger expression:
	[([([(col("mydict_nc").struct.field_by_name(age!)()) == (col("len"))]) | ([([(col("mydict_min").struct.field_by_name(age!)()) > (17)]) & (col("mydict_min").struct.field_by_name(age!)().alias("").is_not_null())])]) | ([([(col("mydict_max").struct.field_by_name(age!)()) < (17)]) & (col("mydict_max").struct.field_by_name(age!)().alias("").is_not_null())])]
```

Reproduces on current `main` (`66d85d5088`, `2.0.0-rc.2`). `age` (no special character) does not trigger this. Not specific to `!` - `.` triggers the same error.

### Ruled out (all confirmed correct in isolation)
* `pl_dtype_from_iceberg_field` for the field - correctly returns `Struct({'age!': Int64})`.
* `pl.repeat(None, ..., dtype=Struct({'age!': Int64}))`, both eager and lazy - correct.
* The exact DataFrame-construction pattern `IcebergColumnStatisticsLoader.finish()` uses (a shared `Expr` aliased twice into `_min`/`_max` in one `with_columns()` call) - reproduced standalone, correct; `.schema` on the result genuinely shows `Struct({'age!': Int64})`.
* A plain in-memory `df.lazy().filter(pl.col("s").struct.field("age!") == 1).collect()` - works fine.

So the field is present with the correct name at every point inspectable from Python, including the concrete `mydict_min`/`mydict_max`/`mydict_nc` statistics frame `predicate_to_pa`'s Iceberg path (`crates/polars-plan/src/plans/aexpr/predicates/skip_batches.rs`) builds via `struct_field_path`/`resolve_stat_target`. The failure is in **evaluating** the resulting skip-batches predicate against real per-batch data during a native Iceberg scan, not in constructing the predicate or the statistics frame's declared schema.

### Where to look next
Since ordinary in-memory struct-field evaluation works, the bug is likely specific to whatever executes the skip-batches predicate against real batches during a multi-file scan - probably in `polars-stream`'s multi-scan machinery, not `polars-plan`'s `skip_batches.rs` (which was inspected and looks structurally correct - it builds the same `IRStructFunction::FieldByName` node type regardless of field-name content).
