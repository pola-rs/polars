### Description (updated - root cause found)

**Not actually about special characters.** The real defect: when `missing_struct_fields="insert"` (or any struct schema-evolution reconciliation) widens a struct column to add a field the physical Parquet file doesn't have, the widening is applied to the skip-batches `min`/`max` statistics columns but **not** to the `null_count` (`_nc`) statistics column. Filtering on the inserted field then crashes with `StructFieldNotFoundError` when the skip-batches predicate looks it up in `_nc`.

The original special-character symptom (below) was a coincidental second trigger of the same underlying pattern (two independently-computed schemas for `min`/`max` vs `null_count`, which can silently drift apart), not the cause itself.

### Confirmed root cause

**File:** `crates/polars-stream/src/nodes/io_sources/parquet/statistics.rs`, function `static_skip_mask`:

```rust
let mut statistics = load_parquet_column_statistics(&metadata, row_group_slice.clone(), projection)?;

// Note: Order is important here. We re-use the transform for the output column, meaning
// that it may set the column name.
statistics.min = projection.apply_transform(statistics.min)?;
statistics.max = projection.apply_transform(statistics.max)?;
//              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ never called on statistics.null_count

let statistics = statistics.with_base_column_name(c);
columns.extend([statistics.min, statistics.max, statistics.null_count]);
```

`projection.apply_transform(...)` is the step that reconciles a column against schema evolution, including `missing_struct_fields="insert"`. It's called on `min` and `max`. It is **never called on `null_count`**, which passes straight through from `load_parquet_column_statistics` untouched.

**Why the fix isn't just "call `apply_transform` on `null_count` too":** `apply_transform`'s `ColumnSelector` is built against `output_dtype` - the *real* column's dtype (e.g. `Struct({a: Int64, b: Int64})`). Reusing it verbatim on `null_count` (whose leaves should be `UInt32`/`IDX_DTYPE`, not `Int64`) would insert the new field with the wrong dtype, or try to cast the existing leaf's dtype and corrupt real null-counts.

### Minimal reproduction (pure Parquet, no Iceberg)

```python
import tempfile
from pathlib import Path
import polars as pl

path = Path(tempfile.mkdtemp()) / "data.parquet"
pl.DataFrame({"s": [{"a": 1}, {"a": 2}]}).write_parquet(path)

pl.scan_parquet(
    path,
    schema={"s": pl.Struct({"a": pl.Int64, "b": pl.Int64})},
    cast_options=pl.ScanCastOptions(missing_struct_fields="insert"),
    use_statistics=True,
).filter(pl.col("s").struct.field("b").is_null()).collect()
```

```
polars.exceptions.StructFieldNotFoundError: b

This error occurred in the following expression:
	col("s_nc").struct.field_by_name(b)()
while evaluating this larger expression:
	[(col("s_nc").struct.field_by_name(b)()) == (0)]
```

`use_statistics=False` correctly returns both rows with `b=null`.

**Verified empirically** with debug instrumentation in `static_skip_mask`:

```
before transform: min={'a': Int64} max={'a': Int64} nc={'a': UInt32}
after transform:  min={'a': Int64, 'b': Int64} max={'a': Int64, 'b': Int64} nc={'a': UInt32}
CRASHED: StructFieldNotFoundError b
```

`min`/`max` correctly widen to include `b`; `nc` is untouched.

### Step-by-step fix plan

1. **Don't reuse `apply_transform` verbatim for `null_count`** - its `ColumnSelector` targets the real column's dtype, not the null-count-shaped dtype (`null_count_dtype()` maps every leaf to `IDX_DTYPE`).
2. **Two viable approaches:**
   - **(a)** Parametrize `apply_transform` to accept an optional per-leaf dtype-mapping function, so the same insertion structure (which fields, at which positions) can be reused with `null_count_dtype()` substituted for the target dtype. Cleaner long-term, more invasive.
   - **(b) (recommended starting point)** Add a standalone function `widen_null_count_struct(null_count: Column, target_dtype: &DataType) -> PolarsResult<Column>` near `StatisticsColumns` in `statistics.rs` that widens the narrow `null_count` struct to `null_count_dtype(output_dtype)`, inserting missing leaves as all-null `UInt32`. Self-contained, doesn't touch the `ColumnSelector`/transform machinery, lower risk to review/test in isolation.
3. **Wire it in**, in `static_skip_mask`, alongside the existing `min`/`max` transform calls:
   ```rust
   statistics.null_count = widen_null_count_struct(statistics.null_count, &null_count_dtype(&output_dtype))?;
   ```
   (`output_dtype` already exists via `ArrowFieldProjection::output_dtype()`.)
4. **Test**: the minimal repro above as a regression test - `use_statistics=True` vs `False` must return identical, correct results.
5. **Check nested-nested case**: a struct-in-a-struct with a missing leaf several levels down (`build_struct_statistics_arrays` recurses), so the widening fix should be recursive too.
6. **Re-verify the original Iceberg "age!" case separately** afterward - it goes through a different code path (`IcebergColumnStatisticsLoader.finish()` in Python, computing `null_count_dtype()` independently from `min`/`max`'s dtype). Check whether fixing the Parquet-level bug incidentally fixes it, or whether it needs its own analogous fix.

### Original special-character repro (secondary trigger, same underlying pattern)

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

Reproduces on current `main` (`66d85d5088`, `2.0.0-rc.2`). `age` (no special character) does not trigger this; `.` triggers the same error as `!`. Scoped precisely: a top-level column with a special character in its name works fine - the crash is specific to nested struct field access.

**Elimination steps confirmed correct in isolation** (all before finding the real root cause above): `pl_dtype_from_iceberg_field`, `pl.repeat(None, ..., dtype=...)` (eager and lazy), the exact same-`Expr`-aliased-twice DataFrame-construction pattern `IcebergColumnStatisticsLoader.finish()` uses, and a plain in-memory `df.lazy().filter(pl.col("s").struct.field("age!") == 1).collect()`.
