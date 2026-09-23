use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::*;
use polars_ops::prelude::{ListNameSpaceImpl, map_contains_key, map_get};
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRMapFunction;
use polars_utils::broadcast::broadcast_len;

pub fn function_expr_to_udf(func: IRMapFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRMapFunction::*;
    match func {
        Entries => map!(map_entries),
        Keys => map!(map_keys),
        Values => map!(map_values),
        Length => map!(map_len),
        ContainsKey { needle_cast } => map_as_slice!(contains_key, needle_cast.as_ref()),
        Get { needle_cast } => map_as_slice!(get, needle_cast.as_ref()),
    }
}

fn map_entries(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| Ok(s.map()?.live_storage().into_owned().into_series()))
}

fn map_keys(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| Ok(s.map()?.key_lists().into_series()))
}

fn map_values(c: &Column) -> PolarsResult<Column> {
    c.try_apply_unary_elementwise(|s| Ok(s.map()?.value_lists().into_series()))
}

fn map_len(c: &Column) -> PolarsResult<Column> {
    // `lst_lengths` preserves null rows, so compaction is unnecessary.
    c.try_apply_unary_elementwise(|s| Ok(s.map()?.storage().list()?.lst_lengths().into_series()))
}

fn contains_key(args: &mut [Column], needle_cast: Option<&DataType>) -> PolarsResult<Column> {
    super::membership::cast_map_key(&mut args[1], needle_cast)?;
    with_map_and_key(args, |map, key| {
        map_contains_key(map, key).map(IntoColumn::into_column)
    })
}

fn get(args: &mut [Column], needle_cast: Option<&DataType>) -> PolarsResult<Column> {
    super::membership::cast_map_key(&mut args[1], needle_cast)?;
    with_map_and_key(args, |map, key| {
        map_get(map, key).map(IntoColumn::into_column)
    })
}

/// Pass scalar inputs to the lookup kernel without expanding them.
fn with_map_and_key(
    args: &[Column],
    f: impl FnOnce(&MapChunked, &Series) -> PolarsResult<Column>,
) -> PolarsResult<Column> {
    let (map, key) = (&args[0], &args[1]);
    let (map_s, key_s) = (
        map.as_materialized_series_maintain_scalar(),
        key.as_materialized_series_maintain_scalar(),
    );
    f(map_s.map()?, &key_s)?.broadcast_owned_to(broadcast_len([map, key])?)
}
