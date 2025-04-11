use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::schema::{Schema, SchemaRef};
use polars_plan::plans::hive::HivePartitionsDf;

/// Returns the schema containing columns to project from the file.
///
/// Note: This is used during IR lowering.
///
/// # Returns
/// `projected_file_schema, full_file_schema`
pub fn resolve_projections(
    // Final output schema of the MultiScan. Includes e.g. row index, missing columns etc, and the
    // projection is applied.
    final_output_schema: &Schema,
    // Full schema inferred for the file
    file_schema: &SchemaRef,
    // Hive parts
    hive_parts: &mut Option<HivePartitionsDf>,

    // TODO: One day update IR conversion to avoid attaching these to the file schema :')
    row_index_name: Option<&str>,
    include_file_paths: Option<&str>,
) -> (SchemaRef, SchemaRef) {
    if let Some(hive_parts) = hive_parts.as_mut() {
        let projected_hive_parts: HivePartitionsDf = hive_parts
            .df()
            .get_columns()
            .iter()
            .filter(|c| final_output_schema.contains(c.name()))
            .cloned()
            .collect::<DataFrame>()
            .into();

        *hive_parts = projected_hive_parts;
    }

    let hive_schema = hive_parts.as_ref().map(|x| x.schema().as_ref());

    // We will assume that aside from hive, there are no duplicate column names with other column
    // adding functions (e.g. row index, include file paths).

    let projected_file_schema: Schema = file_schema
        .iter()
        .filter_map(|(name, dtype)| {
            let in_final = final_output_schema.contains(name);
            let in_hive = hive_schema.is_some_and(|x| x.contains(name));
            let is_row_index_col = row_index_name.is_some_and(|x| name == x);
            let is_file_path_col = include_file_paths.is_some_and(|x| name == x);
            (in_final && !(in_hive || is_file_path_col || is_row_index_col))
                .then(|| (name.clone(), dtype.clone()))
        })
        .collect();

    // Resolve full file schema.
    // TODO: Update IR conversion to avoid attaching row index / file path
    let mut full_file_schema = file_schema.clone();

    if row_index_name.is_some_and(|x| full_file_schema.contains(x)) {
        Arc::make_mut(&mut full_file_schema).shift_remove(row_index_name.unwrap());
    }

    if include_file_paths.is_some_and(|x| full_file_schema.contains(x)) {
        Arc::make_mut(&mut full_file_schema).shift_remove(include_file_paths.unwrap());
    }

    (projected_file_schema.into(), full_file_schema)
}
