use polars_core::frame::DataFrame;
use polars_core::schema::{Schema, SchemaRef};
use polars_plan::plans::hive::HivePartitionsDf;

/// Returns the schema containing columns to project from the file.
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
) -> SchemaRef {
    // TODO: Does hive df already have projections applied earlier?
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

    let skip = row_index_name.map_or(0, |x| match file_schema.index_of(x) {
        Some(0) => 1,
        // If it exists it should always be the first column.
        Some(_) => panic!(),
        None => 0,
    });

    let projected_file_schema: Schema = file_schema
        .iter()
        .skip(skip)
        .filter_map(|(name, dtype)| {
            let in_final = final_output_schema.contains(name);
            let in_hive = hive_schema.is_some_and(|x| x.contains(name));
            let is_file_path_col = include_file_paths.is_some_and(|x| name == x);

            (in_final && !in_hive && !is_file_path_col).then(|| (name.clone(), dtype.clone()))
        })
        .collect();

    projected_file_schema.into()
}
