use polars_core::utils::try_get_supertype;

use super::*;

// Has functions that create schema's for both the `LogicalPlan` and the `AlogicalPlan` builders.

pub(super) fn explode_schema(schema: &mut Schema, columns: &[Arc<str>]) -> PolarsResult<()> {
    // columns to string
    columns.iter().try_for_each(|name| {
        if let DataType::List(inner) = schema.try_get(name)? {
            let inner = *inner.clone();
            schema.with_column(name.as_ref().into(), inner);
        };
        Ok(())
    })
}

pub(super) fn det_melt_schema(args: &MeltArgs, input_schema: &Schema) -> SchemaRef {
    let mut new_schema = args
        .id_vars
        .iter()
        .map(|id| Field::new(id, input_schema.get(id).unwrap().clone()))
        .collect::<Schema>();
    let variable_name = args
        .variable_name
        .as_ref()
        .cloned()
        .unwrap_or_else(|| "variable".into());
    let value_name = args
        .value_name
        .as_ref()
        .cloned()
        .unwrap_or_else(|| "value".into());

    new_schema.with_column(variable_name, DataType::String);

    // We need to determine the supertype of all value columns.
    let mut supertype = DataType::Null;

    // take all columns that are not in `id_vars` as `value_var`
    if args.value_vars.is_empty() {
        let id_vars = PlHashSet::from_iter(&args.id_vars);
        for (name, dtype) in input_schema.iter() {
            if !id_vars.contains(name) {
                supertype = try_get_supertype(&supertype, dtype).unwrap();
            }
        }
    } else {
        for name in &args.value_vars {
            let dtype = input_schema.get(name).unwrap();
            supertype = try_get_supertype(&supertype, dtype).unwrap();
        }
    }
    new_schema.with_column(value_name, supertype);
    Arc::new(new_schema)
}

pub(super) fn row_index_schema(schema: &mut Schema, name: &str) {
    schema.insert_at_index(0, name.into(), IDX_DTYPE).unwrap();
}
