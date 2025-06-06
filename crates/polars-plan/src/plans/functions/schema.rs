#[cfg(feature = "pivot")]
use polars_core::utils::try_get_supertype;

use super::*;
use crate::constants::get_len_name;

impl FunctionIR {
    pub(crate) fn clear_cached_schema(&self) {
        use FunctionIR::*;
        // We will likely add more branches later
        #[allow(clippy::single_match)]
        match self {
            #[cfg(feature = "pivot")]
            Unpivot { schema, .. } => {
                let mut guard = schema.lock().unwrap();
                *guard = None;
            },
            RowIndex { schema, .. } | Explode { schema, .. } => {
                let mut guard = schema.lock().unwrap();
                *guard = None;
            },
            _ => {},
        }
    }

    pub(crate) fn schema<'a>(
        &self,
        input_schema: &'a SchemaRef,
    ) -> PolarsResult<Cow<'a, SchemaRef>> {
        use FunctionIR::*;
        match self {
            Opaque { schema, .. } => match schema {
                None => Ok(Cow::Borrowed(input_schema)),
                Some(schema_fn) => {
                    let output_schema = schema_fn.get_schema(input_schema)?;
                    Ok(Cow::Owned(output_schema))
                },
            },
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { schema, .. }) => Ok(schema
                .as_ref()
                .map(|schema| Cow::Owned(schema.clone()))
                .unwrap_or_else(|| Cow::Borrowed(input_schema))),
            FastCount { alias, .. } => {
                let mut schema: Schema = Schema::with_capacity(1);
                let name = alias.clone().unwrap_or_else(get_len_name);
                schema.insert_at_index(0, name, IDX_DTYPE)?;
                Ok(Cow::Owned(Arc::new(schema)))
            },
            Rechunk => Ok(Cow::Borrowed(input_schema)),
            Unnest { columns: _columns } => {
                #[cfg(feature = "dtype-struct")]
                {
                    let mut new_schema = Schema::with_capacity(input_schema.len() * 2);
                    for (name, dtype) in input_schema.iter() {
                        if _columns.iter().any(|item| item == name) {
                            match dtype {
                                DataType::Struct(flds) => {
                                    for fld in flds {
                                        new_schema
                                            .with_column(fld.name().clone(), fld.dtype().clone());
                                    }
                                },
                                DataType::Unknown(_) => {
                                    // pass through unknown
                                },
                                _ => {
                                    polars_bail!(
                                        SchemaMismatch: "expected struct dtype, got: `{}`", dtype
                                    );
                                },
                            }
                        } else {
                            new_schema.with_column(name.clone(), dtype.clone());
                        }
                    }

                    Ok(Cow::Owned(Arc::new(new_schema)))
                }
                #[cfg(not(feature = "dtype-struct"))]
                {
                    panic!("activate feature 'dtype-struct'")
                }
            },
            RowIndex { schema, name, .. } => Ok(Cow::Owned(row_index_schema(
                schema,
                input_schema,
                name.clone(),
            ))),
            Explode { schema, columns } => explode_schema(schema, input_schema, columns),
            #[cfg(feature = "pivot")]
            Unpivot { schema, args } => unpivot_schema(args, schema, input_schema),
        }
    }
}

fn row_index_schema(
    cached_schema: &CachedSchema,
    input_schema: &SchemaRef,
    name: PlSmallStr,
) -> SchemaRef {
    let mut guard = cached_schema.lock().unwrap();
    if let Some(schema) = &*guard {
        return schema.clone();
    }
    let mut schema = (**input_schema).clone();
    schema.insert_at_index(0, name, IDX_DTYPE).unwrap();
    let schema_ref = Arc::new(schema);
    *guard = Some(schema_ref.clone());
    schema_ref
}

fn explode_schema<'a>(
    cached_schema: &CachedSchema,
    schema: &'a Schema,
    columns: &[PlSmallStr],
) -> PolarsResult<Cow<'a, SchemaRef>> {
    let mut guard = cached_schema.lock().unwrap();
    if let Some(schema) = &*guard {
        return Ok(Cow::Owned(schema.clone()));
    }
    let mut schema = schema.clone();

    // columns to string
    columns.iter().try_for_each(|name| {
        match schema.try_get(name)? {
            DataType::List(inner) => {
                schema.with_column(name.clone(), inner.as_ref().clone());
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner, _) => {
                schema.with_column(name.clone(), inner.as_ref().clone());
            },
            _ => {},
        }

        PolarsResult::Ok(())
    })?;
    let schema = Arc::new(schema);
    *guard = Some(schema.clone());
    Ok(Cow::Owned(schema))
}

#[cfg(feature = "pivot")]
fn unpivot_schema<'a>(
    args: &UnpivotArgsIR,
    cached_schema: &CachedSchema,
    input_schema: &'a Schema,
) -> PolarsResult<Cow<'a, SchemaRef>> {
    let mut guard = cached_schema.lock().unwrap();
    if let Some(schema) = &*guard {
        return Ok(Cow::Owned(schema.clone()));
    }

    let mut new_schema = args
        .index
        .iter()
        .map(|id| Ok(Field::new(id.clone(), input_schema.try_get(id)?.clone())))
        .collect::<PolarsResult<Schema>>()?;
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
    if args.on.is_empty() {
        let id_vars = PlHashSet::from_iter(&args.index);
        for (name, dtype) in input_schema.iter() {
            if !id_vars.contains(name) {
                supertype = try_get_supertype(&supertype, dtype)?;
            }
        }
    } else {
        for name in &args.on {
            let dtype = input_schema.try_get(name)?;
            supertype = try_get_supertype(&supertype, dtype)?;
        }
    }
    new_schema.with_column(value_name, supertype);
    let schema = Arc::new(new_schema);
    *guard = Some(schema.clone());
    Ok(Cow::Owned(schema))
}
