use super::*;

impl FunctionNode {
    pub(crate) fn clear_cached_schema(&self) {
        use FunctionNode::*;
        // We will likely add more branches later
        #[allow(clippy::single_match)]
        match self {
            RowIndex { schema, .. } => {
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
        use FunctionNode::*;
        match self {
            Opaque { schema, .. } => match schema {
                None => Ok(Cow::Borrowed(input_schema)),
                Some(schema_fn) => {
                    let output_schema = schema_fn.get_schema(input_schema)?;
                    Ok(Cow::Owned(output_schema))
                },
            },
            #[cfg(feature = "python")]
            OpaquePython { schema, .. } => Ok(schema
                .as_ref()
                .map(|schema| Cow::Owned(schema.clone()))
                .unwrap_or_else(|| Cow::Borrowed(input_schema))),
            Pipeline { schema, .. } => Ok(Cow::Owned(schema.clone())),
            DropNulls { .. } => Ok(Cow::Borrowed(input_schema)),
            Count { alias, .. } => {
                let mut schema: Schema = Schema::with_capacity(1);
                let name = SmartString::from(
                    alias
                        .as_ref()
                        .map(|alias| alias.as_ref())
                        .unwrap_or(crate::constants::LEN),
                );
                schema.insert_at_index(0, name, IDX_DTYPE)?;
                Ok(Cow::Owned(Arc::new(schema)))
            },
            Rechunk => Ok(Cow::Borrowed(input_schema)),
            Unnest { columns: _columns } => {
                #[cfg(feature = "dtype-struct")]
                {
                    let mut new_schema = Schema::with_capacity(input_schema.len() * 2);
                    for (name, dtype) in input_schema.iter() {
                        if _columns.iter().any(|item| item.as_ref() == name.as_str()) {
                            match dtype {
                                DataType::Struct(flds) => {
                                    for fld in flds {
                                        new_schema.with_column(
                                            fld.name().clone(),
                                            fld.data_type().clone(),
                                        );
                                    }
                                },
                                DataType::Unknown => {
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
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => Ok(Cow::Borrowed(input_schema)),
            Rename { existing, new, .. } => rename::rename_schema(input_schema, existing, new),
            RowIndex { schema, name, .. } => {
                Ok(Cow::Owned(row_index_schema(schema, input_schema, name)))
            },
            Explode { schema, .. } | Melt { schema, .. } => Ok(Cow::Owned(schema.clone())),
        }
    }
}

fn row_index_schema(
    cached_schema: &CachedSchema,
    input_schema: &SchemaRef,
    name: &str,
) -> SchemaRef {
    let mut guard = cached_schema.lock().unwrap();
    if let Some(schema) = &*guard {
        return schema.clone();
    }
    let mut schema = (**input_schema).clone();
    schema.insert_at_index(0, name.into(), IDX_DTYPE).unwrap();
    let schema_ref = Arc::new(schema);
    *guard = Some(schema_ref.clone());
    schema_ref
}

// We don't use an `Arc<Mutex>` because caches should live in different query plans.
// For that reason we have a specialized deep clone.
#[derive(Default)]
pub struct CachedSchema(Mutex<Option<SchemaRef>>);

impl AsRef<Mutex<Option<SchemaRef>>> for CachedSchema {
    fn as_ref(&self) -> &Mutex<Option<SchemaRef>> {
        &self.0
    }
}

impl Deref for CachedSchema {
    type Target = Mutex<Option<SchemaRef>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Clone for CachedSchema {
    fn clone(&self) -> Self {
        let inner = self.0.lock().unwrap();
        Self(Mutex::new(inner.clone()))
    }
}
