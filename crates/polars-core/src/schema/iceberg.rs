//! TODO
//!
//! This should ideally be moved to `polars-schema`, currently it cannot due to dependency on
//! `polars_core::DataType`.
use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::datatypes::{ArrowDataType, ArrowSchema, Field as ArrowField};
use polars_error::{PolarsResult, feature_gated, polars_bail, polars_err};
use polars_utils::aliases::{InitHashMaps, PlHashMap};
use polars_utils::pl_str::PlSmallStr;

use crate::prelude::{DataType, Field, PlIndexMap};

pub const LIST_ELEMENT_DEFAULT_ID: u32 = u32::MAX;

/// Maps Iceberg physical IDs to columns.
///
/// Note: This doesn't use `Schema<F>` as the keys are u32's.
#[derive(Debug, Clone, Eq, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct IcebergSchema(PlIndexMap<u32, IcebergColumn>);
pub type IcebergSchemaRef = Arc<IcebergSchema>;

impl IcebergSchema {
    /// Constructs a schema keyed by the physical ID stored in the arrow field metadata.
    pub fn from_arrow_schema(schema: &ArrowSchema) -> PolarsResult<Self> {
        Self::try_from_arrow_fields_iter(schema.iter_values())
    }

    pub fn try_from_arrow_fields_iter<'a, I>(iter: I) -> PolarsResult<Self>
    where
        I: IntoIterator<Item = &'a ArrowField>,
    {
        let iter = iter.into_iter();

        let mut out = PlIndexMap::with_capacity(iter.size_hint().0);

        for arrow_field in iter {
            let col: IcebergColumn = arrow_field_to_iceberg_column_rec(arrow_field, None)?;
            let existing = out.insert(col.physical_id, col);

            if let Some(existing) = existing {
                polars_bail!(
                    Duplicate:
                    "IcebergSchema: duplicate physical ID {:?}",
                    existing,
                )
            }
        }

        Ok(Self(out))
    }
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct IcebergColumn {
    /// Output name
    pub name: PlSmallStr,
    /// This is expected to map from 'PARQUET:field_id'
    pub physical_id: u32,
    pub type_: IcebergColumnType,
    /// Names of the field in data files without field IDs, from the table's name mapping
    /// (`schema.name-mapping.default`).
    pub mapped_names: Vec<PlSmallStr>,
}

#[derive(Debug, Clone, Eq, Hash, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum IcebergColumnType {
    Primitive {
        /// This must not be a nested data type.
        dtype: DataType,
    },
    List(Box<IcebergColumn>),
    /// (values, width)
    FixedSizeList(Box<IcebergColumn>, usize),
    /// (keys, values)
    Map(Box<IcebergColumn>, Box<IcebergColumn>),
    Struct(IcebergSchema),
}

impl IcebergColumnType {
    pub fn to_polars_dtype(&self) -> DataType {
        use IcebergColumnType::*;

        match self {
            Primitive { dtype } => dtype.clone(),
            List(inner) => DataType::List(Box::new(inner.type_.to_polars_dtype())),
            FixedSizeList(inner, width) => {
                feature_gated!("dtype-array", {
                    DataType::Array(Box::new(inner.type_.to_polars_dtype()), *width)
                })
            },
            Map(key, value) => feature_gated!("dtype-map", {
                DataType::Map(
                    Box::new(key.type_.to_polars_dtype()),
                    Box::new(value.type_.to_polars_dtype()),
                )
            }),
            Struct(fields) => feature_gated!("dtype-struct", {
                DataType::Struct(
                    fields
                        .values()
                        .map(|col| Field::new(col.name.clone(), col.type_.to_polars_dtype()))
                        .collect(),
                )
            }),
        }
    }

    pub fn is_nested(&self) -> bool {
        use IcebergColumnType::*;

        match self {
            List(_) | FixedSizeList(..) | Map(..) | Struct(_) => true,
            Primitive { .. } => false,
        }
    }
}

fn arrow_field_to_iceberg_column_rec(
    field: &ArrowField,
    field_id_override: Option<u32>,
) -> PolarsResult<IcebergColumn> {
    let physical_id: u32 = field_id_override.ok_or(Cow::Borrowed("")).or_else(|_| {
        field
            .metadata
            .as_deref()
            .ok_or(Cow::Borrowed("metadata was None"))
            .and_then(|md| {
                md.get(PARQUET_FIELD_ID_KEY)
                    .ok_or(Cow::Borrowed("key not found in metadata"))
            })
            .and_then(|x| {
                x.parse()
                    .map_err(|_| Cow::Owned(format!("could not parse value as u32: '{x}'")))
            })
            .map_err(|failed_reason: Cow<'_, str>| {
                polars_err!(
                    SchemaFieldNotFound:
                    "IcebergSchema: failed to load '{PARQUET_FIELD_ID_KEY}' for field {}: {}",
                    &field.name,
                    failed_reason,
                )
            })
    })?;

    // Prevent accidental re-use.
    #[expect(unused)]
    let field_id_override: ();

    use ArrowDataType as ADT;

    let name = field.name.clone();

    let list_column_type = |field: &ArrowField| -> PolarsResult<IcebergColumnType> {
        let field_id_override = field
            .metadata
            .as_ref()
            .is_none_or(|x| !x.contains_key(PARQUET_FIELD_ID_KEY))
            .then_some(LIST_ELEMENT_DEFAULT_ID);

        Ok(IcebergColumnType::List(Box::new(
            arrow_field_to_iceberg_column_rec(field, field_id_override)?,
        )))
    };

    let type_ = match &field.dtype {
        ADT::List(field) | ADT::LargeList(field) => list_column_type(field)?,

        ADT::Map(entries, _) => {
            #[cfg(feature = "dtype-map")]
            {
                // The `entries` field itself carries no physical ID - the IDs sit on the `key` and
                // `value` fields underneath it.
                let ADT::Struct(entry_fields) = &entries.dtype else {
                    polars_bail!(
                        ComputeError:
                        "IcebergSchema: expected struct under arrow map type, got: {:?}",
                        &entries.dtype,
                    )
                };

                let [key, value] = entry_fields.as_slice() else {
                    polars_bail!(
                        ComputeError:
                        "IcebergSchema: expected 2 fields under arrow map type, got: {}",
                        entry_fields.len(),
                    )
                };

                IcebergColumnType::Map(
                    Box::new(arrow_field_to_iceberg_column_rec(key, None)?),
                    Box::new(arrow_field_to_iceberg_column_rec(value, None)?),
                )
            }
            #[cfg(not(feature = "dtype-map"))]
            {
                // Without the `Map` dtype these columns load as `List(Struct { key, value })`.
                list_column_type(entries)?
            }
        },

        #[cfg(feature = "dtype-array")]
        ADT::FixedSizeList(field, width) => IcebergColumnType::FixedSizeList(
            Box::new(arrow_field_to_iceberg_column_rec(field, None)?),
            *width,
        ),

        #[cfg(feature = "dtype-struct")]
        ADT::Struct(fields) => {
            IcebergColumnType::Struct(IcebergSchema::try_from_arrow_fields_iter(fields)?)
        },

        dtype => {
            if let ADT::Dictionary(_key_type, value_type, _is_ordered) = dtype
                && !value_type.is_nested()
            {
                let mut new_field = ArrowField::new(name.clone(), dtype.clone(), true);
                if let Some(metadata) = field.metadata.as_ref() {
                    new_field = new_field.with_metadata((**metadata).clone());
                }
                let dtype = DataType::from_arrow_field(&new_field);

                IcebergColumnType::Primitive { dtype }
            } else if let ADT::Extension(ext_type) = dtype
                && let DataType::Binary = DataType::from_arrow_dtype(&ext_type.inner)
            {
                // Iceberg UUID type will hit this branch.
                IcebergColumnType::Primitive {
                    dtype: DataType::Binary,
                }
            } else if dtype.is_nested() {
                polars_bail!(
                    ComputeError:
                    "IcebergSchema: unsupported arrow type: {:?}",
                    dtype,
                )
            } else {
                let dtype =
                    DataType::from_arrow_field(&ArrowField::new(name.clone(), dtype.clone(), true));

                IcebergColumnType::Primitive { dtype }
            }
        },
    };

    let out = IcebergColumn {
        name,
        physical_id,
        type_,
        mapped_names: vec![],
    };

    Ok(out)
}

const PARQUET_FIELD_ID_KEY: &str = "PARQUET:field_id";

impl IcebergSchema {
    /// Sets the mapped names of the columns (and their nested fields) from the table's name
    /// mapping, keyed by field ID.
    pub fn apply_name_mapping(&mut self, names: &PlHashMap<u32, Vec<PlSmallStr>>) {
        fn apply_column(col: &mut IcebergColumn, names: &PlHashMap<u32, Vec<PlSmallStr>>) {
            if let Some(n) = names.get(&col.physical_id) {
                col.mapped_names = n.clone();
            }
            match &mut col.type_ {
                IcebergColumnType::Primitive { .. } => {},
                IcebergColumnType::List(inner) | IcebergColumnType::FixedSizeList(inner, _) => {
                    apply_column(inner, names)
                },
                IcebergColumnType::Map(key, value) => {
                    apply_column(key, names);
                    apply_column(value, names);
                },
                IcebergColumnType::Struct(fields) => fields.apply_name_mapping(names),
            }
        }

        for col in self.values_mut() {
            apply_column(col, names);
        }
    }

    /// The schema of a data file, as [`Self::from_arrow_schema`]. A file written without field
    /// IDs (e.g. imported with `add_files`) is read with the name mapping of `table`: its fields
    /// take the ID of the table field whose mapped names hold their name, as in Iceberg Java.
    /// Unmapped fields get IDs that match no table field.
    pub fn from_arrow_schema_with_name_mapping(
        schema: &ArrowSchema,
        table: &IcebergSchema,
    ) -> PolarsResult<Self> {
        let has_field_ids = schema.iter_values().any(|f| {
            f.metadata
                .as_ref()
                .is_some_and(|md| md.contains_key(PARQUET_FIELD_ID_KEY))
        });
        let has_name_mapping = table.values().any(|c| !c.mapped_names.is_empty());

        if has_field_ids || !has_name_mapping {
            return Self::from_arrow_schema(schema);
        }

        let fields = with_mapped_ids_of_level(schema.iter_values(), table);
        Self::try_from_arrow_fields_iter(fields.iter())
    }
}

/// The fields of one struct level with the field IDs of the `table` columns they are mapped to.
fn with_mapped_ids_of_level<'a>(
    fields: impl IntoIterator<Item = &'a ArrowField>,
    table: &IcebergSchema,
) -> Vec<ArrowField> {
    // IDs of unmapped fields, unique within the level; they keep their position, which struct
    // field selection relies on.
    let mut unmapped_id = LIST_ELEMENT_DEFAULT_ID;

    fields
        .into_iter()
        .map(|field| {
            match table
                .values()
                .find(|c| c.mapped_names.iter().any(|n| n == &field.name))
            {
                Some(column) => with_mapped_ids(field, column),
                None => {
                    unmapped_id -= 1;
                    with_field_id(field.clone(), unmapped_id)
                },
            }
        })
        .collect()
}

/// `field` with the field IDs of `column` and its nested fields. List elements and map keys and
/// values are matched by position.
fn with_mapped_ids(field: &ArrowField, column: &IcebergColumn) -> ArrowField {
    use ArrowDataType as ADT;
    use IcebergColumnType as ICT;

    let dtype = match (&field.dtype, &column.type_) {
        (ADT::Struct(fields), ICT::Struct(children)) => {
            ADT::Struct(with_mapped_ids_of_level(fields, children))
        },
        (ADT::List(inner), ICT::List(c)) => ADT::List(Box::new(with_mapped_ids(inner, c))),
        (ADT::LargeList(inner), ICT::List(c)) => {
            ADT::LargeList(Box::new(with_mapped_ids(inner, c)))
        },
        (ADT::FixedSizeList(inner, width), ICT::FixedSizeList(c, _)) => {
            ADT::FixedSizeList(Box::new(with_mapped_ids(inner, c)), *width)
        },
        (ADT::Map(entries, sorted), ICT::Map(key, value)) => match &entries.dtype {
            ADT::Struct(kv) if kv.len() == 2 => {
                let entries = ArrowField {
                    dtype: ADT::Struct(vec![
                        with_mapped_ids(&kv[0], key),
                        with_mapped_ids(&kv[1], value),
                    ]),
                    ..(**entries).clone()
                };
                ADT::Map(Box::new(entries), *sorted)
            },
            dtype => dtype.clone(),
        },
        // Other (e.g. mismatched) types are left for the schema checks.
        (dtype, _) => dtype.clone(),
    };

    with_field_id(
        ArrowField {
            dtype,
            ..field.clone()
        },
        column.physical_id,
    )
}

fn with_field_id(mut field: ArrowField, id: u32) -> ArrowField {
    let mut metadata = field.metadata.as_deref().cloned().unwrap_or_default();
    metadata.insert(
        PlSmallStr::from_static(PARQUET_FIELD_ID_KEY),
        id.to_string().into(),
    );
    field.metadata = Some(Arc::new(metadata));
    field
}

impl<T> FromIterator<T> for IcebergSchema
where
    PlIndexMap<u32, IcebergColumn>: FromIterator<T>,
{
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self(PlIndexMap::<u32, IcebergColumn>::from_iter(iter))
    }
}

impl std::hash::Hash for IcebergSchema {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        for col in self.values() {
            col.hash(state);
        }
    }
}

impl std::ops::Deref for IcebergSchema {
    type Target = PlIndexMap<u32, IcebergColumn>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl std::ops::DerefMut for IcebergSchema {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}
