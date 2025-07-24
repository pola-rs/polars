//! TODO
//!
//! This should ideally be moved to `polars-schema`, currently it cannot due to dependency on
//! `polars_core::DataType`.
use std::borrow::Cow;
use std::sync::Arc;

use arrow::datatypes::{ArrowDataType, ArrowSchema, Field as ArrowField};
use polars_error::{PolarsResult, feature_gated, polars_bail, polars_err};
use polars_utils::aliases::InitHashMaps;
use polars_utils::pl_str::PlSmallStr;

use crate::prelude::{DataType, Field, PlIndexMap};

/// Maps Iceberg physical IDs to columns.
///
/// Note: This doesn't use `Schema<D>` as the keys are u32's.
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
        let size_hint = iter.size_hint();

        let mut out = PlIndexMap::with_capacity(size_hint.1.unwrap_or(size_hint.0));

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
            List(_) | FixedSizeList(..) | Struct(_) => true,
            Primitive { .. } => false,
        }
    }
}

fn arrow_field_to_iceberg_column_rec(
    field: &ArrowField,
    field_id_override: Option<u32>,
) -> PolarsResult<IcebergColumn> {
    const PARQUET_FIELD_ID_KEY: &str = "PARQUET:field_id";
    const MAP_DEFAULT_ID: u32 = u32::MAX; // u32::MAX

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

    let type_ = match &field.dtype {
        ADT::List(field) | ADT::LargeList(field) | ADT::Map(field, _) => {
            // The `field` directly under the `Map` type does not contain a physical ID, so we add one in here.
            // Note that this branch also catches `(Large)List` as the `Map` columns get loaded as that type
            // from Parquet (currently unsure if this is intended).
            let field_id_override = field
                .metadata
                .as_ref()
                .is_none_or(|x| !x.contains_key(PARQUET_FIELD_ID_KEY))
                .then_some(MAP_DEFAULT_ID);

            IcebergColumnType::List(Box::new(arrow_field_to_iceberg_column_rec(
                field,
                field_id_override,
            )?))
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
            if dtype.is_nested() {
                polars_bail!(
                    ComputeError:
                    "IcebergSchema: unsupported arrow type: {:?}",
                    dtype,
                )
            }

            let dtype =
                DataType::from_arrow_field(&ArrowField::new(name.clone(), dtype.clone(), true));

            IcebergColumnType::Primitive { dtype }
        },
    };

    let out = IcebergColumn {
        name,
        physical_id,
        type_,
    };

    Ok(out)
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
