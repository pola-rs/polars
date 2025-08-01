use polars_core::schema::iceberg::{IcebergSchema, IcebergSchemaRef};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_plan::dsl::{ColumnMapping, ExtraColumnsPolicy};

use crate::nodes::io_sources::multi_scan::components::errors::extra_column_err;

#[derive(Debug, Clone)]
pub enum ForbidExtraColumns {
    /// Full file schema in the IR.
    Plain(SchemaRef),
    /// Full iceberg file schema in the IR.
    Iceberg(IcebergSchemaRef),
}

impl ForbidExtraColumns {
    pub fn opt_new(
        extra_columns_policy: &ExtraColumnsPolicy,
        full_file_schema: &SchemaRef,
        column_mapping: Option<&ColumnMapping>,
    ) -> Option<Self> {
        if matches!(extra_columns_policy, ExtraColumnsPolicy::Ignore) {
            return None;
        }

        Some(match column_mapping {
            Some(ColumnMapping::Iceberg(schema)) => Self::Iceberg(schema.clone()),
            None => Self::Plain(full_file_schema.clone()),
        })
    }

    /// # Panics
    /// Panics if `self` is an `Iceberg` variant and `file_iceberg_schema` is `None`.
    pub fn check_file_schema(
        &self,
        file_schema: &Schema,
        file_iceberg_schema: Option<&IcebergSchema>,
    ) -> PolarsResult<()> {
        let Some(extra_column_name) = (match self {
            Self::Plain(schema) => file_schema.iter_names().find(|x| !schema.contains(x)),
            Self::Iceberg(schema) => file_iceberg_schema
                .unwrap()
                .values()
                .find_map(|x| (!schema.contains_key(&x.physical_id)).then_some(&x.name)),
        }) else {
            return Ok(());
        };

        Err(extra_column_err(extra_column_name))
    }
}
