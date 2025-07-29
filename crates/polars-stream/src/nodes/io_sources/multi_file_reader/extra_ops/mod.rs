//! Extra operations applied during reads.
pub mod apply;
pub mod column_selector;

use polars_core::schema::iceberg::{IcebergSchema, IcebergSchemaRef};
use polars_core::schema::{Schema, SchemaRef};
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{ColumnMapping, ExtraColumnsPolicy};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;

/// Anything aside from reading columns from the file. E.g. row_index, slice, predicate etc.
///
/// Note that hive partition columns are tracked separately.
///
/// This struct is mainly used as a data model / IR.
#[derive(Debug, Default, Clone)]
pub struct ExtraOperations {
    // Note: These fields are ordered according to when they (should be) applied.
    pub row_index: Option<RowIndex>,
    /// Index of the row index column in the final output.
    pub row_index_col_idx: usize,
    pub pre_slice: Option<Slice>,
    pub include_file_paths: Option<PlSmallStr>,
    /// Index of the file path column in the final output.
    pub file_path_col_idx: usize,
    pub predicate: Option<ScanIOPredicate>,
}

impl ExtraOperations {
    pub fn has_row_index_or_slice(&self) -> bool {
        self.row_index.is_some() || self.pre_slice.is_some()
    }
}

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

pub fn missing_column_err(missing_column_name: &str) -> PolarsError {
    polars_err!(
        ColumnNotFound:
        "did not find column {}, consider passing `missing_columns='insert'`",
        missing_column_name,
    )
}

pub fn extra_column_err(extra_column_name: &str) -> PolarsError {
    polars_err!(
        SchemaMismatch:
        "extra column in file outside of expected schema: {}, \
        hint: specify this column in the schema, or pass \
        extra_columns='ignore' in scan options",
        extra_column_name,
    )
}
