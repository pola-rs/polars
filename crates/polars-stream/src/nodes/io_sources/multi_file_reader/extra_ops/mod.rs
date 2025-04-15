//! Extra operations applied during reads.
pub mod apply;
pub mod cast_columns;
pub mod missing_columns;
pub mod reorder_columns;

use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_bail};
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{CastColumnsPolicy, ExtraColumnsPolicy, MissingColumnsPolicy};
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
    pub pre_slice: Option<Slice>,
    pub cast_columns_policy: CastColumnsPolicy,
    pub missing_columns_policy: MissingColumnsPolicy,
    pub include_file_paths: Option<PlSmallStr>,
    pub predicate: Option<ScanIOPredicate>,
}

impl ExtraOperations {
    pub fn has_row_index_or_slice(&self) -> bool {
        self.row_index.is_some() || self.pre_slice.is_some()
    }
}

pub fn apply_extra_columns_policy(
    policy: &ExtraColumnsPolicy,
    target_schema: SchemaRef,
    incoming_schema: SchemaRef,
) -> PolarsResult<()> {
    use ExtraColumnsPolicy::*;
    match policy {
        Ignore => {},

        Raise => {
            if let Some(extra_col) = incoming_schema
                .iter_names()
                .find(|x| !target_schema.contains(x))
            {
                polars_bail!(
                    SchemaMismatch:
                    "extra column in file outside of expected schema: {}",
                    extra_col,
                )
            }
        },
    }

    Ok(())
}
