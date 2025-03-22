//! Extra operations applied during reads.
pub mod apply;
pub mod cast_columns;
pub mod missing_columns;
pub mod reorder_columns;

use cast_columns::CastColumnsPolicy;
use missing_columns::MissingColumnsPolicy;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_bail};
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
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

/// TODO: Eventually move this enum to polars-plan
#[derive(Clone)]
pub enum SchemaNamesMatchPolicy {
    /// * If the schema lengths match, ensure that all columns match in the same order
    /// * Otherwise, ensure that there are no extra columns in the incoming schema that
    ///   cannot be found in the target schema.
    ///   * Ignores if the incoming schema is missing columns, this is handled by a separate module.
    RequireOrderedExact,
}

impl SchemaNamesMatchPolicy {
    pub fn apply_policy(
        &self,
        target_schema: SchemaRef,
        incoming_schema: SchemaRef,
    ) -> PolarsResult<()> {
        use SchemaNamesMatchPolicy::*;
        match self {
            RequireOrderedExact => {
                if incoming_schema.len() == target_schema.len() {
                    if incoming_schema
                        .iter_names()
                        .zip(target_schema.iter_names())
                        .any(|(l, r)| l != r)
                    {
                        polars_bail!(
                            SchemaMismatch:
                            "column name ordering of file differs: {:?} != {:?}",
                            incoming_schema.iter_names().collect::<Vec<_>>(), target_schema.iter_names().collect::<Vec<_>>()
                        )
                    }
                } else if let Some(extra_col) = incoming_schema
                    .iter_names()
                    .find(|x| !target_schema.contains(x))
                {
                    polars_bail!(
                        SchemaMismatch:
                        "extra column in file outside of expected schema: {}",
                        extra_col,
                    )
                }

                Ok(())
            },
        }
    }
}
