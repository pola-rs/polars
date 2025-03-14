//! Extra operations applied during reads.
pub mod apply;
pub mod cast_columns;
pub mod missing_columns;
pub mod pipeline;
pub mod reorder_columns;
pub mod row_index;

use cast_columns::CastColumnsPolicy;
use missing_columns::MissingColumnsPolicy;
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
    pub cast_columns: Option<CastColumnsPolicy>,
    pub predicate: Option<ScanIOPredicate>,
    pub missing_columns: Option<MissingColumnsPolicy>,
    pub include_file_paths: Option<PlSmallStr>,
}

impl ExtraOperations {
    pub fn has_row_index_or_slice(&self) -> bool {
        self.row_index.is_some() || self.pre_slice.is_some()
    }
}
