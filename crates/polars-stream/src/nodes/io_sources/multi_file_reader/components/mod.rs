//! Extra operations applied during reads.
pub mod apply_extra_ops;
pub mod column_selector;
pub mod errors;
pub mod forbid_extra_columns;
pub mod projection_builder;
pub mod row_deletions;
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
