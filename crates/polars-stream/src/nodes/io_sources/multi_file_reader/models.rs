//! Structures for holding data.
use std::collections::VecDeque;

use components::row_counter::RowCounter;
use components::row_deletions::ExternalFilterMask;
use polars_core::prelude::PlHashMap;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::slice_enum::Slice;

use crate::nodes::io_sources::multi_file_reader::components;
use crate::nodes::io_sources::multi_file_reader::reader_interface::FileReader;

/// Anything aside from reading columns from the file. E.g. row_index, slice, predicate etc.
///
/// Note that hive partition columns are tracked separately.
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

pub struct ResolvedSliceInfo {
    /// In the negative slice case this can be a non-zero starting position.
    pub scan_source_idx: usize,
    /// In the negative slice case this will hold a row index with the offset adjusted.
    pub row_index: Option<RowIndex>,
    /// Resolved positive slice.
    pub pre_slice: Option<Slice>,
    /// If we resolved a negative slice we keep the initialized readers here (with a limit). For
    /// Parquet this can save a duplicate metadata fetch/decode.
    ///
    /// This will be in-order - i.e. `pop_front()` corresponds to the next reader.
    ///
    /// `Option(scan_source_idx, Deque(file_reader, n_rows))`
    #[expect(clippy::type_complexity)]
    pub initialized_readers: Option<(usize, VecDeque<(Box<dyn FileReader>, RowCounter)>)>,
    pub row_deletions: PlHashMap<usize, ExternalFilterMask>,
}
