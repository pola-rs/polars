//! Implementation of applying the operations during execution.
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::{AnyValue, DataType, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::ScanSources;
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::IdxSize;

use super::ExtraOperations;
use super::cast_columns::CastColumns;
use super::reorder_columns::ReorderColumns;
use super::row_index::materialize_row_index_checked;

/// Apply extra operations onto morsels originating from a reader. This should be initialized
/// per-reader (it contains e.g. file path).
#[derive(Debug)]
pub enum ApplyExtraOps {
    /// Intended to be initialized once, as we expect all morsels coming from a single reader to have
    /// the same schema. The initialized state can then be executed potentially in parallel by wrapping
    /// in Arc.
    Uninitialized {
        final_output_schema: SchemaRef,
        projected_file_schema: SchemaRef,
        extra_ops: ExtraOperations,
        scan_source_idx: usize,
        /// This here so that we can get the include file path name if needed.
        sources: ScanSources,
        hive_parts: Option<Arc<HivePartitionsDf>>,
    },

    Initialized {
        // Note: These fields are ordered according to when they (should be) applied.
        row_index: Option<RowIndex>,
        cast_columns: Option<CastColumns>,
        /// This will have include_file_paths, hive columns, missing columns.
        extra_columns: Vec<ScalarColumn>,
        predicate: Option<ScanIOPredicate>,
        reorder: ReorderColumns,
    },

    /// No-op.
    Noop,
}

impl ApplyExtraOps {
    pub fn initialize(
        self,
        // Schema of the incoming morsels.
        incoming_schema: &SchemaRef,
    ) -> PolarsResult<Self> {
        use ApplyExtraOps::*;
        match self {
            Initialized { .. } => panic!("ApplyExtraOps already initialized"),
            Noop => Ok(Noop),

            Uninitialized {
                final_output_schema,
                projected_file_schema,
                extra_ops:
                    ExtraOperations {
                        row_index,
                        pre_slice,
                        cast_columns,
                        predicate,
                        missing_columns,
                        include_file_paths,
                    },
                scan_source_idx,
                sources,
                hive_parts,
            } => {
                todo!()
            },
        }
    }

    /// # Panics
    /// Panics if `self` is `Uninitialized`
    pub fn apply_to_df(
        &self,
        df: &mut DataFrame,
        current_row_position: IdxSize,
    ) -> PolarsResult<()> {
        let Self::Initialized {
            row_index,
            cast_columns,
            extra_columns,
            predicate,
            reorder,
        } = ({
            use ApplyExtraOps::*;

            match self {
                Noop => return Ok(()),
                Uninitialized { .. } => panic!("ApplyExtraOps not initialized"),
                Initialized { .. } => self,
            }
        })
        else {
            unreachable!();
        };

        todo!()
    }
}
