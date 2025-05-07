//! Implementation of applying the operations during execution.
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::frame::column::ScalarColumn;
use polars_core::prelude::{AnyValue, Column, DataType, IntoColumn};
use polars_core::scalar::Scalar;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::ScanSource;
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::IdxSize;
use polars_utils::slice_enum::Slice;

use super::ExtraOperations;
use super::cast_columns::CastColumns;
use super::reorder_columns::ReorderColumns;
use crate::nodes::io_sources::multi_file_reader::extra_ops::missing_columns::initialize_missing_columns_policy;

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
        /// This here so that we can get the include file path name if needed.
        scan_source: ScanSource,
        scan_source_idx: usize,
        hive_parts: Option<Arc<HivePartitionsDf>>,
    },

    Initialized {
        // Note: These fields are ordered according to when they (should be) applied.
        row_index: Option<RowIndex>,
        pre_slice: Option<Slice>,
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
                        cast_columns_policy,
                        missing_columns_policy,
                        include_file_paths,
                        predicate,
                    },
                scan_source,
                scan_source_idx,
                hive_parts,
            } => {
                // Negative slice should have been resolved earlier.
                if let Some(Slice::Negative { .. }) = pre_slice {
                    panic!("impl error: negative pre_slice at post")
                }

                let cast_columns = CastColumns::try_init_from_policy(
                    &cast_columns_policy,
                    &final_output_schema,
                    incoming_schema,
                )?;

                let n_expected_extra_columns = final_output_schema.len()
                    - incoming_schema.len()
                    - row_index.is_some() as usize;

                let mut extra_columns: Vec<ScalarColumn> =
                    Vec::with_capacity(n_expected_extra_columns);

                initialize_missing_columns_policy(
                    &missing_columns_policy,
                    &projected_file_schema,
                    incoming_schema,
                    &mut extra_columns,
                )?;

                if let Some(hive_parts) = hive_parts {
                    extra_columns.extend(hive_parts.df().get_columns().iter().map(|c| {
                        c.new_from_index(scan_source_idx, 1)
                            .as_scalar_column()
                            .unwrap()
                            .clone()
                    }))
                }

                if let Some(file_path_col) = include_file_paths {
                    extra_columns.push(ScalarColumn::new(
                        file_path_col,
                        Scalar::new(
                            DataType::String,
                            AnyValue::StringOwned(
                                scan_source
                                    .as_scan_source_ref()
                                    .to_include_path_name()
                                    .into(),
                            ),
                        ),
                        1,
                    ))
                }

                debug_assert_eq!(extra_columns.len(), n_expected_extra_columns);

                let mut slf = Self::Initialized {
                    row_index,
                    pre_slice,
                    cast_columns,
                    extra_columns,
                    predicate,
                    // Initialized below
                    reorder: ReorderColumns::Passthrough,
                };

                let schema_before_reorder = if incoming_schema.len() == final_output_schema.len() {
                    // Incoming schema already has all of the columns, either because no extra columns were needed, or
                    // the extra columns were attached by the reader (which is just Parquet when it has a predicate).
                    incoming_schema.clone()
                } else {
                    // We use a trick to determine our schema state before reordering by applying onto an empty DataFrame.
                    // This is much less error prone compared determining it separately.
                    let mut df = DataFrame::empty_with_schema(incoming_schema);
                    slf.apply_to_df(&mut df, IdxSize::MAX)?;
                    df.schema().clone()
                };

                if cfg!(debug_assertions)
                    && schema_before_reorder.len() != final_output_schema.len()
                {
                    assert_eq!(schema_before_reorder, final_output_schema);
                    unreachable!()
                }

                let initialized_reorder =
                    ReorderColumns::initialize(&final_output_schema, &schema_before_reorder);

                let Self::Initialized { reorder, .. } = &mut slf else {
                    unreachable!()
                };

                *reorder = initialized_reorder;

                // Return a `Noop` if our initialized state does not have any operations. Downstream
                // can see the `Noop` and avoid running through an extra distributor pipeline.
                let slf = match slf {
                    Initialized {
                        row_index: None,
                        pre_slice: None,
                        cast_columns: None,
                        extra_columns,
                        predicate: None,
                        reorder: ReorderColumns::Passthrough,
                    } if extra_columns.is_empty() => Self::Noop,

                    Initialized { .. } => slf,

                    _ => unreachable!(),
                };

                Ok(slf)
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
            pre_slice,
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

        if let Some(ri) = row_index {
            unsafe {
                df.with_column_unchecked(Column::new_row_index(
                    ri.name.clone(),
                    ri.offset.saturating_add(current_row_position),
                    df.height(),
                )?)
            };
        }

        if let Some(pre_slice) = pre_slice.clone() {
            let Slice::Positive { offset, len } = pre_slice
                .offsetted(usize::try_from(current_row_position).unwrap())
                .restrict_to_bounds(df.height())
            else {
                unreachable!()
            };

            *df = df.slice(i64::try_from(offset).unwrap(), len)
        }

        if let Some(cast_columns) = cast_columns {
            cast_columns.apply_cast(df)?;
        }

        if !extra_columns.is_empty() {
            df.clear_schema();
            let h = df.height();
            let cols = unsafe { df.get_columns_mut() };
            cols.extend(extra_columns.iter().map(|c| c.resize(h).into_column()));
        }

        if let Some(predicate) = predicate {
            let mask = predicate.predicate.evaluate_io(df)?;
            *df = df._filter_seq(mask.bool().expect("predicate not boolean"))?;
        }

        reorder.reorder_columns(df);

        df.clear_schema();

        Ok(())
    }
}
