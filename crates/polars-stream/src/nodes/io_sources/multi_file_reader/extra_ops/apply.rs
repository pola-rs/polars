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
use polars_utils::slice_enum::Slice;

use super::ExtraOperations;
use super::cast_columns::CastColumns;
use super::reorder_columns::ReorderColumns;
use crate::nodes::io_sources::multi_file_reader::extra_ops::missing_columns::initialize_missing_columns_policy;
use crate::nodes::io_sources::multi_file_reader::initialization::deletion_files::ExternalFilterMask;
use crate::nodes::io_sources::multi_file_reader::row_counter::RowCounter;

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
        /// E.g. Iceberg deletion files.
        external_filter_mask: Option<ExternalFilterMask>,
    },

    /// Note: These fields are ordered according to the order in which they are applied.
    Initialized {
        /// Physical - i.e. applied before `external_filter_mask`. This is calculated in `initialize()` if needed.
        physical_pre_slice: Option<Slice>,
        external_filter_mask: Option<ExternalFilterMask>,
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
    pub fn variant_name(&self) -> &'static str {
        match self {
            ApplyExtraOps::Uninitialized { .. } => "Uninitialized",
            ApplyExtraOps::Initialized { .. } => "Initialized",
            ApplyExtraOps::Noop => "Noop",
        }
    }

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
                external_filter_mask,
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
                    physical_pre_slice: pre_slice,
                    external_filter_mask,
                    row_index,
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
                    slf.apply_to_df(&mut df, RowCounter::MAX)?;
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
                        physical_pre_slice: None,
                        external_filter_mask: None,
                        row_index: None,
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
        // Row position of this morsel relative to the start of the current file.
        current_row_position: RowCounter,
    ) -> PolarsResult<()> {
        let Self::Initialized {
            physical_pre_slice,
            external_filter_mask,
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

        // Note, this counts physical rows
        let mut local_slice_offset: usize = 0;

        if let Some(physical_pre_slice) = physical_pre_slice.clone() {
            let Slice::Positive { offset, len } = physical_pre_slice
                .offsetted(current_row_position.num_physical_rows())
                .restrict_to_bounds(df.height())
            else {
                unreachable!()
            };

            local_slice_offset = offset;

            *df = df.slice(i64::try_from(offset).unwrap(), len)
        }

        if let Some(external_filter_mask) = external_filter_mask {
            let offset = current_row_position
                .num_physical_rows()
                .saturating_add(local_slice_offset);

            let Slice::Positive { offset, len } = Slice::Positive {
                offset,
                len: df.height(),
            }
            .restrict_to_bounds(external_filter_mask.len()) else {
                unreachable!()
            };

            let local_filter_mask = external_filter_mask.slice(offset, len);
            local_filter_mask.filter_df(df)?;
        };

        // Note: This branch is hit if we have negative slice or predicate + row index and the reader
        // does not support them.
        if let Some(ri) = row_index {
            // Adjustment needed for `current_row_position`.
            let local_offset_adjustment = RowCounter::new(
                // Number of physical rows skipped in the current function
                local_slice_offset,
                // How many of those skipped rows were deleted
                external_filter_mask.as_ref().map_or(0, |mask| {
                    if local_slice_offset == 0 {
                        0
                    } else {
                        mask.slice(current_row_position.num_physical_rows(), local_slice_offset)
                            .num_deleted_rows()
                    }
                }),
            );

            let offset = ri.offset.saturating_add(
                current_row_position
                    .add(local_offset_adjustment)
                    .num_rows_idxsize_saturating()?,
            );

            unsafe {
                df.with_column_unchecked(Column::new_row_index(
                    ri.name.clone(),
                    offset,
                    df.height(),
                )?)
            };
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
