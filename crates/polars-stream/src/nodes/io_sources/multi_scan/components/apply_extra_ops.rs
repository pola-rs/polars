//! Implementation of applying the operations during execution.
use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, Column, DataType};
use polars_core::scalar::Scalar;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::dsl::{CastColumnsPolicy, MissingColumnsPolicy, ScanSource};
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::slice_enum::Slice;

use crate::nodes::io_sources::multi_scan::components::column_selector::ColumnSelector;
use crate::nodes::io_sources::multi_scan::components::column_selector::builder::ColumnSelectorBuilder;
use crate::nodes::io_sources::multi_scan::components::errors::missing_column_err;
use crate::nodes::io_sources::multi_scan::components::projection::Projection;
use crate::nodes::io_sources::multi_scan::components::row_counter::RowCounter;
use crate::nodes::io_sources::multi_scan::components::row_deletions::ExternalFilterMask;
use crate::nodes::io_sources::multi_scan::pipeline::models::ExtraOperations;

/// Apply extra operations onto morsels originating from a reader. This should be initialized
/// per-reader (it contains e.g. file path).
#[derive(Debug)]
pub enum ApplyExtraOps {
    /// Intended to be initialized once, as we expect all morsels coming from a single reader to have
    /// the same schema. The initialized state can then be executed potentially in parallel by wrapping
    /// in Arc.
    Uninitialized {
        final_output_schema: SchemaRef,
        projection: Projection,
        cast_columns_policy: CastColumnsPolicy,
        missing_columns_policy: MissingColumnsPolicy,
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
        /// `(_, insertion_position)`
        row_index: Option<(RowIndex, usize)>,
        /// This will have include_file_paths, hive columns, missing columns.
        column_selectors: Option<Vec<ColumnSelector>>,
        predicate: Option<ScanIOPredicate>,
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
                projection,
                cast_columns_policy,
                missing_columns_policy,
                extra_ops:
                    ExtraOperations {
                        row_index,
                        row_index_col_idx,
                        pre_slice,
                        include_file_paths,
                        file_path_col_idx,
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

                let mut column_selectors = Vec::with_capacity(final_output_schema.len());
                let selector_builder = ColumnSelectorBuilder {
                    cast_columns_policy,
                    missing_columns_policy,
                };
                // Tracks if the input already has all columns in the right order and type.
                let mut is_input_passthrough = incoming_schema.len() == final_output_schema.len();

                for (output_index, (output_name, output_dtype)) in
                    final_output_schema.iter().enumerate()
                {
                    let selector = if output_index == file_path_col_idx {
                        ColumnSelector::Constant(Box::new((
                            include_file_paths.clone().unwrap(),
                            Scalar::new(
                                DataType::String,
                                AnyValue::StringOwned(
                                    scan_source
                                        .as_scan_source_ref()
                                        .to_include_path_name()
                                        .into(),
                                ),
                            ),
                        )))
                    } else if output_index == row_index_col_idx {
                        if let Some(ri) = &row_index {
                            // Row index is done by us (ApplyExtraOps). Insert a placeholder column.
                            ColumnSelector::Constant(Box::new((
                                ri.name.clone(),
                                Scalar::null(DataType::Null),
                            )))
                        } else {
                            debug_assert_eq!(
                                incoming_schema.get(output_name),
                                Some(&DataType::IDX_DTYPE)
                            );

                            ColumnSelector::Position(incoming_schema.index_of(output_name).unwrap())
                        }
                    } else if let Some(hive_parts) = &hive_parts
                        && let Ok(hive_column) = hive_parts.df().column(output_name)
                    {
                        ColumnSelector::Constant(Box::new((
                            output_name.clone(),
                            Scalar::new(
                                hive_column.dtype().clone(),
                                hive_column.get(scan_source_idx)?.into_static(),
                            ),
                        )))
                    } else if let Some((mapped_projection, incoming_idx, incoming_dtype)) = (|| {
                        let mapped_projection =
                            projection.get_mapped_projection_ref_by_output_name(output_name)?;

                        let (incoming_idx, _, incoming_dtype) =
                            incoming_schema.get_full(mapped_projection.source_name)?;

                        Some((mapped_projection, incoming_idx, incoming_dtype))
                    })(
                    ) {
                        debug_assert_eq!(mapped_projection.output_dtype, output_dtype);

                        if let Some(resolved_transform) = mapped_projection.resolved_transform {
                            debug_assert_eq!(resolved_transform.source_dtype, incoming_dtype);

                            resolved_transform
                                .attach_transforms(ColumnSelector::Position(incoming_idx))
                        } else {
                            selector_builder.build_column_selector(
                                incoming_schema,
                                output_name,
                                output_dtype,
                            )?
                        }
                    } else {
                        match &selector_builder.missing_columns_policy {
                            MissingColumnsPolicy::Insert => ColumnSelector::Constant(Box::new((
                                output_name.clone(),
                                projection
                                    .get_default_value_by_output_name(output_name)
                                    .cloned()
                                    .unwrap_or_else(|| Scalar::null(output_dtype.clone())),
                            ))),
                            MissingColumnsPolicy::Raise => {
                                return Err(missing_column_err(output_name));
                            },
                        }
                    };

                    is_input_passthrough &= match &selector {
                        ColumnSelector::Position(input_index) => *input_index == output_index,
                        _ => false,
                    };

                    column_selectors.push(selector);
                }

                let column_selectors = if is_input_passthrough {
                    None
                } else {
                    Some(column_selectors)
                };

                let out = Self::Initialized {
                    physical_pre_slice: pre_slice,
                    external_filter_mask,
                    row_index: row_index.map(|ri| (ri, row_index_col_idx)),
                    column_selectors,
                    predicate,
                };

                // Return a `Noop` if our initialized state does not have any operations. Downstream
                // can see the `Noop` and avoid running through an extra distributor pipeline.
                let out = match out {
                    Initialized {
                        physical_pre_slice: None,
                        external_filter_mask: None,
                        row_index: None,
                        column_selectors: None,
                        predicate: None,
                    } => Self::Noop,

                    Initialized { .. } => out,

                    _ => unreachable!(),
                };

                Ok(out)
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
            column_selectors,
            predicate,
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

        if let Some(column_selectors) = column_selectors.as_deref() {
            *df = column_selectors
                .iter()
                .map(|x| x.select_from_columns(df.get_columns(), df.height()))
                .collect::<PolarsResult<DataFrame>>()?;
        }

        // Note: This branch is hit if we have negative slice or predicate + row index and the reader
        // does not support them.
        if let Some((ri, col_idx)) = row_index {
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

            let row_index_col = Column::new_row_index(ri.name.clone(), offset, df.height())?;

            debug_assert_eq!(df.get_columns()[*col_idx].name(), &ri.name);

            unsafe { *df.get_columns_mut().get_mut(*col_idx).unwrap() = row_index_col }
        }

        if let Some(predicate) = predicate {
            let mask = predicate.predicate.evaluate_io(df)?;
            *df = df._filter_seq(mask.bool().expect("predicate not boolean"))?;
        }

        df.clear_schema();

        Ok(())
    }
}
