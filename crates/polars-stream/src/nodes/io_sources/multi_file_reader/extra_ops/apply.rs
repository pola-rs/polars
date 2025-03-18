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
use polars_plan::dsl::ScanSources;
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::IdxSize;

use super::ExtraOperations;
use super::cast_columns::CastColumns;
use super::reorder_columns::ReorderColumns;

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
                        missing_columns,
                        include_file_paths,
                        predicate,
                    },
                scan_source_idx,
                sources,
                hive_parts,
            } => {
                // This should always be pushed to the reader, or otherwise handled separately.
                assert!(pre_slice.is_none());

                let cast_columns = if let Some(policy) = cast_columns {
                    CastColumns::try_init_from_policy(
                        policy,
                        &final_output_schema,
                        incoming_schema,
                    )?
                } else {
                    None
                };

                let mut extra_columns: Vec<ScalarColumn> = Vec::with_capacity(
                    final_output_schema.len()
                        - incoming_schema.len()
                        - row_index.is_some() as usize,
                );

                if let Some(policy) = missing_columns {
                    policy.initialize_policy(
                        &projected_file_schema,
                        incoming_schema,
                        &mut extra_columns,
                    )?;
                }

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
                                sources
                                    .get(scan_source_idx)
                                    .unwrap()
                                    .to_include_path_name()
                                    .into(),
                            ),
                        ),
                        1,
                    ))
                }

                let mut slf = Self::Initialized {
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
                    slf.apply_to_df(&mut df, IdxSize::MAX)?;
                    df.schema().clone()
                };

                assert_eq!(schema_before_reorder.len(), final_output_schema.len());

                let initialized_reorder =
                    ReorderColumns::initialize(&final_output_schema, &schema_before_reorder);

                let Self::Initialized { reorder, .. } = &mut slf else {
                    unreachable!()
                };

                *reorder = initialized_reorder;

                // Return a `Noop` if our initialized state does not have any operations.
                let slf = match slf {
                    Initialized {
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

        if let Some(ri) = row_index {
            unsafe {
                df.with_column_unchecked(Column::new_row_index(
                    ri.name.clone(),
                    ri.offset.saturating_add(current_row_position),
                    df.height(),
                )?)
            };
            df.clear_schema();
        }

        if let Some(cast_columns) = cast_columns {
            cast_columns.apply_cast(df)?;
        }

        if !extra_columns.is_empty() {
            let h = df.height();
            let cols = unsafe { df.get_columns_mut() };
            cols.extend(extra_columns.iter().map(|c| c.resize(h).into_column()));
            df.clear_schema();
        }

        if let Some(predicate) = predicate {
            let mask = predicate.predicate.evaluate_io(df)?;
            *df = df._filter_seq(mask.bool().expect("predicate not boolean"))?;
        }

        reorder.reorder_columns(df);

        Ok(())
    }
}
