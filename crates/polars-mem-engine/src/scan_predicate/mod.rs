pub mod functions;
pub mod skip_files_mask;
#[cfg(feature = "dtype-categorical")]
mod table_statistics;
use core::fmt;
use std::sync::Arc;

pub use functions::{create_scan_predicate, initialize_scan_predicate};
use polars_core::frame::DataFrame;
use polars_core::prelude::{
    AnyValue, Column, Field, GroupPositions, PlBitmap, PlHashMap, PlIndexSet,
};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_expr::prelude::{AggregationContext, PhysicalExpr, phys_expr_to_io_expr};
use polars_expr::state::ExecutionState;
use polars_io::predicates::{
    ColumnPredicates, RuntimeRangeHint, ScanIOPredicate, SkipBatchPredicate,
    SpecializedColumnPredicate, StagedScanIOPredicate,
};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, format_pl_smallstr};

/// [`ScanPredicate::predicate`] as two conjunctions, see [`StagedScanIOPredicate`].
#[derive(Clone)]
pub struct StagedScanPredicate {
    pub first: Arc<dyn PhysicalExpr>,
    pub first_columns: Arc<PlIndexSet<PlSmallStr>>,
    pub second: Arc<dyn PhysicalExpr>,
    /// Partial predicates for each column of `first`.
    pub column_predicates: PhysicalColumnPredicates,
}

impl StagedScanPredicate {
    fn with_constant_columns(&self, constants: &[(PlSmallStr, Scalar)]) -> Self {
        let mut first_columns = self.first_columns.as_ref().clone();
        let mut column_predicates = self.column_predicates.clone();
        for (name, _) in constants {
            first_columns.swap_remove(name);
            column_predicates.predicates.remove(name);
        }
        Self {
            first: Arc::new(PhysicalExprWithConstCols {
                constants: constants.to_vec(),
                child: self.first.clone(),
            }),
            first_columns: Arc::new(first_columns),
            second: Arc::new(PhysicalExprWithConstCols {
                constants: constants.to_vec(),
                child: self.second.clone(),
            }),
            column_predicates,
        }
    }

    fn to_io(&self) -> StagedScanIOPredicate {
        StagedScanIOPredicate {
            first: phys_expr_to_io_expr(self.first.clone()),
            first_columns: self.first_columns.clone(),
            second: phys_expr_to_io_expr(self.second.clone()),
            column_predicates: self.column_predicates.to_io(),
        }
    }
}

/// All the expressions and metadata used to filter out rows using predicates.
#[derive(Clone)]
pub struct ScanPredicate {
    pub predicate: Arc<dyn PhysicalExpr>,

    pub staged: Option<StagedScanPredicate>,

    /// Whether `predicate` filters rows at all. False when the scan only has
    /// runtime ranges to skip batches by.
    pub filters_rows: bool,

    /// Column names that are used in the predicate.
    pub live_columns: Arc<PlIndexSet<PlSmallStr>>,

    /// A predicate expression used to skip record batches based on its statistics.
    ///
    /// This expression will be given a batch size along with a `min`, `max` and `null count` for
    /// each live column (set to `null` when it is not known) and the expression evaluates to
    /// `true` if the whole batch can for sure be skipped. This may be conservative and evaluate to
    /// `false` even when the batch could theoretically be skipped.
    pub skip_batch_predicate: Option<Arc<dyn PhysicalExpr>>,

    /// Columns whose batches are skipped by a range published at run time.
    pub runtime_ranges: Vec<RuntimeRangeHint>,

    /// Partial predicates for each column for filter when loading columnar formats.
    pub column_predicates: PhysicalColumnPredicates,

    /// Predicate only referring to hive columns.
    pub hive_predicate: Option<Arc<dyn PhysicalExpr>>,
    pub hive_predicate_is_full_predicate: bool,
}

impl fmt::Debug for ScanPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("scan_predicate")
    }
}

#[derive(Clone)]
pub struct PhysicalColumnPredicates {
    pub predicates:
        PlHashMap<PlSmallStr, (Arc<dyn PhysicalExpr>, Option<SpecializedColumnPredicate>)>,
    pub is_sumwise_complete: bool,
}

impl PhysicalColumnPredicates {
    fn to_io(&self) -> Arc<ColumnPredicates> {
        Arc::new(ColumnPredicates {
            predicates: self
                .predicates
                .iter()
                .map(|(n, (p, s))| (n.clone(), (phys_expr_to_io_expr(p.clone()), s.clone())))
                .collect(),
            is_sumwise_complete: self.is_sumwise_complete,
        })
    }
}

/// Helper to implement [`SkipBatchPredicate`].
struct SkipBatchPredicateHelper {
    skip_batch_predicate: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
}

/// Helper for the [`PhysicalExpr`] trait to include constant columns.
pub struct PhysicalExprWithConstCols {
    constants: Vec<(PlSmallStr, Scalar)>,
    child: Arc<dyn PhysicalExpr>,
}

impl PhysicalExpr for PhysicalExprWithConstCols {
    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut df = df.clone();
        for (name, scalar) in &self.constants {
            df.with_column(Column::new_scalar(
                name.clone(),
                scalar.clone(),
                df.height(),
            ))?;
        }

        self.child.evaluate(&df, state)
    }

    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut df = df.clone();
        for (name, scalar) in &self.constants {
            df.with_column(Column::new_scalar(
                name.clone(),
                scalar.clone(),
                df.height(),
            ))?;
        }

        self.child.evaluate_on_groups(&df, groups, state)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.child.to_field(input_schema)
    }
    fn is_scalar(&self) -> bool {
        self.child.is_scalar()
    }
}

impl ScanPredicate {
    pub fn with_constant_columns(
        &self,
        constant_columns: impl IntoIterator<Item = (PlSmallStr, Scalar)>,
    ) -> Self {
        let constant_columns = constant_columns.into_iter();

        let mut live_columns = self.live_columns.as_ref().clone();
        let mut runtime_ranges = self.runtime_ranges.clone();
        let mut skip_batch_predicate_constants =
            Vec::with_capacity(if self.skip_batch_predicate.is_some() {
                1 + constant_columns.size_hint().0 * 3
            } else {
                Default::default()
            });

        let predicate_constants: Vec<(PlSmallStr, Scalar)> = constant_columns
            .filter_map(|(name, scalar): (PlSmallStr, Scalar)| {
                RuntimeRangeHint::set_constant(&mut runtime_ranges, &name, &scalar);
                if !live_columns.swap_remove(&name) {
                    return None;
                }

                if self.skip_batch_predicate.is_some() {
                    let mut null_count: Scalar = (0 as IdxSize).into();

                    // If the constant value is Null, we don't know how many nulls there are
                    // because the length of the batch may vary.
                    if scalar.is_null() {
                        null_count.update(AnyValue::Null);
                    }

                    skip_batch_predicate_constants.extend([
                        (format_pl_smallstr!("{name}_min"), scalar.clone()),
                        (format_pl_smallstr!("{name}_max"), scalar.clone()),
                        (format_pl_smallstr!("{name}_nc"), null_count),
                    ]);
                }

                Some((name, scalar))
            })
            .collect();

        let staged = self
            .staged
            .as_ref()
            .map(|staged| staged.with_constant_columns(&predicate_constants));
        let predicate = Arc::new(PhysicalExprWithConstCols {
            constants: predicate_constants,
            child: self.predicate.clone(),
        });
        let skip_batch_predicate = self.skip_batch_predicate.as_ref().map(|skp| {
            Arc::new(PhysicalExprWithConstCols {
                constants: skip_batch_predicate_constants,
                child: skp.clone(),
            }) as _
        });

        Self {
            predicate,
            staged,
            filters_rows: self.filters_rows,
            live_columns: Arc::new(live_columns),
            skip_batch_predicate,
            runtime_ranges,
            column_predicates: self.column_predicates.clone(), // Q? Maybe this should cull
            // predicates.
            hive_predicate: None,
            hive_predicate_is_full_predicate: false,
        }
    }

    /// Create a predicate to skip batches using statistics.
    pub(crate) fn to_dyn_skip_batch_predicate(
        &self,
        schema: SchemaRef,
    ) -> Option<Arc<dyn SkipBatchPredicate>> {
        let skip_batch_predicate = self.skip_batch_predicate.as_ref()?.clone();
        Some(Arc::new(SkipBatchPredicateHelper {
            skip_batch_predicate,
            schema,
        }))
    }

    pub fn to_io(
        &self,
        skip_batch_predicate: Option<&Arc<dyn SkipBatchPredicate>>,
        schema: SchemaRef,
    ) -> ScanIOPredicate {
        ScanIOPredicate {
            predicate: phys_expr_to_io_expr(self.predicate.clone()),
            staged: self.staged.as_ref().map(StagedScanPredicate::to_io),
            filters_rows: self.filters_rows,
            live_columns: self.live_columns.clone(),
            skip_batch_predicate: skip_batch_predicate
                .cloned()
                .or_else(|| self.to_dyn_skip_batch_predicate(schema)),
            runtime_ranges: self.runtime_ranges.clone(),
            column_predicates: self.column_predicates.to_io(),
            hive_predicate: self.hive_predicate.clone().map(phys_expr_to_io_expr),
            hive_predicate_is_full_predicate: self.hive_predicate_is_full_predicate,
        }
    }
}

impl SkipBatchPredicate for SkipBatchPredicateHelper {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn evaluate_with_stat_df(&self, df: &DataFrame) -> PolarsResult<PlBitmap> {
        if df.height() == 0 {
            return Ok(PlBitmap::new_empty());
        }
        let array = self
            .skip_batch_predicate
            .evaluate(df, &Default::default())?;
        let array = array.bool()?.rechunk();
        let array = array.downcast_as_array();

        // Nulls count as false.
        let mask = array.true_and_valid();

        // @NOTE: Certain predicates like `1 == 1` will only output 1 value. We need to broadcast
        // the result back to the dataframe length — which the mask does by keeping the one bit it
        // holds rather than writing it out per row.
        if mask.len() == 1 {
            return Ok(PlBitmap::new_scalar(mask.get(0), df.height()));
        }

        assert_eq!(mask.len(), df.height());
        Ok(mask)
    }
}
