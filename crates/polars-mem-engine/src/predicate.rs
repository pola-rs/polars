use core::fmt;
use std::sync::Arc;

use arrow::bitmap::Bitmap;
use polars_core::frame::DataFrame;
use polars_core::prelude::{AnyValue, Column, Field, GroupPositions, PlHashMap, PlIndexSet};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_expr::prelude::{AggregationContext, PhysicalExpr, phys_expr_to_io_expr};
use polars_expr::state::ExecutionState;
use polars_io::predicates::{
    ColumnPredicates, ScanIOPredicate, SkipBatchPredicate, SpecializedColumnPredicate,
};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, format_pl_smallstr};

/// All the expressions and metadata used to filter out rows using predicates.
#[derive(Clone)]
pub struct ScanPredicate {
    pub predicate: Arc<dyn PhysicalExpr>,

    /// Column names that are used in the predicate.
    pub live_columns: Arc<PlIndexSet<PlSmallStr>>,

    /// A predicate expression used to skip record batches based on its statistics.
    ///
    /// This expression will be given a batch size along with a `min`, `max` and `null count` for
    /// each live column (set to `null` when it is not known) and the expression evaluates to
    /// `true` if the whole batch can for sure be skipped. This may be conservative and evaluate to
    /// `false` even when the batch could theoretically be skipped.
    pub skip_batch_predicate: Option<Arc<dyn PhysicalExpr>>,

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
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
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

    fn evaluate_on_groups<'a>(
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
        let mut skip_batch_predicate_constants =
            Vec::with_capacity(if self.skip_batch_predicate.is_some() {
                1 + constant_columns.size_hint().0 * 3
            } else {
                Default::default()
            });

        let predicate_constants = constant_columns
            .filter_map(|(name, scalar): (PlSmallStr, Scalar)| {
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
            live_columns: Arc::new(live_columns),
            skip_batch_predicate,
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
            live_columns: self.live_columns.clone(),
            skip_batch_predicate: skip_batch_predicate
                .cloned()
                .or_else(|| self.to_dyn_skip_batch_predicate(schema)),
            column_predicates: Arc::new(ColumnPredicates {
                predicates: self
                    .column_predicates
                    .predicates
                    .iter()
                    .map(|(n, (p, s))| (n.clone(), (phys_expr_to_io_expr(p.clone()), s.clone())))
                    .collect(),
                is_sumwise_complete: self.column_predicates.is_sumwise_complete,
            }),
            hive_predicate: self.hive_predicate.clone().map(phys_expr_to_io_expr),
            hive_predicate_is_full_predicate: self.hive_predicate_is_full_predicate,
        }
    }
}

impl SkipBatchPredicate for SkipBatchPredicateHelper {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn evaluate_with_stat_df(&self, df: &DataFrame) -> PolarsResult<Bitmap> {
        let array = self
            .skip_batch_predicate
            .evaluate(df, &Default::default())?;
        let array = array.bool()?;
        let array = array.downcast_as_array();

        let array = if let Some(validity) = array.validity() {
            array.values() & validity
        } else {
            array.values().clone()
        };

        // @NOTE: Certain predicates like `1 == 1` will only output 1 value. We need to broadcast
        // the result back to the dataframe length.
        if array.len() == 1 && df.height() != 0 {
            return Ok(Bitmap::new_with_value(array.get_bit(0), df.height()));
        }

        assert_eq!(array.len(), df.height());
        Ok(array)
    }
}
