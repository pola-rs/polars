use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::prelude::{
    AnyValue, Column, Field, GroupPositions, PlIndexMap, PlIndexSet, IDX_DTYPE,
};
use polars_core::scalar::Scalar;
use polars_core::schema::{Schema, SchemaRef};
use polars_error::PolarsResult;
use polars_expr::prelude::{phys_expr_to_io_expr, AggregationContext, PhysicalExpr};
use polars_expr::state::ExecutionState;
use polars_io::predicates::{ColumnStatistics, ScanIOPredicate, SkipBatchPredicate};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{format_pl_smallstr, IdxSize};

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
}

/// Helper to implement [`SkipBatchPredicate`].
struct SkipBatchPredicateHelper {
    skip_batch_predicate: Arc<dyn PhysicalExpr>,
    live_columns: Arc<PlIndexSet<PlSmallStr>>,

    /// A cached dataframe that gets used to evaluate all the expressions.
    df: DataFrame,
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

    fn isolate_column_expr(
        &self,
        _name: &str,
    ) -> Option<(
        Arc<dyn PhysicalExpr>,
        Option<polars_io::predicates::SpecializedColumnPredicateExpr>,
    )> {
        None
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
        let mut skip_batch_predicate_constants = Vec::with_capacity(
            self.skip_batch_predicate
                .is_some()
                .then_some(1 + constant_columns.size_hint().0 * 3)
                .unwrap_or_default(),
        );

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
        }
    }

    /// Create a predicate to skip batches using statistics.
    pub(crate) fn to_dyn_skip_batch_predicate(
        &self,
        schema: &Schema,
    ) -> Option<Arc<dyn SkipBatchPredicate>> {
        let skip_batch_predicate = self.skip_batch_predicate.as_ref()?;

        let mut columns = Vec::with_capacity(1 + self.live_columns.len() * 3);

        columns.push(Column::new_scalar(
            PlSmallStr::from_static("len"),
            Scalar::null(IDX_DTYPE),
            1,
        ));
        for col in self.live_columns.as_ref() {
            let dtype = schema.get(col).unwrap();
            columns.extend([
                Column::new_scalar(
                    format_pl_smallstr!("{col}_min"),
                    Scalar::null(dtype.clone()),
                    1,
                ),
                Column::new_scalar(
                    format_pl_smallstr!("{col}_max"),
                    Scalar::null(dtype.clone()),
                    1,
                ),
                Column::new_scalar(format_pl_smallstr!("{col}_nc"), Scalar::null(IDX_DTYPE), 1),
            ]);
        }

        // SAFETY:
        // * Each column is length = 1
        // * We have an IndexSet, so each column name is unique
        let df = unsafe { DataFrame::new_no_checks(1, columns) };

        Some(Arc::new(SkipBatchPredicateHelper {
            skip_batch_predicate: skip_batch_predicate.clone(),
            live_columns: self.live_columns.clone(),
            df,
        }))
    }

    pub fn to_io(
        &self,
        skip_batch_predicate: Option<&Arc<dyn SkipBatchPredicate>>,
        schema: &SchemaRef,
    ) -> ScanIOPredicate {
        ScanIOPredicate {
            predicate: phys_expr_to_io_expr(self.predicate.clone()),
            live_columns: self.live_columns.clone(),
            skip_batch_predicate: skip_batch_predicate
                .cloned()
                .or_else(|| self.to_dyn_skip_batch_predicate(schema)),
        }
    }
}

impl SkipBatchPredicate for SkipBatchPredicateHelper {
    fn can_skip_batch(
        &self,
        batch_size: IdxSize,
        statistics: PlIndexMap<PlSmallStr, ColumnStatistics>,
    ) -> PolarsResult<bool> {
        // This is the DF with all nulls.
        let mut df = self.df.clone();

        // SAFETY: We don't update the dtype, name or length of columns.
        let columns = unsafe { df.get_columns_mut() };

        // Set `len` statistic.
        columns[0]
            .as_scalar_column_mut()
            .unwrap()
            .with_value(batch_size.into());

        for (col, stat) in statistics {
            // Skip all statistics of columns that are not used in the predicate.
            let Some(idx) = self.live_columns.get_index_of(col.as_str()) else {
                continue;
            };

            let nc = stat.null_count.map_or(AnyValue::Null, |nc| nc.into());

            // Set `min`, `max` and `null_count` statistics.
            let col_idx = (idx * 3) + 1;
            columns[col_idx]
                .as_scalar_column_mut()
                .unwrap()
                .with_value(stat.min);
            columns[col_idx + 1]
                .as_scalar_column_mut()
                .unwrap()
                .with_value(stat.max);
            columns[col_idx + 2]
                .as_scalar_column_mut()
                .unwrap()
                .with_value(nc);
        }

        Ok(self
            .skip_batch_predicate
            .evaluate(&df, &Default::default())?
            .bool()?
            .first()
            .unwrap())
    }
}
