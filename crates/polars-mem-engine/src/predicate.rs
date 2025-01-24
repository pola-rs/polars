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
use polars_io::predicates::{ColumnStatistics, IOPredicate, SkipBatchPredicate};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{format_pl_smallstr, IdxSize};

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

    fn collect_live_columns(&self, lv: &mut PlIndexSet<PlSmallStr>) {
        self.child.collect_live_columns(lv)
    }
    fn is_scalar(&self) -> bool {
        self.child.is_scalar()
    }
}

/// All the expressions and metadata used to filter out rows using predicates.
#[derive(Clone)]
pub struct FilePredicate {
    pub predicate: Arc<dyn PhysicalExpr>,

    /// Column names that are used in the predicate.
    pub live_columns: Arc<PlIndexSet<PlSmallStr>>,

    /// A predicate expression used to skip record batches based on its statistics.
    ///
    /// This expression will be given a `min`, `max` and `null count` for each live column (set to
    /// `null` when it is not known) and the expression evaluates to `true` if the whole batch can for
    /// sure be skipped. This may be conservative and evaluate to `false` even when the batch could
    /// theorically be skipped.
    pub skip_batch_predicate: Option<Arc<dyn PhysicalExpr>>,
}

struct SkipBatchPredicateHelper {
    skip_batch_predicate: Arc<dyn PhysicalExpr>,
    live_columns: Arc<PlIndexSet<PlSmallStr>>,
    schema: SchemaRef,
}

impl FilePredicate {
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

    pub(crate) fn to_dyn_skip_batch_predicate(
        &self,
        schema: SchemaRef,
    ) -> Option<Arc<dyn SkipBatchPredicate>> {
        let skip_batch_predicate = self.skip_batch_predicate.as_ref()?;

        Some(Arc::new(SkipBatchPredicateHelper {
            skip_batch_predicate: skip_batch_predicate.clone(),
            live_columns: self.live_columns.clone(),
            schema,
        }))
    }

    pub fn to_io(
        &self,
        skip_batch_predicate: Option<&Arc<dyn SkipBatchPredicate>>,
        schema: &SchemaRef,
    ) -> IOPredicate {
        IOPredicate {
            expr: phys_expr_to_io_expr(self.predicate.clone()),
            live_columns: self.live_columns.clone(),
            skip_batch_predicate: skip_batch_predicate
                .cloned()
                .or_else(|| self.to_dyn_skip_batch_predicate(schema.clone())),
        }
    }
}

impl SkipBatchPredicate for SkipBatchPredicateHelper {
    fn can_skip_batch(
        &self,
        batch_size: IdxSize,
        statistics: PlIndexMap<PlSmallStr, ColumnStatistics>,
    ) -> PolarsResult<bool> {
        let mut columns = Vec::with_capacity(1 + self.live_columns.len() * 3);
        columns.push(Column::new_scalar(
            PlSmallStr::from_static("len"),
            batch_size.into(),
            1,
        ));

        for col in self.live_columns.as_ref() {
            if statistics.contains_key(col) {
                continue;
            }

            let dtype = self.schema.get(col).unwrap();
            columns.extend([
                Column::new_scalar(
                    format_pl_smallstr!("{col}_min"),
                    Scalar::new(dtype.clone(), AnyValue::Null),
                    1,
                ),
                Column::new_scalar(
                    format_pl_smallstr!("{col}_max"),
                    Scalar::new(dtype.clone(), AnyValue::Null),
                    1,
                ),
                Column::new_scalar(
                    format_pl_smallstr!("{col}_nc"),
                    Scalar::new(IDX_DTYPE, AnyValue::Null),
                    1,
                ),
            ]);
        }

        for (col, stat) in statistics {
            columns.extend([
                Column::new_scalar(
                    format_pl_smallstr!("{col}_min"),
                    Scalar::new(stat.dtype.clone(), stat.min),
                    1,
                ),
                Column::new_scalar(
                    format_pl_smallstr!("{col}_max"),
                    Scalar::new(stat.dtype, stat.max),
                    1,
                ),
                Column::new_scalar(
                    format_pl_smallstr!("{col}_nc"),
                    Scalar::new(
                        IDX_DTYPE,
                        stat.null_count.map_or(AnyValue::Null, |nc| nc.into()),
                    ),
                    1,
                ),
            ]);
        }

        let df = DataFrame::new(columns).unwrap();
        Ok(self
            .skip_batch_predicate
            .evaluate(&df, &Default::default())?
            .bool()?
            .first()
            .unwrap())
    }
}
