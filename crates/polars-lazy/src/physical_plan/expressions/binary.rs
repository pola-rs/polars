use std::sync::Arc;

use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;

use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct BinaryExpr {
    pub(crate) left: Arc<dyn PhysicalExpr>,
    pub(crate) op: Operator,
    pub(crate) right: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl BinaryExpr {
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
        expr: Expr,
    ) -> Self {
        Self {
            left,
            op,
            right,
            expr,
        }
    }
}

/// Can partially do operations in place.
fn apply_operator_owned(left: Series, right: Series, op: Operator) -> PolarsResult<Series> {
    match op {
        Operator::Plus => Ok(left + right),
        Operator::Minus => Ok(left - right),
        Operator::Multiply => Ok(left * right),
        _ => apply_operator(&left, &right, op),
    }
}

pub fn apply_operator(left: &Series, right: &Series, op: Operator) -> PolarsResult<Series> {
    use DataType::*;
    match op {
        Operator::Gt => ChunkCompare::<&Series>::gt(left, right).map(|ca| ca.into_series()),
        Operator::GtEq => ChunkCompare::<&Series>::gt_eq(left, right).map(|ca| ca.into_series()),
        Operator::Lt => ChunkCompare::<&Series>::lt(left, right).map(|ca| ca.into_series()),
        Operator::LtEq => ChunkCompare::<&Series>::lt_eq(left, right).map(|ca| ca.into_series()),
        Operator::Eq => ChunkCompare::<&Series>::equal(left, right).map(|ca| ca.into_series()),
        Operator::NotEq => {
            ChunkCompare::<&Series>::not_equal(left, right).map(|ca| ca.into_series())
        }
        Operator::Plus => Ok(left + right),
        Operator::Minus => Ok(left - right),
        Operator::Multiply => Ok(left * right),
        Operator::Divide => Ok(left / right),
        Operator::TrueDivide => match left.dtype() {
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => Ok(left / right),
            Date | Datetime(_, _) | Float32 | Float64 => Ok(left / right),
            _ => Ok(&left.cast(&Float64)? / &right.cast(&Float64)?),
        },
        Operator::FloorDivide => {
            #[cfg(feature = "round_series")]
            {
                floor_div_series(left, right)
            }
            #[cfg(not(feature = "round_series"))]
            {
                panic!("activate 'round_series' feature")
            }
        }
        Operator::And => left.bitand(right),
        Operator::Or => left.bitor(right),
        Operator::Xor => left.bitxor(right),
        Operator::Modulus => Ok(left % right),
        Operator::EqValidity => left.equal_missing(right).map(|ca| ca.into_series()),
        Operator::NotEqValidity => left.not_equal_missing(right).map(|ca| ca.into_series()),
    }
}

impl BinaryExpr {
    fn apply_elementwise<'a>(
        &self,
        mut ac_l: AggregationContext<'a>,
        ac_r: AggregationContext,
        aggregated: bool,
    ) -> PolarsResult<AggregationContext<'a>> {
        // we want to be able to mutate in place
        // so we take the lhs to make sure that we drop
        let lhs = ac_l.series().clone();
        let rhs = ac_r.series().clone();

        // drop lhs so that we might operate in place
        {
            let _ = ac_l.take();
        }

        let out = apply_operator_owned(lhs, rhs, self.op)?;

        ac_l.with_series(out, aggregated, Some(&self.expr))?;
        Ok(ac_l)
    }

    fn apply_group_aware<'a>(
        &self,
        mut ac_l: AggregationContext<'a>,
        mut ac_r: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let name = ac_l.series().name().to_string();
        let mut ca: ListChunked = ac_l
            .iter_groups(false)
            .zip(ac_r.iter_groups(false))
            .map(|(l, r)| {
                match (l, r) {
                    (Some(l), Some(r)) => {
                        let l = l.as_ref();
                        let r = r.as_ref();
                        Some(apply_operator(l, r, self.op))
                    }
                    _ => None,
                }
                .transpose()
            })
            .collect::<PolarsResult<_>>()?;
        ca.rename(&name);

        // try if we can reuse the groups
        use AggState::*;
        match (ac_l.agg_state(), ac_r.agg_state()) {
            // no need to change update groups
            (AggregatedList(_), _) => {}
            // we can take the groups of the rhs
            (_, AggregatedList(_)) if matches!(ac_r.update_groups, UpdateGroups::No) => {
                ac_l.groups = ac_r.groups
            }
            // we must update the groups
            _ => {
                ac_l.with_update_groups(UpdateGroups::WithSeriesLen);
            }
        }

        ac_l.with_series(ca.into_series(), true, Some(&self.expr))?;
        Ok(ac_l)
    }
}

impl PhysicalExpr for BinaryExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        // window functions may set a global state that determine their output
        // state, so we don't let them run in parallel as they race
        // they also saturate the thread pool by themselves, so that's fine
        let has_window = state.has_window();
        // streaming takes care of parallelism, don't parallelize here, as it
        // increases contention

        #[cfg(feature = "streaming")]
        let in_streaming = state.in_streaming_engine();

        #[cfg(not(feature = "streaming"))]
        let in_streaming = false;

        let (lhs, rhs) = if has_window {
            let mut state = state.split();
            state.remove_cache_window_flag();
            (
                self.left.evaluate(df, &state),
                self.right.evaluate(df, &state),
            )
        } else if in_streaming {
            (
                self.left.evaluate(df, state),
                self.right.evaluate(df, state),
            )
        } else {
            POOL.install(|| {
                rayon::join(
                    || self.left.evaluate(df, state),
                    || self.right.evaluate(df, state),
                )
            })
        };
        let lhs = lhs?;
        let rhs = rhs?;
        polars_ensure!(
            lhs.len() == rhs.len() || lhs.len() == 1 || rhs.len() == 1,
            expr = self.expr,
            ComputeError: "cannot evaluate two series of different lengths ({} and {})",
            lhs.len(), rhs.len(),
        );
        apply_operator_owned(lhs, rhs, self.op)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let (result_a, result_b) = POOL.install(|| {
            rayon::join(
                || self.left.evaluate_on_groups(df, groups, state),
                || self.right.evaluate_on_groups(df, groups, state),
            )
        });
        let mut ac_l = result_a?;
        let ac_r = result_b?;

        match (ac_l.agg_state(), ac_r.agg_state()) {
            (
                AggState::Literal(_) | AggState::NotAggregated(_),
                AggState::Literal(_) | AggState::NotAggregated(_),
            ) => self.apply_elementwise(ac_l, ac_r, false),
            (
                AggState::AggregatedFlat(_) | AggState::Literal(_),
                AggState::AggregatedFlat(_) | AggState::Literal(_),
            ) => self.apply_elementwise(ac_l, ac_r, true),
            (AggState::AggregatedFlat(_), AggState::NotAggregated(_))
            | (AggState::NotAggregated(_), AggState::AggregatedFlat(_)) => {
                self.apply_group_aware(ac_l, ac_r)
            }
            (AggState::AggregatedList(lhs), AggState::AggregatedList(rhs)) => {
                let lhs = lhs.list().unwrap();
                let rhs = rhs.list().unwrap();
                let out =
                    lhs.apply_to_inner(&|lhs| apply_operator(&lhs, &rhs.get_inner(), self.op))?;
                ac_l.with_series(out.into_series(), true, Some(&self.expr))?;
                Ok(ac_l)
            }
            _ => self.apply_group_aware(ac_l, ac_r),
        }
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.expr.to_field(input_schema, Context::Default)
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        Some(self)
    }

    fn is_valid_aggregation(&self) -> bool {
        // we don't want:
        // col(a) == lit(1)

        // we do want
        // col(a).sum() == lit(1)
        (!self.left.is_literal() && self.left.is_valid_aggregation())
            | (!self.right.is_literal() && self.right.is_valid_aggregation())
    }
}

#[cfg(feature = "parquet")]
mod stats {
    use polars_io::parquet::predicates::BatchStats;
    use polars_io::predicates::StatsEvaluator;

    use super::*;

    fn apply_operator_stats_eq(min_max: &Series, literal: &Series) -> bool {
        // literal is greater than max, don't need to read
        if ChunkCompare::<&Series>::gt(literal, min_max)
            .ok()
            .map(|s| s.all())
            == Some(true)
        {
            return false;
        }

        // literal is smaller than min, don't need to read
        if ChunkCompare::<&Series>::lt(literal, min_max)
            .ok()
            .map(|s| s.all())
            == Some(true)
        {
            return false;
        }

        true
    }

    fn apply_operator_stats_rhs_lit(min_max: &Series, literal: &Series, op: Operator) -> bool {
        match op {
            Operator::Eq => apply_operator_stats_eq(min_max, literal),
            // col > lit
            // e.g.
            // [min,
            // max] > 0
            //
            // [-1,
            // 2] > 0
            //
            // [false, true] -> true -> read
            Operator::Gt => {
                // literal is bigger than max value
                // selection needs all rows
                ChunkCompare::<&Series>::gt(min_max, literal)
                    .ok()
                    .map(|s| s.any())
                    == Some(true)
            }
            // col >= lit
            Operator::GtEq => {
                // literal is bigger than max value
                // selection needs all rows
                ChunkCompare::<&Series>::gt_eq(min_max, literal)
                    .ok()
                    .map(|ca| ca.any())
                    == Some(true)
            }
            // col < lit
            Operator::Lt => {
                // literal is smaller than min value
                // selection needs all rows
                ChunkCompare::<&Series>::lt(min_max, literal)
                    .ok()
                    .map(|ca| ca.any())
                    == Some(true)
            }
            // col <= lit
            Operator::LtEq => {
                // literal is smaller than min value
                // selection needs all rows
                ChunkCompare::<&Series>::lt_eq(min_max, literal)
                    .ok()
                    .map(|ca| ca.any())
                    == Some(true)
            }
            // default: read the file
            _ => true,
        }
    }

    fn apply_operator_stats_lhs_lit(literal: &Series, min_max: &Series, op: Operator) -> bool {
        match op {
            Operator::Eq => apply_operator_stats_eq(min_max, literal),
            Operator::Gt => {
                // literal is bigger than max value
                // selection needs all rows
                ChunkCompare::<&Series>::gt(literal, min_max)
                    .ok()
                    .map(|ca| ca.any())
                    == Some(true)
            }
            Operator::GtEq => {
                // literal is bigger than max value
                // selection needs all rows
                ChunkCompare::<&Series>::gt_eq(literal, min_max)
                    .ok()
                    .map(|ca| ca.any())
                    == Some(true)
            }
            Operator::Lt => {
                // literal is smaller than min value
                // selection needs all rows
                ChunkCompare::<&Series>::lt(literal, min_max)
                    .ok()
                    .map(|ca| ca.any())
                    == Some(true)
            }
            Operator::LtEq => {
                // literal is smaller than min value
                // selection needs all rows
                ChunkCompare::<&Series>::lt_eq(literal, min_max)
                    .ok()
                    .map(|ca| ca.any())
                    == Some(true)
            }
            // default: read the file
            _ => true,
        }
    }

    impl BinaryExpr {
        fn impl_should_read(&self, stats: &BatchStats) -> PolarsResult<bool> {
            // See: #5864 for the rationale behind this.
            use Expr::*;
            use Operator::*;
            if !self.expr.into_iter().all(|e| match e {
                BinaryExpr { op, .. } => !matches!(
                    op,
                    Multiply | Divide | TrueDivide | FloorDivide | Modulus | NotEq
                ),
                Column(_) | Literal(_) | Alias(_, _) => true,
                _ => false,
            }) {
                return Ok(true);
            }

            let schema = stats.schema();
            let fld_l = self.left.to_field(schema)?;
            let fld_r = self.right.to_field(schema)?;

            #[cfg(debug_assertions)]
            {
                match (fld_l.data_type(), fld_r.data_type()) {
                    #[cfg(feature = "dtype-categorical")]
                    (DataType::Utf8, DataType::Categorical(_)) => {}
                    #[cfg(feature = "dtype-categorical")]
                    (DataType::Categorical(_), DataType::Utf8) => {}
                    (l, r) if l != r => panic!("implementation error: {l:?}, {r:?}"),
                    _ => {}
                }
            }

            let dummy = DataFrame::new_no_checks(vec![]);
            let state = ExecutionState::new();

            let out = match (self.left.is_literal(), self.right.is_literal()) {
                (false, true) => {
                    let l = stats.get_stats(fld_l.name())?;
                    match l.to_min_max() {
                        None => Ok(true),
                        Some(min_max_s) => {
                            // will be incorrect if not
                            debug_assert_eq!(min_max_s.null_count(), 0);
                            let lit_s = self.right.evaluate(&dummy, &state).unwrap();
                            Ok(apply_operator_stats_rhs_lit(&min_max_s, &lit_s, self.op))
                        }
                    }
                }
                (true, false) => {
                    let r = stats.get_stats(fld_r.name())?;
                    match r.to_min_max() {
                        None => Ok(true),
                        Some(min_max_s) => {
                            // will be incorrect if not
                            debug_assert_eq!(min_max_s.null_count(), 0);
                            let lit_s = self.left.evaluate(&dummy, &state).unwrap();
                            Ok(apply_operator_stats_lhs_lit(&lit_s, &min_max_s, self.op))
                        }
                    }
                }
                // default: read the file
                _ => Ok(true),
            };
            out.map(|read| {
                if state.verbose() && read {
                    eprintln!("parquet file must be read, statistics not sufficient for predicate.")
                } else if state.verbose() && !read {
                    eprintln!("parquet file can be skipped, the statistics were sufficient to apply the predicate.")
                };
                read
            })
        }
    }

    impl StatsEvaluator for BinaryExpr {
        fn should_read(&self, stats: &BatchStats) -> PolarsResult<bool> {
            if std::env::var("POLARS_NO_PARQUET_STATISTICS").is_ok() {
                return Ok(true);
            }

            match (
                self.left.as_stats_evaluator(),
                self.right.as_stats_evaluator(),
            ) {
                (Some(l), Some(r)) => match self.op {
                    Operator::And => Ok(l.should_read(stats)? && r.should_read(stats)?),
                    Operator::Or => Ok(l.should_read(stats)? || r.should_read(stats)?),
                    _ => Ok(true),
                },
                _ => self.impl_should_read(stats),
            }
        }
    }
}

impl PartitionedAggregation for BinaryExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let left = self.left.as_partitioned_aggregator().unwrap();
        let right = self.right.as_partitioned_aggregator().unwrap();
        let left = left.evaluate_partitioned(df, groups, state)?;
        let right = right.evaluate_partitioned(df, groups, state)?;
        apply_operator(&left, &right, self.op)
    }

    fn finalize(
        &self,
        partitioned: Series,
        _groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        Ok(partitioned)
    }
}
