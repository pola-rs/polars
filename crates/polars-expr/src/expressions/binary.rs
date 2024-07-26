use polars_core::prelude::*;
use polars_core::POOL;
#[cfg(feature = "round_series")]
use polars_ops::prelude::floor_div_series;

use super::*;
use crate::expressions::{
    AggState, AggregationContext, PartitionedAggregation, PhysicalExpr, UpdateGroups,
};

pub struct BinaryExpr {
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
    expr: Expr,
    has_literal: bool,
    allow_threading: bool,
}

impl BinaryExpr {
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
        expr: Expr,
        has_literal: bool,
        allow_threading: bool,
    ) -> Self {
        Self {
            left,
            op,
            right,
            expr,
            has_literal,
            allow_threading,
        }
    }
}

/// Can partially do operations in place.
fn apply_operator_owned(left: Series, right: Series, op: Operator) -> PolarsResult<Series> {
    match op {
        Operator::Plus => left.try_add_owned(right),
        Operator::Minus => left.try_sub_owned(right),
        Operator::Multiply if left.dtype().is_numeric() && right.dtype().is_numeric() => {
            left.try_mul_owned(right)
        },
        _ => apply_operator(&left, &right, op),
    }
}

pub fn apply_operator(left: &Series, right: &Series, op: Operator) -> PolarsResult<Series> {
    use DataType::*;
    match op {
        Operator::Gt => ChunkCompare::gt(left, right).map(|ca| ca.into_series()),
        Operator::GtEq => ChunkCompare::gt_eq(left, right).map(|ca| ca.into_series()),
        Operator::Lt => ChunkCompare::lt(left, right).map(|ca| ca.into_series()),
        Operator::LtEq => ChunkCompare::lt_eq(left, right).map(|ca| ca.into_series()),
        Operator::Eq => ChunkCompare::equal(left, right).map(|ca| ca.into_series()),
        Operator::NotEq => ChunkCompare::not_equal(left, right).map(|ca| ca.into_series()),
        Operator::Plus => left + right,
        Operator::Minus => left - right,
        Operator::Multiply => left * right,
        Operator::Divide => left / right,
        Operator::TrueDivide => match left.dtype() {
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => left / right,
            Duration(_) | Date | Datetime(_, _) | Float32 | Float64 => left / right,
            #[cfg(feature = "dtype-array")]
            dt @ Array(_, _) => {
                let left_dt = dt.cast_leaf(Float64);
                let right_dt = right.dtype().cast_leaf(Float64);
                left.cast(&left_dt)? / right.cast(&right_dt)?
            },
            _ => {
                if right.dtype().is_temporal() {
                    return left / right;
                }
                left.cast(&Float64)? / right.cast(&Float64)?
            },
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
        },
        Operator::And => left.bitand(right),
        Operator::Or => left.bitor(right),
        Operator::LogicalOr => left
            .cast(&DataType::Boolean)?
            .bitor(&right.cast(&DataType::Boolean)?),
        Operator::LogicalAnd => left
            .cast(&DataType::Boolean)?
            .bitand(&right.cast(&DataType::Boolean)?),
        Operator::Xor => left.bitxor(right),
        Operator::Modulus => left % right,
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
        // We want to be able to mutate in place, so we take the lhs to make sure that we drop.
        let lhs = ac_l.series().clone();
        let rhs = ac_r.series().clone();

        // Drop lhs so that we might operate in place.
        drop(ac_l.take());

        let out = apply_operator_owned(lhs, rhs, self.op)?;
        ac_l.with_series(out, aggregated, Some(&self.expr))?;
        Ok(ac_l)
    }

    fn apply_all_literal<'a>(
        &self,
        mut ac_l: AggregationContext<'a>,
        mut ac_r: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let name = ac_l.series().name().to_string();
        ac_l.groups();
        ac_r.groups();
        polars_ensure!(ac_l.groups.len() == ac_r.groups.len(), ComputeError: "lhs and rhs should have same group length");
        let left_s = ac_l.series().rechunk();
        let right_s = ac_r.series().rechunk();
        let res_s = apply_operator(&left_s, &right_s, self.op)?;
        ac_l.with_update_groups(UpdateGroups::WithSeriesLen);
        let res_s = if res_s.len() == 1 {
            res_s.new_from_index(0, ac_l.groups.len())
        } else {
            ListChunked::full(&name, &res_s, ac_l.groups.len()).into_series()
        };
        ac_l.with_series(res_s, true, Some(&self.expr))?;
        Ok(ac_l)
    }

    fn apply_group_aware<'a>(
        &self,
        mut ac_l: AggregationContext<'a>,
        mut ac_r: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let name = ac_l.series().name().to_string();
        // SAFETY: unstable series never lives longer than the iterator.
        let ca = unsafe {
            ac_l.iter_groups(false)
                .zip(ac_r.iter_groups(false))
                .map(|(l, r)| Some(apply_operator(l?.as_ref(), r?.as_ref(), self.op)))
                .map(|opt_res| opt_res.transpose())
                .collect::<PolarsResult<ListChunked>>()?
                .with_name(&name)
        };

        ac_l.with_update_groups(UpdateGroups::WithSeriesLen);
        ac_l.with_agg_state(AggState::AggregatedList(ca.into_series()));
        Ok(ac_l)
    }
}

impl PhysicalExpr for BinaryExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        // Window functions may set a global state that determine their output
        // state, so we don't let them run in parallel as they race
        // they also saturate the thread pool by themselves, so that's fine.
        let has_window = state.has_window();

        let (lhs, rhs);
        if has_window {
            let mut state = state.split();
            state.remove_cache_window_flag();
            lhs = self.left.evaluate(df, &state)?;
            rhs = self.right.evaluate(df, &state)?;
        } else if !self.allow_threading || self.has_literal {
            // Literals are free, don't pay par cost.
            lhs = self.left.evaluate(df, state)?;
            rhs = self.right.evaluate(df, state)?;
        } else {
            let (opt_lhs, opt_rhs) = POOL.install(|| {
                rayon::join(
                    || self.left.evaluate(df, state),
                    || self.right.evaluate(df, state),
                )
            });
            (lhs, rhs) = (opt_lhs?, opt_rhs?);
        };
        polars_ensure!(
            lhs.len() == rhs.len() || lhs.len() == 1 || rhs.len() == 1,
            expr = self.expr,
            ComputeError: "cannot evaluate two Series of different lengths ({} and {})",
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
            (AggState::Literal(s), AggState::NotAggregated(_))
            | (AggState::NotAggregated(_), AggState::Literal(s)) => match s.len() {
                1 => self.apply_elementwise(ac_l, ac_r, false),
                _ => self.apply_group_aware(ac_l, ac_r),
            },
            (AggState::Literal(_), AggState::Literal(_)) => self.apply_all_literal(ac_l, ac_r),
            (AggState::NotAggregated(_), AggState::NotAggregated(_)) => {
                self.apply_elementwise(ac_l, ac_r, false)
            },
            (
                AggState::AggregatedScalar(_) | AggState::Literal(_),
                AggState::AggregatedScalar(_) | AggState::Literal(_),
            ) => self.apply_elementwise(ac_l, ac_r, true),
            (AggState::AggregatedScalar(_), AggState::NotAggregated(_))
            | (AggState::NotAggregated(_), AggState::AggregatedScalar(_)) => {
                self.apply_group_aware(ac_l, ac_r)
            },
            (AggState::AggregatedList(lhs), AggState::AggregatedList(rhs)) => {
                let lhs = lhs.list().unwrap();
                let rhs = rhs.list().unwrap();
                let out =
                    lhs.apply_to_inner(&|lhs| apply_operator(&lhs, &rhs.get_inner(), self.op))?;
                ac_l.with_series(out.into_series(), true, Some(&self.expr))?;
                Ok(ac_l)
            },
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
}

#[cfg(feature = "parquet")]
mod stats {
    use polars_io::predicates::{BatchStats, StatsEvaluator};

    use super::*;

    fn apply_operator_stats_eq(min_max: &Series, literal: &Series) -> bool {
        use ChunkCompare as C;
        // Literal is greater than max, don't need to read.
        if C::gt(literal, min_max).map(|s| s.all()).unwrap_or(false) {
            return false;
        }

        // Literal is smaller than min, don't need to read.
        if C::lt(literal, min_max).map(|s| s.all()).unwrap_or(false) {
            return false;
        }

        true
    }

    fn apply_operator_stats_neq(min_max: &Series, literal: &Series) -> bool {
        if min_max.len() < 2 || min_max.null_count() > 0 {
            return true;
        }
        use ChunkCompare as C;

        // First check proofs all values are the same (e.g. min/max is the same)
        // Second check proofs all values are equal, so we can skip as we search
        // for non-equal values.
        if min_max.get(0).unwrap() == min_max.get(1).unwrap()
            && C::equal(literal, min_max).map(|s| s.all()).unwrap_or(false)
        {
            return false;
        }
        true
    }

    fn apply_operator_stats_rhs_lit(min_max: &Series, literal: &Series, op: Operator) -> bool {
        use ChunkCompare as C;
        match op {
            Operator::Eq => apply_operator_stats_eq(min_max, literal),
            Operator::NotEq => apply_operator_stats_neq(min_max, literal),
            // col > lit
            // e.g.
            // [min, max] > 0
            //
            // [-1, 2] > 0
            //
            // [false, true] -> true -> read
            Operator::Gt => {
                // Literal is bigger than max value, selection needs all rows.
                C::gt(min_max, literal).map(|s| s.any()).unwrap_or(false)
            },
            // col >= lit
            Operator::GtEq => {
                // Literal is bigger than max value, selection needs all rows.
                C::gt_eq(min_max, literal).map(|s| s.any()).unwrap_or(false)
            },
            // col < lit
            Operator::Lt => {
                // Literal is smaller than min value, selection needs all rows.
                C::lt(min_max, literal).map(|s| s.any()).unwrap_or(false)
            },
            // col <= lit
            Operator::LtEq => {
                // Literal is smaller than min value, selection needs all rows.
                C::lt_eq(min_max, literal).map(|s| s.any()).unwrap_or(false)
            },
            // Default: read the file
            _ => true,
        }
    }

    fn apply_operator_stats_lhs_lit(literal: &Series, min_max: &Series, op: Operator) -> bool {
        use ChunkCompare as C;
        match op {
            Operator::Eq => apply_operator_stats_eq(min_max, literal),
            Operator::NotEq => apply_operator_stats_eq(min_max, literal),
            Operator::Gt => {
                // Literal is bigger than max value, selection needs all rows.
                C::gt(literal, min_max).map(|ca| ca.any()).unwrap_or(false)
            },
            Operator::GtEq => {
                // Literal is bigger than max value, selection needs all rows.
                C::gt_eq(literal, min_max)
                    .map(|ca| ca.any())
                    .unwrap_or(false)
            },
            Operator::Lt => {
                // Literal is smaller than min value, selection needs all rows.
                C::lt(literal, min_max).map(|ca| ca.any()).unwrap_or(false)
            },
            Operator::LtEq => {
                // Literal is smaller than min value, selection needs all rows.
                C::lt_eq(literal, min_max)
                    .map(|ca| ca.any())
                    .unwrap_or(false)
            },
            // Default: read the file.
            _ => true,
        }
    }

    impl BinaryExpr {
        fn impl_should_read(&self, stats: &BatchStats) -> PolarsResult<bool> {
            // See: #5864 for the rationale behind this.
            use Expr::*;
            use Operator::*;
            if !self.expr.into_iter().all(|e| match e {
                BinaryExpr { op, .. } => {
                    !matches!(op, Multiply | Divide | TrueDivide | FloorDivide | Modulus)
                },
                Column(_) | Literal(_) | Alias(_, _) => true,
                _ => false,
            }) {
                return Ok(true);
            }
            let schema = stats.schema();
            let Some(fld_l) = self.left.to_field(schema).ok() else {
                return Ok(true);
            };
            let Some(fld_r) = self.right.to_field(schema).ok() else {
                return Ok(true);
            };

            #[cfg(debug_assertions)]
            {
                match (fld_l.data_type(), fld_r.data_type()) {
                    #[cfg(feature = "dtype-categorical")]
                    (DataType::String, DataType::Categorical(_, _) | DataType::Enum(_, _)) => {},
                    #[cfg(feature = "dtype-categorical")]
                    (DataType::Categorical(_, _) | DataType::Enum(_, _), DataType::String) => {},
                    (l, r) if l != r => panic!("implementation error: {l:?}, {r:?}"),
                    _ => {},
                }
            }

            let dummy = DataFrame::empty();
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
                        },
                    }
                },
                (true, false) => {
                    let r = stats.get_stats(fld_r.name())?;
                    match r.to_min_max() {
                        None => Ok(true),
                        Some(min_max_s) => {
                            // will be incorrect if not
                            debug_assert_eq!(min_max_s.null_count(), 0);
                            let lit_s = self.left.evaluate(&dummy, &state).unwrap();
                            Ok(apply_operator_stats_lhs_lit(&lit_s, &min_max_s, self.op))
                        },
                    }
                },
                // Default: read the file
                _ => Ok(true),
            };
            out.inspect(|read| {
                if state.verbose() && *read {
                    eprintln!("parquet file must be read, statistics not sufficient for predicate.")
                } else if state.verbose() && !*read {
                    eprintln!("parquet file can be skipped, the statistics were sufficient to apply the predicate.")
                }
            })
        }
    }

    impl StatsEvaluator for BinaryExpr {
        fn should_read(&self, stats: &BatchStats) -> PolarsResult<bool> {
            if std::env::var("POLARS_NO_PARQUET_STATISTICS").is_ok() {
                return Ok(true);
            }

            use Operator::*;
            match (
                self.left.as_stats_evaluator(),
                self.right.as_stats_evaluator(),
            ) {
                (Some(l), Some(r)) => match self.op {
                    And | LogicalAnd => Ok(l.should_read(stats)? && r.should_read(stats)?),
                    Or | LogicalOr => Ok(l.should_read(stats)? || r.should_read(stats)?),
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
