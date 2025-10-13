use polars_core::POOL;
use polars_core::prelude::*;
#[cfg(feature = "round_series")]
use polars_ops::prelude::floor_div_series;

use super::*;
use crate::expressions::{
    AggState, AggregationContext, PartitionedAggregation, PhysicalExpr, UpdateGroups,
};

#[derive(Clone)]
pub struct BinaryExpr {
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
    expr: Expr,
    has_literal: bool,
    allow_threading: bool,
    is_scalar: bool,
    output_field: Field,
}

impl BinaryExpr {
    #[expect(clippy::too_many_arguments)]
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
        expr: Expr,
        has_literal: bool,
        allow_threading: bool,
        is_scalar: bool,
        output_field: Field,
    ) -> Self {
        Self {
            left,
            op,
            right,
            expr,
            has_literal,
            allow_threading,
            is_scalar,
            output_field,
        }
    }
}

/// Can partially do operations in place.
fn apply_operator_owned(left: Column, right: Column, op: Operator) -> PolarsResult<Column> {
    match op {
        Operator::Plus => left.try_add_owned(right),
        Operator::Minus => left.try_sub_owned(right),
        Operator::Multiply
            if left.dtype().is_primitive_numeric() && right.dtype().is_primitive_numeric() =>
        {
            left.try_mul_owned(right)
        },
        _ => apply_operator(&left, &right, op),
    }
}

pub fn apply_operator(left: &Column, right: &Column, op: Operator) -> PolarsResult<Column> {
    use DataType::*;
    match op {
        Operator::Gt => ChunkCompareIneq::gt(left, right).map(|ca| ca.into_column()),
        Operator::GtEq => ChunkCompareIneq::gt_eq(left, right).map(|ca| ca.into_column()),
        Operator::Lt => ChunkCompareIneq::lt(left, right).map(|ca| ca.into_column()),
        Operator::LtEq => ChunkCompareIneq::lt_eq(left, right).map(|ca| ca.into_column()),
        Operator::Eq => ChunkCompareEq::equal(left, right).map(|ca| ca.into_column()),
        Operator::NotEq => ChunkCompareEq::not_equal(left, right).map(|ca| ca.into_column()),
        Operator::Plus => left + right,
        Operator::Minus => left - right,
        Operator::Multiply => left * right,
        Operator::Divide => left / right,
        Operator::TrueDivide => match left.dtype() {
            #[cfg(feature = "dtype-decimal")]
            Decimal(_, _) => left / right,
            Duration(_) | Date | Datetime(_, _) | Float32 | Float64 => left / right,
            #[cfg(feature = "dtype-array")]
            Array(..) => left / right,
            #[cfg(feature = "dtype-array")]
            _ if right.dtype().is_array() => left / right,
            List(_) => left / right,
            _ if right.dtype().is_list() => left / right,
            _ if left.dtype().is_string() || right.dtype().is_string() => {
                polars_bail!(InvalidOperation: "cannot divide using strings")
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
                floor_div_series(
                    left.as_materialized_series(),
                    right.as_materialized_series(),
                )
                .map(Column::from)
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
        Operator::EqValidity => left.equal_missing(right).map(|ca| ca.into_column()),
        Operator::NotEqValidity => left.not_equal_missing(right).map(|ca| ca.into_column()),
    }
}

impl BinaryExpr {
    fn apply_elementwise<'a>(
        &self,
        mut ac_l: AggregationContext<'a>,
        mut ac_r: AggregationContext<'a>,
        aggregated: bool,
    ) -> PolarsResult<AggregationContext<'a>> {
        // At this stage, there is no combination of AggregatedList and NotAggregated ACs.

        // Check group lengths in case of all AggList
        if [&ac_l, &ac_r]
            .iter()
            .all(|ac| matches!(ac.state, AggState::AggregatedList(_)))
        {
            ac_l.groups().check_lengths(ac_r.groups())?;
        }

        match (ac_l.agg_state(), ac_r.agg_state()) {
            (_, AggState::AggregatedList(s)) | (AggState::AggregatedList(s), _) => {
                let ca = s.list().unwrap();
                let [col_l, col_r] = [&ac_l, &ac_r].map(|ac| ac.flat_naive().into_owned());

                let out = ca.apply_to_inner(&|_| {
                    apply_operator(&col_l, &col_r, self.op).map(|c| c.take_materialized_series())
                })?;
                let out = out.into_column();

                if ac_l.is_literal() {
                    std::mem::swap(&mut ac_l, &mut ac_r);
                }

                ac_l.with_values(out.into_column(), true, Some(&self.expr))?;
                Ok(ac_l)
            },

            _ => {
                // We want to be able to mutate in place, so we take the lhs to make sure that we drop.
                let lhs = ac_l.get_values().clone();
                let rhs = ac_r.get_values().clone();

                let out = apply_operator_owned(lhs, rhs, self.op)?;

                if ac_l.is_literal() {
                    std::mem::swap(&mut ac_l, &mut ac_r);
                }

                // Drop lhs so that we might operate in place.
                drop(ac_l.take());

                ac_l.with_values(out, aggregated, Some(&self.expr))?;
                Ok(ac_l)
            },
        }
    }

    fn apply_all_literal<'a>(
        &self,
        mut ac_l: AggregationContext<'a>,
        ac_r: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        debug_assert!(ac_l.is_literal() && ac_r.is_literal());
        polars_ensure!(ac_l.groups.len() == ac_r.groups.len(),
            ComputeError: "lhs and rhs should have same number of groups");

        let left_c = ac_l.get_values().rechunk().into_column();
        let right_c = ac_r.get_values().rechunk().into_column();
        let res_c = apply_operator(&left_c, &right_c, self.op)?;
        polars_ensure!(res_c.len() == 1,
            ComputeError: "binary operation on literals expected 1 value, found {}", res_c.len());

        ac_l.with_literal(res_c);
        Ok(ac_l)
    }

    fn apply_group_aware<'a>(
        &self,
        mut ac_l: AggregationContext<'a>,
        mut ac_r: AggregationContext<'a>,
    ) -> PolarsResult<AggregationContext<'a>> {
        let name = self.output_field.name().clone();
        let mut ca = ac_l
            .iter_groups(false)
            .zip(ac_r.iter_groups(false))
            .map(|(l, r)| {
                Some(apply_operator(
                    &l?.as_ref().clone().into_column(),
                    &r?.as_ref().clone().into_column(),
                    self.op,
                ))
            })
            .map(|opt_res| opt_res.transpose())
            .collect::<PolarsResult<ListChunked>>()?
            .with_name(name.clone());
        if ca.is_empty() {
            ca = ListChunked::full_null_with_dtype(name, 0, self.output_field.dtype());
        }

        ac_l.with_update_groups(UpdateGroups::WithSeriesLen);
        ac_l.with_agg_state(AggState::AggregatedList(ca.into_column()));
        Ok(ac_l)
    }
}

impl PhysicalExpr for BinaryExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
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
            ShapeMismatch: "cannot evaluate two Series of different lengths ({} and {})",
            lhs.len(), rhs.len(),
        );
        apply_operator_owned(lhs, rhs, self.op)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let (result_a, result_b) = POOL.install(|| {
            rayon::join(
                || self.left.evaluate_on_groups(df, groups, state),
                || self.right.evaluate_on_groups(df, groups, state),
            )
        });
        let mut ac_l = result_a?;
        let mut ac_r = result_b?;

        // Aggregate NotAggregated into AggregatedList, but only if strictly required AND
        // when there is no risk of memory explosion.
        // See ApplyExpr for additional context
        let mut has_agg_list = false;
        let mut has_agg_scalar = false;
        let mut has_not_agg = false;
        let mut has_not_agg_with_overlapping_groups = false;
        let mut not_agg_groups_may_diverge = false;

        let mut previous: Option<&AggregationContext<'_>> = None;
        for ac in [&ac_l, &ac_r] {
            match ac.state {
                AggState::AggregatedList(_) => {
                    has_agg_list = true;
                },
                AggState::AggregatedScalar(_) => has_agg_scalar = true,
                AggState::NotAggregated(_) => {
                    has_not_agg = true;
                    if let Some(p) = previous {
                        not_agg_groups_may_diverge |=
                            !std::ptr::eq(p.groups.as_ref(), ac.groups.as_ref());
                    }
                    previous = Some(ac);
                    if ac.groups.is_overlapping() {
                        has_not_agg_with_overlapping_groups = true;
                    }
                },
                _ => {},
            }
        }

        let all_literal = !(has_agg_list || has_agg_scalar || has_not_agg);
        let elementwise_must_aggregate =
            has_not_agg && (has_agg_list || not_agg_groups_may_diverge);
        let mut aggregated = has_agg_list || has_agg_scalar;

        // Arithmetic on Decimal is fallible
        let has_decimal_dtype =
            ac_l.get_values().dtype().is_decimal() || ac_r.get_values().dtype().is_decimal();
        let is_fallible = has_decimal_dtype && self.op.is_arithmetic();

        // Dispatch
        // See ApplyExpr for reference logic, except that we do any required
        // aggregation inline. All BinaryExprs are elementwise.
        if all_literal {
            // Fast path
            self.apply_all_literal(ac_l, ac_r)
        } else if has_agg_scalar && (has_agg_list || has_not_agg) {
            // Not compatible
            self.apply_group_aware(ac_l, ac_r)
        } else if elementwise_must_aggregate && has_not_agg_with_overlapping_groups {
            // Compatible but calling aggregated() is too expensive
            self.apply_group_aware(ac_l, ac_r)
        } else if is_fallible
            && (!ac_l.groups_cover_all_values() || !ac_r.groups_cover_all_values())
        {
            // Fallible expression and there are elements that are masked out.
            self.apply_group_aware(ac_l, ac_r)
        } else {
            if elementwise_must_aggregate {
                for ac in [&mut ac_l, &mut ac_r] {
                    if matches!(ac.state, AggState::NotAggregated(_)) {
                        ac.aggregated();
                    }
                }
                aggregated = true;
            }
            self.apply_elementwise(ac_l, ac_r, aggregated)
        }
    }

    fn to_field(&self, _input_schema: &Schema) -> PolarsResult<Field> {
        Ok(self.output_field.clone())
    }

    fn is_scalar(&self) -> bool {
        self.is_scalar
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for BinaryExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let left = self.left.as_partitioned_aggregator().unwrap();
        let right = self.right.as_partitioned_aggregator().unwrap();
        let left = left.evaluate_partitioned(df, groups, state)?;
        let right = right.evaluate_partitioned(df, groups, state)?;
        apply_operator(&left, &right, self.op)
    }

    fn finalize(
        &self,
        partitioned: Column,
        _groups: &GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<Column> {
        Ok(partitioned)
    }
}
