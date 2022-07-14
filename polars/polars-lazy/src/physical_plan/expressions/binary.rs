use crate::physical_plan::state::{ExecutionState, StateFlags};
use crate::prelude::*;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::series::unstable::UnstableSeries;
use polars_core::{prelude::*, POOL};
use std::convert::TryFrom;
use std::sync::Arc;

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
fn apply_operator_owned(left: Series, right: Series, op: Operator) -> Result<Series> {
    match op {
        Operator::Gt => ChunkCompare::<&Series>::gt(&left, &right).map(|ca| ca.into_series()),
        Operator::GtEq => ChunkCompare::<&Series>::gt_eq(&left, &right).map(|ca| ca.into_series()),
        Operator::Lt => ChunkCompare::<&Series>::lt(&left, &right).map(|ca| ca.into_series()),
        Operator::LtEq => ChunkCompare::<&Series>::lt_eq(&left, &right).map(|ca| ca.into_series()),
        Operator::Eq => ChunkCompare::<&Series>::equal(&left, &right).map(|ca| ca.into_series()),
        Operator::NotEq => {
            ChunkCompare::<&Series>::not_equal(&left, &right).map(|ca| ca.into_series())
        }
        Operator::Plus => Ok(left + right),
        Operator::Minus => Ok(left - right),
        Operator::Multiply => Ok(left * right),
        Operator::Divide => Ok(&left / &right),
        Operator::TrueDivide => {
            use DataType::*;
            match left.dtype() {
                Date | Datetime(_, _) | Float32 | Float64 => Ok(&left / &right),
                _ => Ok(&left.cast(&Float64)? / &right.cast(&Float64)?),
            }
        }
        Operator::And => left.bitand(&right),
        Operator::Or => left.bitor(&right),
        Operator::Xor => left.bitxor(&right),
        Operator::Modulus => Ok(&left % &right),
    }
}

pub fn apply_operator(left: &Series, right: &Series, op: Operator) -> Result<Series> {
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
        Operator::TrueDivide => {
            use DataType::*;
            match left.dtype() {
                Date | Datetime(_, _) | Float32 | Float64 => Ok(left / right),
                _ => Ok(&left.cast(&Float64)? / &right.cast(&Float64)?),
            }
        }
        Operator::And => left.bitand(right),
        Operator::Or => left.bitor(right),
        Operator::Xor => left.bitxor(right),
        Operator::Modulus => Ok(left % right),
    }
}

impl PhysicalExpr for BinaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let mut state = state.split();
        // don't cache window functions as they run in parallel
        state.flags.remove(StateFlags::CACHE_WINDOW_EXPR);
        let (lhs, rhs) = POOL.install(|| {
            rayon::join(
                || self.left.evaluate(df, &state),
                || self.right.evaluate(df, &state),
            )
        });
        apply_operator_owned(lhs?, rhs?, self.op)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let (result_a, result_b) = POOL.install(|| {
            rayon::join(
                || self.left.evaluate_on_groups(df, groups, state),
                || self.right.evaluate_on_groups(df, groups, state),
            )
        });
        let mut ac_l = result_a?;
        let mut ac_r = result_b?;

        if !ac_l.can_combine(&ac_r) {
            return Err(PolarsError::InvalidOperation(
                "\
            cannot combine this binary expression, the groups do not match"
                    .into(),
            ));
        }

        match (ac_l.agg_state(), ac_r.agg_state(), self.op) {
            // Some aggregations must return boolean masks that fit the group. That's why not all literals can take this path.
            // only literals that are used in arithmetic
            (
                AggState::AggregatedFlat(lhs),
                AggState::Literal(rhs),
                Operator::Plus
                | Operator::Minus
                | Operator::Divide
                | Operator::Multiply
                | Operator::Modulus
                | Operator::TrueDivide,
            )
            | (
                AggState::Literal(lhs),
                AggState::AggregatedFlat(rhs),
                Operator::Plus
                | Operator::Minus
                | Operator::Divide
                | Operator::Multiply
                | Operator::Modulus
                | Operator::TrueDivide,
            ) => {
                // we want to be able to mutate in place
                // so we take the lhs to make sure that we drop
                let lhs = lhs.clone();
                let rhs = rhs.clone();

                // drop lhs so that we might operate in place
                {
                    let _ = ac_l.take();
                }

                let out = apply_operator_owned(lhs, rhs, self.op)?;

                ac_l.with_series(out, true);
                Ok(ac_l)
            }
            // One of the two exprs is aggregated with flat aggregation, e.g. `e.min(), e.max(), e.first()`
            // the other is a literal value. In that case it is unlikely we want to expand this to the
            // group sizes.
            //
            (AggState::AggregatedFlat(_), AggState::Literal(_), _)
            | (AggState::Literal(_), AggState::AggregatedFlat(_), _) => {
                let l = ac_l.series().clone();
                let r = ac_r.series().clone();

                // drop lhs so that we might operate in place
                {
                    let _ = ac_l.take();
                }
                let out = apply_operator_owned(l, r, self.op)?;

                ac_l.with_series(out, true);
                Ok(ac_l)
            }
            // One of the two exprs is aggregated with flat aggregation, e.g. `e.min(), e.max(), e.first()`

            // if the groups_len == df.len we can just apply all flat.
            // within an aggregation a `col().first() - lit(0)` must still produce a boolean array of group length,
            // that's why a literal also takes this branch
            (AggState::AggregatedFlat(s), AggState::NotAggregated(_), _)
                if s.len() != df.height() =>
            {
                // this is a flat series of len eq to group tuples
                let l = ac_l.aggregated_arity_operation();
                let l = l.as_ref();
                let arr_l = &l.chunks()[0];

                // we create a dummy Series that is not cloned nor moved
                // so we can swap the ArrayRef during the hot loop
                // this prevents a series Arc alloc and a vec alloc per iteration
                let dummy = Series::try_from(("dummy", vec![arr_l.clone()])).unwrap();
                // keep logical type info
                let dummy = dummy.cast(l.dtype()).unwrap();
                let mut us = UnstableSeries::new(&dummy);

                // this is now a list
                let r = ac_r.aggregated_arity_operation();
                let r = r.list().unwrap();

                let mut ca: ListChunked = r
                    .amortized_iter()
                    .enumerate()
                    .map(|(idx, opt_s)| {
                        opt_s
                            .map(|s| {
                                let r = s.as_ref();
                                // TODO: optimize this?

                                // Safety:
                                // we are in bounds
                                let arr = unsafe { arr_l.slice_unchecked(idx, 1) };
                                us.swap(arr);

                                let l = us.as_ref();

                                apply_operator(l, r, self.op)
                            })
                            .transpose()
                    })
                    .collect::<Result<_>>()?;
                ca.rename(l.name());

                ac_l.with_series(ca.into_series(), true);
                ac_l.with_update_groups(UpdateGroups::WithGroupsLen);
                Ok(ac_l)
            }
            // if the groups_len == df.len we can just apply all flat.
            (
                AggState::AggregatedList(_) | AggState::NotAggregated(_),
                AggState::AggregatedFlat(s),
                _,
            ) if s.len() != df.height() => {
                // this is now a list
                let l = ac_l.aggregated_arity_operation();
                let l = l.list().unwrap();

                // this is a flat series of len eq to group tuples
                let r = ac_r.aggregated_arity_operation();
                assert_eq!(l.len(), groups.len());
                let r = r.as_ref();
                let arr_r = &r.chunks()[0];

                // we create a dummy Series that is not cloned nor moved
                // so we can swap the ArrayRef during the hot loop
                // this prevents a series Arc alloc and a vec alloc per iteration
                let dummy = Series::try_from(("dummy", vec![arr_r.clone()])).unwrap();
                // keep logical type info
                let dummy = dummy.cast(r.dtype()).unwrap();
                let mut us = UnstableSeries::new(&dummy);

                let mut ca: ListChunked = l
                    .amortized_iter()
                    .enumerate()
                    .map(|(idx, opt_s)| {
                        opt_s
                            .map(|s| {
                                let l = s.as_ref();
                                // TODO: optimize this? Its slow.
                                // Safety:
                                // we are in bounds
                                let arr = unsafe { arr_r.slice_unchecked(idx, 1) };
                                us.swap(arr);
                                let r = us.as_ref();

                                apply_operator(l, r, self.op)
                            })
                            .transpose()
                    })
                    .collect::<Result<_>>()?;
                ca.rename(l.name());

                ac_l.with_series(ca.into_series(), true);
                // Todo! maybe always update with groups len here?
                if matches!(ac_l.update_groups, UpdateGroups::WithSeriesLen)
                    || matches!(ac_r.update_groups, UpdateGroups::WithSeriesLen)
                {
                    ac_l.with_update_groups(UpdateGroups::WithSeriesLen);
                } else {
                    ac_l.with_update_groups(UpdateGroups::WithGroupsLen);
                }
                Ok(ac_l)
            }
            (AggState::AggregatedList(_), AggState::NotAggregated(_) | AggState::Literal(_), _)
            | (AggState::NotAggregated(_) | AggState::Literal(_), AggState::AggregatedList(_), _) =>
            {
                ac_l.sort_by_groups();
                ac_r.sort_by_groups();

                let lhs = ac_l.flat_naive().as_ref().clone();
                let rhs = ac_r.flat_naive().as_ref().clone();

                // drop lhs so that we might operate in place
                {
                    let _ = ac_l.take();
                }

                let out = apply_operator_owned(lhs, rhs, self.op)?;

                // we flattened the series, so that sorts by group
                ac_l.with_update_groups(UpdateGroups::WithGroupsLen);
                ac_l.with_series(out, false);
                Ok(ac_l)
            }
            // flatten the Series and apply the operators
            (AggState::AggregatedList(_), AggState::AggregatedList(_), _) => {
                let lhs = ac_l.flat_naive().as_ref().clone();
                let rhs = ac_r.flat_naive().as_ref().clone();

                // drop lhs so that we might operate in place
                {
                    let _ = ac_l.take();
                }

                let out = apply_operator_owned(lhs, rhs, self.op)?;

                ac_l.combine_groups(ac_r).with_series(out, false);
                ac_l.with_update_groups(UpdateGroups::WithGroupsLen);
                Ok(ac_l)
            }
            // Both are or a flat series
            // so we can flatten the Series and apply the operators
            _ => {
                // Check if the group state of `ac_a` differs from the original `GroupTuples`.
                // If this is the case we might need to align the groups. But only if `ac_b` is not a
                // `Literal` as literals don't have any groups, the changed group order does not matter
                // in that case
                let different_group_state =
                    |ac_a: &AggregationContext, ac_b: &AggregationContext| {
                        (ac_a.update_groups != UpdateGroups::No)
                            && !matches!(ac_b.agg_state(), AggState::Literal(_))
                    };

                // the groups state differs, so we aggregate both and flatten again to make them align
                if different_group_state(&ac_l, &ac_r) || different_group_state(&ac_r, &ac_l) {
                    // use the aggregated state to determine the new groups
                    let lhs = ac_l.aggregated();
                    ac_l.with_update_groups(UpdateGroups::WithSeriesLenOwned(lhs.clone()));

                    // we should only explode lists
                    // not aggregated flat states
                    let flatten = |s: Series| match s.dtype() {
                        DataType::List(_) => s.explode(),
                        _ => Ok(s),
                    };

                    let out =
                        apply_operator(&flatten(lhs)?, &flatten(ac_r.aggregated())?, self.op)?;
                    ac_l.with_series(out, false);
                    Ok(ac_l)
                } else {
                    let lhs = ac_l.flat_naive().as_ref().clone();
                    let rhs = ac_r.flat_naive().as_ref().clone();

                    // drop lhs so that we might operate in place
                    {
                        let _ = ac_l.take();
                    }

                    let out = apply_operator_owned(lhs, rhs, self.op)?;

                    ac_l.combine_groups(ac_r).with_series(out, false);

                    Ok(ac_l)
                }
            }
        }
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
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
    use super::*;
    use polars_io::parquet::predicates::BatchStats;
    use polars_io::predicates::StatsEvaluator;

    fn apply_operator_stats_rhs_lit(min_max: &Series, literal: &Series, op: Operator) -> bool {
        match op {
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
        fn impl_should_read(&self, stats: &BatchStats) -> Result<bool> {
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
                    (l, r) if l != r => panic!("implementation error: {:?}, {:?}", l, r),
                    _ => {}
                }
            }

            let dummy = DataFrame::new_no_checks(vec![]);
            let state = ExecutionState::new();

            let out = match (fld_l.name().as_str(), fld_r.name().as_str()) {
                (_, "literal") => {
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
                ("literal", _) => {
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
                    eprintln!("parquet file must be read, statistics not sufficient to for predicate.")
                } else if state.verbose() && !read {
                    eprintln!("parquet file can be skipped, the statistics were sufficient to apply the predicate.")
                };
                read
            })
        }
    }

    impl StatsEvaluator for BinaryExpr {
        fn should_read(&self, stats: &BatchStats) -> Result<bool> {
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
    ) -> Result<Series> {
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
    ) -> Result<Series> {
        Ok(partitioned)
    }
}
