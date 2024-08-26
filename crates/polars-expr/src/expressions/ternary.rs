use polars_core::prelude::*;
use polars_core::POOL;
use polars_plan::prelude::*;

use super::*;
use crate::expressions::{AggregationContext, PhysicalExpr};

pub struct TernaryExpr {
    predicate: Arc<dyn PhysicalExpr>,
    truthy: Arc<dyn PhysicalExpr>,
    falsy: Arc<dyn PhysicalExpr>,
    expr: Expr,
    // Can be expensive on small data to run literals in parallel.
    run_par: bool,
}

impl TernaryExpr {
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        truthy: Arc<dyn PhysicalExpr>,
        falsy: Arc<dyn PhysicalExpr>,
        expr: Expr,
        run_par: bool,
    ) -> Self {
        Self {
            predicate,
            truthy,
            falsy,
            expr,
            run_par,
        }
    }
}

fn finish_as_iters<'a>(
    mut ac_truthy: AggregationContext<'a>,
    mut ac_falsy: AggregationContext<'a>,
    mut ac_mask: AggregationContext<'a>,
) -> PolarsResult<AggregationContext<'a>> {
    let ca = ac_truthy
        .iter_groups(false)
        .zip(ac_falsy.iter_groups(false))
        .zip(ac_mask.iter_groups(false))
        .map(|((truthy, falsy), mask)| {
            match (truthy, falsy, mask) {
                (Some(truthy), Some(falsy), Some(mask)) => Some(
                    truthy
                        .as_ref()
                        .zip_with(mask.as_ref().bool()?, falsy.as_ref()),
                ),
                _ => None,
            }
            .transpose()
        })
        .collect::<PolarsResult<ListChunked>>()?
        .with_name(ac_truthy.series().name());

    // Aggregation leaves only a single chunk.
    let arr = ca.downcast_iter().next().unwrap();
    let list_vals_len = arr.values().len();

    let mut out = ca.into_series();
    if ac_truthy.arity_should_explode() && ac_falsy.arity_should_explode() && ac_mask.arity_should_explode() &&
        // Exploded list should be equal to groups length.
        list_vals_len == ac_truthy.groups.len()
    {
        out = out.explode()?
    }

    ac_truthy.with_series(out, true, None)?;
    Ok(ac_truthy)
}

impl PhysicalExpr for TernaryExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let mut state = state.split();
        // Don't cache window functions as they run in parallel.
        state.remove_cache_window_flag();
        let mask_series = self.predicate.evaluate(df, &state)?;
        let mask = mask_series.bool()?.clone();

        let op_truthy = || self.truthy.evaluate(df, &state);
        let op_falsy = || self.falsy.evaluate(df, &state);
        let (truthy, falsy) = if self.run_par {
            POOL.install(|| rayon::join(op_truthy, op_falsy))
        } else {
            (op_truthy(), op_falsy())
        };
        let truthy = truthy?;
        let falsy = falsy?;

        truthy.zip_with(&mask, &falsy)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.truthy.to_field(input_schema)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let op_mask = || self.predicate.evaluate_on_groups(df, groups, state);
        let op_truthy = || self.truthy.evaluate_on_groups(df, groups, state);
        let op_falsy = || self.falsy.evaluate_on_groups(df, groups, state);
        let (ac_mask, (ac_truthy, ac_falsy)) = if self.run_par {
            POOL.install(|| rayon::join(op_mask, || rayon::join(op_truthy, op_falsy)))
        } else {
            (op_mask(), (op_truthy(), op_falsy()))
        };

        let mut ac_mask = ac_mask?;
        let mut ac_truthy = ac_truthy?;
        let mut ac_falsy = ac_falsy?;

        use AggState::*;

        // Check if there are any:
        // - non-unit literals
        // - AggregatedScalar or AggregatedList
        let mut has_non_unit_literal = false;
        let mut has_aggregated = false;
        // If the length has changed then we must not apply on the flat values
        // as ternary broadcasting is length-sensitive.
        let mut non_aggregated_len_modified = false;

        for ac in [&ac_mask, &ac_truthy, &ac_falsy].into_iter() {
            match ac.agg_state() {
                Literal(s) => {
                    has_non_unit_literal = s.len() != 1;

                    if has_non_unit_literal {
                        break;
                    }
                },
                NotAggregated(_) => {
                    non_aggregated_len_modified |= !ac.original_len;
                },
                AggregatedScalar(_) | AggregatedList(_) => {
                    has_aggregated = true;
                },
            }
        }

        if has_non_unit_literal {
            // finish_as_iters for non-unit literals to avoid materializing the
            // literal inputs per-group.
            if state.verbose() {
                eprintln!("ternary agg: finish as iters due to non-unit literal")
            }
            return finish_as_iters(ac_truthy, ac_falsy, ac_mask);
        }

        if !has_aggregated && !non_aggregated_len_modified {
            // Everything is flat (either NotAggregated or a unit literal).
            if state.verbose() {
                eprintln!("ternary agg: finish all not-aggregated or unit literal");
            }

            let out = ac_truthy
                .series()
                .zip_with(ac_mask.series().bool()?, ac_falsy.series())?;

            for ac in [&ac_mask, &ac_truthy, &ac_falsy].into_iter() {
                if matches!(ac.agg_state(), NotAggregated(_)) {
                    let ac_target = ac;

                    return Ok(AggregationContext {
                        state: NotAggregated(out),
                        groups: ac_target.groups.clone(),
                        sorted: ac_target.sorted,
                        update_groups: ac_target.update_groups,
                        original_len: ac_target.original_len,
                    });
                }
            }

            ac_truthy.with_agg_state(Literal(out));

            return Ok(ac_truthy);
        }

        for ac in [&mut ac_mask, &mut ac_truthy, &mut ac_falsy].into_iter() {
            if matches!(ac.agg_state(), NotAggregated(_)) {
                let _ = ac.aggregated();
            }
        }

        // At this point the input agg states are one of the following:
        // * `Literal` where `s.len() == 1`
        // * `AggregatedList`
        // * `AggregatedScalar`

        let mut non_literal_acs = Vec::<&AggregationContext>::with_capacity(3);

        // non_literal_acs will have at least 1 item because has_aggregated was
        // true from above.
        for ac in [&ac_mask, &ac_truthy, &ac_falsy].into_iter() {
            if !matches!(ac.agg_state(), Literal(_)) {
                non_literal_acs.push(ac);
            }
        }

        for (ac_l, ac_r) in non_literal_acs.iter().zip(non_literal_acs.iter().skip(1)) {
            if std::mem::discriminant(ac_l.agg_state()) != std::mem::discriminant(ac_r.agg_state())
            {
                // Mix of AggregatedScalar and AggregatedList is done per group,
                // as every row of the AggregatedScalar must be broadcasted to a
                // list of the same length as the corresponding AggregatedList
                // row.
                if state.verbose() {
                    eprintln!("ternary agg: finish as iters due to mix of AggregatedScalar and AggregatedList")
                }
                return finish_as_iters(ac_truthy, ac_falsy, ac_mask);
            }
        }

        // At this point, the possible combinations are:
        // * mix of unit literals and AggregatedScalar
        //   * `zip_with` can be called directly with the series
        // * mix of unit literals and AggregatedList
        //   * `zip_with` can be called with the flat values after the offsets
        //     have been been checked for alignment
        let ac_target = non_literal_acs.first().unwrap();

        let agg_state_out = match ac_target.agg_state() {
            AggregatedList(_) => {
                // Ternary can be applied directly on the flattened series,
                // given that their offsets have been checked to be equal.
                if state.verbose() {
                    eprintln!("ternary agg: finish AggregatedList")
                }

                for (ac_l, ac_r) in non_literal_acs.iter().zip(non_literal_acs.iter().skip(1)) {
                    match (ac_l.agg_state(), ac_r.agg_state()) {
                        (AggregatedList(s_l), AggregatedList(s_r)) => {
                            let check = s_l.list().unwrap().offsets()?.as_slice()
                                == s_r.list().unwrap().offsets()?.as_slice();

                            polars_ensure!(
                                check,
                                ShapeMismatch: "shapes of `self`, `mask` and `other` are not suitable for `zip_with` operation"
                            );
                        },
                        _ => unreachable!(),
                    }
                }

                let truthy = if let AggregatedList(s) = ac_truthy.agg_state() {
                    s.list().unwrap().get_inner()
                } else {
                    ac_truthy.series().clone()
                };

                let falsy = if let AggregatedList(s) = ac_falsy.agg_state() {
                    s.list().unwrap().get_inner()
                } else {
                    ac_falsy.series().clone()
                };

                let mask = if let AggregatedList(s) = ac_mask.agg_state() {
                    s.list().unwrap().get_inner()
                } else {
                    ac_mask.series().clone()
                };

                let out = truthy.zip_with(mask.bool()?, &falsy)?;

                // The output series is guaranteed to be aligned with expected
                // offsets buffer of the result, so we construct the result
                // ListChunked directly from the 2.
                let out = out.rechunk();
                let values = out.array_ref(0);
                let offsets = ac_target.series().list().unwrap().offsets()?;
                let inner_type = out.dtype();
                let data_type = LargeListArray::default_datatype(values.data_type().clone());

                // SAFETY: offsets are correct.
                let out = LargeListArray::new(data_type, offsets, values.clone(), None);

                let mut out = ListChunked::with_chunk(truthy.name(), out);
                unsafe { out.to_logical(inner_type.clone()) };

                if ac_target.series().list().unwrap()._can_fast_explode() {
                    out.set_fast_explode();
                };

                let out = out.into_series();

                AggregatedList(out)
            },
            AggregatedScalar(_) => {
                if state.verbose() {
                    eprintln!("ternary agg: finish AggregatedScalar")
                }

                let out = ac_truthy
                    .series()
                    .zip_with(ac_mask.series().bool()?, ac_falsy.series())?;
                AggregatedScalar(out)
            },
            _ => {
                unreachable!()
            },
        };

        Ok(AggregationContext {
            state: agg_state_out,
            groups: ac_target.groups.clone(),
            sorted: ac_target.sorted,
            update_groups: ac_target.update_groups,
            original_len: ac_target.original_len,
        })
    }
    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for TernaryExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let truthy = self.truthy.as_partitioned_aggregator().unwrap();
        let falsy = self.falsy.as_partitioned_aggregator().unwrap();
        let mask = self.predicate.as_partitioned_aggregator().unwrap();

        let truthy = truthy.evaluate_partitioned(df, groups, state)?;
        let falsy = falsy.evaluate_partitioned(df, groups, state)?;
        let mask = mask.evaluate_partitioned(df, groups, state)?;
        let mask = mask.bool()?.clone();

        truthy.zip_with(&mask, &falsy)
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
