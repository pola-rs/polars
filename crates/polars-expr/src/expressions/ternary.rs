use std::ops::Not;

use polars_core::prelude::*;
use polars_core::runtime::RAYON;
use polars_plan::prelude::*;
use recursive::recursive;

use super::*;
use crate::expressions::{AggregationContext, PhysicalExpr};

pub struct TernaryExpr {
    predicate: Arc<dyn PhysicalExpr>,
    truthy: Arc<dyn PhysicalExpr>,
    falsy: Arc<dyn PhysicalExpr>,
    expr: Expr,
    // Can be expensive on small data to run literals in parallel.
    run_par: bool,
    returns_scalar: bool,
    short_circuit: bool,
}

impl TernaryExpr {
    pub fn new(
        predicate: Arc<dyn PhysicalExpr>,
        truthy: Arc<dyn PhysicalExpr>,
        falsy: Arc<dyn PhysicalExpr>,
        expr: Expr,
        run_par: bool,
        returns_scalar: bool,
        short_circuit: bool,
    ) -> Self {
        Self {
            predicate,
            truthy,
            falsy,
            expr,
            run_par,
            returns_scalar,
            short_circuit,
        }
    }

    fn finish_short_circuit_fast_path(
        &self,
        column: Column,
        df: &DataFrame,
    ) -> PolarsResult<Column> {
        let name = self.to_field(df.schema())?.name().clone();
        let length = if self.returns_scalar { 1 } else { df.height() };
        Ok(broadcast_scalar_output(column, length).with_name(name))
    }
}

fn broadcast_scalar_output(column: Column, length: usize) -> Column {
    if column.len() == 1 && length != 1 {
        column.new_from_index(0, length)
    } else {
        column
    }
}

fn prepare_branch(column: Column, expected_len: usize, branch_name: &str) -> PolarsResult<Column> {
    match column.len() {
        1 => Ok(broadcast_scalar_output(column, expected_len)),
        len if len == expected_len => Ok(column),
        len => polars_bail!(
            ShapeMismatch:
            "short-circuit ternary branch produced length {len}, expected either 1 or {expected_len} for the {branch_name} branch"
        ),
    }
}

fn finish_short_circuit(
    mask: &BooleanChunked,
    truthy: Column,
    falsy: Column,
    output_field: Field,
) -> PolarsResult<Column> {
    polars_ensure!(
        mask.len() <= IdxSize::MAX as usize,
        ComputeError: "short-circuit ternary output exceeds the index size"
    );

    let true_count = mask.num_trues();
    let mut values = prepare_branch(truthy, true_count, "truthy")?;
    let falsy = prepare_branch(falsy, mask.len() - true_count, "falsy")?;
    values.append(&falsy)?;

    let mut truthy_idx = 0 as IdxSize;
    let mut falsy_idx = true_count as IdxSize;
    let indices = mask
        .iter()
        .map(|mask_value| {
            if mask_value == Some(true) {
                let idx = truthy_idx;
                truthy_idx += 1;
                idx
            } else {
                let idx = falsy_idx;
                falsy_idx += 1;
                idx
            }
        })
        .collect();
    let indices = IdxCa::from_vec(PlSmallStr::EMPTY, indices);

    Ok(values
        .take(&indices)?
        .with_name(output_field.name().clone()))
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
        .with_name(ac_truthy.get_values().name().clone());

    // Aggregation leaves only a single chunk.
    let arr = ca.downcast_iter().next().unwrap();
    let list_vals_len = arr.values().len();

    let mut out = ca.into_column();
    if ac_truthy.arity_should_explode() && ac_falsy.arity_should_explode() && ac_mask.arity_should_explode() &&
        // Exploded list should be equal to groups length.
        list_vals_len == ac_truthy.groups.len()
    {
        out = out.explode(ExplodeOptions {
            empty_as_null: true,
            keep_nulls: true,
        })?
    }

    ac_truthy.with_agg_state(AggState::AggregatedList(out));
    ac_truthy.with_update_groups(UpdateGroups::WithSeriesLen);

    Ok(ac_truthy)
}

impl PhysicalExpr for TernaryExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    #[recursive]
    fn evaluate_impl(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let mut state = state.split();
        // Don't cache window functions as they run in parallel.
        state.remove_cache_window_flag();
        let mask_series = self.predicate.evaluate(df, &state)?;
        let mask = mask_series.bool()?.clone();

        if !self.short_circuit {
            let op_truthy = || self.truthy.evaluate(df, &state);
            let op_falsy = || self.falsy.evaluate(df, &state);
            let (truthy, falsy) = if self.run_par {
                RAYON.install(|| rayon::join(op_truthy, op_falsy))
            } else {
                (op_truthy(), op_falsy())
            };
            let truthy = truthy?;
            let falsy = falsy?;

            return truthy.zip_with(&mask, &falsy);
        }

        let mask = mask.fill_null_with_values(false)?;

        if mask.len() == 1 {
            let branch = if mask.get(0) == Some(true) {
                self.truthy.evaluate(df, &state)
            } else {
                self.falsy.evaluate(df, &state)
            }?;
            return self.finish_short_circuit_fast_path(branch, df);
        }

        if df.height() == 0 {
            let output_field = self.to_field(df.schema())?;
            return Ok(Column::full_null(
                output_field.name().clone(),
                0,
                output_field.dtype(),
            ));
        }

        let true_count = mask.num_trues();

        if true_count == df.height() {
            let truthy = self.truthy.evaluate(df, &state)?;
            return self.finish_short_circuit_fast_path(truthy, df);
        }

        if true_count == 0 {
            let falsy = self.falsy.evaluate(df, &state)?;
            return self.finish_short_circuit_fast_path(falsy, df);
        }

        let falsy_mask = (&mask).not();
        let truthy_df = df.filter(&mask)?;
        let falsy_df = df.filter(&falsy_mask)?;

        let mut truthy_state = state.split();
        truthy_state.remove_cache_window_flag();
        let mut falsy_state = state.split();
        falsy_state.remove_cache_window_flag();

        let truthy = self.truthy.evaluate(&truthy_df, &truthy_state)?;
        let falsy = self.falsy.evaluate(&falsy_df, &falsy_state)?;
        let output_field = self.to_field(df.schema())?;

        finish_short_circuit(&mask, truthy, falsy, output_field)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.truthy.to_field(input_schema)
    }

    #[allow(clippy::ptr_arg)]
    #[recursive]
    fn evaluate_on_groups_impl<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        polars_ensure!(
            !self.short_circuit,
            InvalidOperation:
                "short-circuit when-then-otherwise is not yet supported in grouped execution"
        );

        let op_mask = || self.predicate.evaluate_on_groups(df, groups, state);
        let op_truthy = || self.truthy.evaluate_on_groups(df, groups, state);
        let op_falsy = || self.falsy.evaluate_on_groups(df, groups, state);
        let (ac_mask, (ac_truthy, ac_falsy)) = if self.run_par {
            RAYON.install(|| rayon::join(op_mask, || rayon::join(op_truthy, op_falsy)))
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
        // Unknown groups (rows and their positions do not match initial groups).
        let mut non_aggregated_unknown_groups = false;

        for ac in [&ac_mask, &ac_truthy, &ac_falsy].into_iter() {
            match ac.agg_state() {
                LiteralScalar(s) => {
                    has_non_unit_literal = s.len() != 1;

                    if has_non_unit_literal {
                        break;
                    }
                },
                NotAggregated(_) => {
                    non_aggregated_unknown_groups |= !ac.original_groups;
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

        if !has_aggregated && !non_aggregated_unknown_groups {
            // Everything is flat (either NotAggregated or a unit literal).
            if state.verbose() {
                eprintln!("ternary agg: finish all not-aggregated or unit literal");
            }

            let out = ac_truthy
                .get_values()
                .zip_with(ac_mask.get_values().bool()?, ac_falsy.get_values())?;

            for ac in [&ac_mask, &ac_truthy, &ac_falsy].into_iter() {
                if matches!(ac.agg_state(), NotAggregated(_)) {
                    let ac_target = ac;

                    return Ok(AggregationContext {
                        state: NotAggregated(out),
                        groups: ac_target.groups.clone(),
                        update_groups: ac_target.update_groups,
                        original_groups: ac_target.original_groups,
                    });
                }
            }

            ac_truthy.with_agg_state(LiteralScalar(out));

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
            if !matches!(ac.agg_state(), LiteralScalar(_)) {
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
                    eprintln!(
                        "ternary agg: finish as iters due to mix of AggregatedScalar and AggregatedList"
                    )
                }
                return finish_as_iters(ac_truthy, ac_falsy, ac_mask);
            }
        }

        // At this point, the possible combinations are:
        // * mix of unit literals and AggregatedScalar
        //   * `zip_with` can be called directly with the series
        // * mix of unit literals and AggregatedList
        //   * `zip_with` can be called with the flat values after the offsets
        //     have been checked for alignment
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
                    s.list().unwrap().get_inner().into_column()
                } else {
                    ac_truthy.get_values().clone()
                };

                let falsy = if let AggregatedList(s) = ac_falsy.agg_state() {
                    s.list().unwrap().get_inner().into_column()
                } else {
                    ac_falsy.get_values().clone()
                };

                let mask = if let AggregatedList(s) = ac_mask.agg_state() {
                    s.list().unwrap().get_inner().into_column()
                } else {
                    ac_mask.get_values().clone()
                };

                let out = truthy.zip_with(mask.bool()?, &falsy)?;

                // The output series is guaranteed to be aligned with expected
                // offsets buffer of the result, so we construct the result
                // ListChunked directly from the 2.
                let out = out.rechunk();
                // @scalar-opt
                // @partition-opt
                let values = out.as_materialized_series().array_ref(0);
                let offsets = ac_target.get_values().list().unwrap().offsets()?;
                let inner_type = out.dtype();
                let dtype = LargeListArray::default_datatype(values.dtype().clone());

                // SAFETY: offsets are correct.
                let out = LargeListArray::new(dtype, offsets, values.clone(), None);

                let mut out = ListChunked::with_chunk(truthy.name().clone(), out);
                unsafe { out.to_logical(inner_type.clone()) };

                if ac_target.get_values().list().unwrap()._can_fast_explode() {
                    out.set_fast_explode();
                };

                let out = out.into_column();

                AggregatedList(out)
            },
            AggregatedScalar(_) => {
                if state.verbose() {
                    eprintln!("ternary agg: finish AggregatedScalar")
                }

                let out = ac_truthy
                    .get_values()
                    .zip_with(ac_mask.get_values().bool()?, ac_falsy.get_values())?;
                AggregatedScalar(out)
            },
            _ => {
                unreachable!()
            },
        };

        Ok(AggregationContext {
            state: agg_state_out,
            groups: ac_target.groups.clone(),
            update_groups: ac_target.update_groups,
            original_groups: ac_target.original_groups,
        })
    }

    fn is_scalar(&self) -> bool {
        self.returns_scalar
    }
}
