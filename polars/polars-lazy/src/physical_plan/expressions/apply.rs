use std::sync::Arc;

use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;
#[cfg(feature = "parquet")]
use polars_io::parquet::predicates::BatchStats;
#[cfg(feature = "parquet")]
use polars_io::predicates::StatsEvaluator;
#[cfg(feature = "parquet")]
use polars_plan::dsl::FunctionExpr;
use rayon::prelude::*;

use crate::physical_plan::expression_err;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct ApplyExpr {
    pub inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub function: SpecialEq<Arc<dyn SeriesUdf>>,
    pub expr: Expr,
    pub collect_groups: ApplyOptions,
    pub auto_explode: bool,
    pub allow_rename: bool,
}

impl ApplyExpr {
    pub(crate) fn new_minimal(
        inputs: Vec<Arc<dyn PhysicalExpr>>,
        function: SpecialEq<Arc<dyn SeriesUdf>>,
        expr: Expr,
        collect_groups: ApplyOptions,
    ) -> Self {
        Self {
            inputs,
            function,
            expr,
            collect_groups,
            auto_explode: false,
            allow_rename: false,
        }
    }

    #[allow(clippy::ptr_arg)]
    fn prepare_multiple_inputs<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Vec<AggregationContext<'a>>> {
        POOL.install(|| {
            self.inputs
                .par_iter()
                .map(|e| e.evaluate_on_groups(df, groups, state))
                .collect()
        })
    }

    fn finish_apply_groups<'a>(
        &self,
        mut ac: AggregationContext<'a>,
        ca: ListChunked,
    ) -> AggregationContext<'a> {
        let all_unit_len = all_unit_length(&ca);
        if all_unit_len && self.auto_explode {
            ac.with_series(ca.explode().unwrap().into_series(), true);
            ac.update_groups = UpdateGroups::No;
        } else {
            ac.with_series(ca.into_series(), true);
            ac.with_update_groups(UpdateGroups::WithSeriesLen);
        }
        ac
    }
}

fn all_unit_length(ca: &ListChunked) -> bool {
    assert_eq!(ca.chunks().len(), 1);
    let list_arr = ca.downcast_iter().next().unwrap();
    let offset = list_arr.offsets().as_slice();
    (offset[offset.len() - 1] as usize) == list_arr.len()
}

fn check_map_output_len(input_len: usize, output_len: usize, expr: &Expr) -> PolarsResult<()> {
    if input_len != output_len {
        let msg = "A 'map' functions output length must be equal to that of the input length. Consider using 'apply' in favor of 'map'.";
        Err(expression_err!(msg, expr, ComputeError))
    } else {
        Ok(())
    }
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let mut inputs = self
            .inputs
            .par_iter()
            .map(|e| e.evaluate(df, state))
            .collect::<PolarsResult<Vec<_>>>()?;

        if self.allow_rename {
            return self.function.call_udf(&mut inputs);
        }
        let in_name = inputs[0].name().to_string();
        let mut out = self.function.call_udf(&mut inputs)?;
        if in_name != out.name() {
            out.rename(&in_name);
        }
        Ok(out)
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        if self.inputs.len() == 1 {
            let mut ac = self.inputs[0].evaluate_on_groups(df, groups, state)?;

            match (state.overlapping_groups(), self.collect_groups) {
                (_, ApplyOptions::ApplyList) => {
                    let s = self.function.call_udf(&mut [ac.aggregated()])?;
                    ac.with_series(s, true);
                    Ok(ac)
                }
                // overlapping groups always take this branch as explode/flat_naive bloats data size
                (_, ApplyOptions::ApplyGroups) | (true, _) => {
                    let s = ac.series();

                    if matches!(ac.agg_state(), AggState::AggregatedFlat(_)) {
                        let msg = format!(
                            "Cannot aggregate {:?}. The column is already aggregated.",
                            self.expr
                        );
                        return Err(expression_err!(msg, self.expr, ComputeError));
                    }

                    // collection of empty list leads to a null dtype
                    // see: #3687
                    if s.len() == 0 {
                        // create input for the function to determine the output dtype
                        // see #3946
                        let agg = ac.aggregated();
                        let agg = agg.list().unwrap();
                        let input_dtype = agg.inner_dtype();

                        let input = Series::full_null("", 0, &input_dtype);

                        let output = self.function.call_udf(&mut [input])?;
                        let ca = ListChunked::full(ac.series().name(), &output, 0);
                        return Ok(self.finish_apply_groups(ac, ca));
                    }

                    let name = s.name().to_string();

                    let mut ca: ListChunked = ac
                        .aggregated()
                        .list()
                        .unwrap()
                        .par_iter()
                        .map(|opt_s| {
                            opt_s.and_then(|s| {
                                let mut container = [s];
                                self.function.call_udf(&mut container).ok()
                            })
                        })
                        .collect();

                    ca.rename(&name);
                    Ok(self.finish_apply_groups(ac, ca))
                }
                (_, ApplyOptions::ApplyFlat) => {
                    // make sure the groups are updated because we are about to throw away
                    // the series' length information
                    let set_update_groups = match ac.update_groups {
                        UpdateGroups::WithSeriesLen => {
                            ac.groups();
                            true
                        }
                        UpdateGroups::WithSeriesLenOwned(_) => false,
                        UpdateGroups::No | UpdateGroups::WithGroupsLen => false,
                    };

                    if let UpdateGroups::WithSeriesLen = ac.update_groups {
                        ac.groups();
                    }

                    let input = ac.flat_naive().into_owned();
                    let input_len = input.len();
                    let s = self.function.call_udf(&mut [input])?;

                    check_map_output_len(input_len, s.len(), &self.expr)?;
                    ac.with_series(s, false);

                    if set_update_groups {
                        // The flat_naive orders by groups, so we must create new groups
                        // not by series length as we don't have an agg_list, but by original
                        // groups length
                        ac.update_groups = UpdateGroups::WithGroupsLen;
                    }
                    Ok(ac)
                }
            }
        } else {
            let mut acs = self.prepare_multiple_inputs(df, groups, state)?;

            match (state.overlapping_groups(), self.collect_groups) {
                (_, ApplyOptions::ApplyList) => {
                    let mut s = acs.iter_mut().map(|ac| ac.aggregated()).collect::<Vec<_>>();
                    let s = self.function.call_udf(&mut s)?;
                    // take the first aggregation context that as that is the input series
                    let mut ac = acs.swap_remove(0);
                    ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    ac.with_series(s, true);
                    Ok(ac)
                }

                // overlapping groups always take this branch as explode bloats data size
                (_, ApplyOptions::ApplyGroups) | (true, _) => {
                    // if
                    // - there are overlapping groups
                    // - can do elementwise operations
                    // - we don't have to explode
                    // then apply flat
                    if let (
                        true,
                        ApplyOptions::ApplyFlat,
                        AggState::AggregatedFlat(_) | AggState::NotAggregated(_),
                    ) = (
                        state.overlapping_groups(),
                        self.collect_groups,
                        acs[0].agg_state(),
                    ) {
                        apply_multiple_flat(acs, self.function.as_ref(), &self.expr)
                    } else {
                        let mut container = vec![Default::default(); acs.len()];
                        let name = acs[0].series().name().to_string();

                        // aggregate representation of the aggregation contexts
                        // then unpack the lists and finally create iterators from this list chunked arrays.
                        let mut iters = acs
                            .iter_mut()
                            .map(|ac| ac.iter_groups())
                            .collect::<Vec<_>>();

                        // length of the items to iterate over
                        let len = iters[0].size_hint().0;

                        let mut ca: ListChunked = (0..len)
                            .map(|_| {
                                container.clear();
                                for iter in &mut iters {
                                    match iter.next().unwrap() {
                                        None => return None,
                                        Some(s) => container.push(s.deep_clone()),
                                    }
                                }
                                self.function.call_udf(&mut container).ok()
                            })
                            .collect_trusted();
                        ca.rename(&name);
                        drop(iters);

                        // take the first aggregation context that as that is the input series
                        let ac = acs.swap_remove(0);
                        let ac = self.finish_apply_groups(ac, ca);
                        Ok(ac)
                    }
                }
                (_, ApplyOptions::ApplyFlat) => {
                    apply_multiple_flat(acs, self.function.as_ref(), &self.expr)
                }
            }
        }
    }
    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.expr.to_field(input_schema, Context::Default)
    }
    fn is_valid_aggregation(&self) -> bool {
        matches!(self.collect_groups, ApplyOptions::ApplyGroups)
    }
    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn polars_io::predicates::StatsEvaluator> {
        if matches!(
            self.expr,
            Expr::Function {
                function: FunctionExpr::IsNull,
                ..
            }
        ) {
            Some(self)
        } else {
            None
        }
    }
    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        if self.inputs.len() == 1 && matches!(self.collect_groups, ApplyOptions::ApplyFlat) {
            Some(self)
        } else {
            None
        }
    }
}

fn apply_multiple_flat<'a>(
    mut acs: Vec<AggregationContext<'a>>,
    function: &dyn SeriesUdf,
    expr: &Expr,
) -> PolarsResult<AggregationContext<'a>> {
    let mut s = acs
        .iter_mut()
        .map(|ac| {
            // make sure the groups are updated because we are about to throw away
            // the series length information
            if let UpdateGroups::WithSeriesLen = ac.update_groups {
                ac.groups();
            }

            ac.flat_naive().into_owned()
        })
        .collect::<Vec<_>>();

    let input_len = s[0].len();
    let s = function.call_udf(&mut s)?;
    check_map_output_len(input_len, s.len(), expr)?;

    // take the first aggregation context that as that is the input series
    let mut ac = acs.swap_remove(0);
    ac.with_series(s, false);
    Ok(ac)
}

#[cfg(feature = "parquet")]
impl StatsEvaluator for ApplyExpr {
    fn should_read(&self, stats: &BatchStats) -> PolarsResult<bool> {
        if matches!(
            self.expr,
            Expr::Function {
                function: FunctionExpr::IsNull,
                ..
            }
        ) {
            let root = expr_to_leaf_column_name(&self.expr)?;

            let read = true;
            let skip = false;

            match stats.get_stats(&root).ok() {
                Some(st) => match st.null_count() {
                    Some(0) => Ok(skip),
                    _ => Ok(read),
                },
                None => Ok(read),
            }
        } else {
            Ok(true)
        }
    }
}

impl PartitionedAggregation for ApplyExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let a = self.inputs[0].as_partitioned_aggregator().unwrap();
        let s = a.evaluate_partitioned(df, groups, state)?;

        if self.allow_rename {
            return self.function.call_udf(&mut [s]);
        }
        let in_name = s.name().to_string();
        let mut out = self.function.call_udf(&mut [s])?;
        if in_name != out.name() {
            out.rename(&in_name);
        }
        Ok(out)
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
