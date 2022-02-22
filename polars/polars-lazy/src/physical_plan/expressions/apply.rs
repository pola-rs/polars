use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;
use std::sync::Arc;

pub struct ApplyExpr {
    pub inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub function: NoEq<Arc<dyn SeriesUdf>>,
    pub expr: Expr,
    pub collect_groups: ApplyOptions,
    pub auto_explode: bool,
}

impl ApplyExpr {
    #[allow(clippy::ptr_arg)]
    fn prepare_multiple_inputs<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Vec<AggregationContext<'a>>> {
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
        all_unit_len: bool,
    ) -> AggregationContext<'a> {
        if all_unit_len && self.auto_explode {
            ac.with_series(ca.explode().unwrap().into_series(), true);
        } else {
            ac.with_series(ca.into_series(), true);
        }
        ac.with_all_unit_len(all_unit_len);
        ac.with_update_groups(UpdateGroups::WithSeriesLen);
        ac
    }
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let mut inputs = self
            .inputs
            .par_iter()
            .map(|e| e.evaluate(df, state))
            .collect::<Result<Vec<_>>>()?;
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
    ) -> Result<AggregationContext<'a>> {
        if self.inputs.len() == 1 {
            let mut ac = self.inputs[0].evaluate_on_groups(df, groups, state)?;

            // a unique or a sort
            let mut update_group_tuples = false;

            match self.collect_groups {
                ApplyOptions::ApplyGroups => {
                    let mut container = [Default::default()];
                    let name = ac.series().name().to_string();

                    let mut all_unit_len = true;

                    let mut ca: ListChunked = ac
                        .aggregated()
                        .list()
                        .unwrap()
                        .into_iter()
                        .map(|opt_s| {
                            opt_s.and_then(|s| {
                                let in_len = s.len();
                                container[0] = s;
                                self.function.call_udf(&mut container).ok().map(|s| {
                                    let len = s.len();
                                    if len != in_len {
                                        update_group_tuples = true;
                                    };
                                    if len != 1 {
                                        all_unit_len = false;
                                    }

                                    s
                                })
                            })
                        })
                        .collect();

                    ca.rename(&name);
                    let ac = self.finish_apply_groups(ac, ca, all_unit_len);
                    Ok(ac)
                }
                ApplyOptions::ApplyFlat => {
                    let input = ac.flat_naive().into_owned();
                    let input_len = input.len();
                    let s = self.function.call_udf(&mut [input])?;

                    if s.len() != input_len {
                        return Err(PolarsError::ComputeError("A map function may never return a Series of a different length than its input".into()));
                    }

                    if ac.is_aggregated() {
                        ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    }
                    ac.with_series(s, false);
                    Ok(ac)
                }
                ApplyOptions::ApplyList => {
                    let s = self.function.call_udf(&mut [ac.aggregated()])?;
                    ac.with_series(s, true);
                    Ok(ac)
                }
            }
        } else {
            let mut acs = self.prepare_multiple_inputs(df, groups, state)?;

            match self.collect_groups {
                ApplyOptions::ApplyGroups => {
                    let mut container = vec![Default::default(); acs.len()];
                    let name = acs[0].series().name().to_string();

                    // aggregate representation of the aggregation contexts
                    // then unpack the lists and finaly create iterators from this list chunked arrays.
                    let lists = acs
                        .iter_mut()
                        .map(|ac| {
                            let s = ac.aggregated();
                            s.list().unwrap().clone()
                        })
                        .collect::<Vec<_>>();
                    let mut iters = lists.iter().map(|ca| ca.into_iter()).collect::<Vec<_>>();

                    // length of the items to iterate over
                    let len = lists[0].len();
                    let mut all_unit_len = true;

                    let mut ca: ListChunked = (0..len)
                        .map(|_| {
                            container.clear();
                            for iter in &mut iters {
                                match iter.next().unwrap() {
                                    None => return None,
                                    Some(s) => container.push(s),
                                }
                            }
                            self.function.call_udf(&mut container).ok().map(|s| {
                                if s.len() != 1 {
                                    all_unit_len = false;
                                }
                                s
                            })
                        })
                        .collect_trusted();
                    ca.rename(&name);
                    let ac = acs.pop().unwrap();
                    let ac = self.finish_apply_groups(ac, ca, all_unit_len);
                    Ok(ac)
                }
                ApplyOptions::ApplyFlat => {
                    let mut s = acs
                        .iter()
                        .map(|ac| ac.flat_naive().into_owned())
                        .collect::<Vec<_>>();

                    let s = self.function.call_udf(&mut s)?;
                    let mut ac = acs.pop().unwrap();
                    if ac.is_aggregated() {
                        ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    }
                    ac.with_series(s, false);
                    Ok(ac)
                }
                ApplyOptions::ApplyList => {
                    let mut s = acs.iter_mut().map(|ac| ac.aggregated()).collect::<Vec<_>>();
                    let s = self.function.call_udf(&mut s)?;
                    let mut ac = acs.pop().unwrap();
                    ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    ac.with_series(s, true);
                    Ok(ac)
                }
            }
        }
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.inputs[0].to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}

impl PhysicalAggregation for ApplyExpr {
    fn aggregate(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        let mut ac = self.evaluate_on_groups(df, groups, state)?;
        let s = ac.aggregated();
        Ok(Some(s))
    }
}
