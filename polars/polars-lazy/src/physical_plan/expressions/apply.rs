use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PhysicalAggregation;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use rayon::prelude::*;
use std::sync::Arc;

pub struct ApplyExpr {
    pub inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub function: NoEq<Arc<dyn SeriesUdf>>,
    pub expr: Expr,
    pub collect_groups: ApplyOptions,
}

impl ApplyExpr {
    #[allow(clippy::ptr_arg)]
    fn prepare_multiple_inputs<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<Vec<AggregationContext<'a>>> {
        self.inputs
            .par_iter()
            .map(|e| e.evaluate_on_groups(df, groups, state))
            .collect()
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
        groups: &'a GroupTuples,
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
                                    if s.len() != in_len {
                                        update_group_tuples = true;
                                    }
                                    s
                                })
                            })
                        })
                        .collect();

                    ca.rename(&name);
                    ac.with_series(ca.into_series(), true);
                    ac.with_update_groups(UpdateGroups::WithSeriesLen);
                    Ok(ac)
                }
                ApplyOptions::ApplyFlat => {
                    let s = self.function.call_udf(&mut [ac.flat().into_owned()])?;
                    if ac.is_aggregated() {
                        ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    }
                    ac.with_series(s, false);
                    Ok(ac)
                }
                ApplyOptions::ApplyList => {
                    let s = self
                        .function
                        .call_udf(&mut [ac.aggregated().into_owned()])?;
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

                    let mut ca: ListChunked = (0..len)
                        .map(|_| {
                            container.clear();
                            for iter in &mut iters {
                                match iter.next().unwrap() {
                                    None => return None,
                                    Some(s) => container.push(s),
                                }
                            }
                            self.function.call_udf(&mut container).ok()
                        })
                        .collect();
                    ca.rename(&name);
                    let mut ac = acs.pop().unwrap();
                    ac.with_series(ca.into_series(), true);
                    Ok(ac)
                }
                ApplyOptions::ApplyFlat => {
                    let mut s = acs
                        .iter()
                        .map(|ac| ac.flat().into_owned())
                        .collect::<Vec<_>>();

                    let s = self.function.call_udf(&mut s)?;
                    let mut ac = acs.pop().unwrap();
                    ac.with_update_groups(UpdateGroups::WithGroupsLen);
                    ac.with_series(s, true);
                    Ok(ac)
                }
                ApplyOptions::ApplyList => {
                    let mut s = acs
                        .iter_mut()
                        .map(|ac| ac.aggregated().into_owned())
                        .collect::<Vec<_>>();
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
        groups: &GroupTuples,
        state: &ExecutionState,
    ) -> Result<Option<Series>> {
        if self.inputs.len() == 1 {
            let mut ac = self.inputs[0].evaluate_on_groups(df, groups, state)?;

            match self.collect_groups {
                ApplyOptions::ApplyGroups => {
                    let mut container = [Default::default()];
                    let name = ac.series().name().to_string();

                    let mut ca: ListChunked = ac
                        .aggregated()
                        .list()
                        .unwrap()
                        .into_iter()
                        .map(|opt_s| {
                            opt_s.and_then(|s| {
                                container[0] = s;
                                self.function.call_udf(&mut container).ok()
                            })
                        })
                        .collect();
                    ca.rename(&name);
                    Ok(Some(ca.into_series()))
                }
                ApplyOptions::ApplyFlat => {
                    // the function needs to be called on a flat series
                    // but the series may be flat or aggregated
                    // if its flat, we just apply and return
                    // if not flat, the flattening sorts by group, so we must create new group tuples
                    // and again aggregate.
                    let out = self.function.call_udf(&mut [ac.flat().into_owned()]);

                    if ac.is_not_aggregated() || !matches!(ac.series().dtype(), DataType::List(_)) {
                        out.map(Some)
                    } else {
                        // TODO! maybe just apply over list?
                        ac.with_update_groups(UpdateGroups::WithGroupsLen);
                        Ok(out?.agg_list(ac.groups()))
                    }
                }
                ApplyOptions::ApplyList => self
                    .function
                    .call_udf(&mut [ac.aggregated().into_owned()])
                    .map(Some),
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

                    let mut ca: ListChunked = (0..len)
                        .map(|_| {
                            container.clear();
                            for iter in &mut iters {
                                match iter.next().unwrap() {
                                    None => return None,
                                    Some(s) => container.push(s),
                                }
                            }
                            self.function.call_udf(&mut container).ok()
                        })
                        .collect();
                    ca.rename(&name);
                    Ok(Some(ca.into_series()))
                }
                ApplyOptions::ApplyFlat => {
                    // the function needs to be called on a flat series
                    // but the series may be flat or aggregated
                    // if its flat, we just apply and return
                    // if not flat, the flattening sorts by group, so we must create new group tuples
                    // and again aggregate.
                    let name = acs[0].series().name().to_string();

                    // get the flat representation of the aggregation contexts
                    let mut container = acs
                        .iter_mut()
                        .map(|ac| {
                            // this is hard because the flattening sorts by group
                            assert!(
                                ac.is_not_aggregated(),
                                "flat apply on any expression that is already \
                            in aggregated state is not yet suported"
                            );
                            ac.flat().into_owned()
                        })
                        .collect::<Vec<_>>();

                    let out = self.function.call_udf(&mut container)?;
                    let out = out.agg_list(acs[0].groups().as_ref()).map(|mut out| {
                        out.rename(&name);
                        out
                    });

                    Ok(out)
                }
                ApplyOptions::ApplyList => {
                    let mut s = acs
                        .iter_mut()
                        .map(|ac| ac.aggregated().into_owned())
                        .collect::<Vec<_>>();
                    self.function.call_udf(&mut s).map(Some)
                }
            }
        }
    }
}
