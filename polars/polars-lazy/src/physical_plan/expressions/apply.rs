use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::GroupsProxy;
use polars_core::prelude::*;
use polars_core::POOL;
use rayon::prelude::*;
use std::sync::Arc;

pub struct ApplyExpr {
    pub inputs: Vec<Arc<dyn PhysicalExpr>>,
    pub function: SpecialEq<Arc<dyn SeriesUdf>>,
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
    (offset[offset.len() - 1] as usize) == list_arr.len() as usize
}

fn check_map_output_len(input_len: usize, output_len: usize) -> Result<()> {
    if input_len != output_len {
        Err(PolarsError::ComputeError("A 'map' functions output length must be equal to that of the input length. Consider using 'apply' in favor of 'map'.".into()))
    } else {
        Ok(())
    }
}

impl PhysicalExpr for ApplyExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
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

            match self.collect_groups {
                ApplyOptions::ApplyGroups => {
                    let s = ac.series();

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
                ApplyOptions::ApplyFlat => {
                    // make sure the groups are updated because we are about to throw away
                    // the series length information
                    if let UpdateGroups::WithSeriesLen = ac.update_groups {
                        ac.groups();
                    }
                    let input = ac.flat_naive().into_owned();
                    let input_len = input.len();
                    let s = self.function.call_udf(&mut [input])?;

                    check_map_output_len(input_len, s.len())?;
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
                    // then unpack the lists and finally create iterators from this list chunked arrays.
                    let lists = acs
                        .iter_mut()
                        .map(|ac| {
                            let s = match ac.agg_state() {
                                AggState::AggregatedFlat(s) => s.reshape(&[-1, 1]).unwrap(),
                                _ => ac.aggregated(),
                            };
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
                        .collect_trusted();
                    ca.rename(&name);
                    // take the first aggregation context that as that is the input series
                    let ac = acs.swap_remove(0);
                    let ac = self.finish_apply_groups(ac, ca);
                    Ok(ac)
                }
                ApplyOptions::ApplyFlat => {
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
                    let s = self.function.call_udf(&mut s)?;
                    check_map_output_len(input_len, s.len())?;

                    // take the first aggregation context that as that is the input series
                    let mut ac = acs.swap_remove(0);
                    ac.with_series(s, false);
                    Ok(ac)
                }
                ApplyOptions::ApplyList => {
                    let mut s = acs.iter_mut().map(|ac| ac.aggregated()).collect::<Vec<_>>();
                    let s = self.function.call_udf(&mut s)?;
                    // take the first aggregation context that as that is the input series
                    let mut ac = acs.swap_remove(0);
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
    fn is_valid_aggregation(&self) -> bool {
        matches!(self.collect_groups, ApplyOptions::ApplyGroups)
    }
}
