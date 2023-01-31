use std::borrow::Cow;
use std::sync::Arc;

use polars_arrow::export::arrow::array::*;
use polars_arrow::export::arrow::compute::concatenate::concatenate;
use polars_arrow::export::arrow::offset::Offsets;
use polars_arrow::prelude::QuantileInterpolOptions;
use polars_arrow::utils::CustomIterTools;
use polars_core::frame::groupby::{GroupByMethod, GroupsProxy};
use polars_core::prelude::*;
use polars_core::utils::NoNull;
#[cfg(feature = "dtype-struct")]
use polars_core::POOL;

use crate::physical_plan::state::ExecutionState;
use crate::physical_plan::PartitionedAggregation;
use crate::prelude::*;

pub(crate) struct AggregationExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: GroupByMethod,
}

impl AggregationExpr {
    pub fn new(expr: Arc<dyn PhysicalExpr>, agg_type: GroupByMethod) -> Self {
        Self {
            input: expr,
            agg_type,
        }
    }
}

impl PhysicalExpr for AggregationExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, _df: &DataFrame, _state: &ExecutionState) -> PolarsResult<Series> {
        unimplemented!()
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();

        let check_flat = || {
            if !ac.null_propagated && matches!(ac.agg_state(), AggState::AggregatedFlat(_)) {
                Err(PolarsError::ComputeError(
                    format!(
                        "Cannot aggregate as {}. The column is already aggregated.",
                        self.agg_type
                    )
                    .into(),
                ))
            } else {
                Ok(())
            }
        };

        // Safety:
        // groups must always be in bounds.
        let out = unsafe {
            match self.agg_type {
                GroupByMethod::Min => {
                    check_flat()?;
                    let agg_s = ac.flat_naive().into_owned().agg_min(ac.groups());
                    rename_series(agg_s, &keep_name)
                }
                GroupByMethod::Max => {
                    check_flat()?;
                    let agg_s = ac.flat_naive().into_owned().agg_max(ac.groups());
                    rename_series(agg_s, &keep_name)
                }
                GroupByMethod::Median => {
                    check_flat()?;
                    let agg_s = ac.flat_naive().into_owned().agg_median(ac.groups());
                    rename_series(agg_s, &keep_name)
                }
                GroupByMethod::Mean => {
                    check_flat()?;
                    let agg_s = ac.flat_naive().into_owned().agg_mean(ac.groups());
                    rename_series(agg_s, &keep_name)
                }
                GroupByMethod::Sum => {
                    check_flat()?;
                    let agg_s = ac.flat_naive().into_owned().agg_sum(ac.groups());
                    rename_series(agg_s, &keep_name)
                }
                GroupByMethod::Count => {
                    // a few fast paths that prevent materializing new groups
                    match ac.update_groups {
                        UpdateGroups::WithSeriesLen => {
                            let list = ac
                                .series()
                                .list()
                                .expect("impl error, should be a list at this point");

                            let mut s = match list.chunks().len() {
                                1 => {
                                    let arr = list.downcast_iter().next().unwrap();
                                    let offsets = arr.offsets().as_slice();

                                    let mut previous = 0i64;
                                    let counts: NoNull<IdxCa> = offsets[1..]
                                        .iter()
                                        .map(|&o| {
                                            let len = (o - previous) as IdxSize;
                                            previous = o;
                                            len
                                        })
                                        .collect_trusted();
                                    counts.into_inner()
                                }
                                _ => {
                                    let counts: NoNull<IdxCa> = list
                                        .amortized_iter()
                                        .map(|s| {
                                            if let Some(s) = s {
                                                s.as_ref().len() as IdxSize
                                            } else {
                                                1
                                            }
                                        })
                                        .collect_trusted();
                                    counts.into_inner()
                                }
                            };
                            s.rename(&keep_name);
                            s.into_series()
                        }
                        UpdateGroups::WithGroupsLen => {
                            // no need to update the groups
                            // we can just get the attribute, because we only need the length,
                            // not the correct order
                            let mut ca = ac.groups.group_count();
                            ca.rename(&keep_name);
                            ca.into_series()
                        }
                        // materialize groups
                        _ => {
                            let mut ca = ac.groups().group_count();
                            ca.rename(&keep_name);
                            ca.into_series()
                        }
                    }
                }
                GroupByMethod::First => {
                    check_flat()?;
                    let mut agg_s = ac.flat_naive().into_owned().agg_first(ac.groups());
                    agg_s.rename(&keep_name);
                    agg_s
                }
                GroupByMethod::Last => {
                    check_flat()?;
                    let mut agg_s = ac.flat_naive().into_owned().agg_last(ac.groups());
                    agg_s.rename(&keep_name);
                    agg_s
                }
                GroupByMethod::NUnique => {
                    check_flat()?;
                    let agg_s = ac.flat_naive().into_owned().agg_n_unique(ac.groups());
                    rename_series(agg_s, &keep_name)
                }
                GroupByMethod::List => {
                    let agg = ac.aggregated();

                    if state.unset_finalize_window_as_list() {
                        rename_series(agg, &keep_name)
                    } else {
                        let ca = agg.list().unwrap();
                        let s = run_list_agg(ca);
                        rename_series(s, &keep_name)
                    }
                }
                GroupByMethod::Groups => {
                    let mut column: ListChunked = ac.groups().as_list_chunked();
                    column.rename(&keep_name);
                    column.into_series()
                }
                GroupByMethod::Std(ddof) => {
                    check_flat()?;
                    let agg_s = ac.flat_naive().into_owned().agg_std(ac.groups(), ddof);
                    rename_series(agg_s, &keep_name)
                }
                GroupByMethod::Var(ddof) => {
                    check_flat()?;
                    let agg_s = ac.flat_naive().into_owned().agg_var(ac.groups(), ddof);
                    rename_series(agg_s, &keep_name)
                }
                GroupByMethod::Quantile(_, _) => {
                    // implemented explicitly in AggQuantile struct
                    unimplemented!()
                }
                GroupByMethod::NanMin => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        check_flat()?;
                        let agg_s = ac.flat_naive().into_owned();
                        let groups = ac.groups();
                        let agg_s = if agg_s.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_min_s(&agg_s, groups)
                        } else {
                            agg_s.agg_min(groups)
                        };
                        rename_series(agg_s, &keep_name)
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                }
                GroupByMethod::NanMax => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        check_flat()?;
                        let agg_s = ac.flat_naive().into_owned();
                        let groups = ac.groups();
                        let agg_s = if agg_s.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_max_s(&agg_s, groups)
                        } else {
                            agg_s.agg_max(groups)
                        };
                        rename_series(agg_s, &keep_name)
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                }
            }
        };

        Ok(AggregationContext::new(out, Cow::Borrowed(groups), true))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }

    fn is_valid_aggregation(&self) -> bool {
        true
    }
}

fn rename_series(mut s: Series, name: &str) -> Series {
    s.rename(name);
    s
}

impl PartitionedAggregation for AggregationExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<Series> {
        let expr = self.input.as_partitioned_aggregator().unwrap();
        let series = expr.evaluate_partitioned(df, groups, state)?;

        // Safety:
        // groups are in bounds
        unsafe {
            match self.agg_type {
                #[cfg(feature = "dtype-struct")]
                GroupByMethod::Mean => {
                    let new_name = series.name().to_string();

                    // ensure we don't overflow
                    // the all 8 and 16 bits integers are already upcasted to int16 on `agg_sum`
                    let mut agg_s = if matches!(series.dtype(), DataType::Int32 | DataType::UInt32)
                    {
                        series.cast(&DataType::Int64).unwrap().agg_sum(groups)
                    } else {
                        series.agg_sum(groups)
                    };
                    agg_s.rename(&new_name);

                    if !agg_s.dtype().is_numeric() {
                        Ok(agg_s)
                    } else {
                        let agg_s = match agg_s.dtype() {
                            DataType::Float32 => agg_s,
                            _ => agg_s.cast(&DataType::Float64).unwrap(),
                        };
                        let mut count_s = series.agg_valid_count(groups);
                        count_s.rename("count");
                        Ok(StructChunked::new(&new_name, &[agg_s, count_s])
                            .unwrap()
                            .into_series())
                    }
                }
                GroupByMethod::List => {
                    let new_name = series.name();
                    let mut agg = series.agg_list(groups);
                    agg.rename(new_name);
                    Ok(agg)
                }
                GroupByMethod::First => {
                    let mut agg = series.agg_first(groups);
                    agg.rename(series.name());
                    Ok(agg)
                }
                GroupByMethod::Last => {
                    let mut agg = series.agg_last(groups);
                    agg.rename(series.name());
                    Ok(agg)
                }
                GroupByMethod::Max => {
                    let mut agg = series.agg_max(groups);
                    agg.rename(series.name());
                    Ok(agg)
                }
                GroupByMethod::Min => {
                    let mut agg = series.agg_min(groups);
                    agg.rename(series.name());
                    Ok(agg)
                }
                GroupByMethod::Sum => {
                    let mut agg = series.agg_sum(groups);
                    agg.rename(series.name());
                    Ok(agg)
                }
                GroupByMethod::Count => {
                    let mut ca = groups.group_count();
                    ca.rename(series.name());
                    Ok(ca.into_series())
                }
                _ => {
                    unimplemented!()
                }
            }
        }
    }

    fn finalize(
        &self,
        partitioned: Series,
        groups: &GroupsProxy,
        _state: &ExecutionState,
    ) -> PolarsResult<Series> {
        match self.agg_type {
            GroupByMethod::Count | GroupByMethod::Sum => {
                let mut agg = unsafe { partitioned.agg_sum(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            }
            #[cfg(feature = "dtype-struct")]
            GroupByMethod::Mean => {
                let new_name = partitioned.name();
                match partitioned.dtype() {
                    DataType::Struct(_) => {
                        let ca = partitioned.struct_().unwrap();
                        let sum = &ca.fields()[0];
                        let count = &ca.fields()[1];
                        let (agg_count, agg_s) =
                            unsafe { POOL.join(|| count.agg_sum(groups), || sum.agg_sum(groups)) };
                        let agg_s = &agg_s / &agg_count;
                        Ok(rename_series(agg_s, new_name))
                    }
                    _ => Ok(Series::full_null(
                        new_name,
                        groups.len(),
                        partitioned.dtype(),
                    )),
                }
            }
            GroupByMethod::List => {
                // the groups are scattered over multiple groups/sub dataframes.
                // we now must collect them into a single group
                let ca = partitioned.list().unwrap();
                let new_name = partitioned.name().to_string();

                let mut values = Vec::with_capacity(groups.len());
                let mut can_fast_explode = true;

                let mut offsets = Vec::<i64>::with_capacity(groups.len() + 1);
                let mut length_so_far = 0i64;
                offsets.push(length_so_far);

                let mut process_group = |ca: ListChunked| -> PolarsResult<()> {
                    let s = ca.explode()?;
                    length_so_far += s.len() as i64;
                    offsets.push(length_so_far);
                    values.push(s.chunks()[0].clone());

                    if s.len() == 0 {
                        can_fast_explode = false;
                    }
                    Ok(())
                };

                match groups {
                    GroupsProxy::Idx(groups) => {
                        for (_, idx) in groups {
                            let ca = unsafe {
                                // Safety
                                // The indexes of the groupby operation are never out of bounds
                                ca.take_unchecked(idx.into())
                            };
                            process_group(ca)?;
                        }
                    }
                    GroupsProxy::Slice { groups, .. } => {
                        for [first, len] in groups {
                            let len = *len as usize;
                            let ca = ca.slice(*first as i64, len);
                            process_group(ca)?;
                        }
                    }
                }

                let vals = values.iter().map(|arr| &**arr).collect::<Vec<_>>();
                let values = concatenate(&vals).unwrap();

                let data_type = ListArray::<i64>::default_datatype(values.data_type().clone());
                // Safety:
                // offsets are monotonically increasing
                let arr = unsafe {
                    Box::new(ListArray::<i64>::new(
                        data_type,
                        Offsets::new_unchecked(offsets).into(),
                        values,
                        None,
                    )) as ArrayRef
                };
                let mut ca = unsafe { ListChunked::from_chunks(&new_name, vec![arr]) };
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                Ok(run_list_agg(&ca))
            }
            GroupByMethod::First => {
                let mut agg = unsafe { partitioned.agg_first(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            }
            GroupByMethod::Last => {
                let mut agg = unsafe { partitioned.agg_last(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            }
            GroupByMethod::Max => {
                let mut agg = unsafe { partitioned.agg_max(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            }
            GroupByMethod::Min => {
                let mut agg = unsafe { partitioned.agg_min(groups) };
                agg.rename(partitioned.name());
                Ok(agg)
            }
            _ => unimplemented!(),
        }
    }
}

pub struct AggQuantileExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) quantile: Arc<dyn PhysicalExpr>,
    pub(crate) interpol: QuantileInterpolOptions,
}

impl AggQuantileExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        quantile: Arc<dyn PhysicalExpr>,
        interpol: QuantileInterpolOptions,
    ) -> Self {
        Self {
            input,
            quantile,
            interpol,
        }
    }

    fn get_quantile(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<f64> {
        let quantile = self.quantile.evaluate(df, state)?;
        if quantile.len() > 1 {
            return Err(PolarsError::ComputeError(
                "Polars only supports computing a single quantile. \
            Make sure the 'quantile' expression input produces a single quantile."
                    .into(),
            ));
        }
        quantile.get(0).unwrap().try_extract::<f64>()
    }
}

impl PhysicalExpr for AggQuantileExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let input = self.input.evaluate(df, state)?;
        let quantile = self.get_quantile(df, state)?;
        input.quantile_as_series(quantile, self.interpol)
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.series().name().to_string();

        let quantile = self.get_quantile(df, state)?;

        // safety:
        // groups are in bounds
        let mut agg = unsafe {
            ac.flat_naive()
                .into_owned()
                .agg_quantile(ac.groups(), quantile, self.interpol)
        };
        agg.rename(&keep_name);
        Ok(AggregationContext::new(agg, Cow::Borrowed(groups), true))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_valid_aggregation(&self) -> bool {
        true
    }
}

fn run_list_agg(ca: &ListChunked) -> Series {
    assert_eq!(ca.chunks().len(), 1);
    let arr = ca.chunks()[0].clone();

    let offsets = (0i64..(ca.len() as i64 + 1)).collect::<Vec<_>>();
    let offsets = unsafe { Offsets::new_unchecked(offsets) };

    let new_arr = LargeListArray::new(
        DataType::List(Box::new(ca.dtype().clone())).to_arrow(),
        offsets.into(),
        arr,
        None,
    );
    unsafe { ListChunked::from_chunks(ca.name(), vec![Box::new(new_arr)]).into_series() }
}
