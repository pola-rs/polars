use std::borrow::Cow;

use arrow::array::*;
use arrow::compute::concatenate::concatenate;
use arrow::legacy::utils::CustomIterTools;
use arrow::offset::Offsets;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::{NoNull, _split_offsets};
use polars_core::POOL;
#[cfg(feature = "propagate_nans")]
use polars_ops::prelude::nan_propagating_aggregate;
use rayon::prelude::*;

use super::*;
use crate::expressions::AggState::{AggregatedList, AggregatedScalar};
use crate::expressions::{
    AggState, AggregationContext, PartitionedAggregation, PhysicalExpr, UpdateGroups,
};

#[derive(Debug, Clone, Copy)]
pub struct AggregationType {
    pub(crate) groupby: GroupByMethod,
    pub(crate) allow_threading: bool,
}

pub(crate) struct AggregationExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) agg_type: AggregationType,
    field: Option<Field>,
}

impl AggregationExpr {
    pub fn new(
        expr: Arc<dyn PhysicalExpr>,
        agg_type: AggregationType,
        field: Option<Field>,
    ) -> Self {
        Self {
            input: expr,
            agg_type,
            field,
        }
    }
}

impl PhysicalExpr for AggregationExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let s = self.input.evaluate(df, state)?;

        let AggregationType {
            groupby,
            allow_threading,
        } = self.agg_type;

        let is_float = s.dtype().is_float();
        let group_by = match groupby {
            GroupByMethod::NanMin if !is_float => GroupByMethod::Min,
            GroupByMethod::NanMax if !is_float => GroupByMethod::Max,
            gb => gb,
        };

        match group_by {
            GroupByMethod::Min => match s.is_sorted_flag() {
                IsSorted::Ascending | IsSorted::Descending => {
                    s.min_reduce().map(|sc| sc.into_column(s.name().clone()))
                },
                IsSorted::Not => parallel_op_columns(
                    |s| s.min_reduce().map(|sc| sc.into_column(s.name().clone())),
                    s,
                    allow_threading,
                ),
            },
            #[cfg(feature = "propagate_nans")]
            GroupByMethod::NanMin => parallel_op_columns(
                |s| {
                    Ok(polars_ops::prelude::nan_propagating_aggregate::nan_min_s(
                        s.as_materialized_series(),
                        s.name().clone(),
                    )
                    .into_column())
                },
                s,
                allow_threading,
            ),
            #[cfg(not(feature = "propagate_nans"))]
            GroupByMethod::NanMin => {
                panic!("activate 'propagate_nans' feature")
            },
            GroupByMethod::Max => match s.is_sorted_flag() {
                IsSorted::Ascending | IsSorted::Descending => {
                    s.max_reduce().map(|sc| sc.into_column(s.name().clone()))
                },
                IsSorted::Not => parallel_op_columns(
                    |s| s.max_reduce().map(|sc| sc.into_column(s.name().clone())),
                    s,
                    allow_threading,
                ),
            },
            #[cfg(feature = "propagate_nans")]
            GroupByMethod::NanMax => parallel_op_columns(
                |s| {
                    Ok(polars_ops::prelude::nan_propagating_aggregate::nan_max_s(
                        s.as_materialized_series(),
                        s.name().clone(),
                    )
                    .into_column())
                },
                s,
                allow_threading,
            ),
            #[cfg(not(feature = "propagate_nans"))]
            GroupByMethod::NanMax => {
                panic!("activate 'propagate_nans' feature")
            },
            GroupByMethod::Median => s.median_reduce().map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Mean => Ok(s.mean_reduce().into_column(s.name().clone())),
            GroupByMethod::First => Ok(if s.is_empty() {
                Column::full_null(s.name().clone(), 1, s.dtype())
            } else {
                s.head(Some(1))
            }),
            GroupByMethod::Last => Ok(if s.is_empty() {
                Column::full_null(s.name().clone(), 1, s.dtype())
            } else {
                s.tail(Some(1))
            }),
            GroupByMethod::Sum => parallel_op_columns(
                |s| s.sum_reduce().map(|sc| sc.into_column(s.name().clone())),
                s,
                allow_threading,
            ),
            GroupByMethod::Groups => unreachable!(),
            GroupByMethod::NUnique => s.n_unique().map(|count| {
                IdxCa::from_slice(s.name().clone(), &[count as IdxSize]).into_column()
            }),
            GroupByMethod::Count { include_nulls } => {
                let count = s.len() - s.null_count() * !include_nulls as usize;

                Ok(IdxCa::from_slice(s.name().clone(), &[count as IdxSize]).into_column())
            },
            GroupByMethod::Implode => s.implode().map(|ca| ca.into_column()),
            GroupByMethod::Std(ddof) => s
                .std_reduce(ddof)
                .map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Var(ddof) => s
                .var_reduce(ddof)
                .map(|sc| sc.into_column(s.name().clone())),
            GroupByMethod::Quantile(_, _) => unimplemented!(),
        }
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.get_values().name().clone();
        polars_ensure!(!matches!(ac.agg_state(), AggState::Literal(_)), ComputeError: "cannot aggregate a literal");

        if let AggregatedScalar(_) = ac.agg_state() {
            match self.agg_type.groupby {
                GroupByMethod::Implode => {},
                _ => {
                    polars_bail!(ComputeError: "cannot aggregate as {}, the column is already aggregated", self.agg_type.groupby);
                },
            }
        }

        // SAFETY:
        // groups must always be in bounds.
        let out = unsafe {
            match self.agg_type.groupby {
                GroupByMethod::Min => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_min(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Max => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_max(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Median => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_median(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Mean => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_mean(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Sum => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_sum(&groups);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Count { include_nulls } => {
                    if include_nulls || ac.get_values().null_count() == 0 {
                        // a few fast paths that prevent materializing new groups
                        match ac.update_groups {
                            UpdateGroups::WithSeriesLen => {
                                let list = ac
                                    .get_values()
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
                                    },
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
                                    },
                                };
                                s.rename(keep_name);
                                AggregatedScalar(s.into_column())
                            },
                            UpdateGroups::WithGroupsLen => {
                                // no need to update the groups
                                // we can just get the attribute, because we only need the length,
                                // not the correct order
                                let mut ca = ac.groups.group_count();
                                ca.rename(keep_name);
                                AggregatedScalar(ca.into_column())
                            },
                            // materialize groups
                            _ => {
                                let mut ca = ac.groups().group_count();
                                ca.rename(keep_name);
                                AggregatedScalar(ca.into_column())
                            },
                        }
                    } else {
                        // TODO: optimize this/and write somewhere else.
                        match ac.agg_state() {
                            AggState::Literal(s) | AggState::AggregatedScalar(s) => {
                                AggregatedScalar(Column::new(
                                    keep_name,
                                    [(s.len() as IdxSize - s.null_count() as IdxSize)],
                                ))
                            },
                            AggState::AggregatedList(s) => {
                                let ca = s.list()?;
                                let out: IdxCa = ca
                                    .into_iter()
                                    .map(|opt_s| {
                                        opt_s
                                            .map(|s| s.len() as IdxSize - s.null_count() as IdxSize)
                                    })
                                    .collect();
                                AggregatedScalar(out.into_column().with_name(keep_name))
                            },
                            AggState::NotAggregated(s) => {
                                let s = s.clone();
                                let groups = ac.groups();
                                let out: IdxCa = if matches!(s.dtype(), &DataType::Null) {
                                    IdxCa::full(s.name().clone(), 0, groups.len())
                                } else {
                                    match groups.as_ref().as_ref() {
                                        GroupsType::Idx(idx) => {
                                            let s = s.rechunk();
                                            // @scalar-opt
                                            // @partition-opt
                                            let array = &s.as_materialized_series().chunks()[0];
                                            let validity = array.validity().unwrap();
                                            idx.iter()
                                                .map(|(_, g)| {
                                                    let mut count = 0 as IdxSize;
                                                    // Count valid values
                                                    g.iter().for_each(|i| {
                                                        count += validity
                                                            .get_bit_unchecked(*i as usize)
                                                            as IdxSize;
                                                    });
                                                    count
                                                })
                                                .collect_ca_trusted_with_dtype(keep_name, IDX_DTYPE)
                                        },
                                        GroupsType::Slice { groups, .. } => {
                                            // Slice and use computed null count
                                            groups
                                                .iter()
                                                .map(|g| {
                                                    let start = g[0];
                                                    let len = g[1];
                                                    len - s
                                                        .slice(start as i64, len as usize)
                                                        .null_count()
                                                        as IdxSize
                                                })
                                                .collect_ca_trusted_with_dtype(keep_name, IDX_DTYPE)
                                        },
                                    }
                                };
                                AggregatedScalar(out.into_column())
                            },
                        }
                    }
                },
                GroupByMethod::First => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_first(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::Last => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_last(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::NUnique => {
                    let (s, groups) = ac.get_final_aggregation();
                    let agg_s = s.agg_n_unique(&groups);
                    AggregatedScalar(agg_s.with_name(keep_name))
                },
                GroupByMethod::Implode => {
                    // if the aggregation is already
                    // in an aggregate flat state for instance by
                    // a mean aggregation, we simply convert to list
                    //
                    // if it is not, we traverse the groups and create
                    // a list per group.
                    let c = match ac.agg_state() {
                        // mean agg:
                        // -> f64 -> list<f64>
                        AggState::AggregatedScalar(c) => c
                            .reshape_list(&[
                                ReshapeDimension::Infer,
                                ReshapeDimension::new_dimension(1),
                            ])
                            .unwrap(),
                        _ => {
                            let agg = ac.aggregated();
                            agg.as_list().into_column()
                        },
                    };
                    AggregatedList(c.with_name(keep_name))
                },
                GroupByMethod::Groups => {
                    let mut column: ListChunked = ac.groups().as_list_chunked();
                    column.rename(keep_name);
                    AggregatedScalar(column.into_column())
                },
                GroupByMethod::Std(ddof) => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_std(&groups, ddof);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Var(ddof) => {
                    let (c, groups) = ac.get_final_aggregation();
                    let agg_c = c.agg_var(&groups, ddof);
                    AggregatedScalar(agg_c.with_name(keep_name))
                },
                GroupByMethod::Quantile(_, _) => {
                    // implemented explicitly in AggQuantile struct
                    unimplemented!()
                },
                GroupByMethod::NanMin => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (c, groups) = ac.get_final_aggregation();
                        let agg_c = if c.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_min_s(
                                c.as_materialized_series(),
                                &groups,
                            )
                            .into_column()
                        } else {
                            c.agg_min(&groups)
                        };
                        AggregatedScalar(agg_c.with_name(keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
                GroupByMethod::NanMax => {
                    #[cfg(feature = "propagate_nans")]
                    {
                        let (c, groups) = ac.get_final_aggregation();
                        let agg_c = if c.dtype().is_float() {
                            nan_propagating_aggregate::group_agg_nan_max_s(
                                c.as_materialized_series(),
                                &groups,
                            )
                            .into_column()
                        } else {
                            c.agg_max(&groups)
                        };
                        AggregatedScalar(agg_c.with_name(keep_name))
                    }
                    #[cfg(not(feature = "propagate_nans"))]
                    {
                        panic!("activate 'propagate_nans' feature")
                    }
                },
            }
        };

        Ok(AggregationContext::from_agg_state(
            out,
            Cow::Borrowed(groups),
        ))
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        if let Some(field) = self.field.as_ref() {
            Ok(field.clone())
        } else {
            self.input.to_field(input_schema)
        }
    }

    fn collect_live_columns(&self, lv: &mut PlIndexSet<PlSmallStr>) {
        self.input.collect_live_columns(lv);
    }

    fn is_scalar(&self) -> bool {
        true
    }

    fn as_partitioned_aggregator(&self) -> Option<&dyn PartitionedAggregation> {
        Some(self)
    }
}

impl PartitionedAggregation for AggregationExpr {
    fn evaluate_partitioned(
        &self,
        df: &DataFrame,
        groups: &GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<Column> {
        let expr = self.input.as_partitioned_aggregator().unwrap();
        let column = expr.evaluate_partitioned(df, groups, state)?;

        // SAFETY:
        // groups are in bounds
        unsafe {
            match self.agg_type.groupby {
                #[cfg(feature = "dtype-struct")]
                GroupByMethod::Mean => {
                    let new_name = column.name().clone();

                    // ensure we don't overflow
                    // the all 8 and 16 bits integers are already upcasted to int16 on `agg_sum`
                    let mut agg_s = if matches!(column.dtype(), DataType::Int32 | DataType::UInt32)
                    {
                        column.cast(&DataType::Int64).unwrap().agg_sum(groups)
                    } else {
                        column.agg_sum(groups)
                    };
                    agg_s.rename(new_name.clone());

                    if !agg_s.dtype().is_primitive_numeric() {
                        Ok(agg_s)
                    } else {
                        let agg_s = match agg_s.dtype() {
                            DataType::Float32 => agg_s,
                            _ => agg_s.cast(&DataType::Float64).unwrap(),
                        };
                        let mut count_s = column.agg_valid_count(groups);
                        count_s.rename(PlSmallStr::from_static("__POLARS_COUNT"));
                        Ok(
                            StructChunked::from_columns(new_name, agg_s.len(), &[agg_s, count_s])
                                .unwrap()
                                .into_column(),
                        )
                    }
                },
                GroupByMethod::Implode => {
                    let new_name = column.name().clone();
                    let mut agg = column.agg_list(groups);
                    agg.rename(new_name);
                    Ok(agg)
                },
                GroupByMethod::First => {
                    let mut agg = column.agg_first(groups);
                    agg.rename(column.name().clone());
                    Ok(agg)
                },
                GroupByMethod::Last => {
                    let mut agg = column.agg_last(groups);
                    agg.rename(column.name().clone());
                    Ok(agg)
                },
                GroupByMethod::Max => {
                    let mut agg = column.agg_max(groups);
                    agg.rename(column.name().clone());
                    Ok(agg)
                },
                GroupByMethod::Min => {
                    let mut agg = column.agg_min(groups);
                    agg.rename(column.name().clone());
                    Ok(agg)
                },
                GroupByMethod::Sum => {
                    let mut agg = column.agg_sum(groups);
                    agg.rename(column.name().clone());
                    Ok(agg)
                },
                GroupByMethod::Count {
                    include_nulls: true,
                } => {
                    let mut ca = groups.group_count();
                    ca.rename(column.name().clone());
                    Ok(ca.into_column())
                },
                _ => {
                    unimplemented!()
                },
            }
        }
    }

    fn finalize(
        &self,
        partitioned: Column,
        groups: &GroupPositions,
        _state: &ExecutionState,
    ) -> PolarsResult<Column> {
        match self.agg_type.groupby {
            GroupByMethod::Count {
                include_nulls: true,
            }
            | GroupByMethod::Sum => {
                let mut agg = unsafe { partitioned.agg_sum(groups) };
                agg.rename(partitioned.name().clone());
                Ok(agg)
            },
            #[cfg(feature = "dtype-struct")]
            GroupByMethod::Mean => {
                let new_name = partitioned.name().clone();
                match partitioned.dtype() {
                    DataType::Struct(_) => {
                        let ca = partitioned.struct_().unwrap();
                        let fields = ca.fields_as_series();
                        let sum = &fields[0];
                        let count = &fields[1];
                        let (agg_count, agg_s) =
                            unsafe { POOL.join(|| count.agg_sum(groups), || sum.agg_sum(groups)) };
                        let agg_s = &agg_s / &agg_count;
                        Ok(agg_s?.with_name(new_name).into_column())
                    },
                    _ => Ok(Column::full_null(
                        new_name,
                        groups.len(),
                        partitioned.dtype(),
                    )),
                }
            },
            GroupByMethod::Implode => {
                // the groups are scattered over multiple groups/sub dataframes.
                // we now must collect them into a single group
                let ca = partitioned.list().unwrap();
                let new_name = partitioned.name().clone();

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

                match groups.as_ref() {
                    GroupsType::Idx(groups) => {
                        for (_, idx) in groups {
                            let ca = unsafe {
                                // SAFETY:
                                // The indexes of the group_by operation are never out of bounds
                                ca.take_unchecked(idx)
                            };
                            process_group(ca)?;
                        }
                    },
                    GroupsType::Slice { groups, .. } => {
                        for [first, len] in groups {
                            let len = *len as usize;
                            let ca = ca.slice(*first as i64, len);
                            process_group(ca)?;
                        }
                    },
                }

                let vals = values.iter().map(|arr| &**arr).collect::<Vec<_>>();
                let values = concatenate(&vals).unwrap();

                let dtype = ListArray::<i64>::default_datatype(values.dtype().clone());
                // SAFETY: offsets are monotonically increasing.
                let arr = ListArray::<i64>::new(
                    dtype,
                    unsafe { Offsets::new_unchecked(offsets).into() },
                    values,
                    None,
                );
                let mut ca = ListChunked::with_chunk(new_name, arr);
                if can_fast_explode {
                    ca.set_fast_explode()
                }
                Ok(ca.into_series().as_list().into_column())
            },
            GroupByMethod::First => {
                let mut agg = unsafe { partitioned.agg_first(groups) };
                agg.rename(partitioned.name().clone());
                Ok(agg)
            },
            GroupByMethod::Last => {
                let mut agg = unsafe { partitioned.agg_last(groups) };
                agg.rename(partitioned.name().clone());
                Ok(agg)
            },
            GroupByMethod::Max => {
                let mut agg = unsafe { partitioned.agg_max(groups) };
                agg.rename(partitioned.name().clone());
                Ok(agg)
            },
            GroupByMethod::Min => {
                let mut agg = unsafe { partitioned.agg_min(groups) };
                agg.rename(partitioned.name().clone());
                Ok(agg)
            },
            _ => unimplemented!(),
        }
    }
}

pub struct AggQuantileExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) quantile: Arc<dyn PhysicalExpr>,
    pub(crate) method: QuantileMethod,
}

impl AggQuantileExpr {
    pub fn new(
        input: Arc<dyn PhysicalExpr>,
        quantile: Arc<dyn PhysicalExpr>,
        method: QuantileMethod,
    ) -> Self {
        Self {
            input,
            quantile,
            method,
        }
    }

    fn get_quantile(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<f64> {
        let quantile = self.quantile.evaluate(df, state)?;
        polars_ensure!(quantile.len() <= 1, ComputeError:
            "polars only supports computing a single quantile; \
            make sure the 'quantile' expression input produces a single quantile"
        );
        quantile.get(0).unwrap().try_extract()
    }
}

impl PhysicalExpr for AggQuantileExpr {
    fn as_expression(&self) -> Option<&Expr> {
        None
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let input = self.input.evaluate(df, state)?;
        let quantile = self.get_quantile(df, state)?;
        input
            .quantile_reduce(quantile, self.method)
            .map(|sc| sc.into_column(input.name().clone()))
    }
    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut ac = self.input.evaluate_on_groups(df, groups, state)?;
        // don't change names by aggregations as is done in polars-core
        let keep_name = ac.get_values().name().clone();

        let quantile = self.get_quantile(df, state)?;

        // SAFETY:
        // groups are in bounds
        let mut agg = unsafe {
            ac.flat_naive()
                .into_owned()
                .agg_quantile(ac.groups(), quantile, self.method)
        };
        agg.rename(keep_name);
        Ok(AggregationContext::from_agg_state(
            AggregatedScalar(agg),
            Cow::Borrowed(groups),
        ))
    }

    fn collect_live_columns(&self, lv: &mut PlIndexSet<PlSmallStr>) {
        self.input.collect_live_columns(lv);
        self.quantile.collect_live_columns(lv);
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        true
    }
}

/// Simple wrapper to parallelize functions that can be divided over threads aggregated and
/// finally aggregated in the main thread. This can be done for sum, min, max, etc.
fn parallel_op_columns<F>(f: F, s: Column, allow_threading: bool) -> PolarsResult<Column>
where
    F: Fn(Column) -> PolarsResult<Column> + Send + Sync,
{
    // set during debug low so
    // we mimic production size data behavior
    #[cfg(debug_assertions)]
    let thread_boundary = 0;

    #[cfg(not(debug_assertions))]
    let thread_boundary = 100_000;

    // threading overhead/ splitting work stealing is costly..
    if allow_threading
        || s.len() < thread_boundary
        || POOL.current_thread_has_pending_tasks().unwrap_or(false)
    {
        return f(s);
    }
    let n_threads = POOL.current_num_threads();
    let splits = _split_offsets(s.len(), n_threads);

    let chunks = POOL.install(|| {
        splits
            .into_par_iter()
            .map(|(offset, len)| {
                let s = s.slice(offset as i64, len);
                f(s)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut iter = chunks.into_iter();
    let first = iter.next().unwrap();
    let dtype = first.dtype();
    let out = iter.fold(first.to_physical_repr(), |mut acc, s| {
        acc.append(&s.to_physical_repr()).unwrap();
        acc
    });

    unsafe { f(out.from_physical_unchecked(dtype).unwrap()) }
}
