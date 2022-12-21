use std::sync::Arc;

use polars_core::frame::groupby::{GroupsProxy, IdxItem};
use polars_core::prelude::*;
use polars_core::utils::{slice_offsets, CustomIterTools};
use polars_core::POOL;
use rayon::prelude::*;
use AnyValue::Null;

use crate::physical_plan::expression_err;
use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;

pub struct SliceExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) offset: Arc<dyn PhysicalExpr>,
    pub(crate) length: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
}

fn extract_offset(offset: &Series, expr: &Expr) -> PolarsResult<i64> {
    if offset.len() > 1 {
        let msg = format!(
            "Invalid argument to slice; expected an offset literal but got a Series of length {}.",
            offset.len()
        );
        return Err(expression_err!(msg, expr, ComputeError));
    }
    offset.get(0).unwrap().extract::<i64>().ok_or_else(|| {
        PolarsError::ComputeError(format!("could not get an offset from {offset:?}").into())
    })
}

fn extract_length(length: &Series, expr: &Expr) -> PolarsResult<usize> {
    if length.len() > 1 {
        let msg = format!(
            "Invalid argument to slice; expected a length literal but got a Series of length {}.",
            length.len()
        );
        return Err(expression_err!(msg, expr, ComputeError));
    }
    match length.get(0).unwrap() {
        Null => Ok(usize::MAX),
        v => v.extract::<usize>().ok_or_else(|| {
            let msg = format!("Could not get a length from {length:?}.");
            expression_err!(msg, expr, ComputeError)
        }),
    }
}

fn extract_args(offset: &Series, length: &Series, expr: &Expr) -> PolarsResult<(i64, usize)> {
    Ok((extract_offset(offset, expr)?, extract_length(length, expr)?))
}

fn check_argument(arg: &Series, groups: &GroupsProxy, name: &str, expr: &Expr) -> PolarsResult<()> {
    if let DataType::List(_) = arg.dtype() {
        let msg = format!("Invalid slice argument: cannot use an array as {name} argument.",);
        Err(expression_err!(msg, expr, ComputeError))
    } else if arg.len() != groups.len() {
        let msg = format!("Invalid slice argument: the evaluated length expression was of different {name} than the number of groups.");
        Err(expression_err!(msg, expr, ComputeError))
    } else if arg.null_count() > 0 {
        let msg =
            format!("Invalid slice argument: the {name} expression should not have null values.",);
        Err(expression_err!(msg, expr, ComputeError))
    } else {
        Ok(())
    }
}

fn slice_groups_idx(offset: i64, length: usize, first: IdxSize, idx: &[IdxSize]) -> IdxItem {
    let (offset, len) = slice_offsets(offset, length, idx.len());
    (
        first + offset as IdxSize,
        idx[offset..offset + len].to_vec(),
    )
}

fn slice_groups_slice(offset: i64, length: usize, first: IdxSize, len: IdxSize) -> [IdxSize; 2] {
    let (offset, len) = slice_offsets(offset, length, len as usize);
    [first + offset as IdxSize, len as IdxSize]
}

impl PhysicalExpr for SliceExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Series> {
        let results = POOL.install(|| {
            [&self.offset, &self.length, &self.input]
                .par_iter()
                .map(|e| e.evaluate(df, state))
                .collect::<PolarsResult<Vec<_>>>()
        })?;
        let offset = &results[0];
        let length = &results[1];
        let series = &results[2];
        let (offset, length) = extract_args(offset, length, &self.expr)?;

        Ok(series.slice(offset, length))
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let mut results = POOL.install(|| {
            [&self.offset, &self.length, &self.input]
                .par_iter()
                .map(|e| e.evaluate_on_groups(df, groups, state))
                .collect::<PolarsResult<Vec<_>>>()
        })?;
        let mut ac = results.pop().unwrap();
        let mut ac_length = results.pop().unwrap();
        let mut ac_offset = results.pop().unwrap();

        let groups = ac.groups();

        use AggState::*;
        let groups = match (&ac_offset.state, &ac_length.state) {
            (Literal(offset), Literal(length)) => {
                let (offset, length) = extract_args(offset, length, &self.expr)?;

                match groups.as_ref() {
                    GroupsProxy::Idx(groups) => {
                        let groups = groups
                            .iter()
                            .map(|(first, idx)| slice_groups_idx(offset, length, first, idx))
                            .collect();
                        GroupsProxy::Idx(groups)
                    }
                    GroupsProxy::Slice { groups, .. } => {
                        let groups = groups
                            .iter()
                            .map(|&[first, len]| slice_groups_slice(offset, length, first, len))
                            .collect_trusted();
                        GroupsProxy::Slice {
                            groups,
                            rolling: false,
                        }
                    }
                }
            }
            (Literal(offset), _) => {
                let offset = extract_offset(offset, &self.expr)?;
                let length = ac_length.aggregated();
                check_argument(&length, groups, "length", &self.expr)?;

                let length = length.cast(&IDX_DTYPE)?;
                let length = length.idx().unwrap();

                match groups.as_ref() {
                    GroupsProxy::Idx(groups) => {
                        let groups = groups
                            .iter()
                            .zip(length.into_no_null_iter())
                            .map(|((first, idx), length)| {
                                slice_groups_idx(offset, length as usize, first, idx)
                            })
                            .collect();
                        GroupsProxy::Idx(groups)
                    }
                    GroupsProxy::Slice { groups, .. } => {
                        let groups = groups
                            .iter()
                            .zip(length.into_no_null_iter())
                            .map(|(&[first, len], length)| {
                                slice_groups_slice(offset, length as usize, first, len)
                            })
                            .collect_trusted();
                        GroupsProxy::Slice {
                            groups,
                            rolling: false,
                        }
                    }
                }
            }
            (_, Literal(length)) => {
                let length = extract_length(length, &self.expr)?;
                let offset = ac_offset.aggregated();
                check_argument(&offset, groups, "offset", &self.expr)?;

                let offset = offset.cast(&DataType::Int64)?;
                let offset = offset.i64().unwrap();

                match groups.as_ref() {
                    GroupsProxy::Idx(groups) => {
                        let groups = groups
                            .iter()
                            .zip(offset.into_no_null_iter())
                            .map(|((first, idx), offset)| {
                                slice_groups_idx(offset, length, first, idx)
                            })
                            .collect();
                        GroupsProxy::Idx(groups)
                    }
                    GroupsProxy::Slice { groups, .. } => {
                        let groups = groups
                            .iter()
                            .zip(offset.into_no_null_iter())
                            .map(|(&[first, len], offset)| {
                                slice_groups_slice(offset, length, first, len)
                            })
                            .collect_trusted();
                        GroupsProxy::Slice {
                            groups,
                            rolling: false,
                        }
                    }
                }
            }
            _ => {
                let length = ac_length.aggregated();
                let offset = ac_offset.aggregated();
                check_argument(&length, groups, "length", &self.expr)?;
                check_argument(&offset, groups, "offset", &self.expr)?;

                let offset = offset.cast(&DataType::Int64)?;
                let offset = offset.i64().unwrap();

                let length = length.cast(&IDX_DTYPE)?;
                let length = length.idx().unwrap();

                match groups.as_ref() {
                    GroupsProxy::Idx(groups) => {
                        let groups = groups
                            .iter()
                            .zip(offset.into_no_null_iter())
                            .zip(length.into_no_null_iter())
                            .map(|(((first, idx), offset), length)| {
                                slice_groups_idx(offset, length as usize, first, idx)
                            })
                            .collect();
                        GroupsProxy::Idx(groups)
                    }
                    GroupsProxy::Slice { groups, .. } => {
                        let groups = groups
                            .iter()
                            .zip(offset.into_no_null_iter())
                            .zip(length.into_no_null_iter())
                            .map(|((&[first, len], offset), length)| {
                                slice_groups_slice(offset, length as usize, first, len)
                            })
                            .collect_trusted();
                        GroupsProxy::Slice {
                            groups,
                            rolling: false,
                        }
                    }
                }
            }
        };

        ac.with_groups(groups);

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_valid_aggregation(&self) -> bool {
        true
    }
}
