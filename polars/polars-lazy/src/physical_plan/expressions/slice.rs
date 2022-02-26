use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::{GroupsProxy, IdxItem};
use polars_core::prelude::*;
use polars_core::utils::{slice_offsets, CustomIterTools};
use polars_core::POOL;
use rayon::prelude::*;
use std::sync::Arc;

pub struct SliceExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) offset: Arc<dyn PhysicalExpr>,
    pub(crate) length: Arc<dyn PhysicalExpr>,
    pub(crate) expr: Expr,
}

fn extract_offset(offset: &Series) -> Result<i64> {
    if offset.len() > 1 {
        return Err(PolarsError::ComputeError(format!("Invalid argument to slice; expected an offset literal but got an Series of length {}", offset.len()).into()));
    }
    offset.get(0).extract::<i64>().ok_or_else(|| {
        PolarsError::ComputeError(format!("could not get an offset from {:?}", offset).into())
    })
}

fn extract_length(length: &Series) -> Result<usize> {
    if length.len() > 1 {
        return Err(PolarsError::ComputeError(format!("Invalid argument to slice; expected a length literal but got an Series of length {}", length.len()).into()));
    }
    length.get(0).extract::<usize>().ok_or_else(|| {
        PolarsError::ComputeError(format!("could not get a length from {:?}", length).into())
    })
}

fn extract_args(offset: &Series, length: &Series) -> Result<(i64, usize)> {
    Ok((extract_offset(offset)?, extract_length(length)?))
}

fn check_argument(arg: &Series, groups: &GroupsProxy, name: &str) -> Result<()> {
    if let DataType::List(_) = arg.dtype() {
        Err(PolarsError::ComputeError(
            format!(
                "Invalid slice argument: cannot use an array as {} argument",
                name
            )
            .into(),
        ))
    } else if arg.len() != groups.len() {
        Err(PolarsError::ComputeError(format!("Invalid slice argument: the evaluated length expression was of different {} than the number of groups", name).into()))
    } else if arg.null_count() > 0 {
        Err(PolarsError::ComputeError(
            format!(
                "Invalid slice argument: the {} expression should not have null values",
                name
            )
            .into(),
        ))
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
    let (offset, len) = slice_offsets(offset, length as usize, len as usize);
    [first + offset as IdxSize, len as IdxSize]
}

impl PhysicalExpr for SliceExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let results = POOL.install(|| {
            [&self.offset, &self.length, &self.input]
                .par_iter()
                .map(|e| e.evaluate(df, state))
                .collect::<Result<Vec<_>>>()
        })?;
        let offset = &results[0];
        let length = &results[1];
        let series = &results[2];
        let (offset, length) = extract_args(offset, length)?;

        Ok(series.slice(offset, length))
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupsProxy,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let mut results = POOL.install(|| {
            [&self.offset, &self.length, &self.input]
                .par_iter()
                .map(|e| e.evaluate_on_groups(df, groups, state))
                .collect::<Result<Vec<_>>>()
        })?;
        let mut ac = results.pop().unwrap();
        let mut ac_length = results.pop().unwrap();
        let mut ac_offset = results.pop().unwrap();

        let groups = ac.groups();

        use AggState::*;
        let groups = match (&ac_offset.state, &ac_length.state) {
            (Literal(offset), Literal(length)) => {
                let (offset, length) = extract_args(offset, length)?;

                match groups.as_ref() {
                    GroupsProxy::Idx(groups) => {
                        let groups = groups
                            .iter()
                            .map(|(first, idx)| slice_groups_idx(offset, length, first, idx))
                            .collect();
                        GroupsProxy::Idx(groups)
                    }
                    GroupsProxy::Slice(groups) => {
                        let groups = groups
                            .iter()
                            .map(|&[first, len]| slice_groups_slice(offset, length, first, len))
                            .collect_trusted();
                        GroupsProxy::Slice(groups)
                    }
                }
            }
            (Literal(offset), _) => {
                let offset = extract_offset(offset)?;
                let length = ac_length.aggregated();
                check_argument(&length, groups, "length")?;

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
                    GroupsProxy::Slice(groups) => {
                        let groups = groups
                            .iter()
                            .zip(length.into_no_null_iter())
                            .map(|(&[first, len], length)| {
                                slice_groups_slice(offset, length as usize, first, len)
                            })
                            .collect_trusted();
                        GroupsProxy::Slice(groups)
                    }
                }
            }
            (_, Literal(length)) => {
                let length = extract_length(length)?;
                let offset = ac_offset.aggregated();
                check_argument(&offset, groups, "offset")?;

                let offset = offset.cast(&DataType::Int64)?;
                let offset = offset.i64().unwrap();

                match groups.as_ref() {
                    GroupsProxy::Idx(groups) => {
                        let groups = groups
                            .iter()
                            .zip(offset.into_no_null_iter())
                            .map(|((first, idx), offset)| {
                                slice_groups_idx(offset, length as usize, first, idx)
                            })
                            .collect();
                        GroupsProxy::Idx(groups)
                    }
                    GroupsProxy::Slice(groups) => {
                        let groups = groups
                            .iter()
                            .zip(offset.into_no_null_iter())
                            .map(|(&[first, len], offset)| {
                                slice_groups_slice(offset, length, first, len)
                            })
                            .collect_trusted();
                        GroupsProxy::Slice(groups)
                    }
                }
            }
            _ => {
                let length = ac_length.aggregated();
                let offset = ac_offset.aggregated();
                check_argument(&length, groups, "length")?;
                check_argument(&offset, groups, "offset")?;

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
                    GroupsProxy::Slice(groups) => {
                        let groups = groups
                            .iter()
                            .zip(offset.into_no_null_iter())
                            .zip(length.into_no_null_iter())
                            .map(|((&[first, len], offset), length)| {
                                slice_groups_slice(offset, length as usize, first, len)
                            })
                            .collect_trusted();
                        GroupsProxy::Slice(groups)
                    }
                }
            }
        };

        ac.with_groups(groups);

        Ok(ac)
    }

    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.input.to_field(input_schema)
    }

    fn as_agg_expr(&self) -> Result<&dyn PhysicalAggregation> {
        Ok(self)
    }
}
impl PhysicalAggregation for SliceExpr {
    // As a final aggregation a Slice returns a list array.
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
