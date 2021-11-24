use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_arrow::arrow::array::ArrayRef;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use std::convert::TryFrom;
use std::sync::Arc;

pub struct TernaryExpr {
    pub predicate: Arc<dyn PhysicalExpr>,
    pub truthy: Arc<dyn PhysicalExpr>,
    pub falsy: Arc<dyn PhysicalExpr>,
    pub expr: Expr,
}

impl PhysicalExpr for TernaryExpr {
    fn as_expression(&self) -> &Expr {
        &self.expr
    }
    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> Result<Series> {
        let mask_series = self.predicate.evaluate(df, state)?;
        let mask = mask_series.bool()?;
        let truthy = self.truthy.evaluate(df, state)?;
        let falsy = self.falsy.evaluate(df, state)?;
        truthy.zip_with(mask, &falsy)
    }
    fn to_field(&self, input_schema: &Schema) -> Result<Field> {
        self.truthy.to_field(input_schema)
    }

    #[allow(clippy::ptr_arg)]
    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupTuples,
        state: &ExecutionState,
    ) -> Result<AggregationContext<'a>> {
        let required_height = df.height();
        let ac_mask = self.predicate.evaluate_on_groups(df, groups, state)?;
        let mask_s = ac_mask.flat();

        assert!(
            !(mask_s.len() != required_height),
            "The predicate is of a different length than the groups.\
The predicate produced {} values. Where the original DataFrame has {} values",
            mask_s.len(),
            required_height
        );

        let mask = mask_s.bool()?;
        let mut ac_truthy = self.truthy.evaluate_on_groups(df, groups, state)?;
        let mut ac_falsy = self.falsy.evaluate_on_groups(df, groups, state)?;

        if !ac_truthy.can_combine(&ac_falsy) {
            return Err(PolarsError::InvalidOperation(
                "\
            cannot combine this ternary expression, the groups do not match"
                    .into(),
            ));
        }

        match (ac_truthy.agg_state(), ac_falsy.agg_state()) {
            (AggState::AggregatedFlat(_), AggState::NotAggregated(_)) => {
                // this is a flat series of len eq to group tuples
                let truthy = ac_truthy.aggregated();
                let truthy = truthy.as_ref();
                let arr_truthy = &truthy.chunks()[0];
                assert_eq!(truthy.len(), groups.len());

                // we create a dummy Series that is not cloned nor moved
                // so we can swap the ArrayRef during the hot loop
                // this prevents a series Arc alloc and a vec alloc per iteration
                let dummy = Series::try_from(("dummy", vec![arr_truthy.clone()])).unwrap();
                let chunks = unsafe {
                    let chunks = dummy.chunks();
                    let ptr = chunks.as_ptr() as *mut ArrayRef;
                    let len = chunks.len();
                    std::slice::from_raw_parts_mut(ptr, len)
                };

                // this is now a list
                let falsy = ac_falsy.aggregated();
                let falsy = falsy.list().unwrap();

                let mut ca: ListChunked = falsy
                    .amortized_iter()
                    .enumerate()
                    .map(|(idx, opt_s)| {
                        opt_s
                            .map(|s| {
                                let falsy = s.as_ref();

                                // Safety:
                                // we are in bounds
                                let mut arr =
                                    unsafe { Arc::from(arr_truthy.slice_unchecked(idx, 1)) };
                                std::mem::swap(&mut chunks[0], &mut arr);
                                let truthy = &dummy;

                                truthy.zip_with(mask, falsy)
                            })
                            .transpose()
                    })
                    .collect::<Result<_>>()?;
                ca.rename(truthy.name());

                ac_truthy.with_series(ca.into_series(), true);
                Ok(ac_truthy)
            }
            (AggState::NotAggregated(_), AggState::AggregatedFlat(_)) => {
                // this is now a list
                let truthy = ac_truthy.aggregated();
                let truthy = truthy.list().unwrap();

                // this is a flat series of len eq to group tuples
                let falsy = ac_falsy.aggregated();
                assert_eq!(falsy.len(), groups.len());
                let falsy = falsy.as_ref();
                let arr_falsy = &falsy.chunks()[0];

                // we create a dummy Series that is not cloned nor moved
                // so we can swap the ArrayRef during the hot loop
                // this prevents a series Arc alloc and a vec alloc per iteration
                let dummy = Series::try_from(("dummy", vec![arr_falsy.clone()])).unwrap();
                let chunks = unsafe {
                    let chunks = dummy.chunks();
                    let ptr = chunks.as_ptr() as *mut ArrayRef;
                    let len = chunks.len();
                    std::slice::from_raw_parts_mut(ptr, len)
                };

                let mut ca: ListChunked = truthy
                    .amortized_iter()
                    .enumerate()
                    .map(|(idx, opt_s)| {
                        opt_s
                            .map(|s| {
                                let truthy = s.as_ref();
                                // Safety:
                                // we are in bounds
                                let mut arr =
                                    unsafe { Arc::from(arr_falsy.slice_unchecked(idx, 1)) };
                                std::mem::swap(&mut chunks[0], &mut arr);
                                let falsy = &dummy;

                                truthy.zip_with(mask, falsy)
                            })
                            .transpose()
                    })
                    .collect::<Result<_>>()?;
                ca.rename(truthy.name());

                ac_truthy.with_series(ca.into_series(), true);
                Ok(ac_truthy)
            }
            // Both are or a flat series or aggreagated into a list
            // so we can flatten the Series an apply the operators
            _ => {
                let out = ac_truthy.flat().zip_with(mask, ac_falsy.flat().as_ref())?;

                assert!(!(out.len() != required_height), "The output of the `when -> then -> otherwise-expr` is of a different length than the groups.\
The expr produced {} values. Where the original DataFrame has {} values",
                        out.len(),
                        required_height);

                ac_truthy.with_series(out, false);

                Ok(ac_truthy)
            }
        }
    }
}
