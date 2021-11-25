use crate::physical_plan::state::ExecutionState;
use crate::prelude::*;
use polars_core::frame::groupby::GroupTuples;
use polars_core::prelude::*;
use polars_core::series::unstable::UnstableSeries;
use polars_core::POOL;
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

        let op_mask = || self.predicate.evaluate_on_groups(df, groups, state);
        let op_truthy = || self.truthy.evaluate_on_groups(df, groups, state);
        let op_falsy = || self.falsy.evaluate_on_groups(df, groups, state);

        let (ac_mask, (ac_truthy, ac_falsy)) =
            POOL.install(|| rayon::join(op_mask, || rayon::join(op_truthy, op_falsy)));
        let mut ac_mask = ac_mask?;
        let mut ac_truthy = ac_truthy?;
        let mut ac_falsy = ac_falsy?;

        let mask_s = ac_mask.flat();

        assert!(
            !(mask_s.len() != required_height),
            "The predicate is of a different length than the groups.\
The predicate produced {} values. Where the original DataFrame has {} values",
            mask_s.len(),
            required_height
        );

        assert!(
            ac_truthy.can_combine(&ac_falsy),
            "cannot combine this ternary expression, the groups do not match"
        );

        match (ac_truthy.agg_state(), ac_falsy.agg_state()) {
            // if the groups_len == df.len we can just apply all flat.
            (AggState::AggregatedFlat(s), AggState::NotAggregated(_)) if s.len() != df.height() => {
                // this is a flat series of len eq to group tuples
                let truthy = ac_truthy.aggregated();
                let truthy = truthy.as_ref();
                let arr_truthy = &truthy.chunks()[0];
                assert_eq!(truthy.len(), groups.len());

                // we create a dummy Series that is not cloned nor moved
                // so we can swap the ArrayRef during the hot loop
                // this prevents a series Arc alloc and a vec alloc per iteration
                let dummy = Series::try_from(("dummy", vec![arr_truthy.clone()])).unwrap();
                let mut us = UnstableSeries::new(&dummy);

                // this is now a list
                let falsy = ac_falsy.aggregated();
                let falsy = falsy.as_ref();
                let falsy = falsy.list().unwrap();

                let mask = ac_mask.aggregated();
                let mask = mask.as_ref();
                let mask = mask.list()?;
                if !matches!(mask.inner_dtype(), DataType::Boolean) {
                    return Err(PolarsError::ComputeError(
                        format!("expected mask of type bool, got {:?}", mask.inner_dtype()).into(),
                    ));
                }

                let mut ca: ListChunked = falsy
                    .amortized_iter()
                    .zip(mask.amortized_iter())
                    .enumerate()
                    .map(|(idx, (opt_falsy, opt_mask))| {
                        match (opt_falsy, opt_mask) {
                            (Some(falsy), Some(mask)) => {
                                let falsy = falsy.as_ref();
                                let mask = mask.as_ref();
                                let mask = mask.bool()?;

                                // Safety:
                                // we are in bounds
                                let arr = unsafe { Arc::from(arr_truthy.slice_unchecked(idx, 1)) };
                                us.swap(arr);
                                let truthy = us.as_ref();

                                Some(truthy.zip_with(mask, falsy))
                            }
                            _ => None,
                        }
                        .transpose()
                    })
                    .collect::<Result<_>>()?;
                ca.rename(truthy.name());

                ac_truthy.with_series(ca.into_series(), true);
                Ok(ac_truthy)
            }
            // if the groups_len == df.len we can just apply all flat.
            (AggState::NotAggregated(_), AggState::AggregatedFlat(s)) if s.len() != df.height() => {
                // this is now a list
                let truthy = ac_truthy.aggregated();
                let truthy = truthy.as_ref();
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
                let mut us = UnstableSeries::new(&dummy);

                let mask = ac_mask.aggregated();
                let mask = mask.as_ref();
                let mask = mask.list()?;
                if !matches!(mask.inner_dtype(), DataType::Boolean) {
                    return Err(PolarsError::ComputeError(
                        format!("expected mask of type bool, got {:?}", mask.inner_dtype()).into(),
                    ));
                }

                let mut ca: ListChunked = truthy
                    .amortized_iter()
                    .zip(mask.amortized_iter())
                    .enumerate()
                    .map(|(idx, (opt_truthy, opt_mask))| {
                        match (opt_truthy, opt_mask) {
                            (Some(truthy), Some(mask)) => {
                                let truthy = truthy.as_ref();
                                let mask = mask.as_ref();
                                let mask = mask.bool()?;

                                // Safety:
                                // we are in bounds
                                let arr = unsafe { Arc::from(arr_falsy.slice_unchecked(idx, 1)) };
                                us.swap(arr);
                                let falsy = us.as_ref();

                                Some(truthy.zip_with(mask, falsy))
                            }
                            _ => None,
                        }
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
                let mask = mask_s.bool()?;
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
