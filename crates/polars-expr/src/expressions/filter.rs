use polars_core::POOL;
use polars_core::prelude::*;
use polars_utils::UnitVec;

use super::*;
use crate::expressions::{AggregationContext, PhysicalExpr};

pub struct FilterExpr {
    pub(crate) input: Arc<dyn PhysicalExpr>,
    pub(crate) by: Arc<dyn PhysicalExpr>,
    expr: Expr,
}

impl FilterExpr {
    pub fn new(input: Arc<dyn PhysicalExpr>, by: Arc<dyn PhysicalExpr>, expr: Expr) -> Self {
        Self { input, by, expr }
    }
}

impl PhysicalExpr for FilterExpr {
    fn as_expression(&self) -> Option<&Expr> {
        Some(&self.expr)
    }

    fn evaluate(&self, df: &DataFrame, state: &ExecutionState) -> PolarsResult<Column> {
        let s_f = || self.input.evaluate(df, state);
        let predicate_f = || self.by.evaluate(df, state);

        let (series, predicate) = POOL.install(|| rayon::join(s_f, predicate_f));
        let (series, predicate) = (series?, predicate?);

        series.filter(predicate.bool()?)
    }

    fn evaluate_on_groups<'a>(
        &self,
        df: &DataFrame,
        groups: &'a GroupPositions,
        state: &ExecutionState,
    ) -> PolarsResult<AggregationContext<'a>> {
        let ac_s_f = || self.input.evaluate_on_groups(df, groups, state);
        let ac_predicate_f = || self.by.evaluate_on_groups(df, groups, state);

        let (ac_s, ac_predicate) = POOL.install(|| rayon::join(ac_s_f, ac_predicate_f));
        let (mut ac_s, mut ac_predicate) = (ac_s?, ac_predicate?);

        ac_s.set_groups_for_undefined_agg_states();
        ac_predicate.set_groups_for_undefined_agg_states();

        ac_s.groups();
        ac_predicate.groups();

        assert_eq!(ac_s.groups.len(), ac_predicate.groups.len());

        // Slow path. Different groups for input and predicate.
        if !std::ptr::eq(ac_s.groups.as_ref(), ac_predicate.groups.as_ref()) {
            let mut needs_broadcast = false;
            for (l, r) in ac_s.groups.iter().zip(ac_predicate.groups.iter()) {
                needs_broadcast |= (l.len() == 1 || r.len() == 1) && l.len() != r.len();
                if l.len() != 1 && r.len() != 1 && l.len() != r.len() {
                    polars_bail!(length_mismatch = "filter", l.len(), r.len());
                }
            }

            fn broadcast(
                groups: &GroupsType,
                other_lengths: impl Iterator<Item = usize>,
            ) -> GroupsIdx {
                match groups {
                    GroupsType::Idx(i) => i
                        .iter()
                        .zip(other_lengths)
                        .map(|((fst, idxs), l)| {
                            if idxs.len() != l && idxs.len() == 1 {
                                (fst, UnitVec::from_iter(std::iter::repeat_n(fst, l)))
                            } else {
                                (fst, idxs.clone())
                            }
                        })
                        .collect(),
                    GroupsType::Slice {
                        groups,
                        overlapping: _,
                    } => groups
                        .iter()
                        .zip(other_lengths)
                        .map(|([start, length], l)| {
                            if *length as usize != l && *length == 1 {
                                (*start, UnitVec::from_iter(std::iter::repeat_n(*start, l)))
                            } else {
                                (*start, UnitVec::from_iter(*start..*start + *length))
                            }
                        })
                        .collect(),
                }
            }

            // If either side needs a broadcast, perform the broadcasting on the groups before the
            // `aggregated`.
            if needs_broadcast {
                ac_s.groups = Cow::Owned(
                    GroupsType::Idx(broadcast(
                        ac_s.groups.as_ref(),
                        ac_predicate.groups.iter().map(|i| i.len()),
                    ))
                    .into_sliceable(),
                );
                ac_predicate.groups = Cow::Owned(
                    GroupsType::Idx(broadcast(
                        ac_predicate.groups.as_ref(),
                        ac_s.groups.iter().map(|i| i.len()),
                    ))
                    .into_sliceable(),
                );
            }

            ac_s.normalize_values();
            ac_predicate.normalize_values();
        }

        let predicate = ac_predicate.flat_naive();
        let predicate = predicate.bool()?;
        let predicate = predicate.rechunk();
        let predicate = predicate.downcast_as_array();
        let predicate = if let Some(validity) = predicate.validity()
            && validity.unset_bits() > 0
        {
            predicate.values() & validity
        } else {
            predicate.values().clone()
        };

        crate::dispatch::drop_items(ac_s, &predicate)
    }

    fn to_field(&self, input_schema: &Schema) -> PolarsResult<Field> {
        self.input.to_field(input_schema)
    }

    fn is_scalar(&self) -> bool {
        false
    }
}
