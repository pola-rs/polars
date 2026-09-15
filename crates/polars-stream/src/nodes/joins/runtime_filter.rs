//! The range of a build-side key, published by a hash join to the scans below
//! its probe side once the build is done.

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_plan::plans::PredicateExpr;

/// Min and max of one build key column. `None` when the build side had no
/// non-null key, so nothing can match.
pub struct KeyRange {
    bounds: Option<(Scalar, Scalar)>,
}

impl KeyRange {
    pub fn new(bounds: Option<(Scalar, Scalar)>) -> Self {
        Self { bounds }
    }

    /// Widen `acc` to cover the non-null values of `column`.
    pub fn extend(acc: &mut Option<(Scalar, Scalar)>, column: &Column) -> PolarsResult<()> {
        let min = column.min_reduce()?;
        let max = column.max_reduce()?;
        if min.is_null() || max.is_null() {
            return Ok(());
        }
        match acc {
            None => *acc = Some((min, max)),
            Some((lo, hi)) => {
                if min.value() < lo.value() {
                    *lo = min;
                }
                if max.value() > hi.value() {
                    *hi = max;
                }
            },
        }
        Ok(())
    }

    /// Merge the ranges of several builders.
    pub fn union(
        ranges: impl IntoIterator<Item = Option<(Scalar, Scalar)>>,
    ) -> Option<(Scalar, Scalar)> {
        ranges.into_iter().flatten().reduce(|(lo, hi), (min, max)| {
            (
                if min.value() < lo.value() { min } else { lo },
                if max.value() > hi.value() { max } else { hi },
            )
        })
    }
}

/// The bounds as length-one series of `dtype`, or `None` when a bound does not
/// survive the cast, in which case the range says nothing.
fn bounds_for(
    bounds: &(Scalar, Scalar),
    dtype: &DataType,
) -> PolarsResult<Option<(Series, Series)>> {
    let cast = |bound: &Scalar| {
        bound
            .clone()
            .cast_with_options(dtype, CastOptions::NonStrict)
    };
    let (lo, hi) = (cast(&bounds.0)?, cast(&bounds.1)?);
    if lo.is_null() || hi.is_null() {
        return Ok(None);
    }
    Ok(Some((
        lo.into_series(PlSmallStr::EMPTY),
        hi.into_series(PlSmallStr::EMPTY),
    )))
}

fn all(name: PlSmallStr, len: usize, value: bool) -> Column {
    Column::new_scalar(name, Scalar::from(value), len)
}

impl PredicateExpr for KeyRange {
    fn evaluate(&self, columns: &[Column]) -> PolarsResult<Option<Column>> {
        let column = &columns[0];
        let Some(bounds) = &self.bounds else {
            return Ok(Some(all(column.name().clone(), column.len(), false)));
        };
        let s = column.as_materialized_series();
        let Some((lo, hi)) = bounds_for(bounds, s.dtype())? else {
            return Ok(None);
        };
        let mask = s.gt_eq(&lo)? & s.lt_eq(&hi)?;
        Ok(Some(mask.fill_null_with_values(false)?.into_column()))
    }

    fn evaluate_stats(
        &self,
        min: &Column,
        max: &Column,
        _null_count: &Column,
    ) -> PolarsResult<Option<Column>> {
        let Some(bounds) = &self.bounds else {
            return Ok(Some(all(min.name().clone(), min.len(), true)));
        };
        let min = min.as_materialized_series();
        let max = max.as_materialized_series();
        let Some((lo, hi)) = bounds_for(bounds, min.dtype())? else {
            return Ok(None);
        };
        // A batch is skipped when it lies entirely below or above the range. An
        // unknown statistic is null and settles nothing.
        let skip = max.lt(&lo)? | min.gt(&hi)?;
        Ok(Some(skip.fill_null_with_values(false)?.into_column()))
    }
}
