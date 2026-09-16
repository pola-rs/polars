//! The range of a build-side key, published by a hash join to the scans below
//! its probe side once the build is done. Scans consult it to skip batches by
//! their statistics only.

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::*;
use polars_plan::plans::PredicateExpr;

/// Min and max of one build key column. Empty until a non-null key is seen; an
/// empty range published after the build means nothing can match.
#[derive(Debug, Default)]
pub struct KeyRange {
    bounds: Option<(Scalar, Scalar)>,
}

impl KeyRange {
    /// Widen the range to cover the non-null values of `column`.
    pub fn extend(&mut self, column: &Column) -> PolarsResult<()> {
        let min = column.min_reduce()?;
        let max = column.max_reduce()?;
        if !min.is_null() && !max.is_null() {
            self.merge(Self {
                bounds: Some((min, max)),
            });
        }
        Ok(())
    }

    /// Widen the range to cover `other`.
    pub fn merge(&mut self, other: Self) {
        let Some((min, max)) = other.bounds else {
            return;
        };
        match &mut self.bounds {
            None => self.bounds = Some((min, max)),
            Some((lo, hi)) => {
                if min.value() < lo.value() {
                    *lo = min;
                }
                if max.value() > hi.value() {
                    *hi = max;
                }
            },
        }
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

impl PredicateExpr for KeyRange {
    fn evaluate_stats(
        &self,
        min: &Column,
        max: &Column,
        _null_count: &Column,
    ) -> PolarsResult<Option<Column>> {
        let Some(bounds) = &self.bounds else {
            return Ok(Some(Column::new_scalar(
                min.name().clone(),
                Scalar::from(true),
                min.len(),
            )));
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
