mod asof;
mod groups;

use crate::prelude::*;
use asof::*;
use num::Bounded;
use std::borrow::Cow;

#[derive(Clone, Debug, PartialEq)]
pub struct AsOfOptions {
    pub strategy: AsofStrategy,
    /// A tolerance in the same unit as the asof column
    pub tolerance: Option<AnyValue<'static>>,
    /// An timedelta given as
    /// - "5m"
    /// - "2h15m"
    /// - "1d6h"
    /// etc
    pub tolerance_str: Option<String>,
    pub left_by: Option<Vec<String>>,
    pub right_by: Option<Vec<String>>,
}

fn check_asof_columns(a: &Series, b: &Series) -> Result<()> {
    if a.dtype() != b.dtype() {
        return Err(PolarsError::ComputeError(
            format!(
                "keys used in asof-join must have equal dtypes. We got: left: {:?}\tright: {:?}",
                a.dtype(),
                b.dtype()
            )
            .into(),
        ));
    }

    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AsofStrategy {
    /// selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key
    Backward,
    /// selects the first row in the right DataFrame whose ‘on’ key is greater than or equal to the left’s key.
    Forward,
}

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Bounded + PartialOrd,
{
    pub(crate) fn join_asof(
        &self,
        other: &Series,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
    ) -> Result<Vec<Option<IdxSize>>> {
        let other = self.unpack_series_matching_type(other)?;

        if self.null_count() > 0 || other.null_count() > 0 {
            return Err(PolarsError::ComputeError(
                "asof join must not have null values in 'on' arguments".into(),
            ));
        }

        let out = match strategy {
            AsofStrategy::Forward => match tolerance {
                None => join_asof_forward(self.cont_slice().unwrap(), other.cont_slice().unwrap()),
                Some(tolerance) => {
                    let tolerance = tolerance.extract::<T::Native>().unwrap();
                    join_asof_forward_with_tolerance(
                        self.cont_slice().unwrap(),
                        other.cont_slice().unwrap(),
                        tolerance,
                    )
                }
            },
            AsofStrategy::Backward => match tolerance {
                None => join_asof_backward(self.cont_slice().unwrap(), other.cont_slice().unwrap()),
                Some(tolerance) => {
                    let tolerance = tolerance.extract::<T::Native>().unwrap();
                    join_asof_backward_with_tolerance(
                        self.cont_slice().unwrap(),
                        other.cont_slice().unwrap(),
                        tolerance,
                    )
                }
            },
        };
        Ok(out)
    }
}

impl DataFrame {
    /// This is similar to a left-join except that we match on nearest key rather than equal keys.
    /// The keys must be sorted to perform an asof join
    #[cfg_attr(docsrs, doc(cfg(feature = "asof_join")))]
    pub fn join_asof(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        suffix: Option<String>,
    ) -> Result<DataFrame> {
        let left_key = self.column(left_on)?;
        let right_key = other.column(right_on)?;

        check_asof_columns(left_key, right_key)?;
        let left_key = left_key.to_physical_repr();
        let right_key = right_key.to_physical_repr();

        let take_idx = match left_key.dtype() {
            DataType::Int64 => left_key
                .i64()
                .unwrap()
                .join_asof(&right_key, strategy, tolerance),
            DataType::Int32 => left_key
                .i32()
                .unwrap()
                .join_asof(&right_key, strategy, tolerance),
            DataType::UInt64 => left_key
                .u64()
                .unwrap()
                .join_asof(&right_key, strategy, tolerance),
            DataType::UInt32 => left_key
                .u32()
                .unwrap()
                .join_asof(&right_key, strategy, tolerance),
            _ => {
                let left_key = left_key.cast(&DataType::Int32).unwrap();
                let right_key = right_key.cast(&DataType::Int32).unwrap();
                left_key
                    .i32()
                    .unwrap()
                    .join_asof(&right_key, strategy, tolerance)
            }
        }?;

        // take_idx are sorted so this is a bound check for all
        if let Some(Some(idx)) = take_idx.last() {
            assert!((*idx as usize) < other.height())
        }

        // drop right join column
        let other = if left_on == right_on {
            Cow::Owned(other.drop(right_on)?)
        } else {
            Cow::Borrowed(other)
        };

        // Safety:
        // join tuples are in bounds
        let right_df = unsafe {
            other.take_opt_iter_unchecked(
                take_idx
                    .into_iter()
                    .map(|opt_idx| opt_idx.map(|idx| idx as usize)),
            )
        };

        self.finish_join(self.clone(), right_df, suffix)
    }
}
