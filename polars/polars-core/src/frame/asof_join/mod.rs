mod asof;
mod groups;

use std::borrow::Cow;

use asof::*;
use num::Bounded;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::*;
use crate::utils::slice_slice;

#[derive(Clone, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
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

fn check_asof_columns(a: &Series, b: &Series) -> PolarsResult<()> {
    if a.dtype() != b.dtype() {
        Err(PolarsError::ComputeError(
            format!(
                "keys used in asof-join must have equal dtypes. We got: left: {:?}\tright: {:?}",
                a.dtype(),
                b.dtype()
            )
            .into(),
        ))
    } else if a.null_count() > 0 || b.null_count() > 0 {
        Err(PolarsError::ComputeError(
            "asof join must not have null values in 'on' arguments".into(),
        ))
    } else {
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AsofStrategy {
    /// selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key
    #[default]
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
    ) -> PolarsResult<Vec<Option<IdxSize>>> {
        let other = self.unpack_series_matching_type(other)?;

        // cont_slice requires a single chunk
        let ca = self.rechunk();
        let other = other.rechunk();

        let out = match strategy {
            AsofStrategy::Forward => match tolerance {
                None => join_asof_forward(ca.cont_slice().unwrap(), other.cont_slice().unwrap()),
                Some(tolerance) => {
                    let tolerance = tolerance.extract::<T::Native>().unwrap();
                    join_asof_forward_with_tolerance(
                        ca.cont_slice().unwrap(),
                        other.cont_slice().unwrap(),
                        tolerance,
                    )
                }
            },
            AsofStrategy::Backward => match tolerance {
                None => join_asof_backward(ca.cont_slice().unwrap(), other.cont_slice().unwrap()),
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
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    pub fn _join_asof(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
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
            DataType::Float32 => left_key
                .f32()
                .unwrap()
                .join_asof(&right_key, strategy, tolerance),
            DataType::Float64 => left_key
                .f64()
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

        let mut left = self.clone();
        let mut take_idx = &*take_idx;

        if let Some((offset, len)) = slice {
            left = left.slice(offset, len);
            take_idx = slice_slice(take_idx, offset, len);
        }

        // Safety:
        // join tuples are in bounds
        let right_df = unsafe {
            other.take_opt_iter_unchecked(
                take_idx
                    .iter()
                    .map(|opt_idx| opt_idx.map(|idx| idx as usize)),
            )
        };

        _finish_join(left, right_df, suffix.as_deref())
    }

    /// This is similar to a left-join except that we match on nearest key rather than equal keys.
    /// The keys must be sorted to perform an asof join
    pub fn join_asof(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        suffix: Option<String>,
    ) -> PolarsResult<DataFrame> {
        self._join_asof(other, left_on, right_on, strategy, tolerance, suffix, None)
    }
}
