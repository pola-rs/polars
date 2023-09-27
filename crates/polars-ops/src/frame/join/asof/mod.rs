mod default;
mod groups;
use std::borrow::Cow;

use default::*;
pub(super) use groups::AsofJoinBy;
use num_traits::Bounded;
use polars_core::prelude::*;
use polars_core::utils::{ensure_sorted_arg, slice_slice};
use polars_core::with_match_physical_numeric_polars_type;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

#[cfg(feature = "dtype-categorical")]
use super::_check_categorical_src;
use super::{
    _finish_join, build_tables, get_hash_tbl_threaded_join_partitioned, multiple_keys as mk,
    prepare_bytes,
};
use crate::frame::IntoDf;

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
    pub tolerance_str: Option<SmartString>,
    pub left_by: Option<Vec<SmartString>>,
    pub right_by: Option<Vec<SmartString>>,
}

fn check_asof_columns(a: &Series, b: &Series, check_sorted: bool) -> PolarsResult<()> {
    let dtype_a = a.dtype();
    let dtype_b = b.dtype();
    polars_ensure!(
        dtype_a.to_physical().is_numeric() && dtype_b.to_physical().is_numeric(),
        InvalidOperation:
        "asof join only supported on numeric/temporal keys"
    );
    polars_ensure!(
        dtype_a == dtype_b,
        ComputeError: "mismatching key dtypes in asof-join: `{}` and `{}`",
        a.dtype(), b.dtype()
    );
    polars_ensure!(
        a.null_count() == 0 && b.null_count() == 0,
        ComputeError: "asof join must not have null values in 'on' arguments"
    );
    if check_sorted {
        ensure_sorted_arg(a, "asof_join")?;
        ensure_sorted_arg(b, "asof_join")?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum AsofStrategy {
    /// selects the last row in the right DataFrame whose ‘on’ key is less than or equal to the left’s key
    #[default]
    Backward,
    /// selects the first row in the right DataFrame whose ‘on’ key is greater than or equal to the left’s key.
    Forward,
    /// selects the right in the right DataFrame whose 'on' key is nearest to the left's key.
    Nearest,
}

pub(crate) fn join_asof<T>(
    input_ca: &ChunkedArray<T>,
    other: &Series,
    strategy: AsofStrategy,
    tolerance: Option<AnyValue<'static>>,
) -> PolarsResult<Vec<Option<IdxSize>>>
where
    T: PolarsNumericType,
    T::Native: Bounded + PartialOrd,
{
    let other = input_ca.unpack_series_matching_type(other)?;

    // cont_slice requires a single chunk
    let ca = input_ca.rechunk();
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
            },
        },
        AsofStrategy::Backward => match tolerance {
            None => join_asof_backward(ca.cont_slice().unwrap(), other.cont_slice().unwrap()),
            Some(tolerance) => {
                let tolerance = tolerance.extract::<T::Native>().unwrap();
                join_asof_backward_with_tolerance(
                    input_ca.cont_slice().unwrap(),
                    other.cont_slice().unwrap(),
                    tolerance,
                )
            },
        },
        AsofStrategy::Nearest => match tolerance {
            None => join_asof_nearest(ca.cont_slice().unwrap(), other.cont_slice().unwrap()),
            Some(tolerance) => {
                let tolerance = tolerance.extract::<T::Native>().unwrap();
                join_asof_nearest_with_tolerance(
                    input_ca.cont_slice().unwrap(),
                    other.cont_slice().unwrap(),
                    tolerance,
                )
            },
        },
    };
    Ok(out)
}

pub trait AsofJoin: IntoDf {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn _join_asof(
        &self,
        other: &DataFrame,
        left_on: &str,
        right_on: &str,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
    ) -> PolarsResult<DataFrame> {
        let self_df = self.to_df();
        let left_key = self_df.column(left_on)?;
        let right_key = other.column(right_on)?;

        check_asof_columns(left_key, right_key, true)?;
        let left_key = left_key.to_physical_repr();
        let right_key = right_key.to_physical_repr();

        let take_idx = match left_key.dtype() {
            DataType::Int64 => {
                let ca = left_key.i64().unwrap();
                join_asof(ca, &right_key, strategy, tolerance)
            },
            DataType::Int32 => {
                let ca = left_key.i32().unwrap();
                join_asof(ca, &right_key, strategy, tolerance)
            },
            DataType::UInt64 => {
                let ca = left_key.u64().unwrap();
                join_asof(ca, &right_key, strategy, tolerance)
            },
            DataType::UInt32 => {
                let ca = left_key.u32().unwrap();
                join_asof(ca, &right_key, strategy, tolerance)
            },
            DataType::Float32 => {
                let ca = left_key.f32().unwrap();
                join_asof(ca, &right_key, strategy, tolerance)
            },
            DataType::Float64 => {
                let ca = left_key.f64().unwrap();
                join_asof(ca, &right_key, strategy, tolerance)
            },
            _ => {
                let left_key = left_key.cast(&DataType::Int32).unwrap();
                let right_key = right_key.cast(&DataType::Int32).unwrap();
                let ca = left_key.i32().unwrap();
                join_asof(ca, &right_key, strategy, tolerance)
            },
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

        let mut left = self_df.clone();
        let mut take_idx = &*take_idx;

        if let Some((offset, len)) = slice {
            left = left.slice(offset, len);
            take_idx = slice_slice(take_idx, offset, len);
        }

        // SAFETY: join tuples are in bounds.
        let right_df = unsafe { other.take_unchecked(&take_idx.iter().copied().collect_ca("")) };

        _finish_join(left, right_df, suffix.as_deref())
    }

    /// This is similar to a left-join except that we match on nearest key rather than equal keys.
    /// The keys must be sorted to perform an asof join
    fn join_asof(
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

impl AsofJoin for DataFrame {}
