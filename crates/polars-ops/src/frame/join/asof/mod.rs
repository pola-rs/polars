mod default;
mod groups;
use std::borrow::Cow;

use default::*;
pub use groups::AsofJoinBy;
use polars_core::prelude::*;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use smartstring::alias::String as SmartString;

#[cfg(feature = "dtype-categorical")]
use super::_check_categorical_src;
use super::{_finish_join, build_tables, prepare_bytes};
use crate::frame::IntoDf;
use crate::series::SeriesMethods;

trait AsofJoinState<T>: Default {
    fn next<F: FnMut(IdxSize) -> Option<T>>(
        &mut self,
        left_val: &T,
        right: F,
        n_right: IdxSize,
    ) -> Option<IdxSize>;
}

#[derive(Default)]
struct AsofJoinForwardState {
    scan_offset: IdxSize,
}

impl<T: PartialOrd> AsofJoinState<T> for AsofJoinForwardState {
    #[inline]
    fn next<F: FnMut(IdxSize) -> Option<T>>(
        &mut self,
        left_val: &T,
        mut right: F,
        n_right: IdxSize,
    ) -> Option<IdxSize> {
        while (self.scan_offset) < n_right {
            if let Some(right_val) = right(self.scan_offset) {
                if right_val >= *left_val {
                    return Some(self.scan_offset);
                }
            }
            self.scan_offset += 1;
        }
        None
    }
}

#[derive(Default)]
struct AsofJoinBackwardState {
    // best_bound is the greatest right index <= left_val.
    best_bound: Option<IdxSize>,
    scan_offset: IdxSize,
}

impl<T: PartialOrd> AsofJoinState<T> for AsofJoinBackwardState {
    #[inline]
    fn next<F: FnMut(IdxSize) -> Option<T>>(
        &mut self,
        left_val: &T,
        mut right: F,
        n_right: IdxSize,
    ) -> Option<IdxSize> {
        while self.scan_offset < n_right {
            if let Some(right_val) = right(self.scan_offset) {
                if right_val <= *left_val {
                    self.best_bound = Some(self.scan_offset);
                } else {
                    break;
                }
            }
            self.scan_offset += 1;
        }
        self.best_bound
    }
}

#[derive(Default)]
struct AsofJoinNearestState {
    // best_bound is the nearest value to left_val, with ties broken towards the last element.
    best_bound: Option<IdxSize>,
    scan_offset: IdxSize,
}

impl<T: NumericNative> AsofJoinState<T> for AsofJoinNearestState {
    #[inline]
    fn next<F: FnMut(IdxSize) -> Option<T>>(
        &mut self,
        left_val: &T,
        mut right: F,
        n_right: IdxSize,
    ) -> Option<IdxSize> {
        // Skipping ahead to the first value greater than left_val. This is
        // cheaper than computing differences.
        while self.scan_offset < n_right {
            if let Some(scan_right_val) = right(self.scan_offset) {
                if scan_right_val <= *left_val {
                    self.best_bound = Some(self.scan_offset);
                } else {
                    // Now we must compute a difference to see if scan_right_val
                    // is closer than our current best bound.
                    let scan_is_better = if let Some(best_idx) = self.best_bound {
                        let best_right_val = unsafe { right(best_idx).unwrap_unchecked() };
                        let best_diff = left_val.abs_diff(best_right_val);
                        let scan_diff = left_val.abs_diff(scan_right_val);

                        scan_diff <= best_diff
                    } else {
                        true
                    };

                    if scan_is_better {
                        self.best_bound = Some(self.scan_offset);
                        self.scan_offset += 1;

                        // It is possible there are later elements equal to our
                        // scan, so keep going on.
                        while self.scan_offset < n_right {
                            if let Some(next_right_val) = right(self.scan_offset) {
                                if next_right_val == scan_right_val {
                                    self.best_bound = Some(self.scan_offset);
                                } else {
                                    break;
                                }
                            }

                            self.scan_offset += 1;
                        }
                    }

                    break;
                }
            }

            self.scan_offset += 1;
        }

        self.best_bound
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Default, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct AsOfOptions {
    pub strategy: AsofStrategy,
    /// A tolerance in the same unit as the asof column
    pub tolerance: Option<AnyValue<'static>>,
    /// A time duration specified as a string, for example:
    /// - "5m"
    /// - "2h15m"
    /// - "1d6h"
    pub tolerance_str: Option<SmartString>,
    pub left_by: Option<Vec<SmartString>>,
    pub right_by: Option<Vec<SmartString>>,
}

fn check_asof_columns(
    a: &Series,
    b: &Series,
    has_tolerance: bool,
    check_sorted: bool,
) -> PolarsResult<()> {
    let dtype_a = a.dtype();
    let dtype_b = b.dtype();
    if has_tolerance {
        polars_ensure!(
            dtype_a.to_physical().is_numeric() && dtype_b.to_physical().is_numeric(),
            InvalidOperation:
            "asof join with tolerance is only supported on numeric/temporal keys"
        );
    } else {
        polars_ensure!(
            dtype_a.to_physical().is_primitive() && dtype_b.to_physical().is_primitive(),
            InvalidOperation:
            "asof join is only supported on primitive key types"
        );
    }
    polars_ensure!(
        dtype_a == dtype_b,
        ComputeError: "mismatching key dtypes in asof-join: `{}` and `{}`",
        a.dtype(), b.dtype()
    );
    if check_sorted {
        a.ensure_sorted_arg("asof_join")?;
        b.ensure_sorted_arg("asof_join")?;
    }
    Ok(())
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default, Hash)]
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

pub trait AsofJoin: IntoDf {
    #[doc(hidden)]
    #[allow(clippy::too_many_arguments)]
    fn _join_asof(
        &self,
        other: &DataFrame,
        left_key: &Series,
        right_key: &Series,
        strategy: AsofStrategy,
        tolerance: Option<AnyValue<'static>>,
        suffix: Option<String>,
        slice: Option<(i64, usize)>,
        coalesce: bool,
    ) -> PolarsResult<DataFrame> {
        let self_df = self.to_df();

        check_asof_columns(left_key, right_key, tolerance.is_some(), true)?;
        let left_key = left_key.to_physical_repr();
        let right_key = right_key.to_physical_repr();

        let mut take_idx = match left_key.dtype() {
            DataType::Int64 => {
                let ca = left_key.i64().unwrap();
                join_asof_numeric(ca, &right_key, strategy, tolerance)
            },
            DataType::Int32 => {
                let ca = left_key.i32().unwrap();
                join_asof_numeric(ca, &right_key, strategy, tolerance)
            },
            DataType::UInt64 => {
                let ca = left_key.u64().unwrap();
                join_asof_numeric(ca, &right_key, strategy, tolerance)
            },
            DataType::UInt32 => {
                let ca = left_key.u32().unwrap();
                join_asof_numeric(ca, &right_key, strategy, tolerance)
            },
            DataType::Float32 => {
                let ca = left_key.f32().unwrap();
                join_asof_numeric(ca, &right_key, strategy, tolerance)
            },
            DataType::Float64 => {
                let ca = left_key.f64().unwrap();
                join_asof_numeric(ca, &right_key, strategy, tolerance)
            },
            DataType::Boolean => {
                let ca = left_key.bool().unwrap();
                join_asof::<BooleanType>(ca, &right_key, strategy)
            },
            DataType::Binary => {
                let ca = left_key.binary().unwrap();
                join_asof::<BinaryType>(ca, &right_key, strategy)
            },
            DataType::String => {
                let ca = left_key.str().unwrap();
                let right_binary = right_key.cast(&DataType::Binary).unwrap();
                join_asof::<BinaryType>(&ca.as_binary(), &right_binary, strategy)
            },
            _ => {
                let left_key = left_key.cast(&DataType::Int32).unwrap();
                let right_key = right_key.cast(&DataType::Int32).unwrap();
                let ca = left_key.i32().unwrap();
                join_asof_numeric(ca, &right_key, strategy, tolerance)
            },
        }?;

        // Drop right join column.
        let other = if coalesce && left_key.name() == right_key.name() {
            Cow::Owned(other.drop(right_key.name())?)
        } else {
            Cow::Borrowed(other)
        };

        let mut left = self_df.clone();
        if let Some((offset, len)) = slice {
            left = left.slice(offset, len);
            take_idx = take_idx.slice(offset, len);
        }

        // SAFETY: join tuples are in bounds.
        let right_df = unsafe { other.take_unchecked(&take_idx) };

        _finish_join(left, right_df, suffix.as_deref())
    }
}

impl AsofJoin for DataFrame {}
