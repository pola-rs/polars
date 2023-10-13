mod default;
mod groups;
use std::borrow::Cow;

use default::*;
pub(super) use groups::AsofJoinBy;
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

// If called with increasing val_l it will increment offset to the first
// right(offset) >= val_l and return it. If offset == n_right, the loop is done
// (but harmless to call again). offset should initially be 0.
fn asof_forward_step<T: PartialOrd, F: FnMut(usize) -> Option<T>>(
    offset: &mut usize,
    val_l: &T,
    mut right: F,
    n_right: usize,
) -> Option<T> {
    while *offset < n_right {
        if let Some(val_r) = right(*offset) {
            if val_r >= *val_l {
                return Some(val_r);
            }
        }
        *offset += 1;
    }
    None
}

// If called with decreasing val_l it will decrement offset to the last
// right(offset - 1) <= val_l and return it. If offset == 0, the loop is done
// (but harmless to call again). offset should initially be right.len().
fn asof_backward_step<T: PartialOrd, F: FnMut(usize) -> Option<T>>(
    offset: &mut usize,
    val_l: &T,
    mut right: F,
) -> Option<T> {
    while *offset > 0 {
        if let Some(val_r) = right(*offset - 1) {
            if val_r <= *val_l {
                return Some(val_r);
            }
        }
        *offset -= 1;
    }
    None
}

// If called with decreasing val_l it will decrement offset to the last
// right(offset - 1) which is nearest to val_l and return it. This loop never
// indicates it is done by itself, there is always a nearest element.
// offset and next_offset should initially be right.len().
fn asof_nearest_step<T: NumericNative, F: FnMut(usize) -> Option<T>>(
    offset: &mut usize,
    next_offset: &mut usize,
    val_l: &T,
    mut right: F,
) -> Option<T> {
    // right(offset - 1) is the best known bound on the nearest value.
    // next_offset is the rightmost value <= val_l, which is a candidate for
    // being closer.
    while *next_offset > 0 {
        if let Some(val_r) = right(*next_offset - 1) {
            if val_r <= *val_l {
                break;
            } else {
                *offset = *next_offset;
            }
        }
        
        *next_offset -= 1;
    }
    
    // No new candidate option for nearest.
    if *next_offset == 0 {
        return right(*offset - 1);
    }
    
    if let Some(val_r) = right(*offset - 1) {
        // SAFETY: if right(offset - 1) is non-null and next_offset > 0, then
        // this must be non-null as well.
        let next_val_r = unsafe { right(*next_offset - 1).unwrap_unchecked() };
        let diff = val_l.abs_diff(val_r);
        let next_diff = val_l.abs_diff(next_val_r);
        // Because diff can be NaN, but next_diff can't be, we use a 'backwards'
        // check to see if next_diff is worse rather than to see if it's better.
        // This way, if diff is NaN it gets replaced.
        if next_diff >= diff {
            Some(val_r)
        } else {
            *offset = *next_offset;
            Some(next_val_r)
        }
    } else {
        *offset = *next_offset;
        right(*offset - 1)
    }
}

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

fn check_asof_columns(a: &Series, b: &Series, has_tolerance: bool, check_sorted: bool) -> PolarsResult<()> {
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
            "asof join is only supported on primitive key typess"
        );

    }
    polars_ensure!(
        dtype_a == dtype_b,
        ComputeError: "mismatching key dtypes in asof-join: `{}` and `{}`",
        a.dtype(), b.dtype()
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
            DataType::Utf8 => {
                let ca = left_key.utf8().unwrap();
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
        let other = if left_on == right_on {
            Cow::Owned(other.drop(right_on)?)
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
