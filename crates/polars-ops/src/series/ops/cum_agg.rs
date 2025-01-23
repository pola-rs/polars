use std::ops::{Add, AddAssign, Mul};

use arity::unary_elementwise_values;
use arrow::array::BooleanArray;
use arrow::bitmap::BitmapBuilder;
use num_traits::{Bounded, One, Zero};
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::{CustomIterTools, NoNull};
use polars_core::with_match_physical_numeric_polars_type;

fn det_max<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + AddAssign + Add<Output = T>,
{
    match v {
        Some(v) => {
            if v > *state {
                *state = v
            }
            Some(Some(*state))
        },
        None => Some(None),
    }
}

fn det_min<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + AddAssign + Add<Output = T>,
{
    match v {
        Some(v) => {
            if v < *state {
                *state = v
            }
            Some(Some(*state))
        },
        None => Some(None),
    }
}

fn det_sum<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + AddAssign + Add<Output = T>,
{
    match v {
        Some(v) => {
            *state += v;
            Some(Some(*state))
        },
        None => Some(None),
    }
}

fn det_prod<T>(state: &mut T, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + Mul<Output = T>,
{
    match v {
        Some(v) => {
            *state = *state * v;
            Some(Some(*state))
        },
        None => Some(None),
    }
}

fn cum_max_numeric<T>(ca: &ChunkedArray<T>, reverse: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = Bounded::min_value();

    let out: ChunkedArray<T> = match reverse {
        false => ca.iter().scan(init, det_max).collect_trusted(),
        true => ca.iter().rev().scan(init, det_max).collect_reversed(),
    };
    out.with_name(ca.name().clone())
}

fn cum_min_numeric<T>(ca: &ChunkedArray<T>, reverse: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = Bounded::max_value();
    let out: ChunkedArray<T> = match reverse {
        false => ca.iter().scan(init, det_min).collect_trusted(),
        true => ca.iter().rev().scan(init, det_min).collect_reversed(),
    };
    out.with_name(ca.name().clone())
}

fn cum_max_bool(ca: &BooleanChunked, reverse: bool) -> BooleanChunked {
    if ca.len() == ca.null_count() {
        return ca.clone();
    }

    let mut out;
    if !reverse {
        // TODO: efficient bitscan.
        let Some(first_true_idx) = ca.iter().position(|x| x == Some(true)) else {
            return ca.clone();
        };
        out = BitmapBuilder::with_capacity(ca.len());
        out.extend_constant(first_true_idx, false);
        out.extend_constant(ca.len() - first_true_idx, true);
    } else {
        // TODO: efficient bitscan.
        let Some(last_true_idx) = ca.iter().rposition(|x| x == Some(true)) else {
            return ca.clone();
        };
        out = BitmapBuilder::with_capacity(ca.len());
        out.extend_constant(last_true_idx + 1, true);
        out.extend_constant(ca.len() - 1 - last_true_idx, false);
    }

    let arr: BooleanArray = out.freeze().into();
    BooleanChunked::with_chunk_like(ca, arr.with_validity(ca.rechunk_validity()))
}

fn cum_min_bool(ca: &BooleanChunked, reverse: bool) -> BooleanChunked {
    if ca.len() == ca.null_count() {
        return ca.clone();
    }

    let mut out;
    if !reverse {
        // TODO: efficient bitscan.
        let Some(first_false_idx) = ca.iter().position(|x| x == Some(false)) else {
            return ca.clone();
        };
        out = BitmapBuilder::with_capacity(ca.len());
        out.extend_constant(first_false_idx, true);
        out.extend_constant(ca.len() - first_false_idx, false);
    } else {
        // TODO: efficient bitscan.
        let Some(last_false_idx) = ca.iter().rposition(|x| x == Some(false)) else {
            return ca.clone();
        };
        out = BitmapBuilder::with_capacity(ca.len());
        out.extend_constant(last_false_idx + 1, false);
        out.extend_constant(ca.len() - 1 - last_false_idx, true);
    }

    let arr: BooleanArray = out.freeze().into();
    BooleanChunked::with_chunk_like(ca, arr.with_validity(ca.rechunk_validity()))
}

fn cum_sum_numeric<T>(ca: &ChunkedArray<T>, reverse: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = T::Native::zero();
    let out: ChunkedArray<T> = match reverse {
        false => ca.iter().scan(init, det_sum).collect_trusted(),
        true => ca.iter().rev().scan(init, det_sum).collect_reversed(),
    };
    out.with_name(ca.name().clone())
}

fn cum_prod_numeric<T>(ca: &ChunkedArray<T>, reverse: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = T::Native::one();
    let out: ChunkedArray<T> = match reverse {
        false => ca.iter().scan(init, det_prod).collect_trusted(),
        true => ca.iter().rev().scan(init, det_prod).collect_reversed(),
    };
    out.with_name(ca.name().clone())
}

/// Get an array with the cumulative product computed at every element.
///
/// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16, Int32, UInt32}` the `Series` is
/// first cast to `Int64` to prevent overflow issues.
pub fn cum_prod(s: &Series, reverse: bool) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        Boolean | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 => {
            let s = s.cast(&Int64)?;
            cum_prod_numeric(s.i64()?, reverse).into_series()
        },
        Int64 => cum_prod_numeric(s.i64()?, reverse).into_series(),
        UInt64 => cum_prod_numeric(s.u64()?, reverse).into_series(),
        #[cfg(feature = "dtype-i128")]
        Int128 => cum_prod_numeric(s.i128()?, reverse).into_series(),
        Float32 => cum_prod_numeric(s.f32()?, reverse).into_series(),
        Float64 => cum_prod_numeric(s.f64()?, reverse).into_series(),
        dt => polars_bail!(opq = cum_prod, dt),
    };
    Ok(out)
}

/// Get an array with the cumulative sum computed at every element
///
/// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
/// first cast to `Int64` to prevent overflow issues.
pub fn cum_sum(s: &Series, reverse: bool) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        Boolean => {
            let s = s.cast(&UInt32)?;
            cum_sum_numeric(s.u32()?, reverse).into_series()
        },
        Int8 | UInt8 | Int16 | UInt16 => {
            let s = s.cast(&Int64)?;
            cum_sum_numeric(s.i64()?, reverse).into_series()
        },
        Int32 => cum_sum_numeric(s.i32()?, reverse).into_series(),
        UInt32 => cum_sum_numeric(s.u32()?, reverse).into_series(),
        Int64 => cum_sum_numeric(s.i64()?, reverse).into_series(),
        UInt64 => cum_sum_numeric(s.u64()?, reverse).into_series(),
        #[cfg(feature = "dtype-i128")]
        Int128 => cum_sum_numeric(s.i128()?, reverse).into_series(),
        Float32 => cum_sum_numeric(s.f32()?, reverse).into_series(),
        Float64 => cum_sum_numeric(s.f64()?, reverse).into_series(),
        #[cfg(feature = "dtype-decimal")]
        Decimal(precision, scale) => {
            let ca = s.decimal().unwrap().as_ref();
            cum_sum_numeric(ca, reverse)
                .into_decimal_unchecked(*precision, scale.unwrap())
                .into_series()
        },
        #[cfg(feature = "dtype-duration")]
        Duration(tu) => {
            let s = s.to_physical_repr();
            let ca = s.i64()?;
            cum_sum_numeric(ca, reverse).cast(&Duration(*tu))?
        },
        dt => polars_bail!(opq = cum_sum, dt),
    };
    Ok(out)
}

/// Get an array with the cumulative min computed at every element.
pub fn cum_min(s: &Series, reverse: bool) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Boolean => Ok(cum_min_bool(s.bool()?, reverse).into_series()),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, scale) => {
            let ca = s.decimal().unwrap().as_ref();
            let out = cum_min_numeric(ca, reverse)
                .into_decimal_unchecked(*precision, scale.unwrap())
                .into_series();
            Ok(out)
        },
        dt if dt.to_physical().is_primitive_numeric() => {
            let s = s.to_physical_repr();
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let out = cum_min_numeric(ca, reverse).into_series();
                if dt.is_logical() {
                    out.cast(dt)
                } else {
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = cum_min, dt),
    }
}

/// Get an array with the cumulative max computed at every element.
pub fn cum_max(s: &Series, reverse: bool) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Boolean => Ok(cum_max_bool(s.bool()?, reverse).into_series()),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, scale) => {
            let ca = s.decimal().unwrap().as_ref();
            let out = cum_max_numeric(ca, reverse)
                .into_decimal_unchecked(*precision, scale.unwrap())
                .into_series();
            Ok(out)
        },
        dt if dt.to_physical().is_primitive_numeric() => {
            let s = s.to_physical_repr();
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let out = cum_max_numeric(ca, reverse).into_series();
                if dt.is_logical() {
                    out.cast(dt)
                } else {
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = cum_max, dt),
    }
}

pub fn cum_count(s: &Series, reverse: bool) -> PolarsResult<Series> {
    let mut out = if s.null_count() == 0 {
        // Fast paths for no nulls
        cum_count_no_nulls(s.name().clone(), s.len(), reverse)
    } else {
        let ca = s.is_not_null();
        let out: IdxCa = if reverse {
            let mut count = (s.len() - s.null_count()) as IdxSize;
            let mut prev = false;
            unary_elementwise_values(&ca, |v: bool| {
                if prev {
                    count -= 1;
                }
                prev = v;
                count
            })
        } else {
            let mut count = 0 as IdxSize;
            unary_elementwise_values(&ca, |v: bool| {
                if v {
                    count += 1;
                }
                count
            })
        };

        out.into()
    };

    out.set_sorted_flag([IsSorted::Ascending, IsSorted::Descending][reverse as usize]);

    Ok(out)
}

fn cum_count_no_nulls(name: PlSmallStr, len: usize, reverse: bool) -> Series {
    let start = 1 as IdxSize;
    let end = len as IdxSize + 1;
    let ca: NoNull<IdxCa> = if reverse {
        (start..end).rev().collect()
    } else {
        (start..end).collect()
    };
    let mut ca = ca.into_inner();
    ca.rename(name);
    ca.into_series()
}
