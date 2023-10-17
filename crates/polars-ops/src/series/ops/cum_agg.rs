use std::iter::FromIterator;
use std::ops::{Add, AddAssign, Mul};

use num_traits::Bounded;
use polars_core::prelude::*;
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

fn det_sum<T>(state: &mut Option<T>, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + AddAssign + Add<Output = T>,
{
    match (*state, v) {
        (Some(state_inner), Some(v)) => {
            *state = Some(state_inner + v);
            Some(*state)
        },
        (None, Some(v)) => {
            *state = Some(v);
            Some(*state)
        },
        (_, None) => Some(None),
    }
}

fn det_prod<T>(state: &mut Option<T>, v: Option<T>) -> Option<Option<T>>
where
    T: Copy + PartialOrd + Mul<Output = T>,
{
    match (*state, v) {
        (Some(state_inner), Some(v)) => {
            *state = Some(state_inner * v);
            Some(*state)
        },
        (None, Some(v)) => {
            *state = Some(v);
            Some(*state)
        },
        (_, None) => Some(None),
    }
}

fn cummax_numeric<T>(ca: &ChunkedArray<T>, reverse: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = Bounded::min_value();

    let out: ChunkedArray<T> = match reverse {
        false => ca.into_iter().scan(init, det_max).collect_trusted(),
        true => ca.into_iter().rev().scan(init, det_max).collect_reversed(),
    };
    out.with_name(ca.name())
}

fn cummin_numeric<T>(ca: &ChunkedArray<T>, reverse: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = Bounded::max_value();
    let out: ChunkedArray<T> = match reverse {
        false => ca.into_iter().scan(init, det_min).collect_trusted(),
        true => ca.into_iter().rev().scan(init, det_min).collect_reversed(),
    };
    out.with_name(ca.name())
}

fn cumsum_numeric<T>(ca: &ChunkedArray<T>, reverse: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = None;
    let out: ChunkedArray<T> = match reverse {
        false => ca.into_iter().scan(init, det_sum).collect_trusted(),
        true => ca.into_iter().rev().scan(init, det_sum).collect_reversed(),
    };
    out.with_name(ca.name())
}

fn cumprod_numeric<T>(ca: &ChunkedArray<T>, reverse: bool) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: FromIterator<Option<T::Native>>,
{
    let init = None;
    let out: ChunkedArray<T> = match reverse {
        false => ca.into_iter().scan(init, det_prod).collect_trusted(),
        true => ca.into_iter().rev().scan(init, det_prod).collect_reversed(),
    };
    out.with_name(ca.name())
}

/// Get an array with the cumulative product computed at every element.
///
/// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16, Int32, UInt32}` the `Series` is
/// first cast to `Int64` to prevent overflow issues.
pub fn cumprod(s: &Series, reverse: bool) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        Boolean | Int8 | UInt8 | Int16 | UInt16 | Int32 | UInt32 => {
            let s = s.cast(&Int64)?;
            cumprod_numeric(s.i64()?, reverse).into_series()
        },
        Int64 => cumprod_numeric(s.i64()?, reverse).into_series(),
        UInt64 => cumprod_numeric(s.u64()?, reverse).into_series(),
        Float32 => cumprod_numeric(s.f32()?, reverse).into_series(),
        Float64 => cumprod_numeric(s.f64()?, reverse).into_series(),
        dt => polars_bail!(opq = cumprod, dt),
    };
    Ok(out)
}

/// Get an array with the cumulative sum computed at every element
///
/// If the [`DataType`] is one of `{Int8, UInt8, Int16, UInt16}` the `Series` is
/// first cast to `Int64` to prevent overflow issues.
pub fn cumsum(s: &Series, reverse: bool) -> PolarsResult<Series> {
    use DataType::*;
    let out = match s.dtype() {
        Boolean => {
            let s = s.cast(&UInt32)?;
            cumsum_numeric(s.u32()?, reverse).into_series()
        },
        Int8 | UInt8 | Int16 | UInt16 => {
            let s = s.cast(&Int64)?;
            cumsum_numeric(s.i64()?, reverse).into_series()
        },
        Int32 => cumsum_numeric(s.i32()?, reverse).into_series(),
        UInt32 => cumsum_numeric(s.u32()?, reverse).into_series(),
        Int64 => cumsum_numeric(s.i64()?, reverse).into_series(),
        UInt64 => cumsum_numeric(s.u64()?, reverse).into_series(),
        Float32 => cumsum_numeric(s.f32()?, reverse).into_series(),
        Float64 => cumsum_numeric(s.f64()?, reverse).into_series(),
        #[cfg(feature = "dtype-duration")]
        Duration(tu) => {
            let s = s.to_physical_repr();
            let ca = s.i64()?;
            cumsum_numeric(ca, reverse).cast(&Duration(*tu))?
        },
        dt => polars_bail!(opq = cumsum, dt),
    };
    Ok(out)
}

/// Get an array with the cumulative min computed at every element.
pub fn cummin(s: &Series, reverse: bool) -> PolarsResult<Series> {
    let original_type = s.dtype();
    let s = s.to_physical_repr();
    match s.dtype() {
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let out = cummin_numeric(ca, reverse).into_series();
                if original_type.is_logical(){
                    out.cast(original_type)
                }else{
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = cummin, dt),
    }
}

/// Get an array with the cumulative max computed at every element.
pub fn cummax(s: &Series, reverse: bool) -> PolarsResult<Series> {
    let original_type = s.dtype();
    let s = s.to_physical_repr();
    match s.dtype() {
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let out = cummax_numeric(ca, reverse).into_series();
                if original_type.is_logical(){
                    out.cast(original_type)
                }else{
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = cummin, dt),
    }
}

pub fn cumcount(s: &Series, reverse: bool) -> PolarsResult<Series> {
    if reverse {
        let ca: NoNull<UInt32Chunked> = (0u32..s.len() as u32).rev().collect();
        let mut ca = ca.into_inner();
        ca.rename(s.name());
        Ok(ca.into_series())
    } else {
        let ca: NoNull<UInt32Chunked> = (0u32..s.len() as u32).collect();
        let mut ca = ca.into_inner();
        ca.rename(s.name());
        Ok(ca.into_series())
    }
}
