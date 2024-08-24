use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise, unary_elementwise};
use polars_core::prelude::*;
use polars_core::with_match_physical_numeric_polars_type;

/// Set values outside the given boundaries to the boundary value.
pub fn clip(s: &Series, min: &Series, max: &Series) -> PolarsResult<Series> {
    polars_ensure!(
        s.dtype().to_physical().is_numeric(),
        InvalidOperation: "`clip` only supports physical numeric types"
    );

    let original_type = s.dtype();
    let (min, max) = (min.strict_cast(s.dtype())?, max.strict_cast(s.dtype())?);

    let (s, min, max) = (
        s.to_physical_repr(),
        min.to_physical_repr(),
        max.to_physical_repr(),
    );

    match s.dtype() {
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let min: &ChunkedArray<$T> = min.as_ref().as_ref().as_ref();
                let max: &ChunkedArray<$T> = max.as_ref().as_ref().as_ref();
                let out = clip_helper_both_bounds(ca, min, max).into_series();
                if original_type.is_logical() {
                    out.cast(original_type)
                } else {
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = clippy, dt),
    }
}

/// Set values above the given maximum to the maximum value.
pub fn clip_max(s: &Series, max: &Series) -> PolarsResult<Series> {
    polars_ensure!(
        s.dtype().to_physical().is_numeric(),
        InvalidOperation: "`clip` only supports physical numeric types"
    );

    let original_type = s.dtype();
    let max = max.strict_cast(s.dtype())?;

    let (s, max) = (s.to_physical_repr(), max.to_physical_repr());

    match s.dtype() {
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let max: &ChunkedArray<$T> = max.as_ref().as_ref().as_ref();
                let out = clip_helper_single_bound(ca, max, num_traits::clamp_max).into_series();
                if original_type.is_logical() {
                    out.cast(original_type)
                } else {
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = clippy_max, dt),
    }
}

/// Set values below the given minimum to the minimum value.
pub fn clip_min(s: &Series, min: &Series) -> PolarsResult<Series> {
    polars_ensure!(
        s.dtype().to_physical().is_numeric(),
        InvalidOperation: "`clip` only supports physical numeric types"
    );

    let original_type = s.dtype();
    let min = min.strict_cast(s.dtype())?;

    let (s, min) = (s.to_physical_repr(), min.to_physical_repr());

    match s.dtype() {
        dt if dt.is_numeric() => {
            with_match_physical_numeric_polars_type!(s.dtype(), |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                let min: &ChunkedArray<$T> = min.as_ref().as_ref().as_ref();
                let out = clip_helper_single_bound(ca, min, num_traits::clamp_min).into_series();
                if original_type.is_logical() {
                    out.cast(original_type)
                } else {
                    Ok(out)
                }
            })
        },
        dt => polars_bail!(opq = clippy_min, dt),
    }
}

fn clip_helper_both_bounds<T>(
    ca: &ChunkedArray<T>,
    min: &ChunkedArray<T>,
    max: &ChunkedArray<T>,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    match (min.len(), max.len()) {
        (1, 1) => match (min.get(0), max.get(0)) {
            (Some(min), Some(max)) => clip_unary(ca, |v| num_traits::clamp(v, min, max)),
            (Some(min), None) => clip_unary(ca, |v| num_traits::clamp_min(v, min)),
            (None, Some(max)) => clip_unary(ca, |v| num_traits::clamp_max(v, max)),
            (None, None) => ca.clone(),
        },
        (1, _) => match min.get(0) {
            Some(min) => clip_binary(ca, max, |v, b| num_traits::clamp(v, min, b)),
            None => clip_binary(ca, max, num_traits::clamp_max),
        },
        (_, 1) => match max.get(0) {
            Some(max) => clip_binary(ca, min, |v, b| num_traits::clamp(v, b, max)),
            None => clip_binary(ca, min, num_traits::clamp_min),
        },
        _ => clip_ternary(ca, min, max),
    }
}

fn clip_helper_single_bound<T, F>(
    ca: &ChunkedArray<T>,
    bound: &ChunkedArray<T>,
    op: F,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
    F: Fn(T::Native, T::Native) -> T::Native,
{
    match bound.len() {
        1 => match bound.get(0) {
            Some(bound) => clip_unary(ca, |v| op(v, bound)),
            None => ca.clone(),
        },
        _ => clip_binary(ca, bound, op),
    }
}

fn clip_unary<T, F>(ca: &ChunkedArray<T>, op: F) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    F: Fn(T::Native) -> T::Native + Copy,
{
    unary_elementwise(ca, |v| v.map(op))
}

fn clip_binary<T, F>(ca: &ChunkedArray<T>, bound: &ChunkedArray<T>, op: F) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
    F: Fn(T::Native, T::Native) -> T::Native,
{
    binary_elementwise(ca, bound, |opt_s, opt_bound| match (opt_s, opt_bound) {
        (Some(s), Some(bound)) => Some(op(s, bound)),
        (Some(s), None) => Some(s),
        (None, _) => None,
    })
}

fn clip_ternary<T>(
    ca: &ChunkedArray<T>,
    min: &ChunkedArray<T>,
    max: &ChunkedArray<T>,
) -> ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: PartialOrd,
{
    ternary_elementwise(ca, min, max, |opt_v, opt_min, opt_max| {
        match (opt_v, opt_min, opt_max) {
            (Some(v), Some(min), Some(max)) => Some(num_traits::clamp(v, min, max)),
            (Some(v), Some(min), None) => Some(num_traits::clamp_min(v, min)),
            (Some(v), None, Some(max)) => Some(num_traits::clamp_max(v, max)),
            (Some(v), None, None) => Some(v),
            (None, _, _) => None,
        }
    })
}
