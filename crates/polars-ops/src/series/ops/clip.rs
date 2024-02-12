use num_traits::{clamp, clamp_max, clamp_min};
use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise};
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
                let out = clip_helper(ca, min, max).into_series();
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
                let out = clip_min_max_helper(ca, max, clamp_max).into_series();
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
                let out = clip_min_max_helper(ca, min, clamp_min).into_series();
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

fn clip_helper<T>(
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
            (Some(min), Some(max)) => {
                ca.apply_generic(|s| s.map(|s| num_traits::clamp(s, min, max)))
            },
            _ => ChunkedArray::<T>::full_null(ca.name(), ca.len()),
        },
        (1, _) => match min.get(0) {
            Some(min) => binary_elementwise(ca, max, |opt_s, opt_max| match (opt_s, opt_max) {
                (Some(s), Some(max)) => Some(clamp(s, min, max)),
                _ => None,
            }),
            _ => ChunkedArray::<T>::full_null(ca.name(), ca.len()),
        },
        (_, 1) => match max.get(0) {
            Some(max) => binary_elementwise(ca, min, |opt_s, opt_min| match (opt_s, opt_min) {
                (Some(s), Some(min)) => Some(clamp(s, min, max)),
                _ => None,
            }),
            _ => ChunkedArray::<T>::full_null(ca.name(), ca.len()),
        },
        _ => ternary_elementwise(ca, min, max, |opt_s, opt_min, opt_max| {
            match (opt_s, opt_min, opt_max) {
                (Some(s), Some(min), Some(max)) => Some(clamp(s, min, max)),
                _ => None,
            }
        }),
    }
}

fn clip_min_max_helper<T, F>(
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
            Some(bound) => ca.apply_generic(|s| s.map(|s| op(s, bound))),
            _ => ChunkedArray::<T>::full_null(ca.name(), ca.len()),
        },
        _ => binary_elementwise(ca, bound, |opt_s, opt_bound| match (opt_s, opt_bound) {
            (Some(s), Some(bound)) => Some(op(s, bound)),
            _ => None,
        }),
    }
}
