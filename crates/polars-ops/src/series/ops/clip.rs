use num_traits::{clamp, clamp_max, clamp_min};
use polars_core::prelude::arity::{binary_elementwise, ternary_elementwise};
use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_core::with_match_physical_numeric_polars_type;

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

/// Clamp underlying values to the `min` and `max` values.
pub fn clip(s: &Series, min: &Series, max: &Series) -> PolarsResult<Series> {
    let bounds_dtype = try_get_supertype(min.dtype(), max.dtype())?;
    let return_dtype = try_get_supertype(s.dtype(), &bounds_dtype)?;
    let physical_dtype = return_dtype.to_physical();

    polars_ensure!(
        physical_dtype.is_numeric(),
        InvalidOperation: "`clip` only supports physical numeric types"
    );

    let (s, min, max) = (
        s.cast(&physical_dtype)?,
        min.cast(&physical_dtype)?,
        max.cast(&physical_dtype)?,
    );

    let out = with_match_physical_numeric_polars_type!(physical_dtype, |$T| {
        let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        let min: &ChunkedArray<$T> = min.as_ref().as_ref().as_ref();
        let max: &ChunkedArray<$T> = max.as_ref().as_ref().as_ref();
        clip_helper(ca, min, max).into_series()
    });

    if return_dtype.is_logical() {
        out.cast(&return_dtype)
    } else {
        Ok(out)
    }
}

/// Clamp underlying values to the `max` value.
pub fn clip_max(s: &Series, max: &Series) -> PolarsResult<Series> {
    let return_dtype = try_get_supertype(s.dtype(), max.dtype())?;
    let physical_dtype = return_dtype.to_physical();

    polars_ensure!(
        physical_dtype.is_numeric(),
        InvalidOperation: "`clip` only supports physical numeric types"
    );

    let (s, max) = (s.cast(&physical_dtype)?, max.cast(&physical_dtype)?);

    let out = with_match_physical_numeric_polars_type!(physical_dtype, |$T| {
        let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        let max: &ChunkedArray<$T> = max.as_ref().as_ref().as_ref();
        clip_min_max_helper(ca, max, clamp_max).into_series()
    });

    if return_dtype.is_logical() {
        out.cast(&return_dtype)
    } else {
        Ok(out)
    }
}

/// Clamp underlying values to the `min` value.
pub fn clip_min(s: &Series, min: &Series) -> PolarsResult<Series> {
    let return_dtype = try_get_supertype(s.dtype(), min.dtype())?;
    let physical_dtype = return_dtype.to_physical();

    polars_ensure!(
        physical_dtype.is_numeric(),
        InvalidOperation: "`clip` only supports physical numeric types"
    );

    let (s, min) = (s.cast(&physical_dtype)?, min.cast(&physical_dtype)?);

    let out = with_match_physical_numeric_polars_type!(physical_dtype, |$T| {
        let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
        let min: &ChunkedArray<$T> = min.as_ref().as_ref().as_ref();
        clip_min_max_helper(ca, min, clamp_min).into_series()
    });

    if return_dtype.is_logical() {
        out.cast(&return_dtype)
    } else {
        Ok(out)
    }
}
