use polars_core::prelude::*;

use crate::prelude::*;

/// Resolve a single `dt.replace` component (e.g. year, month, ...):
/// 1. No value was supplied (None)       --> Use existing component from the original array.
/// 2. Value was supplied and is a Scalar --> Create full Series of value.
/// 3. Value was supplied and is a Series --> Update all elements with the non-null values.
fn resolve_component<T>(
    component: &ChunkedArray<T>,
    existing: impl FnOnce() -> ChunkedArray<T>,
    n: usize,
) -> PolarsResult<ChunkedArray<T>>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkZip<T>,
{
    if component.len() == 1 {
        if let Some(value) = component.get(0) {
            Ok(ChunkedArray::<T>::full(PlSmallStr::EMPTY, value, n))
        } else {
            Ok(existing())
        }
    } else {
        component.zip_with(&component.is_not_null(), &existing())
    }
}

/// Replace specific time component of a `DatetimeChunked` with a specified value.
#[cfg(feature = "dtype-datetime")]
#[allow(clippy::too_many_arguments)]
pub fn replace_datetime(
    ca: &DatetimeChunked,
    year: &Int32Chunked,
    month: &Int8Chunked,
    day: &Int8Chunked,
    hour: &Int8Chunked,
    minute: &Int8Chunked,
    second: &Int8Chunked,
    nanosecond: &Int32Chunked,
    ambiguous: &StringChunked,
) -> PolarsResult<DatetimeChunked> {
    let n = [
        ca.len(),
        year.len(),
        month.len(),
        day.len(),
        hour.len(),
        minute.len(),
        second.len(),
        nanosecond.len(),
        ambiguous.len(),
    ]
    .into_iter()
    .find(|l| *l != 1)
    .unwrap_or(1);

    for (i, (name, length)) in [
        ("self", ca.len()),
        ("year", year.len()),
        ("month", month.len()),
        ("day", day.len()),
        ("hour", hour.len()),
        ("minute", minute.len()),
        ("second", second.len()),
        ("nanosecond", nanosecond.len()),
        ("ambiguous", ambiguous.len()),
    ]
    .into_iter()
    .enumerate()
    {
        polars_ensure!(
            length == n || length == 1,
            length_mismatch = "dt.replace",
            length,
            n,
            argument = name,
            argument_idx = i
        );
    }

    let year = &resolve_component(year, || ca.year(), n)?;
    let month = &resolve_component(month, || ca.month(), n)?;
    let day = &resolve_component(day, || ca.day(), n)?;
    let hour = &resolve_component(hour, || ca.hour(), n)?;
    let minute = &resolve_component(minute, || ca.minute(), n)?;
    let second = &resolve_component(second, || ca.second(), n)?;
    let nanosecond = &resolve_component(nanosecond, || ca.nanosecond(), n)?;

    let mut out = DatetimeChunked::new_from_parts(
        year,
        month,
        day,
        hour,
        minute,
        second,
        nanosecond,
        ambiguous,
        &ca.time_unit(),
        ca.time_zone().clone(),
        ca.name().clone(),
    )?;

    // Ensure nulls are propagated. A component can only end up null when `ca` is null at that
    // position, so `out`'s nulls are always a subset of `ca`'s.
    if ca.has_nulls() {
        out.physical_mut().set_validity(ca.physical().rechunk_validity());
    }

    Ok(out)
}

/// Replace specific time component of a `DateChunked` with a specified value.
#[cfg(feature = "dtype-date")]
pub fn replace_date(
    ca: &DateChunked,
    year: &Int32Chunked,
    month: &Int8Chunked,
    day: &Int8Chunked,
) -> PolarsResult<DateChunked> {
    let n = ca.len();

    let year = &resolve_component(year, || ca.year(), n)?;
    let month = &resolve_component(month, || ca.month(), n)?;
    let day = &resolve_component(day, || ca.day(), n)?;
    let mut out = DateChunked::new_from_parts(year, month, day, ca.name().clone())?;

    // Ensure nulls are propagated. A component can only end up null when `ca` is null at that
    // position, so `out`'s nulls are always a subset of `ca`'s.
    if ca.has_nulls() {
        out.physical_mut().set_validity(ca.physical().rechunk_validity());
    }

    Ok(out)
}
