use polars_core::prelude::*;

use crate::prelude::*;

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

    // For each argument, we must check if:
    // 1. No value was supplied (None)       --> Use existing year from Series
    // 2. Value was supplied and is a Scalar --> Create full Series of value
    // 3. Value was supplied and is Series   --> Update all elements with the non-null values
    let year = if year.len() == 1 {
        if let Some(value) = year.get(0) {
            &Int32Chunked::full(PlSmallStr::EMPTY, value, n)
        } else {
            &ca.year()
        }
    } else {
        &year.zip_with(&year.is_not_null(), &ca.year())?
    };
    let month = if month.len() == 1 {
        if let Some(value) = month.get(0) {
            &Int8Chunked::full(PlSmallStr::EMPTY, value, n)
        } else {
            &ca.month()
        }
    } else {
        &month.zip_with(&month.is_not_null(), &ca.month())?
    };
    let day = if day.len() == 1 {
        if let Some(value) = day.get(0) {
            &Int8Chunked::full(PlSmallStr::EMPTY, value, n)
        } else {
            &ca.day()
        }
    } else {
        &day.zip_with(&day.is_not_null(), &ca.day())?
    };
    let hour = if hour.len() == 1 {
        if let Some(value) = hour.get(0) {
            &Int8Chunked::full(PlSmallStr::EMPTY, value, n)
        } else {
            &ca.hour()
        }
    } else {
        &hour.zip_with(&hour.is_not_null(), &ca.hour())?
    };
    let minute = if minute.len() == 1 {
        if let Some(value) = minute.get(0) {
            &Int8Chunked::full(PlSmallStr::EMPTY, value, n)
        } else {
            &ca.minute()
        }
    } else {
        &minute.zip_with(&minute.is_not_null(), &ca.minute())?
    };
    let second = if second.len() == 1 {
        if let Some(value) = second.get(0) {
            &Int8Chunked::full(PlSmallStr::EMPTY, value, n)
        } else {
            &ca.second()
        }
    } else {
        &second.zip_with(&second.is_not_null(), &ca.second())?
    };
    let nanosecond = if nanosecond.len() == 1 {
        if let Some(value) = nanosecond.get(0) {
            &Int32Chunked::full(PlSmallStr::EMPTY, value, n)
        } else {
            &ca.nanosecond()
        }
    } else {
        &nanosecond.zip_with(&nanosecond.is_not_null(), &ca.nanosecond())?
    };

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

    // Ensure nulls are propagated.
    if ca.has_nulls() {
        out.merge_validities(ca.chunks());
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

    let year = if year.len() == 1 {
        if let Some(value) = year.get(0) {
            &Int32Chunked::full("".into(), value, n)
        } else {
            &ca.year()
        }
    } else {
        &year.zip_with(&year.is_not_null(), &ca.year())?
    };
    let month = if month.len() == 1 {
        if let Some(value) = month.get(0) {
            &Int8Chunked::full("".into(), value, n)
        } else {
            &ca.month()
        }
    } else {
        &month.zip_with(&month.is_not_null(), &ca.month())?
    };
    let day = if day.len() == 1 {
        if let Some(value) = day.get(0) {
            &Int8Chunked::full("".into(), value, n)
        } else {
            &ca.day()
        }
    } else {
        &day.zip_with(&day.is_not_null(), &ca.day())?
    };
    let mut out = DateChunked::new_from_parts(year, month, day, ca.name().clone())?;

    // Ensure nulls are propagated.
    if ca.has_nulls() {
        out.merge_validities(ca.chunks());
    }

    Ok(out)
}
