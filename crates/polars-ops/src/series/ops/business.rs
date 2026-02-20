use arrow::Either;
use arrow::array::{
    Array, ListArray, MutableListArray, MutablePrimitiveArray, PrimitiveArray, TryPush,
};
#[cfg(feature = "dtype-date")]
use chrono::DateTime;
use polars_core::prelude::arity::{binary_elementwise, try_binary_elementwise, unary_elementwise};
use polars_core::prelude::*;
#[cfg(feature = "dtype-date")]
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
use polars_utils::binary_search::{find_first_ge_index, find_first_gt_index};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::prelude::ListNameSpaceImpl;
#[cfg(feature = "timezones")]
use crate::prelude::replace_time_zone;

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum Roll {
    Forward,
    Backward,
    Raise,
}

/// Count the number of business days between `start` and `end`, excluding `end`.
///
/// # Arguments
/// - `start`: Series holding start dates.
/// - `end`: Series holding end dates.
/// - `week_mask`: A boolean array of length 7, where `true` indicates that the day is a business day.
/// - `holidays`: timestamps that are holidays. Must be provided as i32, i.e. the number of
///   days since the UNIX epoch.
pub fn business_day_count(
    start: &Series,
    end: &Series,
    week_mask: [bool; 7],
    holidays: &Series,
) -> PolarsResult<Series> {
    if !week_mask.iter().any(|&x| x) {
        polars_bail!(ComputeError:"`week_mask` must have at least one business day");
    }
    polars_ensure!(
        holidays.len() == 1 || start.len() == 1 || end.len() == 1 || holidays.len() == start.len(),
        ShapeMismatch: "number of holiday lists must be either 1 or the number of dates"
    );

    let holidays = prep_holidays(holidays, week_mask)?;
    let mut holidays_lists = holidays_lists_iter(&holidays);
    let start_dates = start.date()?;
    let end_dates = end.date()?;
    let n_business_days_in_week_mask = week_mask.iter().filter(|&x| *x).count() as i32;

    let out = match (start_dates.len(), end_dates.len()) {
        (_, 1) => {
            if let Some(end_date) = end_dates.physical().get(0)
                && let Some(Some(holidays_list)) = holidays_lists.next()
            {
                start_dates.physical().apply_values(|start_date| {
                    business_day_count_impl(
                        start_date,
                        end_date,
                        &week_mask,
                        n_business_days_in_week_mask,
                        holidays_list,
                    )
                })
            } else {
                Int32Chunked::full_null(start_dates.name().clone(), start_dates.len())
            }
        },
        (1, _) => {
            if let Some(start_date) = start_dates.physical().get(0)
                && let Some(Some(holidays_list)) = holidays_lists.next()
            {
                end_dates.physical().apply_values(|end_date| {
                    business_day_count_impl(
                        start_date,
                        end_date,
                        &week_mask,
                        n_business_days_in_week_mask,
                        holidays_list,
                    )
                })
            } else {
                Int32Chunked::full_null(start_dates.name().clone(), end_dates.len())
            }
        },
        _ => {
            polars_ensure!(
                start_dates.len() == end_dates.len(),
                length_mismatch = "business_day_count",
                start_dates.len(),
                end_dates.len()
            );
            binary_elementwise(
                start_dates.physical(),
                end_dates.physical(),
                |start_date, end_date| match (start_date, end_date, holidays_lists.next()) {
                    (Some(start_date), Some(end_date), Some(Some(holidays_list))) => {
                        Some(business_day_count_impl(
                            start_date,
                            end_date,
                            &week_mask,
                            n_business_days_in_week_mask,
                            holidays_list,
                        ))
                    },
                    _ => None,
                },
            )
        },
    };
    let out = out.with_name(start_dates.name().clone());
    Ok(out.into_series())
}

/// Ported from:
/// https://github.com/numpy/numpy/blob/e59c074842e3f73483afa5ddef031e856b9fd313/numpy/_core/src/multiarray/datetime_busday.c#L355-L433
fn business_day_count_impl(
    mut start_date: i32,
    mut end_date: i32,
    week_mask: &[bool; 7],
    n_business_days_in_week_mask: i32,
    holidays: &[i32], // Caller's responsibility to ensure it's sorted.
) -> i32 {
    let swapped = start_date > end_date;
    if swapped {
        (start_date, end_date) = (end_date, start_date);
        start_date += 1;
        end_date += 1;
    }

    let holidays_begin = find_first_ge_index(holidays, start_date);
    let holidays_end = find_first_ge_index(&holidays[holidays_begin..], end_date) + holidays_begin;
    let mut start_day_of_week = get_day_of_week(start_date);
    let diff = end_date - start_date;
    let whole_weeks = diff / 7;
    let mut count = -((holidays_end - holidays_begin) as i32);
    count += whole_weeks * n_business_days_in_week_mask;
    start_date += whole_weeks * 7;
    while start_date < end_date {
        // SAFETY: week_mask is length 7, start_day_of_week is between 0 and 6
        if unsafe { *week_mask.get_unchecked(start_day_of_week) } {
            count += 1;
        }
        start_date += 1;
        start_day_of_week = increment_day_of_week(start_day_of_week);
    }
    if swapped { -count } else { count }
}

/// Add a given number of business days.
///
/// # Arguments
/// - `start`: Series holding start dates.
/// - `n`: Number of business days to add.
/// - `week_mask`: A boolean array of length 7, where `true` indicates that the day is a business day.
/// - `holidays`: timestamps that are holidays. Must be provided as i32, i.e. the number of
///   days since the UNIX epoch.
/// - `roll`: what to do when the start date doesn't land on a business day:
///   - `Roll::Forward`: roll forward to the next business day.
///   - `Roll::Backward`: roll backward to the previous business day.
///   - `Roll::Raise`: raise an error.
pub fn add_business_days(
    start: &Series,
    n: &Series,
    week_mask: [bool; 7],
    holidays: &Series,
    roll: Roll,
) -> PolarsResult<Series> {
    if !week_mask.iter().any(|&x| x) {
        polars_bail!(ComputeError:"`week_mask` must have at least one business day");
    }
    polars_ensure!(
        holidays.len() == 1 || holidays.len() == start.len(),
        ShapeMismatch: "number of holiday lists must be either 1 or the number of dates"
    );

    match start.dtype() {
        DataType::Date => {},
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(time_unit, None) => {
            let result_date =
                add_business_days(&start.cast(&DataType::Date)?, n, week_mask, holidays, roll)?;
            let start_time = start
                .cast(&DataType::Time)?
                .cast(&DataType::Duration(*time_unit))?;
            return std::ops::Add::add(
                result_date.cast(&DataType::Datetime(*time_unit, None))?,
                start_time,
            );
        },
        #[cfg(feature = "timezones")]
        DataType::Datetime(time_unit, Some(time_zone)) => {
            let start_naive = replace_time_zone(
                start.datetime().unwrap(),
                None,
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?;
            let result_date = add_business_days(
                &start_naive.cast(&DataType::Date)?,
                n,
                week_mask,
                holidays,
                roll,
            )?;
            let start_time = start_naive
                .cast(&DataType::Time)?
                .cast(&DataType::Duration(*time_unit))?;
            let result_naive = std::ops::Add::add(
                result_date.cast(&DataType::Datetime(*time_unit, None))?,
                start_time,
            )?;
            let result_tz_aware = replace_time_zone(
                result_naive.datetime().unwrap(),
                Some(time_zone),
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?;
            return Ok(result_tz_aware.into_series());
        },
        _ => polars_bail!(InvalidOperation: "expected date or datetime, got {}", start.dtype()),
    }

    let holidays = prep_holidays(holidays, week_mask)?;
    let mut holidays_lists = holidays_lists_iter(&holidays);

    let start_dates = start.date()?;
    let n = match &n.dtype() {
        DataType::Int64 | DataType::UInt64 | DataType::UInt32 => n.cast(&DataType::Int32)?,
        DataType::Int32 => n.clone(),
        _ => {
            polars_bail!(InvalidOperation: "expected Int64, Int32, UInt64, or UInt32, got {}", n.dtype())
        },
    };
    let n = n.i32()?;
    let n_business_days_in_week_mask = week_mask.iter().filter(|&x| *x).count() as i32;

    let out: Int32Chunked = match (start_dates.len(), n.len()) {
        (_, 1) => {
            if let Some(n) = n.get(0)
                && let Some(Some(holidays_list)) = holidays_lists.next()
            {
                start_dates
                    .physical()
                    .try_apply_nonnull_values_generic(|start_date| {
                        let (start_date, day_of_week) =
                            roll_start_date(start_date, roll, &week_mask, holidays_list)?;
                        Ok::<i32, PolarsError>(add_business_days_impl(
                            start_date,
                            day_of_week,
                            n,
                            &week_mask,
                            n_business_days_in_week_mask,
                            holidays_list,
                        ))
                    })?
            } else {
                Int32Chunked::full_null(start_dates.name().clone(), start_dates.len())
            }
        },
        (1, _) => {
            if let Some(start_date) = start_dates.physical().get(0)
                && let Some(Some(holidays_list)) = holidays_lists.next()
            {
                let (start_date, day_of_week) =
                    roll_start_date(start_date, roll, &week_mask, holidays_list)?;
                n.apply_values(|n| {
                    add_business_days_impl(
                        start_date,
                        day_of_week,
                        n,
                        &week_mask,
                        n_business_days_in_week_mask,
                        holidays_list,
                    )
                })
            } else {
                Int32Chunked::full_null(start_dates.name().clone(), n.len())
            }
        },
        _ => {
            polars_ensure!(
                start_dates.len() == n.len(),
                length_mismatch = "dt.add_business_days",
                start_dates.len(),
                n.len()
            );
            try_binary_elementwise(start_dates.physical(), n, |opt_start_date, opt_n| {
                match (opt_start_date, opt_n, holidays_lists.next()) {
                    (Some(start_date), Some(n), Some(Some(holidays_list))) => {
                        let (start_date, day_of_week) =
                            roll_start_date(start_date, roll, &week_mask, holidays_list)?;
                        Ok::<Option<i32>, PolarsError>(Some(add_business_days_impl(
                            start_date,
                            day_of_week,
                            n,
                            &week_mask,
                            n_business_days_in_week_mask,
                            holidays_list,
                        )))
                    },
                    _ => Ok(None),
                }
            })?
        },
    };
    Ok(out.into_date().into_series())
}

/// Ported from:
/// https://github.com/numpy/numpy/blob/e59c074842e3f73483afa5ddef031e856b9fd313/numpy/_core/src/multiarray/datetime_busday.c#L265-L353
fn add_business_days_impl(
    mut date: i32,
    mut day_of_week: usize,
    mut n: i32,
    week_mask: &[bool; 7],
    n_business_days_in_week_mask: i32,
    holidays: &[i32], // Caller's responsibility to ensure it's sorted.
) -> i32 {
    if n > 0 {
        let holidays_begin = find_first_ge_index(holidays, date);
        date += (n / n_business_days_in_week_mask) * 7;
        n %= n_business_days_in_week_mask;
        let holidays_temp = find_first_gt_index(&holidays[holidays_begin..], date) + holidays_begin;
        n += (holidays_temp - holidays_begin) as i32;
        let holidays_begin = holidays_temp;
        while n > 0 {
            date += 1;
            day_of_week = increment_day_of_week(day_of_week);
            // SAFETY: week_mask is length 7, day_of_week is between 0 and 6
            if unsafe {
                (*week_mask.get_unchecked(day_of_week))
                    && (holidays[holidays_begin..].binary_search(&date).is_err())
            } {
                n -= 1;
            }
        }
        date
    } else {
        let holidays_end = find_first_gt_index(holidays, date);
        date += (n / n_business_days_in_week_mask) * 7;
        n %= n_business_days_in_week_mask;
        let holidays_temp = find_first_ge_index(&holidays[..holidays_end], date);
        n -= (holidays_end - holidays_temp) as i32;
        let holidays_end = holidays_temp;
        while n < 0 {
            date -= 1;
            day_of_week = decrement_day_of_week(day_of_week);
            // SAFETY: week_mask is length 7, day_of_week is between 0 and 6
            if unsafe {
                (*week_mask.get_unchecked(day_of_week))
                    && (holidays[..holidays_end].binary_search(&date).is_err())
            } {
                n += 1;
            }
        }
        date
    }
}

/// Determine if a day lands on a business day.
///
/// # Arguments
/// - `week_mask`: A boolean array of length 7, where `true` indicates that the day is a business day.
/// - `holidays`: timestamps that are holidays. Must be provided as i32, i.e. the number of
///   days since the UNIX epoch.
pub fn is_business_day(
    dates: &Series,
    week_mask: [bool; 7],
    holidays: &Series,
) -> PolarsResult<Series> {
    if !week_mask.iter().any(|&x| x) {
        polars_bail!(ComputeError:"`week_mask` must have at least one business day");
    }
    polars_ensure!(
        holidays.len() == 1 || holidays.len() == dates.len(),
        ShapeMismatch: "number of holiday lists must be either 1 or the number of dates"
    );

    match dates.dtype() {
        DataType::Date => {},
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(_, None) => {
            return is_business_day(&dates.cast(&DataType::Date)?, week_mask, holidays);
        },
        #[cfg(feature = "timezones")]
        DataType::Datetime(_, Some(_)) => {
            let dates_local = replace_time_zone(
                dates.datetime().unwrap(),
                None,
                &StringChunked::from_iter(std::iter::once("raise")),
                NonExistent::Raise,
            )?;
            return is_business_day(&dates_local.cast(&DataType::Date)?, week_mask, holidays);
        },
        _ => polars_bail!(InvalidOperation: "expected date or datetime, got {}", dates.dtype()),
    }

    let holidays = prep_holidays(holidays, week_mask)?;
    let mut holidays_lists = holidays_lists_iter(&holidays);

    let dates = dates.date()?;
    let out: BooleanChunked = unary_elementwise(dates.physical(), |date| {
        match (date, holidays_lists.next()) {
            (Some(date), Some(Some(holidays_list))) => {
                let day_of_week = get_day_of_week(date);
                // SAFETY: week_mask is length 7, day_of_week is between 0 and 6
                Some(unsafe {
                    (*week_mask.get_unchecked(day_of_week))
                        && holidays_list.binary_search(&date).is_err()
                })
            },
            _ => None,
        }
    });
    Ok(out.into_series())
}

fn roll_start_date(
    mut date: i32,
    roll: Roll,
    week_mask: &[bool; 7],
    holidays: &[i32], // Caller's responsibility to ensure it's sorted.
) -> PolarsResult<(i32, usize)> {
    let mut day_of_week = get_day_of_week(date);
    match roll {
        Roll::Raise => {
            // SAFETY: week_mask is length 7, day_of_week is between 0 and 6
            if holidays.binary_search(&date).is_ok()
                | unsafe { !*week_mask.get_unchecked(day_of_week) }
            {
                let date = DateTime::from_timestamp(date as i64 * SECONDS_IN_DAY, 0)
                    .unwrap()
                    .format("%Y-%m-%d");
                polars_bail!(ComputeError:
                    "date {} is not a business date; use `roll` to roll forwards (or backwards) to the next (or previous) valid date.", date
                )
            };
        },
        Roll::Forward => {
            // SAFETY: week_mask is length 7, day_of_week is between 0 and 6
            while holidays.binary_search(&date).is_ok()
                | unsafe { !*week_mask.get_unchecked(day_of_week) }
            {
                date += 1;
                day_of_week = increment_day_of_week(day_of_week);
            }
        },
        Roll::Backward => {
            // SAFETY: week_mask is length 7, day_of_week is between 0 and 6
            while holidays.binary_search(&date).is_ok()
                | unsafe { !*week_mask.get_unchecked(day_of_week) }
            {
                date -= 1;
                day_of_week = decrement_day_of_week(day_of_week);
            }
        },
    }
    Ok((date, day_of_week))
}

/// Sort and deduplicate holidays and remove holidays that are not business days.
fn normalise_holidays(holidays: &mut Vec<i32>, week_mask: &[bool; 7]) {
    holidays.sort_unstable();
    let mut previous_holiday: Option<i32> = None;
    holidays.retain(|&x| {
        // SAFETY: week_mask is length 7, get_day_of_week result is between 0 and 6
        if (Some(x) == previous_holiday) || !unsafe { *week_mask.get_unchecked(get_day_of_week(x)) }
        {
            return false;
        }
        previous_holiday = Some(x);
        true
    });
}

/// Convert from List of Date to normalized representation (see
/// `normalize_holidays`), trying to minimize allocations along the way.
///
/// If a list contains a null, that is considered an error.
fn prep_holidays(holidays: &Series, week_mask: [bool; 7]) -> PolarsResult<ListArray<i32>> {
    polars_ensure!(
        holidays.dtype().is_list() && holidays.dtype().inner_dtype() == Some(&DataType::Date),
        ComputeError: "holidays list had wrong data type {}, expected List of Date", holidays.dtype()
    );
    let holidays = holidays.list()?;
    let mut result = MutableListArray::new_with_capacity(
        MutablePrimitiveArray::with_capacity(holidays.lst_lengths().sum().unwrap_or(0) as usize),
        holidays.len(),
    );
    let mut staging = vec![];

    for maybe_list in holidays.to_physical_repr().chunks().iter().flat_map(|arr| {
        arr.as_any()
            .downcast_ref::<ListArray<i64>>()
            .unwrap()
            .iter()
    }) {
        let normalized_list = match maybe_list {
            Some(list) => {
                staging.clear();
                let array = list.as_any().downcast_ref::<PrimitiveArray<i32>>().unwrap();
                polars_ensure!(
                    array.validity().is_none(),
                    ComputeError: "list of holidays contained a null: {:?}", list.as_ref()
                );
                let holidays_list = array.as_slice().unwrap();
                staging.extend_from_slice(holidays_list);
                normalise_holidays(&mut staging, &week_mask);
                Some(staging.iter().map(|v| Some(*v)))
            },
            None => None,
        };
        result.try_push(normalized_list)?;
    }
    Ok(result.into())
}

fn holidays_lists_iter(holidays: &ListArray<i32>) -> impl Iterator<Item = Option<&[i32]>> {
    let length = holidays.len();
    // We know there are no nulls inside lists, so we can just get slices of off
    // this.
    let slice = holidays
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<i32>>()
        .unwrap()
        .as_slice()
        .unwrap();
    let mut iterator = (0..length).map(|i| {
        if holidays.is_null(i) {
            return None;
        }
        // SAFETY: we're iterating only up to length.
        let (start, end) = unsafe { holidays.offsets().start_end_unchecked(i) };
        Some(&slice[start..end])
    });
    if length == 1 {
        // A single list of holidays gets used for all rows:
        let arr = iterator.next().unwrap();
        Either::Left(std::iter::repeat(arr))
    } else {
        Either::Right(iterator)
    }
}

fn get_day_of_week(x: i32) -> usize {
    // the first modulo might return a negative number, so we add 7 and take
    // the modulo again so we're sure we have something between 0 (Monday)
    // and 6 (Sunday)
    (((x - 4) % 7 + 7) % 7) as usize
}

fn increment_day_of_week(x: usize) -> usize {
    if x == 6 { 0 } else { x + 1 }
}

fn decrement_day_of_week(x: usize) -> usize {
    if x == 0 { 6 } else { x - 1 }
}
