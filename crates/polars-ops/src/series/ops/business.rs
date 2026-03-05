use arrow::Either;
use arrow::array::PrimitiveArray;
use arrow::bitmap::Bitmap;
#[cfg(feature = "dtype-date")]
use chrono::DateTime;
use polars_core::prelude::arity::{try_binary_elementwise, unary_elementwise};
use polars_core::prelude::*;
#[cfg(feature = "dtype-date")]
use polars_core::utils::arrow::temporal_conversions::SECONDS_IN_DAY;
use polars_core::{binary_output_height, ternary_output_height};
use polars_utils::binary_search::{find_first_ge_index, find_first_gt_index};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

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

macro_rules! empty_or_all_null {
    ($c:expr) => {
        $c.len() == 0 || $c.len() == $c.null_count()
    };
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

    let start_dates = start.date()?;
    let end_dates = end.date()?;

    ensure_holidays_dtype(holidays.dtype())?;
    let output_height = ternary_output_height!(start, end, holidays, op = "business_day_count")?;
    let output_name = start_dates.name().clone();

    macro_rules! return_all_null {
        () => {
            return Ok(Int32Chunked::full_null(output_name, output_height).into_series())
        };
    }

    // Broadcasted NULL or 0-length
    if empty_or_all_null!(holidays)
        || empty_or_all_null!(start_dates)
        || empty_or_all_null!(end_dates)
    {
        return_all_null!()
    }

    let start_dates = start_dates.physical().rechunk();
    let end_dates = end_dates.physical().rechunk();

    let start_dates: &PrimitiveArray<i32> =
        start_dates.chunks()[0].as_any().downcast_ref().unwrap();
    let end_dates: &PrimitiveArray<i32> = end_dates.chunks()[0].as_any().downcast_ref().unwrap();
    let holidays = holidays.rechunk();
    let holidays_list: &LargeListArray = holidays.chunks()[0].as_any().downcast_ref().unwrap();
    let mut holidays_getter = HolidayListsGetter::new(holidays_list, week_mask);

    let n_business_days_in_week_mask = week_mask.iter().filter(|&x| *x).count() as i32;

    let out: ChunkedArray<Int32Type> = (0..output_height)
        .map(|i| {
            let start = unsafe { start_dates.get_unchecked(i % start_dates.len()) }?;
            let end = unsafe { end_dates.get_unchecked(i % end_dates.len()) }?;
            let holidays = holidays_getter.holiday_at_idx_broadcast(i)?;

            Some(business_day_count_impl(
                start,
                end,
                &week_mask,
                n_business_days_in_week_mask,
                holidays,
            ))
        })
        .collect();

    holidays_getter.ensure_no_nulls_seen_in_values()?;

    Ok(out.with_name(output_name).into_series())
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

    ensure_holidays_dtype(holidays.dtype())?;
    let output_height = ternary_output_height!(start, n, holidays, op = "add_business_days")?;
    let output_name = start.name().clone();

    macro_rules! return_all_null {
        () => {
            return Ok(Int32Chunked::full_null(output_name, output_height).into_series())
        };
    }

    // Broadcasted NULL or 0-length
    if empty_or_all_null!(holidays) || empty_or_all_null!(start) || empty_or_all_null!(n) {
        return_all_null!()
    }

    let start_dates = start.date()?;
    let n = match &n.dtype() {
        DataType::Int64 | DataType::UInt64 | DataType::UInt32 => n.cast(&DataType::Int32)?,
        DataType::Int32 => n.clone(),
        _ => {
            polars_bail!(InvalidOperation: "expected Int64, Int32, UInt64, or UInt32, got {}", n.dtype())
        },
    };
    let n = n.i32()?;

    let holidays = holidays.rechunk();
    let holidays_list: &LargeListArray = holidays.chunks()[0].as_any().downcast_ref().unwrap();
    let mut holidays_getter = HolidayListsGetter::new(holidays_list, week_mask);

    let start_dates = start_dates.physical().rechunk();

    let start_dates: &PrimitiveArray<i32> =
        start_dates.chunks()[0].as_any().downcast_ref().unwrap();

    let n_business_days_in_week_mask = week_mask.iter().filter(|&x| *x).count() as i32;

    let out: Int32Chunked = if start.len() == 1 && holidays.len() == 1 {
        let holidays_list = holidays_getter.holiday_at_idx_broadcast(0).unwrap();
        let (start, day_of_week) =
            roll_start_date(start_dates.get(0).unwrap(), roll, &week_mask, holidays_list)?;

        n.apply_values(|n| {
            add_business_days_impl(
                start,
                day_of_week,
                n,
                &week_mask,
                n_business_days_in_week_mask,
                holidays_list,
            )
        })
    } else {
        let n = n.rechunk();
        let n: &PrimitiveArray<i32> = n.chunks()[0].as_any().downcast_ref().unwrap();

        (0..output_height)
            .map(|i| {
                let start = unsafe { start_dates.get_unchecked(i % start_dates.len()) }?;
                let n = unsafe { n.get_unchecked(i % n.len()) }?;
                let holidays_list = holidays_getter.holiday_at_idx_broadcast(i)?;

                Some(roll_start_date(start, roll, &week_mask, holidays_list).map(
                    |(start, day_of_week)| {
                        add_business_days_impl(
                            start,
                            day_of_week,
                            n,
                            &week_mask,
                            n_business_days_in_week_mask,
                            holidays_list,
                        )
                    },
                ))
            })
            .map(Option::transpose)
            .collect::<PolarsResult<_>>()?
    };

    holidays_getter.ensure_no_nulls_seen_in_values()?;

    Ok(out.with_name(output_name).into_date().into_series())
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

    ensure_holidays_dtype(holidays.dtype())?;
    let output_height = binary_output_height!(dates, holidays, op = "is_business_day")?;
    let output_name = dates.name().clone();

    macro_rules! return_all_null {
        () => {
            return Ok(BooleanChunked::full_null(output_name, output_height).into_series())
        };
    }

    // Broadcasted NULL or 0-length
    if empty_or_all_null!(holidays) || empty_or_all_null!(dates) {
        return_all_null!()
    }

    let holidays = holidays.rechunk();
    let holidays_list: &LargeListArray = holidays.chunks()[0].as_any().downcast_ref().unwrap();
    let mut holidays_getter = HolidayListsGetter::new(holidays_list, week_mask);

    let dates = dates.date()?;
    let dates = dates.physical().rechunk();
    let dates: &PrimitiveArray<i32> = dates.chunks()[0].as_any().downcast_ref().unwrap();

    let out: BooleanChunked = (0..output_height)
        .map(|i| {
            let date = unsafe { dates.get_unchecked(i % dates.len()) }?;
            let holidays_list = holidays_getter.holiday_at_idx_broadcast(i)?;

            let day_of_week = get_day_of_week(date);

            Some(
                // SAFETY: week_mask is length 7, day_of_week is between 0 and 6
                unsafe {
                    (*week_mask.get_unchecked(day_of_week))
                        && holidays_list.binary_search(&date).is_err()
                },
            )
        })
        .collect();

    holidays_getter.ensure_no_nulls_seen_in_values()?;

    Ok(out.with_name(output_name).into_series())
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
fn normalize_holidays(holidays: &mut Vec<i32>, week_mask: &[bool; 7]) {
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

fn ensure_holidays_dtype(holidays_dtype: &DataType) -> PolarsResult<()> {
    match holidays_dtype {
        DataType::List(dtype) if matches!(dtype.as_ref(), DataType::Date) => {},
        dtype => polars_bail!(
            ComputeError:
            "dtype of holidays list must be List(Date), got {dtype:?} instead"
        ),
    }

    Ok(())
}

struct HolidayListsGetter<'a> {
    holidays_list: &'a LargeListArray,
    holidays_list_values: &'a [i32],
    holidays_list_values_validity: Option<&'a Bitmap>,
    week_mask: [bool; 7],
    current_row_values: Vec<i32>,
    null_seen_in_values: bool,
}

impl<'a> HolidayListsGetter<'a> {
    fn new(holidays_list: &'a LargeListArray, week_mask: [bool; 7]) -> Self {
        let holidays_list_values: &PrimitiveArray<i32> =
            holidays_list.values().as_any().downcast_ref().unwrap();
        let holidays_list_values_validity = holidays_list_values.validity();
        let holidays_list_values: &[i32] = holidays_list_values.values().as_slice();

        Self {
            holidays_list,
            holidays_list_values,
            holidays_list_values_validity,
            week_mask,
            current_row_values: vec![],
            null_seen_in_values: false,
        }
    }

    /// If `idx` exceeds the length of holidays_list, the last returned row value is returned.
    fn holiday_at_idx_broadcast(&mut self, idx: usize) -> Option<&[i32]> {
        if idx >= self.holidays_list.len() {
            return Some(&self.current_row_values);
        }

        if self
            .holidays_list
            .validity()
            .is_some_and(|m| unsafe { !m.get_bit_unchecked(idx) })
        {
            return None;
        }

        let (start, end) = self.holidays_list.offsets().start_end(idx);

        self.current_row_values.clear();
        self.current_row_values
            .extend_from_slice(&self.holidays_list_values[start..end]);
        normalize_holidays(&mut self.current_row_values, &self.week_mask);

        let null_in_values = self
            .holidays_list_values_validity
            .is_some_and(|m| m.null_count_range(start, end - start) > 0);

        self.null_seen_in_values |= null_in_values;

        (!null_in_values).then_some(&self.current_row_values)
    }

    fn ensure_no_nulls_seen_in_values(&self) -> PolarsResult<()> {
        polars_ensure!(
            !self.null_seen_in_values,
            ComputeError:
            "nulls found in holiday list values",
        );

        Ok(())
    }
}
