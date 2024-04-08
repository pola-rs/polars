use polars_core::prelude::arity::binary_elementwise_values;
use polars_core::prelude::*;

/// Count the number of business days between `start` and `end`, excluding `end`.
pub fn business_day_count(start: &Series, end: &Series) -> PolarsResult<Series> {
    let start_dates = start.date()?;
    let end_dates = end.date()?;

    // TODO: support customising weekdays
    let week_mask: [bool; 7] = [true, true, true, true, true, false, false];
    let n_business_days_in_week_mask = week_mask.iter().filter(|&x| *x).count() as i32;

    let out = match (start_dates.len(), end_dates.len()) {
        (_, 1) => {
            if let Some(end_date) = end_dates.get(0) {
                start_dates.apply_values(|start_date| {
                    business_day_count_impl(
                        start_date,
                        end_date,
                        &week_mask,
                        n_business_days_in_week_mask,
                    )
                })
            } else {
                Int32Chunked::full_null(start_dates.name(), start_dates.len())
            }
        },
        (1, _) => {
            if let Some(start_date) = start_dates.get(0) {
                end_dates.apply_values(|end_date| {
                    business_day_count_impl(
                        start_date,
                        end_date,
                        &week_mask,
                        n_business_days_in_week_mask,
                    )
                })
            } else {
                Int32Chunked::full_null(start_dates.name(), end_dates.len())
            }
        },
        _ => binary_elementwise_values(start_dates, end_dates, |start_date, end_date| {
            business_day_count_impl(
                start_date,
                end_date,
                &week_mask,
                n_business_days_in_week_mask,
            )
        }),
    };
    Ok(out.into_series())
}

/// Ported from:
/// https://github.com/numpy/numpy/blob/e59c074842e3f73483afa5ddef031e856b9fd313/numpy/_core/src/multiarray/datetime_busday.c#L355-L433
fn business_day_count_impl(
    mut start_date: i32,
    mut end_date: i32,
    week_mask: &[bool; 7],
    n_business_days_in_week_mask: i32,
) -> i32 {
    let swapped = start_date > end_date;
    if swapped {
        (start_date, end_date) = (end_date, start_date);
        start_date += 1;
        end_date += 1;
    }

    let mut start_weekday = weekday(start_date);
    let diff = end_date - start_date;
    let whole_weeks = diff / 7;
    let mut count = 0;
    count += whole_weeks * n_business_days_in_week_mask;
    start_date += whole_weeks * 7;
    while start_date < end_date {
        if unsafe { *week_mask.get_unchecked(start_weekday) } {
            count += 1;
        }
        start_date += 1;
        start_weekday += 1;
        if start_weekday >= 7 {
            start_weekday = 0;
        }
    }
    if swapped {
        -count
    } else {
        count
    }
}

fn weekday(x: i32) -> usize {
    // the first modulo might return a negative number, so we add 7 and take
    // the modulo again so we're sure we have something between 0 (Monday)
    // and 6 (Sunday)
    (((x - 4) % 7 + 7) % 7) as usize
}
