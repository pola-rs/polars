use arrow::legacy::time_zone::Tz;
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_core::series::IsSorted;

use crate::Duration;

fn apply_offsets_to_datetime(
    datetime: &Logical<DatetimeType, Int64Type>,
    offsets: &StringChunked,
    time_zone: Option<&Tz>,
) -> PolarsResult<Int64Chunked> {
    match offsets.len() {
        1 => match offsets.get(0) {
            Some(offset) => {
                let offset = &Duration::parse(offset);
                if offset.is_constant_duration(datetime.time_zone().as_ref()) {
                    // fastpath!
                    let mut duration = match datetime.time_unit() {
                        TimeUnit::Milliseconds => offset.duration_ms(),
                        TimeUnit::Microseconds => offset.duration_us(),
                        TimeUnit::Nanoseconds => offset.duration_ns(),
                    };
                    if offset.negative() {
                        duration = -duration;
                    }
                    Ok(datetime.phys.clone().wrapping_add_scalar(duration))
                } else {
                    let offset_fn = match datetime.time_unit() {
                        TimeUnit::Milliseconds => Duration::add_ms,
                        TimeUnit::Microseconds => Duration::add_us,
                        TimeUnit::Nanoseconds => Duration::add_ns,
                    };
                    datetime
                        .phys
                        .try_apply_nonnull_values_generic(|v| offset_fn(offset, v, time_zone))
                }
            },
            _ => Ok(datetime.phys.apply(|_| None)),
        },
        _ => {
            let offset_fn = match datetime.time_unit() {
                TimeUnit::Milliseconds => Duration::add_ms,
                TimeUnit::Microseconds => Duration::add_us,
                TimeUnit::Nanoseconds => Duration::add_ns,
            };
            broadcast_try_binary_elementwise(datetime, offsets, |timestamp_opt, offset_opt| match (
                timestamp_opt,
                offset_opt,
            ) {
                (Some(timestamp), Some(offset)) => {
                    offset_fn(&Duration::try_parse(offset)?, timestamp, time_zone).map(Some)
                },
                _ => Ok(None),
            })
        },
    }
}

pub fn impl_offset_by(ts: &Series, offsets: &Series) -> PolarsResult<Series> {
    let offsets = offsets.str()?;

    polars_ensure!(
        ts.len() == offsets.len() || offsets.len() == 1 || ts.len() == 1,
        length_mismatch = "dt.offset_by",
        ts.len(),
        offsets.len()
    );

    let dtype = ts.dtype();

    // Sortedness may not be preserved for non-constant durations,
    // see https://github.com/pola-rs/polars/issues/19608 for a counterexample.
    // Constant durations (e.g. 2 hours) always preserve sortedness.
    let tz = match dtype {
        DataType::Date => None,
        DataType::Datetime(_, tz) => tz.clone(),
        _ => polars_bail!(InvalidOperation: "expected Date or Datetime, got {}", dtype),
    };
    let preserve_sortedness = match offsets.len() {
        1 => match offsets.get(0) {
            Some(offset) => {
                let offset = Duration::try_parse(offset)?;
                offset.is_constant_duration(tz.as_ref())
            },
            None => false,
        },
        _ => false,
    };

    let out = match dtype {
        DataType::Date => {
            let ts = ts
                .cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap();
            let datetime = ts.datetime().unwrap();
            let out = apply_offsets_to_datetime(datetime, offsets, None)?;
            out.cast(&DataType::Datetime(TimeUnit::Milliseconds, None))
                .unwrap()
                .cast(&DataType::Date)
        },
        DataType::Datetime(tu, tz) => {
            let datetime = ts.datetime().unwrap();

            let out = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => {
                    apply_offsets_to_datetime(datetime, offsets, tz.parse::<Tz>().ok().as_ref())?
                },
                _ => apply_offsets_to_datetime(datetime, offsets, None)?,
            };
            out.cast(&DataType::Datetime(*tu, tz.clone()))
        },
        dt => polars_bail!(
            ComputeError: "cannot use 'offset_by' on Series of datatype {}", dt,
        ),
    };
    if preserve_sortedness {
        out.map(|mut out| {
            out.set_sorted_flag(ts.is_sorted_flag());
            out
        })
    } else {
        out.map(|mut out| {
            out.set_sorted_flag(IsSorted::Not);
            out
        })
    }
}
