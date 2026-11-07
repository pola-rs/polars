use polars_arrow::legacy::time_zone::Tz;
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_defs::time::duration::Duration;

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
                    let mut duration = offset.duration(datetime.time_unit());
                    if offset.negative() {
                        duration = -duration;
                    }
                    Ok(datetime.phys.clone().wrapping_add_scalar(duration))
                } else {
                    datetime.phys.try_apply_nonnull_values_generic(|v| {
                        offset.add(datetime.time_unit(), v, time_zone)
                    })
                }
            },
            _ => Ok(datetime.phys.apply(|_| None)),
        },
        _ => broadcast_try_binary_elementwise(
            datetime.physical(),
            offsets,
            |timestamp_opt, offset_opt| match (timestamp_opt, offset_opt) {
                (Some(timestamp), Some(offset)) => Duration::try_parse(offset)?
                    .add(datetime.time_unit(), timestamp, time_zone)
                    .map(Some),
                _ => Ok(None),
            },
        ),
    }
}

fn datetime_would_overflow(datetime: &Logical<DatetimeType, Int64Type>, offset: &Duration) -> bool {
    let mut duration = offset.duration(datetime.time_unit());

    if offset.negative() {
        duration = -duration;
    }

    match duration {
        0 => false,
        n if n > 0 => {
            let max_timestamp = datetime.phys.max();
            max_timestamp.is_some_and(|max| max.checked_add(duration).is_none())
        },
        _ => {
            let min_timestamp = datetime.phys.min();
            min_timestamp.is_some_and(|min| min.checked_add(duration).is_none())
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
    // Constant durations preserve sortedness unless timestamp arithmetic overflows.
    let tz = match dtype {
        DataType::Date => None,
        DataType::Datetime(_, tz) => tz.clone(),
        _ => polars_bail!(InvalidOperation: "expected Date or Datetime, got {dtype}"),
    };
    let constant_offset = match offsets.len() {
        1 => match offsets.get(0) {
            Some(offset) => {
                let offset = Duration::try_parse(offset)?;
                offset.is_constant_duration(tz.as_ref()).then_some(offset)
            },
            None => None,
        },
        _ => None,
    };

    let preserve_sortedness = constant_offset.is_some() && ts.is_sorted_flag() != IsSorted::Not;

    let (out, would_overflow) = match dtype {
        DataType::Date => {
            let ts = ts
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap();
            let datetime = ts.datetime().unwrap();

            let would_overflow = if preserve_sortedness {
                datetime_would_overflow(datetime, constant_offset.as_ref().unwrap())
            } else {
                false
            };
            let out = apply_offsets_to_datetime(datetime, offsets, None)?;
            (
                out.cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                    .unwrap()
                    .cast(&DataType::Date),
                would_overflow,
            )
        },
        DataType::Datetime(tu, tz) => {
            let datetime = ts.datetime().unwrap();

            let would_overflow = if preserve_sortedness {
                datetime_would_overflow(datetime, constant_offset.as_ref().unwrap())
            } else {
                false
            };
            let out = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => {
                    apply_offsets_to_datetime(datetime, offsets, tz.parse::<Tz>().ok().as_ref())?
                },
                _ => apply_offsets_to_datetime(datetime, offsets, None)?,
            };
            (
                out.cast(&DataType::Datetime(*tu, tz.clone())),
                would_overflow,
            )
        },
        dt => polars_bail!(
            ComputeError: "cannot use 'offset_by' on Series of datatype {}", dt,
        ),
    };
    if preserve_sortedness && !would_overflow {
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
