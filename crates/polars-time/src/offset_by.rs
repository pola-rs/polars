use polars_arrow::legacy::time_zone::Tz;
use polars_core::prelude::arity::broadcast_try_binary_elementwise;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_defs::time::duration::Duration;

fn apply_offsets_to_datetime(
    datetime: &Logical<DatetimeType, Int64Type>,
    offsets: &StringChunked,
    time_zone: Option<&Tz>,
) -> PolarsResult<(Int64Chunked, bool)> {
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
                    let out = datetime.phys.clone().wrapping_add_scalar(duration);
                    // The wrapping add keeps the values' order, unless a
                    // value overflows the i64 bound (e.g. `Datetime("ns")`
                    // max + 1 day wraps around to the minimum). Only the
                    // extreme value can overflow, so a checked add on it
                    // detects the wrap.
                    let wrapped = if duration > 0 {
                        datetime
                            .phys
                            .max()
                            .is_some_and(|max| max.checked_add(duration).is_none())
                    } else if duration < 0 {
                        datetime
                            .phys
                            .min()
                            .is_some_and(|min| min.checked_add(duration).is_none())
                    } else {
                        false
                    };
                    Ok((out, wrapped))
                } else {
                    datetime
                        .phys
                        .try_apply_nonnull_values_generic(|v| {
                            offset.add(datetime.time_unit(), v, time_zone)
                        })
                        .map(|out| (out, false))
                }
            },
            _ => Ok((datetime.phys.apply(|_| None), false)),
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
        )
        .map(|out| (out, false)),
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
    // Constant durations (e.g. 2 hours) preserve sortedness, unless a value
    // wraps around the i64 bound.
    let tz = match dtype {
        DataType::Date => None,
        DataType::Datetime(_, tz) => tz.clone(),
        _ => polars_bail!(InvalidOperation: "expected Date or Datetime, got {dtype}"),
    };
    let mut preserve_sortedness = match offsets.len() {
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
                .cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap();
            let datetime = ts.datetime().unwrap();
            let (out, wrapped) = apply_offsets_to_datetime(datetime, offsets, None)?;
            // A wrapped-around value breaks the sort order, so the sorted
            // flag must not be preserved for it.
            preserve_sortedness &= !wrapped;
            out.cast(&DataType::Datetime(TimeUnit::Microseconds, None))
                .unwrap()
                .cast(&DataType::Date)
        },
        DataType::Datetime(tu, tz) => {
            let datetime = ts.datetime().unwrap();

            let (out, wrapped) = match tz {
                #[cfg(feature = "timezones")]
                Some(tz) => {
                    apply_offsets_to_datetime(datetime, offsets, tz.parse::<Tz>().ok().as_ref())?
                },
                _ => apply_offsets_to_datetime(datetime, offsets, None)?,
            };
            // A wrapped-around value breaks the sort order, so the sorted
            // flag must not be preserved for it.
            preserve_sortedness &= !wrapped;
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

#[cfg(test)]
mod test {
    use super::*;

    fn datetime_ns(values: &[i64]) -> Series {
        Series::new("dt".into(), values)
            .cast(&DataType::Datetime(TimeUnit::Nanoseconds, None))
            .unwrap()
    }

    fn single_offset(offset: &str) -> Series {
        Series::new("offset".into(), &[offset])
    }

    #[test]
    fn offset_by_overflow_clears_sorted_flag() {
        // 2262-04-11T00:00:00 is the largest whole day representable as
        // `Datetime("ns")`; adding a day wraps it around to 1677-09-21.
        let mut s = datetime_ns(&[0, 9_223_286_400_000_000_000]);
        s.set_sorted_flag(IsSorted::Ascending);

        let out = impl_offset_by(&s, &single_offset("1d")).unwrap();
        assert_eq!(out.is_sorted_flag(), IsSorted::Not);

        // The wraparound itself is kept, matching `Duration` arithmetic.
        let values = out.cast(&DataType::Int64).unwrap();
        assert_eq!(
            values.i64().unwrap().get(1).unwrap(),
            9_223_286_400_000_000_000i64.wrapping_add(86_400_000_000_000)
        );
    }

    #[test]
    fn offset_by_underflow_clears_sorted_flag() {
        // 1677-09-22T00:00:00 is the smallest whole day representable as
        // `Datetime("ns")`; subtracting a day wraps it around to 2262-04-12.
        let mut s = datetime_ns(&[-9_223_286_400_000_000_000, 0]);
        s.set_sorted_flag(IsSorted::Ascending);

        let out = impl_offset_by(&s, &single_offset("-1d")).unwrap();
        assert_eq!(out.is_sorted_flag(), IsSorted::Not);
    }

    #[test]
    fn offset_by_without_overflow_keeps_sorted_flag() {
        let mut s = datetime_ns(&[0, 86_400_000_000_000]);
        s.set_sorted_flag(IsSorted::Ascending);

        let out = impl_offset_by(&s, &single_offset("1d")).unwrap();
        assert_eq!(out.is_sorted_flag(), IsSorted::Ascending);
    }
}
