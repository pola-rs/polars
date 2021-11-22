use crate::prelude::*;
use polars_arrow::trusted_len::PushUnchecked;

#[derive(Debug, Default, Copy, Clone)]
pub struct TimeDelta {
    days: i64,
    seconds: u32,
    microseconds: u32,
}

impl TimeDelta {
    fn to_milliseconds(self) -> i64 {
        let mut milliseconds = self.days * 3600 * 24 * 1000;
        milliseconds += (self.seconds as i64) * 1000;
        milliseconds += (self.microseconds as i64) / 1000;
        milliseconds
    }

    fn to_days(self) -> i64 {
        self.days + (self.seconds as i64 / (3600 * 24))
    }
}

#[derive(Debug, Default, Copy, Clone)]
pub struct TimeDeltaBuilder {
    days: i64,
    seconds: u32,
    microseconds: u32,
}

impl TimeDeltaBuilder {
    pub fn new() -> Self {
        Default::default()
    }

    pub fn days(mut self, days: i64) -> Self {
        self.days += days;
        self
    }

    pub fn seconds(mut self, seconds: u32) -> Self {
        self.seconds += seconds;
        self
    }

    pub fn microseconds(mut self, microseconds: u32) -> Self {
        self.microseconds += microseconds;
        self
    }

    pub fn milliseconds(mut self, milliseconds: u32) -> Self {
        self.microseconds += milliseconds * 1000;
        self
    }

    pub fn hours(mut self, hours: u32) -> Self {
        self.seconds += hours * 3600;
        self
    }

    pub fn weeks(mut self, weeks: i64) -> Self {
        self.days += weeks * 7;
        self
    }

    pub fn finish(self) -> TimeDelta {
        TimeDelta {
            days: self.days,
            seconds: self.seconds,
            microseconds: self.microseconds,
        }
    }
}

#[cfg(feature = "dtype-datetime")]
impl DatetimeChunked {
    pub fn buckets(&self, resolution: TimeDelta) -> Self {
        let ca = self.sort(false);

        match ca.first_non_null() {
            None => self.clone(),
            Some(idx) => {
                let arr = ca.downcast_iter().next().unwrap();
                let ms = arr.values().as_slice();

                let mut new_ms = AlignedVec::with_capacity(self.len());

                // extend nulls
                new_ms.extend_from_slice(&ms[..idx]);

                let timedelta = resolution.to_milliseconds();
                let mut current_lower = ms[idx];
                let mut current_higher = current_lower + timedelta;

                for &val in ms {
                    if val > current_higher {
                        current_lower = current_higher;
                        current_higher += timedelta;
                    }
                    // Safety:
                    // we preallocated
                    unsafe { new_ms.push_unchecked(current_lower) };
                }
                let arr = PrimitiveArray::from_data(
                    ArrowDataType::Int64,
                    new_ms.into(),
                    arr.validity().cloned(),
                );
                let mut ca =
                    Int64Chunked::new_from_chunks(self.name(), vec![Arc::new(arr)]).into_date();
                ca.set_sorted(false);
                ca
            }
        }
    }
}

#[cfg(feature = "dtype-date")]
impl DateChunked {
    pub fn buckets(&self, resolution: TimeDelta) -> Self {
        let ca = self.sort(false);

        match ca.first_non_null() {
            None => self.clone(),
            Some(idx) => {
                let arr = ca.downcast_iter().next().unwrap();
                let days = arr.values().as_slice();

                let mut new_days = AlignedVec::with_capacity(self.len());

                // extend nulls
                new_days.extend_from_slice(&days[..idx]);

                let timedelta = resolution.to_days() as i32;
                let mut current_lower = days[idx];
                let mut current_higher = current_lower + timedelta;

                for &val in days {
                    if val > current_higher {
                        current_lower = current_higher;
                        current_higher += timedelta;
                    }
                    // Safety:
                    // we preallocated
                    unsafe { new_days.push_unchecked(current_lower) };
                }
                let arr = PrimitiveArray::from_data(
                    ArrowDataType::Int32,
                    new_days.into(),
                    arr.validity().cloned(),
                );
                let mut ca =
                    Int32Chunked::new_from_chunks(self.name(), vec![Arc::new(arr)]).into_date();
                ca.set_sorted(false);
                ca
            }
        }
    }
}
