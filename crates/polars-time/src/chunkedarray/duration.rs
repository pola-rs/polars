use arrow::temporal_conversions::{MICROSECONDS, MILLISECONDS, NANOSECONDS, SECONDS_IN_DAY};

use super::*;

const NANOSECONDS_IN_MILLISECOND: i64 = 1_000_000;
const SECONDS_IN_HOUR: i64 = 3600;

pub trait DurationMethods {
    /// Extract the days from a `Duration`
    fn days(&self) -> Int64Chunked;

    /// Extract the days from a `Duration` as a fractional value
    fn days_fractional(&self) -> Float64Chunked;

    /// Extract the hours from a `Duration`
    fn hours(&self) -> Int64Chunked;

    /// Extract the hours from a `Duration` as a fractional value
    fn hours_fractional(&self) -> Float64Chunked;

    /// Extract the minutes from a `Duration`
    fn minutes(&self) -> Int64Chunked;

    /// Extract the minutes from a `Duration` as a fractional value
    fn minutes_fractional(&self) -> Float64Chunked;

    /// Extract the seconds from a `Duration`
    fn seconds(&self) -> Int64Chunked;

    /// Extract the seconds from a `Duration` as a fractional value
    fn seconds_fractional(&self) -> Float64Chunked;

    /// Extract the milliseconds from a `Duration`
    fn milliseconds(&self) -> Int64Chunked;

    /// Extract the milliseconds from a `Duration` as a fractional value
    fn milliseconds_fractional(&self) -> Float64Chunked;

    /// Extract the microseconds from a `Duration`
    fn microseconds(&self) -> Int64Chunked;

    /// Extract the microseconds from a `Duration` as a fractional value
    fn microseconds_fractional(&self) -> Float64Chunked;

    /// Extract the nanoseconds from a `Duration`
    fn nanoseconds(&self) -> Int64Chunked;

    /// Extract the nanoseconds from a `Duration` as a fractional value
    fn nanoseconds_fractional(&self) -> Float64Chunked;
}

impl DurationMethods for DurationChunked {
    /// Extract the hours from a `Duration`
    fn hours(&self) -> Int64Chunked {
        let t = time_units_in_second(self.time_unit());
        (&self.phys).wrapping_trunc_div_scalar(t * SECONDS_IN_HOUR)
    }

    /// Extract the hours from a `Duration` as a fractional value
    fn hours_fractional(&self) -> Float64Chunked {
        let t = time_units_in_second(self.time_unit());
        num_of_unit_fractional(self, t as f64 * SECONDS_IN_HOUR as f64)
    }

    /// Extract the days from a `Duration`
    fn days(&self) -> Int64Chunked {
        let t = time_units_in_second(self.time_unit());
        (&self.phys).wrapping_trunc_div_scalar(t * SECONDS_IN_DAY)
    }

    /// Extract the days from a `Duration` as a fractional value
    fn days_fractional(&self) -> Float64Chunked {
        let t = time_units_in_second(self.time_unit());
        num_of_unit_fractional(self, t as f64 * SECONDS_IN_DAY as f64)
    }

    /// Extract the seconds from a `Duration`
    fn minutes(&self) -> Int64Chunked {
        let t = time_units_in_second(self.time_unit());
        (&self.phys).wrapping_trunc_div_scalar(t * 60)
    }

    /// Extract the minutes from a `Duration` as a fractional value
    fn minutes_fractional(&self) -> Float64Chunked {
        let t = time_units_in_second(self.time_unit());
        num_of_unit_fractional(self, t as f64 * 60.0)
    }

    /// Extract the seconds from a `Duration`
    fn seconds(&self) -> Int64Chunked {
        let t = time_units_in_second(self.time_unit());
        (&self.phys).wrapping_trunc_div_scalar(t)
    }

    /// Extract the seconds from a `Duration` as a fractional value
    fn seconds_fractional(&self) -> Float64Chunked {
        let t = time_units_in_second(self.time_unit());
        num_of_unit_fractional(self, t as f64)
    }

    /// Extract the milliseconds from a `Duration`
    fn milliseconds(&self) -> Int64Chunked {
        let t = match self.time_unit() {
            TimeUnit::Milliseconds => return self.phys.clone(),
            TimeUnit::Microseconds => 1000,
            TimeUnit::Nanoseconds => NANOSECONDS_IN_MILLISECOND,
        };
        (&self.phys).wrapping_trunc_div_scalar(t)
    }

    /// Extract the milliseconds from a `Duration`
    fn milliseconds_fractional(&self) -> Float64Chunked {
        let t = time_units_in_second(self.time_unit());
        num_of_unit_fractional(self, t as f64 / MILLISECONDS as f64)
    }

    /// Extract the microseconds from a `Duration`
    fn microseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.phys * 1000,
            TimeUnit::Microseconds => self.phys.clone(),
            TimeUnit::Nanoseconds => (&self.phys).wrapping_trunc_div_scalar(1000),
        }
    }

    /// Extract the microseconds from a `Duration` as a fractional value
    fn microseconds_fractional(&self) -> Float64Chunked {
        let t = time_units_in_second(self.time_unit());
        num_of_unit_fractional(self, t as f64 / MICROSECONDS as f64)
    }

    /// Extract the nanoseconds from a `Duration`
    fn nanoseconds(&self) -> Int64Chunked {
        match self.time_unit() {
            TimeUnit::Milliseconds => &self.phys * NANOSECONDS_IN_MILLISECOND,
            TimeUnit::Microseconds => &self.phys * 1000,
            TimeUnit::Nanoseconds => self.phys.clone(),
        }
    }

    /// Extract the nanoseconds from a `Duration` as a fractional value
    fn nanoseconds_fractional(&self) -> Float64Chunked {
        let t = time_units_in_second(self.time_unit());
        num_of_unit_fractional(self, t as f64 / NANOSECONDS as f64)
    }
}

fn time_units_in_second(tu: TimeUnit) -> i64 {
    match tu {
        TimeUnit::Milliseconds => MILLISECONDS,
        TimeUnit::Microseconds => MICROSECONDS,
        TimeUnit::Nanoseconds => NANOSECONDS,
    }
}

fn num_of_unit_fractional(ca: &DurationChunked, unit_ns: f64) -> Float64Chunked {
    ca.physical()
        .cast(&DataType::Float64)
        .expect("cast failed")
        .f64()
        .unwrap()
        / unit_ns
}
