use chrono::Duration as ChronoDuration;

use crate::fmt::{fmt_duration_string, iso_duration_string};
use crate::prelude::DataType::Duration;
use crate::prelude::*;

impl DurationChunked {
    pub fn time_unit(&self) -> TimeUnit {
        match &self.dtype {
            DataType::Duration(tu) => *tu,
            _ => unreachable!(),
        }
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    #[must_use]
    pub fn cast_time_unit(&self, tu: TimeUnit) -> Self {
        let current_unit = self.time_unit();
        let mut out = self.clone();
        out.set_time_unit(tu);

        use crate::datatypes::time_unit::TimeUnit::*;
        match (current_unit, tu) {
            (Nanoseconds, Microseconds) => {
                let ca = (&self.phys).wrapping_trunc_div_scalar(1_000);
                out.phys = ca;
                out
            },
            (Nanoseconds, Milliseconds) => {
                let ca = (&self.phys).wrapping_trunc_div_scalar(1_000_000);
                out.phys = ca;
                out
            },
            (Microseconds, Nanoseconds) => {
                let ca = &self.phys * 1_000;
                out.phys = ca;
                out
            },
            (Microseconds, Milliseconds) => {
                let ca = (&self.phys).wrapping_trunc_div_scalar(1_000);
                out.phys = ca;
                out
            },
            (Milliseconds, Nanoseconds) => {
                let ca = &self.phys * 1_000_000;
                out.phys = ca;
                out
            },
            (Milliseconds, Microseconds) => {
                let ca = &self.phys * 1_000;
                out.phys = ca;
                out
            },
            (Nanoseconds, Nanoseconds)
            | (Microseconds, Microseconds)
            | (Milliseconds, Milliseconds) => out,
        }
    }

    /// Change the underlying [`TimeUnit`]. This does not modify the data.
    pub fn set_time_unit(&mut self, tu: TimeUnit) {
        self.dtype = Duration(tu);
    }

    /// Convert from [`Duration`] to String; note that `strftime` format
    /// strings are not supported, only the specifiers 'iso' and 'polars'.
    pub fn to_string(&self, format: &str) -> PolarsResult<StringChunked> {
        let time_unit = self.time_unit();
        match format {
            "iso" | "iso:strict" => {
                let out = self.phys.apply_into_string_amortized(|v: i64, buf| {
                    iso_duration_string(buf, v, time_unit);
                });
                Ok(out)
            },
            "polars" => {
                let out = self.phys.apply_into_string_amortized(|v: i64, buf| {
                    fmt_duration_string(buf, v, time_unit)
                        .map_err(|e| polars_err!(ComputeError: "{:?}", e))
                        .expect("failed to format duration");
                });
                Ok(out)
            },
            _ => {
                polars_bail!(
                    InvalidOperation: "format {:?} not supported for Duration type (expected one of 'iso' or 'polars')",
                    format
                )
            },
        }
    }

    /// Construct a new [`DurationChunked`] from an iterator over [`ChronoDuration`].
    pub fn from_duration<I: IntoIterator<Item = ChronoDuration>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Microseconds => |v: ChronoDuration| v.num_microseconds().unwrap(),
            TimeUnit::Milliseconds => |v: ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.into_iter().map(func).collect::<Vec<_>>();
        Int64Chunked::from_vec(name, vals).into_duration(tu)
    }

    /// Construct a new [`DurationChunked`] from an iterator over optional [`ChronoDuration`].
    pub fn from_duration_options<I: IntoIterator<Item = Option<ChronoDuration>>>(
        name: PlSmallStr,
        v: I,
        tu: TimeUnit,
    ) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Microseconds => |v: ChronoDuration| v.num_microseconds().unwrap(),
            TimeUnit::Milliseconds => |v: ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.into_iter().map(|opt| opt.map(func));
        Int64Chunked::from_iter_options(name, vals).into_duration(tu)
    }
}
