use crate::export::chrono::Duration as ChronoDuration;
use crate::fmt::fmt_duration_string;
use crate::prelude::DataType::Duration;
use crate::prelude::*;

impl DurationChunked {
    pub fn time_unit(&self) -> TimeUnit {
        match self.2.as_ref().unwrap() {
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

        use TimeUnit::*;
        match (current_unit, tu) {
            (Nanoseconds, Microseconds) => {
                let ca = (&self.0).wrapping_trunc_div_scalar(1_000);
                out.0 = ca;
                out
            },
            (Nanoseconds, Milliseconds) => {
                let ca = (&self.0).wrapping_trunc_div_scalar(1_000_000);
                out.0 = ca;
                out
            },
            (Microseconds, Nanoseconds) => {
                let ca = &self.0 * 1_000;
                out.0 = ca;
                out
            },
            (Microseconds, Milliseconds) => {
                let ca = (&self.0).wrapping_trunc_div_scalar(1_000);
                out.0 = ca;
                out
            },
            (Milliseconds, Nanoseconds) => {
                let ca = &self.0 * 1_000_000;
                out.0 = ca;
                out
            },
            (Milliseconds, Microseconds) => {
                let ca = &self.0 * 1_000;
                out.0 = ca;
                out
            },
            (Nanoseconds, Nanoseconds)
            | (Microseconds, Microseconds)
            | (Milliseconds, Milliseconds) => out,
        }
    }

    /// Change the underlying [`TimeUnit`]. This does not modify the data.
    pub fn set_time_unit(&mut self, tu: TimeUnit) {
        self.2 = Some(Duration(tu))
    }

    /// Convert from [`Duration`] to String; note that `strftime` format
    /// strings are not supported, only the specifiers 'iso' and 'polars'.
    pub fn to_string(&self, format: &str) -> PolarsResult<StringChunked> {
        match format {
            "iso" | "polars" => {
                let out: StringChunked = self
                    .0
                    .apply_nonnull_values_generic(DataType::String, |v: i64| {
                        fmt_duration_string(v, self.time_unit(), format == "iso")
                    });
                Ok(out)
            },
            _ => Err(PolarsError::InvalidOperation(
                format!("format {:?} not supported for Duration type (expected one of 'iso' or 'polars')", format).into(),
            )),
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
