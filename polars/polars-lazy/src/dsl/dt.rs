use super::*;
use polars_core::prelude::DataType::{Datetime, Duration};

/// Specialized expressions for [`Series`] with dates/datetimes.
pub struct DateLikeNameSpace(pub(crate) Expr);

impl DateLikeNameSpace {
    /// Format Date/datetime with a formatting rule
    /// See [chrono strftime/strptime](https://docs.rs/chrono/0.4.19/chrono/format/strftime/index.html).
    pub fn strftime(self, fmt: &str) -> Expr {
        let fmt = fmt.to_string();
        let function = move |s: Series| s.strftime(&fmt);
        self.0
            .map(function, GetOutput::from_type(DataType::Utf8))
            .with_fmt("strftime")
    }

    /// Change the underlying [`TimeUnit`]. And update the data accordingly.
    pub fn cast_time_unit(self, tu: TimeUnit) -> Expr {
        self.0.map(
            move |s| match s.dtype() {
                DataType::Datetime(_, _) => {
                    let ca = s.datetime().unwrap();
                    Ok(ca.cast_time_unit(tu).into_series())
                }
                DataType::Duration(_) => {
                    let ca = s.duration().unwrap();
                    Ok(ca.cast_time_unit(tu).into_series())
                }
                dt => Err(PolarsError::ComputeError(
                    format!("Series of dtype {:?} has got no time unit", dt).into(),
                )),
            },
            GetOutput::map_dtype(move |dtype| match dtype {
                DataType::Duration(_) => Duration(tu),
                DataType::Datetime(_, tz) => Datetime(tu, tz.clone()),
                _ => panic!("expected duration or datetime"),
            }),
        )
    }

    /// Change the underlying [`TimeUnit`] of the [`Series`]. This does not modify the data.
    pub fn with_time_unit(self, tu: TimeUnit) -> Expr {
        self.0.map(
            move |s| match s.dtype() {
                DataType::Datetime(_, _) => {
                    let mut ca = s.datetime().unwrap().clone();
                    ca.set_time_unit(tu);
                    Ok(ca.into_series())
                }
                DataType::Duration(_) => {
                    let mut ca = s.duration().unwrap().clone();
                    ca.set_time_unit(tu);
                    Ok(ca.into_series())
                }
                dt => Err(PolarsError::ComputeError(
                    format!("Series of dtype {:?} has got no time unit", dt).into(),
                )),
            },
            GetOutput::same_type(),
        )
    }
}
