use crate::datatypes::{AnyValue, TimeUnit};
#[cfg(feature = "dtype-date")]
use crate::prelude::MS_IN_DAY;
use crate::prelude::{DataType, Scalar};

pub fn mean_reduce(value: Option<f64>, dtype: DataType) -> Scalar {
    match dtype {
        DataType::Float32 => {
            let val = value.map(|m| m as f32);
            Scalar::new(dtype, val.into())
        },
        dt if dt.is_numeric() || dt.is_decimal() || dt.is_bool() => {
            Scalar::new(DataType::Float64, value.into())
        },
        #[cfg(feature = "dtype-date")]
        DataType::Date => {
            let val = value.map(|v| (v * MS_IN_DAY as f64) as i64);
            Scalar::new(DataType::Datetime(TimeUnit::Milliseconds, None), val.into())
        },
        #[cfg(feature = "dtype-datetime")]
        dt @ DataType::Datetime(_, _) => {
            let val = value.map(|v| v as i64);
            Scalar::new(dt, val.into())
        },
        #[cfg(feature = "dtype-duration")]
        dt @ DataType::Duration(_) => {
            let val = value.map(|v| v as i64);
            Scalar::new(dt, val.into())
        },
        #[cfg(feature = "dtype-time")]
        dt @ DataType::Time => {
            let val = value.map(|v| v as i64);
            Scalar::new(dt, val.into())
        },
        dt => Scalar::new(dt, AnyValue::Null),
    }
}
