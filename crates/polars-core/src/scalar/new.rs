use std::sync::Arc;

use super::Scalar;
use crate::datatypes::time_unit::TimeUnit;
use crate::prelude::{AnyValue, DataType, TimeZone};

impl Scalar {
    #[cfg(feature = "dtype-date")]
    pub fn new_date(value: i32) -> Self {
        Scalar::new(DataType::Date, AnyValue::Date(value))
    }

    #[cfg(feature = "dtype-datetime")]
    pub fn new_datetime(value: i64, time_unit: TimeUnit, tz: Option<TimeZone>) -> Self {
        Scalar::new(
            DataType::Datetime(time_unit, tz.clone()),
            AnyValue::DatetimeOwned(value, time_unit, tz.map(Arc::new)),
        )
    }

    #[cfg(feature = "dtype-duration")]
    pub fn new_duration(value: i64, time_unit: TimeUnit) -> Self {
        Scalar::new(
            DataType::Duration(time_unit),
            AnyValue::Duration(value, time_unit),
        )
    }
}
