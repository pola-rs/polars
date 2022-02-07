use super::*;
use crate::prelude::*;

pub type DatetimeChunked = Logical<DatetimeType, Int64Type>;

impl Int64Chunked {
    pub fn into_datetime(self, timeunit: TimeUnit, tz: Option<TimeZone>) -> DatetimeChunked {
        let mut dt = DatetimeChunked::new(self);
        dt.2 = Some(DataType::Datetime(timeunit, tz));
        dt
    }
}

impl LogicalType for DatetimeChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    #[cfg(feature = "dtype-datetime")]
    fn get_any_value(&self, i: usize) -> AnyValue<'_> {
        self.0
            .get_any_value(i)
            .into_datetime(self.time_unit(), self.time_zone())
    }
}
