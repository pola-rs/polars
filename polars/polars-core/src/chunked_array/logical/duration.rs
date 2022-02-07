use super::*;
use crate::prelude::*;

pub type DurationChunked = Logical<DurationType, Int64Type>;

impl Int64Chunked {
    pub fn into_duration(self, timeunit: TimeUnit) -> DurationChunked {
        let mut dt = DurationChunked::new(self);
        dt.2 = Some(DataType::Duration(timeunit));
        dt
    }
}

impl LogicalType for DurationChunked {
    fn dtype(&self) -> &DataType {
        self.2.as_ref().unwrap()
    }

    #[cfg(feature = "dtype-duration")]
    fn get_any_value(&self, i: usize) -> AnyValue<'_> {
        self.0.get_any_value(i).into_duration(self.time_unit())
    }
}
