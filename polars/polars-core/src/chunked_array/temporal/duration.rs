use crate::prelude::DataType::Duration;
use crate::prelude::*;


impl DurationChunked {
    pub fn time_unit(&self) -> TimeUnit {
        match self.2.as_ref().unwrap() {
            DataType::Duration(tu) => *tu,
            _ => unreachable!(),
        }
    }

    pub fn set_time_unit(&mut self, tu: TimeUnit) {
        self.2 = Some(Duration(tu))
    }
}