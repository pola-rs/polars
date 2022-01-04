use crate::export::chrono::Duration as ChronoDuration;
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

    pub fn new_from_duration(name: &str, v: &[ChronoDuration], tu: TimeUnit) -> Self {
        let func = match tu {
            TimeUnit::Nanoseconds => |v: &ChronoDuration| v.num_nanoseconds().unwrap(),
            TimeUnit::Milliseconds => |v: &ChronoDuration| v.num_milliseconds(),
        };
        let vals = v.iter().map(func).collect_trusted::<Vec<_>>();
        Int64Chunked::new_from_aligned_vec(name, vals).into_duration(tu)
    }
}
