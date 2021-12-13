use std::ops::Deref;

#[derive(Copy, Clone, Debug)]
pub struct TimeNanoseconds(pub i64);

impl Deref for TimeNanoseconds {
    type Target = i64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<i64> for TimeNanoseconds {
    fn from(v: i64) -> Self {
        TimeNanoseconds(v)
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TimeMilliseconds(i64);

impl Deref for TimeMilliseconds {
    type Target = i64;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<i64> for TimeMilliseconds {
    fn from(v: i64) -> Self {
        TimeMilliseconds(v)
    }
}

impl TimeNanoseconds {
    pub fn to_millisecs(self) -> TimeMilliseconds {
        (*self / 1000_0000).into()
    }
}

impl TimeMilliseconds {
    pub fn to_nsecs(self) -> TimeNanoseconds {
        (*self * 1000_0000).into()
    }
}
