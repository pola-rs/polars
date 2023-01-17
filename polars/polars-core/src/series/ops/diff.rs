use crate::prelude::*;
use crate::series::ops::NullBehavior;

impl Series {
    pub fn diff(&self, n: usize, null_behavior: NullBehavior) -> Series {
        use DataType::*;
        let s = match self.dtype() {
            UInt8 => self.cast(&Int16).unwrap(),
            UInt16 => self.cast(&Int32).unwrap(),
            UInt32 | UInt64 => self.cast(&Int64).unwrap(),
            _ => self.clone(),
        };

        match null_behavior {
            NullBehavior::Ignore => &s - &s.shift(n as i64),
            NullBehavior::Drop => {
                let len = s.len() - n;
                &self.slice(n as i64, len) - &s.slice(0, len)
            }
        }
    }
}
