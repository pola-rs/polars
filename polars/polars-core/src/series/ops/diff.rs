use crate::prelude::*;
use crate::series::ops::NullBehavior;

impl Series {
    pub fn diff(&self, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
        use DataType::*;
        let s = match self.dtype() {
            UInt8 => self.cast(&Int16).unwrap(),
            UInt16 => self.cast(&Int32).unwrap(),
            UInt32 | UInt64 => self.cast(&Int64).unwrap(),
            _ => self.clone(),
        };

        match null_behavior {
            NullBehavior::Ignore => Ok(&s - &s.shift(n)),
            NullBehavior::Drop => {
                polars_ensure!(n > 0, InvalidOperation: "only positive integer allowed if nulls are dropped in 'diff' operation");
                let n = n as usize;
                let len = s.len() - n;
                Ok(&self.slice(n as i64, len) - &s.slice(0, len))
            }
        }
    }
}
