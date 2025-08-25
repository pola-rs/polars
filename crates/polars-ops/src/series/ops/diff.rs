use polars_core::prelude::*;
use polars_core::series::ops::NullBehavior;

pub fn diff(s: &Series, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
    match null_behavior {
        NullBehavior::Ignore => s.clone() - s.shift(n),
        NullBehavior::Drop => {
            polars_ensure!(n > 0, InvalidOperation: "only positive integer allowed if nulls are dropped in 'diff' operation");
            let n = n as usize;
            let len = s.len() - n;
            s.slice(n as i64, len) - s.slice(0, len)
        },
    }
}
