use polars_core::prelude::*;
use polars_core::series::ops::NullBehavior;

pub fn diff(s: &Series, n: i64, null_behavior: NullBehavior) -> PolarsResult<Series> {
    use DataType::*;
    let s = match s.dtype() {
        UInt8 => s.cast(&Int16)?,
        UInt16 => s.cast(&Int32)?,
        UInt32 | UInt64 => s.cast(&Int64)?,
        _ => s.clone(),
    };

    match null_behavior {
        NullBehavior::Ignore => &s - &s.shift(n),
        NullBehavior::Drop => {
            polars_ensure!(n > 0, InvalidOperation: "only positive integer allowed if nulls are dropped in 'diff' operation");
            let n = n as usize;
            let len = s.len() - n;
            &s.slice(n as i64, len) - &s.slice(0, len)
        },
    }
}
