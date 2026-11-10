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
        // `drop` keeps only positions where both sides of the difference exist.
        // When |n| is longer than the series there are no such positions; subtracting
        // the lengths used to overflow (debug panic) or wrap (release length error).
        NullBehavior::Drop => {
            let offset = n.unsigned_abs() as usize;
            if offset > s.len() {
                return Ok(s.slice(0, 0));
            }
            let len = s.len() - offset;
            if n < 0 {
                &s.slice(0, len) - &s.slice(offset as i64, len)
            } else {
                &s.slice(offset as i64, len) - &s.slice(0, len)
            }
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn drop_diff_longer_than_series_is_empty() {
        let s = Series::new("a".into(), &[1.0f64, 2.0]);

        let dropped = diff(&s, 3, NullBehavior::Drop).unwrap();
        assert_eq!(dropped.len(), 0);
        assert_eq!(dropped.dtype(), s.dtype());

        let dropped_neg = diff(&s, -3, NullBehavior::Drop).unwrap();
        assert_eq!(dropped_neg.len(), 0);
        assert_eq!(dropped_neg.dtype(), s.dtype());

        // |n| == len already produced an empty series; keep that.
        let exact = diff(&s, 2, NullBehavior::Drop).unwrap();
        assert_eq!(exact.len(), 0);

        let empty = Series::new("a".into(), &[] as &[f64]);
        let dropped_empty = diff(&empty, 1, NullBehavior::Drop).unwrap();
        assert_eq!(dropped_empty.len(), 0);
        assert_eq!(dropped_empty.dtype(), empty.dtype());
    }
}
