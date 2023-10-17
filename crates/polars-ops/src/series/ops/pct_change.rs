use polars_core::prelude::*;
use polars_core::series::ops::NullBehavior;

pub fn pct_change(s: &Series, n: &Series) -> PolarsResult<Series> {
    polars_ensure!(
        n.len() == 1,
        ComputeError: "n must be a single value."
    );

    match s.dtype() {
        DataType::Float64 | DataType::Float32 => {},
        _ => return pct_change(&s.cast(&DataType::Float64)?, n),
    }

    let fill_null_s = s.fill_null(FillNullStrategy::Forward(None))?;

    let n_s = n.cast(&DataType::Int64)?;
    if let Some(n) = n_s.i64()?.get(0) {
        fill_null_s
            .diff(n, NullBehavior::Ignore)?
            .divide(&fill_null_s.shift(n))
    } else {
        Ok(Series::full_null(s.name(), s.len(), s.dtype()))
    }
}

#[cfg(test)]
mod test {
    use polars_core::prelude::Series;

    use super::pct_change;

    #[test]
    fn test_nulls() -> PolarsResult<()> {
        let s = Series::new("", &[Some(1), None, Some(2), None, Some(3)]);
        assert_eq!(
            pct_change(s, Series::new("i", &[1]))?,
            Series::new("", &[None, Some(0.0f64), Some(1.0), Some(0.), Some(0.5)])
        );
        Ok(())
    }

    #[test]
    fn test_same() -> PolarsResult<()> {
        let s = Series::new("", &[Some(1), Some(1), Some(1)]);
        assert_eq!(
            pct_change(s, Series::new("i", &[1]))?,
            Series::new("", &[None, Some(0.0f64), Some(0.0)])
        );
        Ok(())
    }

    #[test]
    fn test_two_periods() -> PolarsResult<()> {
        let s = Series::new("", &[Some(1), Some(2), Some(4), Some(8), Some(16)]);
        assert_eq!(
            pct_change(s, Series::new("i", &[2]))?,
            Series::new("", &[None, None, Some(3.0f64), Some(3.0), Some(3.0)])
        );
        Ok(())
    }
}
