use crate::prelude::*;
use crate::series::ops::NullBehavior;

impl Series {
    pub fn pct_change(&self, n: usize) -> PolarsResult<Series> {
        match self.dtype() {
            DataType::Float64 | DataType::Float32 => {}
            _ => return self.cast(&DataType::Float64)?.pct_change(n),
        }
        let nn = self.fill_null(FillNullStrategy::Forward(None))?;
        nn.diff(n, NullBehavior::Ignore).divide(&nn.shift(n as i64))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_nulls() -> PolarsResult<()> {
        let s = Series::new("", &[Some(1), None, Some(2), None, Some(3)]);
        assert_eq!(
            s.pct_change(1)?,
            Series::new("", &[None, Some(0.0f64), Some(1.0), Some(0.), Some(0.5)])
        );
        Ok(())
    }

    #[test]
    fn test_same() -> PolarsResult<()> {
        let s = Series::new("", &[Some(1), Some(1), Some(1)]);
        assert_eq!(
            s.pct_change(1)?,
            Series::new("", &[None, Some(0.0f64), Some(0.0)])
        );
        Ok(())
    }

    #[test]
    fn test_two_periods() -> PolarsResult<()> {
        let s = Series::new("", &[Some(1), Some(2), Some(4), Some(8), Some(16)]);
        assert_eq!(
            s.pct_change(2)?,
            Series::new("", &[None, None, Some(3.0f64), Some(3.0), Some(3.0)])
        );
        Ok(())
    }
}
