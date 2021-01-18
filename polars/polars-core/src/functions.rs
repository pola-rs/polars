use crate::prelude::*;
use num::{Float, NumCast};
use std::ops::Div;

// todo! make numerical stable from catastrophic cancellation
pub fn cov<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<T::Native>
where
    T: PolarsFloatType,
    T::Native: Float + Div + NumCast,
{
    if a.len() != b.len() {
        None
    } else {
        Some((&(a - a.mean()?) * &(b - b.mean()?)).sum()? / NumCast::from(a.len() - 1).unwrap())
    }
}
pub fn pearson_corr<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<T::Native>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: ChunkVar<T::Native>,
{
    Some(cov(a, b)? / (a.std()? * b.std()?))
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_pearson_corr() {
        let a = Series::new("a", &[1.0f32, 2.0]);
        let b = Series::new("b", &[1.0f32, 2.0]);
        assert!((cov(&a.f32().unwrap(), &b.f32().unwrap()).unwrap() - 0.5).abs() < 0.001);
        assert!((pearson_corr(&a.f32().unwrap(), &b.f32().unwrap()).unwrap() - 1.0).abs() < 0.001);
    }
}
