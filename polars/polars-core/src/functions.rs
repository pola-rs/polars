//! # Functions
//!
//! Functions that might be useful.
//!
#[cfg(feature = "sort_multiple")]
use crate::chunked_array::ops::sort::prepare_argsort;
use crate::prelude::*;
use num::{Float, NumCast};
use std::ops::Div;

/// Compute the covariance between two columns.
pub fn cov<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<T::Native>
where
    T: PolarsFloatType,
    T::Native: Float + Div + NumCast,
{
    if a.len() != b.len() {
        None
    } else {
        let tmp = (a - a.mean()?) * (b - b.mean()?);
        let n = tmp.len() - tmp.null_count();
        Some(tmp.sum()? / NumCast::from(n - 1).unwrap())
    }
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<T::Native>
where
    T: PolarsFloatType,
    T::Native: Float,
    ChunkedArray<T>: ChunkVar<T::Native>,
{
    Some(cov(a, b)? / (a.std()? * b.std()?))
}

#[cfg(feature = "sort_multiple")]
/// Find the indexes that would sort these series in order of appearance.
/// That means that the first `Series` will be used to determine the ordering
/// until duplicates are found. Once duplicates are found, the next `Series` will
/// be used and so on.
pub fn argsort_by(by: &[Series], reverse: &[bool]) -> Result<UInt32Chunked> {
    if by.len() != reverse.len() {
        return Err(PolarsError::ValueError(
            format!(
                "The amount of ordering booleans: {} does not match amount of Series: {}",
                reverse.len(),
                by.len()
            )
            .into(),
        ));
    }
    let (first, by, reverse) =
        prepare_argsort(by.to_vec(), reverse.iter().copied().collect()).unwrap();
    first.argsort_multiple(&by, &reverse)
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
