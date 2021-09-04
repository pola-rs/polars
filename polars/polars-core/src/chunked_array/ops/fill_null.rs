use crate::prelude::*;
use arrow::compute;
use arrow::types::simd::Simd;
use arrow::types::NativeType;
use num::{Bounded, Num, NumCast, One, Zero};
use polars_arrow::kernels::set::set_at_nulls;
use std::ops::{Add, Div};

fn fill_forward<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    ca.into_iter()
        .scan(None, |previous, opt_v| match opt_v {
            Some(value) => {
                *previous = Some(value);
                Some(Some(value))
            }
            None => Some(*previous),
        })
        .collect()
}

macro_rules! impl_fill_forward {
    ($ca:ident) => {{
        let ca = $ca
            .into_iter()
            .scan(None, |previous, opt_v| match opt_v {
                Some(value) => {
                    *previous = Some(value);
                    Some(Some(value))
                }
                None => Some(*previous),
            })
            .collect();
        Ok(ca)
    }};
}

fn fill_backward<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    // TODO! improve performance. This is a double scan
    let ca: ChunkedArray<T> = ca
        .into_iter()
        .rev()
        .scan(None, |previous, opt_v| match opt_v {
            Some(value) => {
                *previous = Some(value);
                Some(Some(value))
            }
            None => Some(*previous),
        })
        .collect();
    ca.into_iter().rev().collect()
}

macro_rules! impl_fill_backward {
    ($ca:ident, $ChunkedArray:ty) => {{
        let ca: $ChunkedArray = $ca
            .into_iter()
            .rev()
            .scan(None, |previous, opt_v| match opt_v {
                Some(value) => {
                    *previous = Some(value);
                    Some(Some(value))
                }
                None => Some(*previous),
            })
            .collect();
        Ok(ca.into_iter().rev().collect())
    }};
}

impl<T> ChunkFillNull for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NativeType
        + PartialOrd
        + Num
        + NumCast
        + Zero
        + Simd
        + One
        + Bounded
        + Add<Output = T::Native>
        + Div<Output = T::Native>,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    fn fill_null(&self, strategy: FillNullStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        let ca = match strategy {
            FillNullStrategy::Forward => fill_forward(self),
            FillNullStrategy::Backward => fill_backward(self),
            FillNullStrategy::Min => {
                self.fill_null_with_values(self.min().ok_or_else(|| {
                    PolarsError::ComputeError("Could not determine fill value".into())
                })?)?
            }
            FillNullStrategy::Max => {
                self.fill_null_with_values(self.max().ok_or_else(|| {
                    PolarsError::ComputeError("Could not determine fill value".into())
                })?)?
            }
            FillNullStrategy::Mean => self.fill_null_with_values(
                self.mean()
                    .map(|v| NumCast::from(v).unwrap())
                    .ok_or_else(|| {
                        PolarsError::ComputeError("Could not determine fill value".into())
                    })?,
            )?,
            FillNullStrategy::One => return self.fill_null_with_values(One::one()),
            FillNullStrategy::Zero => return self.fill_null_with_values(Zero::zero()),
            FillNullStrategy::MinBound => return self.fill_null_with_values(Bounded::min_value()),
            FillNullStrategy::MaxBound => return self.fill_null_with_values(Bounded::max_value()),
        };
        Ok(ca)
    }
}

impl<T> ChunkFillNullValue<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native> + PartialOrd + Div<Output = T::Native> + Num + NumCast,
{
    fn fill_null_with_values(&self, value: T::Native) -> Result<Self> {
        Ok(self.apply_kernel(|arr| Arc::new(set_at_nulls(arr, value))))
    }
}

impl ChunkFillNull for BooleanChunked {
    fn fill_null(&self, strategy: FillNullStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        match strategy {
            FillNullStrategy::Forward => impl_fill_forward!(self),
            FillNullStrategy::Backward => impl_fill_backward!(self, BooleanChunked),
            FillNullStrategy::Min => self.fill_null_with_values(
                1 == self.min().ok_or_else(|| {
                    PolarsError::ComputeError("Could not determine fill value".into())
                })?,
            ),
            FillNullStrategy::Max => self.fill_null_with_values(
                1 == self.max().ok_or_else(|| {
                    PolarsError::ComputeError("Could not determine fill value".into())
                })?,
            ),
            FillNullStrategy::Mean => Err(PolarsError::InvalidOperation(
                "mean not supported on array of Boolean type".into(),
            )),
            FillNullStrategy::One | FillNullStrategy::MaxBound => self.fill_null_with_values(true),
            FillNullStrategy::Zero | FillNullStrategy::MinBound => {
                self.fill_null_with_values(false)
            }
        }
    }
}

impl ChunkFillNullValue<bool> for BooleanChunked {
    fn fill_null_with_values(&self, value: bool) -> Result<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNull for Utf8Chunked {
    fn fill_null(&self, strategy: FillNullStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        match strategy {
            FillNullStrategy::Forward => impl_fill_forward!(self),
            FillNullStrategy::Backward => impl_fill_backward!(self, Utf8Chunked),
            strat => Err(PolarsError::InvalidOperation(
                format!("Strategy {:?} not supported", strat).into(),
            )),
        }
    }
}

impl ChunkFillNullValue<&str> for Utf8Chunked {
    fn fill_null_with_values(&self, value: &str) -> Result<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNull for ListChunked {
    fn fill_null(&self, _strategy: FillNullStrategy) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_null not supported for List type".into(),
        ))
    }
}

impl ChunkFillNull for CategoricalChunked {
    fn fill_null(&self, _strategy: FillNullStrategy) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_null not supported for Categorical type".into(),
        ))
    }
}

impl ChunkFillNullValue<&Series> for ListChunked {
    fn fill_null_with_values(&self, _value: &Series) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_null_with_value not supported for List type".into(),
        ))
    }
}
#[cfg(feature = "object")]
impl<T> ChunkFillNull for ObjectChunked<T> {
    fn fill_null(&self, _strategy: FillNullStrategy) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_null not supported for Object type".into(),
        ))
    }
}

#[cfg(feature = "object")]
impl<T> ChunkFillNullValue<ObjectType<T>> for ObjectChunked<T> {
    fn fill_null_with_values(&self, _value: ObjectType<T>) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_null_with_value not supported for Object type".into(),
        ))
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_fill_null() {
        let ca =
            Int32Chunked::new_from_opt_slice("", &[None, Some(2), Some(3), None, Some(4), None]);
        let filled = ca.fill_null(FillNullStrategy::Forward).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[None, Some(2), Some(3), Some(3), Some(4), Some(4)]
        );
        let filled = ca.fill_null(FillNullStrategy::Backward).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(2), Some(2), Some(3), Some(4), Some(4), None]
        );
        let filled = ca.fill_null(FillNullStrategy::Min).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(2), Some(2), Some(3), Some(2), Some(4), Some(2)]
        );
        let filled = ca.fill_null_with_values(10).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(10), Some(2), Some(3), Some(10), Some(4), Some(10)]
        );
        let filled = ca.fill_null(FillNullStrategy::Mean).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(3), Some(2), Some(3), Some(3), Some(4), Some(3)]
        );
        let ca = Int32Chunked::new_from_opt_slice("", &[None, None, None, None, Some(4), None]);
        let filled = ca.fill_null(FillNullStrategy::Backward).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(4), Some(4), Some(4), Some(4), Some(4), None]
        );
    }
}
