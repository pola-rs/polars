use crate::prelude::*;
use arrow::compute;
use arrow::types::simd::Simd;
use num::{Bounded, NumCast, One, Zero};
use polars_arrow::kernels::set::set_at_nulls;
use polars_arrow::utils::CustomIterTools;
use std::ops::Add;

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
        .trust_my_length(ca.len())
        .collect_trusted()
}

macro_rules! impl_fill_forward {
    ($ca:ident) => {{
        $ca.into_iter()
            .scan(None, |previous, opt_v| match opt_v {
                Some(value) => {
                    *previous = Some(value);
                    Some(Some(value))
                }
                None => Some(*previous),
            })
            .trust_my_length($ca.len())
            .collect_trusted()
    }};
}

fn fill_backward<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    ca.into_iter()
        .rev()
        .scan(None, |previous, opt_v| match opt_v {
            Some(value) => {
                *previous = Some(value);
                Some(Some(value))
            }
            None => Some(*previous),
        })
        .trust_my_length(ca.len())
        .collect_reversed()
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
            .trust_my_length($ca.len())
            .collect_trusted();
        ca.into_iter().rev().collect_trusted()
    }};
}

impl<T> ChunkFillNull for ChunkedArray<T>
where
    T: PolarsNumericType,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    fn fill_null(&self, strategy: FillNullStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        let mut ca = match strategy {
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
        ca.rename(self.name());
        Ok(ca)
    }
}

impl<T> ChunkFillNullValue<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
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
            FillNullStrategy::Forward => {
                let mut out: Self = impl_fill_forward!(self);
                out.rename(self.name());
                Ok(out)
            }
            FillNullStrategy::Backward => {
                // TODO: still a double scan. impl collect_reversed for boolean
                let mut out: Self = impl_fill_backward!(self, BooleanChunked);
                out.rename(self.name());
                Ok(out)
            }
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
            FillNullStrategy::Forward => {
                let mut out: Self = impl_fill_forward!(self);
                out.rename(self.name());
                Ok(out)
            }
            FillNullStrategy::Backward => {
                let mut out: Self = impl_fill_backward!(self, Utf8Chunked);
                out.rename(self.name());
                Ok(out)
            }
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

#[cfg(feature = "dtype-categorical")]
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
            Int32Chunked::new_from_opt_slice("a", &[None, Some(2), Some(3), None, Some(4), None]);
        let filled = ca.fill_null(FillNullStrategy::Forward).unwrap();
        assert_eq!(filled.name(), "a");

        assert_eq!(
            Vec::from(&filled),
            &[None, Some(2), Some(3), Some(3), Some(4), Some(4)]
        );
        let filled = ca.fill_null(FillNullStrategy::Backward).unwrap();
        assert_eq!(filled.name(), "a");
        assert_eq!(
            Vec::from(&filled),
            &[Some(2), Some(2), Some(3), Some(4), Some(4), None]
        );
        let filled = ca.fill_null(FillNullStrategy::Min).unwrap();
        assert_eq!(filled.name(), "a");
        assert_eq!(
            Vec::from(&filled),
            &[Some(2), Some(2), Some(3), Some(2), Some(4), Some(2)]
        );
        let filled = ca.fill_null_with_values(10).unwrap();
        assert_eq!(filled.name(), "a");
        assert_eq!(
            Vec::from(&filled),
            &[Some(10), Some(2), Some(3), Some(10), Some(4), Some(10)]
        );
        let filled = ca.fill_null(FillNullStrategy::Mean).unwrap();
        assert_eq!(filled.name(), "a");
        assert_eq!(
            Vec::from(&filled),
            &[Some(3), Some(2), Some(3), Some(3), Some(4), Some(3)]
        );
        let ca = Int32Chunked::new_from_opt_slice("a", &[None, None, None, None, Some(4), None]);
        let filled = ca.fill_null(FillNullStrategy::Backward).unwrap();
        assert_eq!(filled.name(), "a");
        assert_eq!(
            Vec::from(&filled),
            &[Some(4), Some(4), Some(4), Some(4), Some(4), None]
        );
    }
}
