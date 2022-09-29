use std::ops::Add;

use arrow::compute;
use arrow::types::simd::Simd;
use num::{Bounded, NumCast, One, Zero};
use polars_arrow::kernels::set::set_at_nulls;
use polars_arrow::utils::CustomIterTools;

use crate::prelude::*;

fn fill_forward_limit<T>(ca: &ChunkedArray<T>, limit: IdxSize) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    let mut cnt = 0;
    let mut previous = None;
    ca.into_iter()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v);
                Some(v)
            }
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous
                } else {
                    None
                }
            }
        })
        .collect_trusted()
}

fn fill_backward_limit<T>(ca: &ChunkedArray<T>, limit: IdxSize) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    let mut cnt = 0;
    let mut previous = None;
    ca.into_iter()
        .rev()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v);
                Some(v)
            }
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous
                } else {
                    None
                }
            }
        })
        .collect_reversed()
}

fn fill_backward_limit_bool(ca: &BooleanChunked, limit: IdxSize) -> BooleanChunked {
    let mut cnt = 0;
    let mut previous = None;
    ca.into_iter()
        .rev()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v);
                Some(v)
            }
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous
                } else {
                    None
                }
            }
        })
        .collect_reversed()
}

fn fill_backward_limit_utf8(ca: &Utf8Chunked, limit: IdxSize) -> Utf8Chunked {
    let mut cnt = 0;
    let mut previous = None;
    let out: Utf8Chunked = ca
        .into_iter()
        .rev()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v);
                Some(v)
            }
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous
                } else {
                    None
                }
            }
        })
        .collect_trusted();
    out.into_iter().rev().collect_trusted()
}

#[cfg(feature = "dtype-binary")]
fn fill_backward_limit_binary(ca: &BinaryChunked, limit: IdxSize) -> BinaryChunked {
    let mut cnt = 0;
    let mut previous = None;
    let out: BinaryChunked = ca
        .into_iter()
        .rev()
        .map(|opt_v| match opt_v {
            Some(v) => {
                cnt = 0;
                previous = Some(v);
                Some(v)
            }
            None => {
                if cnt < limit {
                    cnt += 1;
                    previous
                } else {
                    None
                }
            }
        })
        .collect_trusted();
    out.into_iter().rev().collect_trusted()
}

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
            .collect_trusted()
    }};
}

macro_rules! impl_fill_forward_limit {
    ($ca:ident, $limit:expr) => {{
        let mut cnt = 0;
        let mut previous = None;
        $ca.into_iter()
            .map(|opt_v| match opt_v {
                Some(v) => {
                    cnt = 0;
                    previous = Some(v);
                    Some(v)
                }
                None => {
                    if cnt < $limit {
                        cnt += 1;
                        previous
                    } else {
                        None
                    }
                }
            })
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
        .collect_reversed()
}

fn fill_backward_bool(ca: &BooleanChunked) -> BooleanChunked {
    ca.into_iter()
        .rev()
        .scan(None, |previous, opt_v| match opt_v {
            Some(value) => {
                *previous = Some(value);
                Some(Some(value))
            }
            None => Some(*previous),
        })
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
    fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        // nothing to fill
        if !self.has_validity() {
            return Ok(self.clone());
        }
        let mut ca = match strategy {
            FillNullStrategy::Forward(None) => fill_forward(self),
            FillNullStrategy::Forward(Some(limit)) => fill_forward_limit(self, limit),
            FillNullStrategy::Backward(None) => fill_backward(self),
            FillNullStrategy::Backward(Some(limit)) => fill_backward_limit(self, limit),
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
    fn fill_null_with_values(&self, value: T::Native) -> PolarsResult<Self> {
        Ok(self.apply_kernel(&|arr| Box::new(set_at_nulls(arr, value))))
    }
}

impl ChunkFillNull for BooleanChunked {
    fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        // nothing to fill
        if !self.has_validity() {
            return Ok(self.clone());
        }
        match strategy {
            FillNullStrategy::Forward(limit) => {
                let mut out: Self = match limit {
                    Some(limit) => impl_fill_forward_limit!(self, limit),
                    None => impl_fill_forward!(self),
                };
                out.rename(self.name());
                Ok(out)
            }
            FillNullStrategy::Backward(limit) => {
                let mut out: Self = match limit {
                    None => fill_backward_bool(self),
                    Some(limit) => fill_backward_limit_bool(self, limit),
                };
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
    fn fill_null_with_values(&self, value: bool) -> PolarsResult<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNull for Utf8Chunked {
    fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        // nothing to fill
        if !self.has_validity() {
            return Ok(self.clone());
        }
        match strategy {
            FillNullStrategy::Forward(limit) => {
                let mut out: Self = match limit {
                    Some(limit) => impl_fill_forward_limit!(self, limit),
                    None => impl_fill_forward!(self),
                };
                out.rename(self.name());
                Ok(out)
            }
            FillNullStrategy::Backward(limit) => {
                let mut out = match limit {
                    None => impl_fill_backward!(self, Utf8Chunked),
                    Some(limit) => fill_backward_limit_utf8(self, limit),
                };
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
    fn fill_null_with_values(&self, value: &str) -> PolarsResult<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkFillNull for BinaryChunked {
    fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        // nothing to fill
        if !self.has_validity() {
            return Ok(self.clone());
        }
        match strategy {
            FillNullStrategy::Forward(limit) => {
                let mut out: Self = match limit {
                    Some(limit) => impl_fill_forward_limit!(self, limit),
                    None => impl_fill_forward!(self),
                };
                out.rename(self.name());
                Ok(out)
            }
            FillNullStrategy::Backward(limit) => {
                let mut out = match limit {
                    None => impl_fill_backward!(self, BinaryChunked),
                    Some(limit) => fill_backward_limit_binary(self, limit),
                };
                out.rename(self.name());
                Ok(out)
            }
            strat => Err(PolarsError::InvalidOperation(
                format!("Strategy {:?} not supported", strat).into(),
            )),
        }
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkFillNullValue<&[u8]> for BinaryChunked {
    fn fill_null_with_values(&self, value: &[u8]) -> PolarsResult<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNull for ListChunked {
    fn fill_null(&self, _strategy: FillNullStrategy) -> PolarsResult<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_null not supported for List type".into(),
        ))
    }
}

impl ChunkFillNullValue<&Series> for ListChunked {
    fn fill_null_with_values(&self, _value: &Series) -> PolarsResult<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_null_with_value not supported for List type".into(),
        ))
    }
}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkFillNull for ObjectChunked<T> {
    fn fill_null(&self, _strategy: FillNullStrategy) -> PolarsResult<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_null not supported for Object type".into(),
        ))
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkFillNullValue<ObjectType<T>> for ObjectChunked<T> {
    fn fill_null_with_values(&self, _value: ObjectType<T>) -> PolarsResult<Self> {
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
        let ca = Int32Chunked::new("a", &[None, Some(2), Some(3), None, Some(4), None]);
        let filled = ca.fill_null(FillNullStrategy::Forward(None)).unwrap();
        assert_eq!(filled.name(), "a");

        assert_eq!(
            Vec::from(&filled),
            &[None, Some(2), Some(3), Some(3), Some(4), Some(4)]
        );
        let filled = ca.fill_null(FillNullStrategy::Backward(None)).unwrap();
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
        let ca = Int32Chunked::new("a", &[None, None, None, None, Some(4), None]);
        let filled = ca.fill_null(FillNullStrategy::Backward(None)).unwrap();
        assert_eq!(filled.name(), "a");
        assert_eq!(
            Vec::from(&filled),
            &[Some(4), Some(4), Some(4), Some(4), Some(4), None]
        );
    }
}
