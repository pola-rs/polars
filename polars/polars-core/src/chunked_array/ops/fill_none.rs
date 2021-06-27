use crate::prelude::*;
use num::{Bounded, Num, NumCast, One, Zero};
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

impl<T> ChunkFillNone for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native>
        + PartialOrd
        + Div<Output = T::Native>
        + Num
        + NumCast
        + Zero
        + One
        + Bounded,
{
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        let ca = match strategy {
            FillNoneStrategy::Forward => fill_forward(self),
            FillNoneStrategy::Backward => fill_backward(self),
            FillNoneStrategy::Min => self
                .fill_none_with_value(self.min().ok_or_else(|| {
                    PolarsError::Other("Could not determine fill value".into())
                })?)?,
            FillNoneStrategy::Max => self
                .fill_none_with_value(self.max().ok_or_else(|| {
                    PolarsError::Other("Could not determine fill value".into())
                })?)?,
            FillNoneStrategy::Mean => self.fill_none_with_value(
                self.mean()
                    .map(|v| NumCast::from(v).unwrap())
                    .ok_or_else(|| PolarsError::Other("Could not determine fill value".into()))?,
            )?,
            FillNoneStrategy::One => return self.fill_none_with_value(One::one()),
            FillNoneStrategy::Zero => return self.fill_none_with_value(Zero::zero()),
            FillNoneStrategy::MinBound => return self.fill_none_with_value(Bounded::min_value()),
            FillNoneStrategy::MaxBound => return self.fill_none_with_value(Bounded::max_value()),
        };
        Ok(ca)
    }
}

impl<T> ChunkFillNoneValue<T::Native> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native> + PartialOrd + Div<Output = T::Native> + Num + NumCast,
{
    fn fill_none_with_value(&self, value: T::Native) -> Result<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNone for BooleanChunked {
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        match strategy {
            FillNoneStrategy::Forward => impl_fill_forward!(self),
            FillNoneStrategy::Backward => impl_fill_backward!(self, BooleanChunked),
            FillNoneStrategy::Min => self.fill_none_with_value(
                1 == self
                    .min()
                    .ok_or_else(|| PolarsError::Other("Could not determine fill value".into()))?,
            ),
            FillNoneStrategy::Max => self.fill_none_with_value(
                1 == self
                    .max()
                    .ok_or_else(|| PolarsError::Other("Could not determine fill value".into()))?,
            ),
            FillNoneStrategy::Mean => Err(PolarsError::InvalidOperation(
                "mean not supported on array of Boolean type".into(),
            )),
            FillNoneStrategy::One | FillNoneStrategy::MaxBound => self.fill_none_with_value(true),
            FillNoneStrategy::Zero | FillNoneStrategy::MinBound => self.fill_none_with_value(false),
        }
    }
}

impl ChunkFillNoneValue<bool> for BooleanChunked {
    fn fill_none_with_value(&self, value: bool) -> Result<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNone for Utf8Chunked {
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        match strategy {
            FillNoneStrategy::Forward => impl_fill_forward!(self),
            FillNoneStrategy::Backward => impl_fill_backward!(self, Utf8Chunked),
            strat => Err(PolarsError::InvalidOperation(
                format!("Strategy {:?} not supported", strat).into(),
            )),
        }
    }
}

impl ChunkFillNoneValue<&str> for Utf8Chunked {
    fn fill_none_with_value(&self, value: &str) -> Result<Self> {
        self.set(&self.is_null(), Some(value))
    }
}

impl ChunkFillNone for ListChunked {
    fn fill_none(&self, _strategy: FillNoneStrategy) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_none not supported for List type".into(),
        ))
    }
}

impl ChunkFillNone for CategoricalChunked {
    fn fill_none(&self, _strategy: FillNoneStrategy) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_none not supported for Categorical type".into(),
        ))
    }
}

impl ChunkFillNoneValue<&Series> for ListChunked {
    fn fill_none_with_value(&self, _value: &Series) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_none_with_value not supported for List type".into(),
        ))
    }
}
#[cfg(feature = "object")]
impl<T> ChunkFillNone for ObjectChunked<T> {
    fn fill_none(&self, _strategy: FillNoneStrategy) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_none not supported for Object type".into(),
        ))
    }
}

#[cfg(feature = "object")]
impl<T> ChunkFillNoneValue<ObjectType<T>> for ObjectChunked<T> {
    fn fill_none_with_value(&self, _value: ObjectType<T>) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_none_with_value not supported for Object type".into(),
        ))
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_fill_none() {
        let ca =
            Int32Chunked::new_from_opt_slice("", &[None, Some(2), Some(3), None, Some(4), None]);
        let filled = ca.fill_none(FillNoneStrategy::Forward).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[None, Some(2), Some(3), Some(3), Some(4), Some(4)]
        );
        let filled = ca.fill_none(FillNoneStrategy::Backward).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(2), Some(2), Some(3), Some(4), Some(4), None]
        );
        let filled = ca.fill_none(FillNoneStrategy::Min).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(2), Some(2), Some(3), Some(2), Some(4), Some(2)]
        );
        let filled = ca.fill_none_with_value(10).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(10), Some(2), Some(3), Some(10), Some(4), Some(10)]
        );
        let filled = ca.fill_none(FillNoneStrategy::Mean).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(3), Some(2), Some(3), Some(3), Some(4), Some(3)]
        );
        let ca = Int32Chunked::new_from_opt_slice("", &[None, None, None, None, Some(4), None]);
        let filled = ca.fill_none(FillNoneStrategy::Backward).unwrap();
        assert_eq!(
            Vec::from(&filled),
            &[Some(4), Some(4), Some(4), Some(4), Some(4), None]
        );
    }
}
