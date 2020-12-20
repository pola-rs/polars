use crate::prelude::*;
use num::{Num, NumCast};
use std::ops::{Add, Div};

fn fill_forward<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    ca.into_iter()
        .scan(None, |previous, opt_v| {
            let val = match opt_v {
                Some(_) => Some(opt_v),
                None => Some(*previous),
            };
            *previous = opt_v;
            val
        })
        .collect()
}

macro_rules! impl_fill_forward {
    ($ca:ident) => {{
        let ca = $ca
            .into_iter()
            .scan(None, |previous, opt_v| {
                let val = match opt_v {
                    Some(_) => Some(opt_v),
                    None => Some(*previous),
                };
                *previous = opt_v;
                val
            })
            .collect();
        Ok(ca)
    }};
}

fn fill_backward<T>(ca: &ChunkedArray<T>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    let mut iter = ca.into_iter().peekable();

    let mut builder = PrimitiveChunkedBuilder::<T>::new(ca.name(), ca.len());
    while let Some(opt_v) = iter.next() {
        match opt_v {
            Some(v) => builder.append_value(v),
            None => {
                match iter.peek() {
                    // end of iterator
                    None => builder.append_null(),
                    Some(opt_v) => builder.append_option(*opt_v),
                }
            }
        }
    }
    builder.finish()
}

macro_rules! impl_fill_backward {
    ($ca:ident, $builder:ident) => {{
        let mut iter = $ca.into_iter().peekable();

        while let Some(opt_v) = iter.next() {
            match opt_v {
                Some(v) => $builder.append_value(v),
                None => {
                    match iter.peek() {
                        // end of iterator
                        None => $builder.append_null(),
                        Some(opt_v) => $builder.append_option(*opt_v),
                    }
                }
            }
        }
        Ok($builder.finish())
    }};
}

fn fill_value<T>(ca: &ChunkedArray<T>, value: Option<T::Native>) -> ChunkedArray<T>
where
    T: PolarsNumericType,
{
    ca.into_iter()
        .map(|opt_v| match opt_v {
            Some(_) => opt_v,
            None => value,
        })
        .collect()
}

macro_rules! impl_fill_value {
    ($ca:ident, $value:expr) => {{
        $ca.into_iter()
            .map(|opt_v| match opt_v {
                Some(_) => opt_v,
                None => $value,
            })
            .collect()
    }};
}

impl<T> ChunkFillNone for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: Add<Output = T::Native> + PartialOrd + Div<Output = T::Native> + Num + NumCast,
{
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        let ca = match strategy {
            FillNoneStrategy::Forward => fill_forward(self),
            FillNoneStrategy::Backward => fill_backward(self),
            FillNoneStrategy::Min => impl_fill_value!(self, self.min()),
            FillNoneStrategy::Max => impl_fill_value!(self, self.max()),
            FillNoneStrategy::Mean => impl_fill_value!(self, self.mean()),
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
        Ok(impl_fill_value!(self, Some(value)))
    }
}

impl ChunkFillNone for BooleanChunked {
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        let mut builder = PrimitiveChunkedBuilder::<BooleanType>::new(self.name(), self.len());
        match strategy {
            FillNoneStrategy::Forward => impl_fill_forward!(self),
            FillNoneStrategy::Backward => impl_fill_backward!(self, builder),
            FillNoneStrategy::Min => Ok(impl_fill_value!(self, self.min().map(|v| v != 0))),
            FillNoneStrategy::Max => Ok(impl_fill_value!(self, self.max().map(|v| v != 0))),
            FillNoneStrategy::Mean => Ok(impl_fill_value!(self, self.mean().map(|v| v != 0))),
        }
    }
}

impl ChunkFillNoneValue<bool> for BooleanChunked {
    fn fill_none_with_value(&self, value: bool) -> Result<Self> {
        Ok(impl_fill_value!(self, Some(value)))
    }
}

impl ChunkFillNone for Utf8Chunked {
    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self> {
        // nothing to fill
        if self.null_count() == 0 {
            return Ok(self.clone());
        }
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len());
        match strategy {
            FillNoneStrategy::Forward => impl_fill_forward!(self),
            FillNoneStrategy::Backward => impl_fill_backward!(self, builder),
            strat => Err(PolarsError::InvalidOperation(
                format!("Strategy {:?} not supported", strat).into(),
            )),
        }
    }
}

impl ChunkFillNoneValue<&str> for Utf8Chunked {
    fn fill_none_with_value(&self, value: &str) -> Result<Self> {
        Ok(impl_fill_value!(self, Some(value)))
    }
}

impl ChunkFillNone for ListChunked {
    fn fill_none(&self, _strategy: FillNoneStrategy) -> Result<Self> {
        Err(PolarsError::InvalidOperation(
            "fill_none not supported for List type".into(),
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
