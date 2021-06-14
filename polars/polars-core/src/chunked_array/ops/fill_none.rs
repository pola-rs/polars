use crate::prelude::*;
use num::{Bounded, Num, NumCast, One, Zero};
use polars_arrow::array::ValueSize;
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
        let mut builder = BooleanChunkedBuilder::new(self.name(), self.len());
        match strategy {
            FillNoneStrategy::Forward => impl_fill_forward!(self),
            FillNoneStrategy::Backward => impl_fill_backward!(self, builder),
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
        let factor = self.len() as f32 / (self.len() - self.null_count()) as f32;
        let value_cap = (self.get_values_size() as f32 * 1.25 * factor) as usize;
        let mut builder = Utf8ChunkedBuilder::new(self.name(), self.len(), value_cap);
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
        println!("{:?}", filled);
    }
}
