use num::{abs, clamp};

use crate::prelude::*;

macro_rules! impl_shift_fill {
    ($self:ident, $periods:expr, $fill_value:expr) => {{
        let periods = clamp($periods, -($self.len() as i64), $self.len() as i64);
        let slice_offset = (-periods).max(0) as i64;
        let length = $self.len() - abs(periods) as usize;
        let mut slice = $self.slice(slice_offset, length);

        let fill_length = abs(periods) as usize;
        let mut fill = match $fill_value {
            Some(val) => Self::full($self.name(), val, fill_length),
            None => Self::full_null($self.name(), fill_length),
        };

        if periods < 0 {
            slice.append(&fill);
            slice
        } else {
            fill.append(&slice);
            fill
        }
    }};
}

impl<T> ChunkShiftFill<T, Option<T::Native>> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn shift_and_fill(&self, periods: i64, fill_value: Option<T::Native>) -> ChunkedArray<T> {
        impl_shift_fill!(self, periods, fill_value)
    }
}
impl<T> ChunkShift<T> for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    fn shift(&self, periods: i64) -> ChunkedArray<T> {
        self.shift_and_fill(periods, None)
    }
}

impl ChunkShiftFill<BooleanType, Option<bool>> for BooleanChunked {
    fn shift_and_fill(&self, periods: i64, fill_value: Option<bool>) -> BooleanChunked {
        impl_shift_fill!(self, periods, fill_value)
    }
}

impl ChunkShift<BooleanType> for BooleanChunked {
    fn shift(&self, periods: i64) -> Self {
        self.shift_and_fill(periods, None)
    }
}

impl ChunkShiftFill<Utf8Type, Option<&str>> for Utf8Chunked {
    fn shift_and_fill(&self, periods: i64, fill_value: Option<&str>) -> Utf8Chunked {
        impl_shift_fill!(self, periods, fill_value)
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkShiftFill<BinaryType, Option<&[u8]>> for BinaryChunked {
    fn shift_and_fill(&self, periods: i64, fill_value: Option<&[u8]>) -> BinaryChunked {
        impl_shift_fill!(self, periods, fill_value)
    }
}

impl ChunkShift<Utf8Type> for Utf8Chunked {
    fn shift(&self, periods: i64) -> Self {
        self.shift_and_fill(periods, None)
    }
}

#[cfg(feature = "dtype-binary")]
impl ChunkShift<BinaryType> for BinaryChunked {
    fn shift(&self, periods: i64) -> Self {
        self.shift_and_fill(periods, None)
    }
}

impl ChunkShiftFill<ListType, Option<&Series>> for ListChunked {
    fn shift_and_fill(&self, periods: i64, fill_value: Option<&Series>) -> ListChunked {
        // This has its own implementation because a ListChunked cannot have a full-null without
        // knowing the inner type
        let periods = clamp(periods, -(self.len() as i64), self.len() as i64);
        let slice_offset = (-periods).max(0);
        let length = self.len() - abs(periods) as usize;
        let mut slice = self.slice(slice_offset, length);

        let fill_length = abs(periods) as usize;
        let mut fill = match fill_value {
            Some(val) => Self::full(self.name(), val, fill_length),
            None => {
                ListChunked::full_null_with_dtype(self.name(), fill_length, &self.inner_dtype())
            }
        };

        if periods < 0 {
            slice.append(&fill).unwrap();
            slice
        } else {
            fill.append(&slice).unwrap();
            fill
        }
    }
}

impl ChunkShift<ListType> for ListChunked {
    fn shift(&self, periods: i64) -> Self {
        self.shift_and_fill(periods, None)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkShiftFill<ObjectType<T>, Option<ObjectType<T>>> for ObjectChunked<T> {
    fn shift_and_fill(
        &self,
        _periods: i64,
        _fill_value: Option<ObjectType<T>>,
    ) -> ChunkedArray<ObjectType<T>> {
        todo!()
    }
}
#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkShift<ObjectType<T>> for ObjectChunked<T> {
    fn shift(&self, periods: i64) -> Self {
        self.shift_and_fill(periods, None)
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_shift() {
        let ca = Int32Chunked::new("", &[1, 2, 3]);

        // shift by 0, 1, 2, 3, 4
        let shifted = ca.shift_and_fill(0, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(1), Some(2), Some(3)]);
        let shifted = ca.shift_and_fill(1, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(5), Some(1), Some(2)]);
        let shifted = ca.shift_and_fill(2, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(5), Some(5), Some(1)]);
        let shifted = ca.shift_and_fill(3, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(5), Some(5), Some(5)]);
        let shifted = ca.shift_and_fill(4, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(5), Some(5), Some(5)]);

        // shift by -1, -2, -3, -4
        let shifted = ca.shift_and_fill(-1, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(2), Some(3), Some(5)]);
        let shifted = ca.shift_and_fill(-2, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(3), Some(5), Some(5)]);
        let shifted = ca.shift_and_fill(-3, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(5), Some(5), Some(5)]);
        let shifted = ca.shift_and_fill(-4, Some(5));
        assert_eq!(Vec::from(&shifted), &[Some(5), Some(5), Some(5)]);

        // fill with None
        let shifted = ca.shift_and_fill(1, None);
        assert_eq!(Vec::from(&shifted), &[None, Some(1), Some(2)]);
        let shifted = ca.shift_and_fill(10, None);
        assert_eq!(Vec::from(&shifted), &[None, None, None]);
        let shifted = ca.shift_and_fill(-2, None);
        assert_eq!(Vec::from(&shifted), &[Some(3), None, None]);

        // string
        let s = Series::new("a", ["a", "b", "c"]);
        let shifted = s.shift(-1);
        assert_eq!(
            Vec::from(shifted.utf8().unwrap()),
            &[Some("b"), Some("c"), None]
        );
    }
}
