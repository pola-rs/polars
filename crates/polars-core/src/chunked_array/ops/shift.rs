use num_traits::{abs, clamp};

use crate::prelude::*;
use crate::series::implementations::null::NullChunked;

impl<'a, T> ChunkShiftFill<T, Option<T::Physical<'a>>> for ChunkedArray<T>
where
    T: PolarsDataType<IsNested = FalseT, IsObject = FalseT>,
    ChunkedArray<T>: ChunkFull<T::Physical<'a>> + ChunkFullNull,
    for<'b> T::Physical<'b>: TotalOrd,
{
    fn shift_and_fill(&self, periods: i64, fill_value: Option<T::Physical<'a>>) -> ChunkedArray<T>
where {
        let fill_length = abs(periods) as usize;
        if fill_length >= self.len() {
            return match fill_value {
                Some(fill) => ChunkedArray::full(self.name().clone(), fill, self.len()),
                None => ChunkedArray::full_null(self.name().clone(), self.len()),
            };
        }
        let slice_offset = (-periods).max(0);
        let length = self.len() - fill_length;
        let mut slice = self.slice(slice_offset, length);

        let mut fill = match fill_value {
            Some(val) => ChunkedArray::full(self.name().clone(), val, fill_length),
            None => ChunkedArray::full_null(self.name().clone(), fill_length),
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

impl<'a, T> ChunkShift<T> for ChunkedArray<T>
where
    T: PolarsDataType<IsNested = FalseT, IsObject = FalseT>,
    ChunkedArray<T>: ChunkFull<T::Physical<'a>> + ChunkFullNull,
    for<'b> T::Physical<'b>: TotalOrd,
{
    fn shift(&self, periods: i64) -> ChunkedArray<T> {
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
            Some(val) => Self::full(self.name().clone(), val, fill_length),
            None => ListChunked::full_null_with_dtype(
                self.name().clone(),
                fill_length,
                self.inner_dtype(),
            ),
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

#[cfg(feature = "dtype-array")]
impl ChunkShiftFill<FixedSizeListType, Option<&Series>> for ArrayChunked {
    fn shift_and_fill(&self, periods: i64, fill_value: Option<&Series>) -> ArrayChunked {
        // This has its own implementation because a ArrayChunked cannot have a full-null without
        // knowing the inner type
        let periods = clamp(periods, -(self.len() as i64), self.len() as i64);
        let slice_offset = (-periods).max(0);
        let length = self.len() - abs(periods) as usize;
        let mut slice = self.slice(slice_offset, length);

        let fill_length = abs(periods) as usize;
        let mut fill = match fill_value {
            Some(val) => Self::full(self.name().clone(), val, fill_length),
            None => ArrayChunked::full_null_with_dtype(
                self.name().clone(),
                fill_length,
                self.inner_dtype(),
                0,
            ),
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

#[cfg(feature = "dtype-array")]
impl ChunkShift<FixedSizeListType> for ArrayChunked {
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

#[cfg(feature = "dtype-struct")]
impl ChunkShift<StructType> for StructChunked {
    fn shift(&self, periods: i64) -> ChunkedArray<StructType> {
        // This has its own implementation because a ArrayChunked cannot have a full-null without
        // knowing the inner type
        let periods = clamp(periods, -(self.len() as i64), self.len() as i64);
        let slice_offset = (-periods).max(0);
        let length = self.len() - abs(periods) as usize;
        let mut slice = self.slice(slice_offset, length);

        let fill_length = abs(periods) as usize;

        // Go via null, so the cast creates the proper struct type.
        let fill = NullChunked::new(self.name().clone(), fill_length)
            .cast(self.dtype(), Default::default())
            .unwrap();
        let mut fill = fill.struct_().unwrap().clone();

        if periods < 0 {
            slice.append(&fill).unwrap();
            slice
        } else {
            fill.append(&slice).unwrap();
            fill
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_shift() {
        let ca = Int32Chunked::new(PlSmallStr::EMPTY, &[1, 2, 3]);

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
        let s = Series::new(PlSmallStr::from_static("a"), ["a", "b", "c"]);
        let shifted = s.shift(-1);
        assert_eq!(
            Vec::from(shifted.str().unwrap()),
            &[Some("b"), Some("c"), None]
        );
    }
}
