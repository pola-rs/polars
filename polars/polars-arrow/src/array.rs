use arrow::array::{
    Array, ArrayData, ArrayDataRef, ArrayRef, BooleanArray, ListArray, PrimitiveArray,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType};
use num::Num;

pub trait GetValues {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num;
}

impl GetValues for ArrayDataRef {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        debug_assert_eq!(&T::DATA_TYPE, self.data_type());
        // the first buffer is the value array
        let value_buf = &self.buffers()[0];
        let offset = self.offset();
        let vals = unsafe { value_buf.typed_data::<T::Native>() };
        &vals[offset..offset + self.len()]
    }
}

impl GetValues for &dyn Array {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        self.data_ref().get_values::<T>()
    }
}

impl GetValues for ArrayRef {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        self.data_ref().get_values::<T>()
    }
}

pub trait ToPrimitive {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType;
}

impl ToPrimitive for ArrayDataRef {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType,
    {
        PrimitiveArray::from(self)
    }
}

impl ToPrimitive for &dyn Array {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType,
    {
        self.data().into_primitive_array()
    }
}

pub trait ValueSize {
    /// Useful for a Utf8 or a List to get underlying value size.
    /// During a rechunk this is handy
    fn get_values_size(&self) -> usize;
}

impl ValueSize for ArrayRef {
    fn get_values_size(&self) -> usize {
        self.data_ref().get_values_size()
    }
}

impl ValueSize for ArrayData {
    fn get_values_size(&self) -> usize {
        match self.data_type() {
            DataType::LargeList(_) | DataType::List(_) => {
                self.child_data()[0].len() - self.offset()
            }
            DataType::LargeUtf8 | DataType::Utf8 => self.buffers()[1].len() - self.offset(),
            _ => unimplemented!(),
        }
    }
}

impl ValueSize for ListArray {
    fn get_values_size(&self) -> usize {
        self.data_ref().get_values_size()
    }
}

pub trait UnsafeValue<T> {
    /// # Safety
    /// no bounds check
    unsafe fn value_unchecked(&self, index: usize) -> T;
}

impl<T> UnsafeValue<T::Native> for PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
{
    #[inline]
    unsafe fn value_unchecked(&self, index: usize) -> T::Native {
        self.value(index)
    }
}

impl UnsafeValue<bool> for BooleanArray {
    #[inline]
    unsafe fn value_unchecked(&self, index: usize) -> bool {
        self.value(index)
    }
}

/// Cheaply get the null mask as BooleanArray.
pub trait IsNull {
    fn is_null_mask(&self) -> BooleanArray;
    fn is_not_null_mask(&self) -> BooleanArray;
}

impl IsNull for &dyn Array {
    fn is_null_mask(&self) -> BooleanArray {
        if self.null_count() == 0 {
            (0..self.len()).map(|_| Some(false)).collect()
        } else {
            let data = self.data();
            let valid = data.null_buffer().unwrap();
            let invert = !valid;

            let array_data = ArrayData::builder(DataType::Boolean)
                .len(self.len())
                .offset(self.offset())
                .add_buffer(invert)
                .build();
            BooleanArray::from(array_data)
        }
    }
    fn is_not_null_mask(&self) -> BooleanArray {
        if self.null_count() == 0 {
            (0..self.len()).map(|_| Some(true)).collect()
        } else {
            let data = self.data();
            let valid = data.null_buffer().unwrap().clone();

            let array_data = ArrayData::builder(DataType::Boolean)
                .len(self.len())
                .offset(self.offset())
                .add_buffer(valid)
                .build();
            BooleanArray::from(array_data)
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use arrow::array::{Array, Int32Array};

    #[test]
    fn test_is_null() {
        let arr = Int32Array::from(vec![Some(0), None, Some(2)]);
        let a: &dyn Array = &arr;
        assert_eq!(
            a.is_null_mask()
                .iter()
                .map(|v| v.unwrap())
                .collect::<Vec<_>>(),
            &[false, true, false]
        );
        assert_eq!(
            a.is_not_null_mask()
                .iter()
                .map(|v| v.unwrap())
                .collect::<Vec<_>>(),
            &[true, false, true]
        );
    }
}
