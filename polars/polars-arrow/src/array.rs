use crate::utils::CustomIterTools;
use arrow::array::{
    Array, ArrayData, ArrayRef, BooleanArray, BooleanBufferBuilder, LargeListArray,
    LargeStringArray, PrimitiveArray,
};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::{ArrowNumericType, ArrowPrimitiveType, DataType, Field};
use num::Num;

pub trait GetValues {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num;
}

impl GetValues for ArrayData {
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

impl ToPrimitive for ArrayData {
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
        self.data().clone().into_primitive_array()
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

impl ValueSize for LargeListArray {
    fn get_values_size(&self) -> usize {
        self.data_ref().get_values_size()
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

fn finish_listarray(
    field: Field,
    child_data: ArrayData,
    offsets: MutableBuffer,
    null_buf: BooleanBufferBuilder,
) -> LargeListArray {
    let data = ArrayData::builder(DataType::LargeList(Box::new(field)))
        .len(null_buf.len())
        .add_buffer(offsets.into())
        .add_child_data(child_data)
        .null_bit_buffer(null_buf.into())
        .build();
    LargeListArray::from(data)
}

macro_rules! iter_to_values {
    ($iterator:expr, $null_buf:expr, $offsets:expr, $length_so_far:expr) => {{
        $iterator
            .filter_map(|opt_iter| match opt_iter {
                Some(x) => {
                    let it = x.into_iter();
                    $length_so_far += it.size_hint().0 as i64;
                    $null_buf.append(true);
                    $offsets.push($length_so_far);
                    Some(it)
                }
                None => {
                    $null_buf.append(false);
                    None
                }
            })
            .flatten()
            .collect()
    }};
}

pub trait ListFromIter {
    /// Create a list-array from an iterator.
    /// Used in groupby agg-list
    ///
    /// # Safety
    /// Will produce incorrect arrays if size hint is incorrect.
    unsafe fn from_iter_primitive_trusted_len<T, P, I>(iter: I) -> LargeListArray
    where
        T: ArrowNumericType,
        P: IntoIterator<Item = Option<T::Native>>,
        I: IntoIterator<Item = Option<P>>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        let mut offsets = MutableBuffer::new((lower + 1) * std::mem::size_of::<i64>());
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        let mut null_buf = BooleanBufferBuilder::new(lower);

        let values: PrimitiveArray<T> = iter_to_values!(iterator, null_buf, offsets, length_so_far);

        let field = Field::new("item", T::DATA_TYPE, true);
        finish_listarray(field, values.data().clone(), offsets, null_buf)
    }

    /// Create a list-array from an iterator.
    /// Used in groupby agg-list
    ///
    /// # Safety
    /// Will produce incorrect arrays if size hint is incorrect.
    unsafe fn from_iter_bool_trusted_len<I, P>(iter: I) -> LargeListArray
    where
        I: IntoIterator<Item = Option<P>>,
        P: IntoIterator<Item = Option<bool>>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        let mut offsets = MutableBuffer::new((lower + 1) * std::mem::size_of::<i64>());
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        let mut null_buf = BooleanBufferBuilder::new(lower);
        let values: BooleanArray = iter_to_values!(iterator, null_buf, offsets, length_so_far);

        let field = Field::new("item", DataType::Boolean, true);
        finish_listarray(field, values.data().clone(), offsets, null_buf)
    }

    /// Create a list-array from an iterator.
    /// Used in groupby agg-list
    ///
    /// # Safety
    /// Will produce incorrect arrays if size hint is incorrect.
    unsafe fn from_iter_utf8_trusted_len<I, P, Ref>(iter: I, n_elements: usize) -> LargeListArray
    where
        I: IntoIterator<Item = Option<P>>,
        P: IntoIterator<Item = Option<Ref>>,
        Ref: AsRef<str>,
    {
        let iterator = iter.into_iter();
        let (lower, _) = iterator.size_hint();

        let mut offsets = MutableBuffer::new((lower + 1) * std::mem::size_of::<i64>());
        let mut length_so_far = 0i64;
        offsets.push(length_so_far);

        let mut null_buf = BooleanBufferBuilder::new(lower);
        let values: LargeStringArray = iterator
            .filter_map(|opt_iter| match opt_iter {
                Some(x) => {
                    let it = x.into_iter();
                    length_so_far += it.size_hint().0 as i64;
                    null_buf.append(true);
                    offsets.push(length_so_far);
                    Some(it)
                }
                None => {
                    null_buf.append(false);
                    None
                }
            })
            .flatten()
            .trust_my_length(n_elements)
            .collect();

        let field = Field::new("item", DataType::LargeUtf8, true);
        finish_listarray(field, values.data().clone(), offsets, null_buf)
    }
}
impl ListFromIter for LargeListArray {}

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
