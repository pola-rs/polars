use crate::trusted_len::TrustedLen;
use crate::utils::FromTrustedLenIterator;
use arrow::array::{ArrayData, BooleanArray};
use arrow::buffer::MutableBuffer;
use arrow::datatypes::DataType;
use arrow::util::bit_util;

impl FromTrustedLenIterator<Option<bool>> for BooleanArray {
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let (_, data_len) = iter.size_hint();
        let data_len = data_len.expect("Iterator must be sized"); // panic if no upper bound.

        let num_bytes = bit_util::ceil(data_len, 8);
        let mut null_buf = MutableBuffer::from_len_zeroed(num_bytes);
        let mut val_buf = MutableBuffer::from_len_zeroed(num_bytes);

        let data = val_buf.as_slice_mut().as_mut_ptr();

        let null_slice = null_buf.as_slice_mut().as_mut_ptr();
        iter.enumerate().for_each(|(i, item)| {
            if let Some(a) = item {
                // Safety
                // iterator is trusted length so data is in bounds
                unsafe {
                    bit_util::set_bit_raw(null_slice, i);
                    if a {
                        bit_util::set_bit_raw(data, i);
                    }
                }
            }
        });

        let data = ArrayData::new(
            DataType::Boolean,
            data_len,
            None,
            Some(null_buf.into()),
            0,
            vec![val_buf.into()],
            vec![],
        );
        BooleanArray::from(data)
    }
}

impl FromTrustedLenIterator<bool> for BooleanArray {
    fn from_iter_trusted_length<I: IntoIterator<Item = bool>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let (_, data_len) = iter.size_hint();
        let data_len = data_len.expect("Iterator must be sized"); // panic if no upper bound.

        let num_bytes = bit_util::ceil(data_len, 8);
        let mut val_buf = MutableBuffer::from_len_zeroed(num_bytes);
        let data = val_buf.as_slice_mut().as_mut_ptr();

        iter.enumerate().for_each(|(i, item)| {
            // Safety
            // iterator is trusted length so data is in bounds
            unsafe {
                if item {
                    bit_util::set_bit_raw(data, i);
                }
            }
        });

        let data = ArrayData::new(
            DataType::Boolean,
            data_len,
            None,
            None,
            0,
            vec![val_buf.into()],
            vec![],
        );
        BooleanArray::from(data)
    }
}
