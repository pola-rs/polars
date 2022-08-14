use arrow::array::BooleanArray;
use arrow::bitmap::MutableBitmap;

use crate::array::default_arrays::FromData;
use crate::trusted_len::TrustedLen;
use crate::utils::FromTrustedLenIterator;

impl FromTrustedLenIterator<Option<bool>> for BooleanArray {
    fn from_iter_trusted_length<I: IntoIterator<Item = Option<bool>>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        // Soundness
        // Trait system bounded to TrustedLen
        unsafe { BooleanArray::from_trusted_len_iter_unchecked(iter.into_iter()) }
    }
}
impl FromTrustedLenIterator<bool> for BooleanArray {
    fn from_iter_trusted_length<I: IntoIterator<Item = bool>>(iter: I) -> Self
    where
        I::IntoIter: TrustedLen,
    {
        // Soundness
        // Trait system bounded to TrustedLen
        unsafe {
            BooleanArray::from_data_default(
                MutableBitmap::from_trusted_len_iter_unchecked(iter.into_iter()).into(),
                None,
            )
        }
    }
}
