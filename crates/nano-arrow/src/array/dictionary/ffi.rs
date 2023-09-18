use crate::{
    array::{FromFfi, PrimitiveArray, ToFfi},
    error::Error,
    ffi,
};

use super::{DictionaryArray, DictionaryKey};

unsafe impl<K: DictionaryKey> ToFfi for DictionaryArray<K> {
    fn buffers(&self) -> Vec<Option<*const u8>> {
        self.keys.buffers()
    }

    fn offset(&self) -> Option<usize> {
        self.keys.offset()
    }

    fn to_ffi_aligned(&self) -> Self {
        Self {
            data_type: self.data_type.clone(),
            keys: self.keys.to_ffi_aligned(),
            values: self.values.clone(),
        }
    }
}

impl<K: DictionaryKey, A: ffi::ArrowArrayRef> FromFfi<A> for DictionaryArray<K> {
    unsafe fn try_from_ffi(array: A) -> Result<Self, Error> {
        // keys: similar to PrimitiveArray, but the datatype is the inner one
        let validity = unsafe { array.validity() }?;
        let values = unsafe { array.buffer::<K>(1) }?;

        let data_type = array.data_type().clone();

        let keys = PrimitiveArray::<K>::try_new(K::PRIMITIVE.into(), values, validity)?;
        let values = array
            .dictionary()?
            .ok_or_else(|| Error::oos("Dictionary Array must contain a dictionary in ffi"))?;
        let values = ffi::try_from(values)?;

        // the assumption of this trait
        DictionaryArray::<K>::try_new_unchecked(data_type, keys, values)
    }
}
