#[cfg(feature = "dtype-categorical")]
use crate::chunked_array::categorical::RevMapping;
#[cfg(not(feature = "dtype-categorical"))]
use crate::chunked_array::RevMapping;
use crate::prelude::*;
use std::convert::TryFrom;

#[inline]
#[allow(unused_variables)]
unsafe fn arr_to_any_value<'a>(
    arr: &'a dyn Array,
    idx: usize,
    categorical_map: &'a Option<Arc<RevMapping>>,
    dtype: &DataType,
) -> AnyValue<'a> {
    if arr.is_null(idx) {
        return AnyValue::Null;
    }

    macro_rules! downcast_and_pack {
        ($casttype:ident, $variant:ident) => {{
            let arr = &*(arr as *const dyn Array as *const $casttype);
            let v = arr.value(idx);
            AnyValue::$variant(v)
        }};
    }
    macro_rules! downcast {
        ($casttype:ident) => {{
            let arr = &*(arr as *const dyn Array as *const $casttype);
            arr.value_unchecked(idx)
        }};
    }
    // TODO: insert types
    match dtype {
        DataType::Utf8 => downcast_and_pack!(LargeStringArray, Utf8),
        DataType::Boolean => downcast_and_pack!(BooleanArray, Boolean),
        DataType::UInt8 => downcast_and_pack!(UInt8Array, UInt8),
        DataType::UInt16 => downcast_and_pack!(UInt16Array, UInt16),
        DataType::UInt32 => downcast_and_pack!(UInt32Array, UInt32),
        DataType::UInt64 => downcast_and_pack!(UInt64Array, UInt64),
        DataType::Int8 => downcast_and_pack!(Int8Array, Int8),
        DataType::Int16 => downcast_and_pack!(Int16Array, Int16),
        DataType::Int32 => downcast_and_pack!(Int32Array, Int32),
        DataType::Int64 => downcast_and_pack!(Int64Array, Int64),
        DataType::Float32 => downcast_and_pack!(Float32Array, Float32),
        DataType::Float64 => downcast_and_pack!(Float64Array, Float64),
        #[cfg(feature = "dtype-date")]
        DataType::Date => downcast_and_pack!(Int32Array, Date),
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime => downcast_and_pack!(Int64Array, Datetime),
        DataType::List(dt) => {
            let v: ArrayRef = downcast!(LargeListArray).into();
            let mut s = Series::try_from(("", v)).unwrap();

            if let DataType::Categorical = **dt {
                let mut s_new = s.cast(&DataType::Categorical).unwrap();
                let ca: &mut CategoricalChunked = s_new.get_inner_mut().as_mut();
                ca.categorical_map = categorical_map.clone();
                s = s_new;
            }
            AnyValue::List(s)
        }
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical => {
            let idx = downcast!(UInt32Array);
            let rev_map = &**categorical_map.as_ref().unwrap();
            AnyValue::Categorical(idx, rev_map)
        }
        #[cfg(feature = "object")]
        DataType::Object(_) => panic!("should not be here"),
        _ => unimplemented!(),
    }
}

macro_rules! get_any_value_unchecked {
    ($self:ident, $index:expr) => {{
        let (chunk_idx, idx) = $self.index_to_chunked_index($index);
        debug_assert!(chunk_idx < $self.chunks.len());
        let arr = &**$self.chunks.get_unchecked(chunk_idx);
        debug_assert!(idx < arr.len());
        arr_to_any_value(arr, idx, &$self.categorical_map, $self.dtype())
    }};
}

macro_rules! get_any_value {
    ($self:ident, $index:expr) => {{
        let (chunk_idx, idx) = $self.index_to_chunked_index($index);
        let arr = &*$self.chunks[chunk_idx];
        assert!(idx < arr.len());
        // SAFETY
        // bounds are checked
        unsafe { arr_to_any_value(arr, idx, &$self.categorical_map, $self.dtype()) }
    }};
}

impl<T> ChunkAnyValue for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> AnyValue {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for BooleanChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> AnyValue {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for Utf8Chunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> AnyValue {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for ListChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> AnyValue {
        get_any_value!(self, index)
    }
}

#[cfg(feature = "dtype-categorical")]
impl ChunkAnyValue for CategoricalChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = &self.chunks[chunk_idx];
        debug_assert!(idx < arr.len());
        arr_to_any_value(&**arr, idx, &self.categorical_map, &DataType::Categorical)
    }

    fn get_any_value(&self, index: usize) -> AnyValue {
        let (chunk_idx, idx) = self.index_to_chunked_index(index);
        let arr = &self.chunks[chunk_idx];
        assert!(idx < arr.len());
        // SAFETY
        // bounds are checked
        unsafe { arr_to_any_value(&**arr, idx, &self.categorical_map, &DataType::Categorical) }
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkAnyValue for ObjectChunked<T> {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue {
        match self.get_object_unchecked(index) {
            None => AnyValue::Null,
            Some(v) => AnyValue::Object(v),
        }
    }

    fn get_any_value(&self, index: usize) -> AnyValue {
        match self.get_object(index) {
            None => AnyValue::Null,
            Some(v) => AnyValue::Object(v),
        }
    }
}
