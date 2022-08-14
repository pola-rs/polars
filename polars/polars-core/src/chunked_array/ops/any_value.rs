use std::convert::TryFrom;

use crate::prelude::*;

#[inline]
#[allow(unused_variables)]
pub(crate) unsafe fn arr_to_any_value<'a>(
    arr: &'a dyn Array,
    idx: usize,
    dtype: &'a DataType,
) -> AnyValue<'a> {
    debug_assert!(idx < arr.len());
    if arr.is_null(idx) {
        return AnyValue::Null;
    }

    macro_rules! downcast_and_pack {
        ($casttype:ident, $variant:ident) => {{
            let arr = &*(arr as *const dyn Array as *const $casttype);
            let v = arr.value_unchecked(idx);
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
        DataType::List(dt) => {
            let v: ArrayRef = downcast!(LargeListArray);
            let mut s = Series::try_from(("", v)).unwrap();

            match &**dt {
                #[cfg(feature = "dtype-categorical")]
                DataType::Categorical(Some(rev_map)) => {
                    let cats = s.u32().unwrap().clone();
                    let out =
                        CategoricalChunked::from_cats_and_rev_map_unchecked(cats, rev_map.clone());
                    s = out.into_series();
                }
                DataType::Date
                | DataType::Datetime(_, _)
                | DataType::Time
                | DataType::Duration(_) => s = s.cast(dt).unwrap(),
                _ => {}
            }

            AnyValue::List(s)
        }
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(rev_map) => {
            let arr = &*(arr as *const dyn Array as *const UInt32Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Categorical(v, rev_map.as_ref().unwrap().as_ref())
        }
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(flds) => {
            let arr = &*(arr as *const dyn Array as *const StructArray);
            let vals = arr
                .values()
                .iter()
                .zip(flds)
                .map(|(arr, fld)| arr_to_any_value(&**arr, idx, fld.data_type()))
                .collect();
            AnyValue::Struct(vals, flds)
        }
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(tu, tz) => {
            let arr = &*(arr as *const dyn Array as *const Int64Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Datetime(v, *tu, tz)
        }
        #[cfg(feature = "dtype-date")]
        DataType::Date => {
            let arr = &*(arr as *const dyn Array as *const Int32Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Date(v)
        }
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(tu) => {
            let arr = &*(arr as *const dyn Array as *const Int64Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Duration(v, *tu)
        }
        #[cfg(feature = "dtype-time")]
        DataType::Time => {
            let arr = &*(arr as *const dyn Array as *const Int64Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Time(v)
        }
        #[cfg(feature = "object")]
        DataType::Object(_) => panic!("should not be here"),
        dt => panic!("not implemented for {:?}", dt),
    }
}

macro_rules! get_any_value_unchecked {
    ($self:ident, $index:expr) => {{
        let (chunk_idx, idx) = $self.index_to_chunked_index($index);
        debug_assert!(chunk_idx < $self.chunks.len());
        let arr = &**$self.chunks.get_unchecked(chunk_idx);
        debug_assert!(idx < arr.len());
        arr_to_any_value(arr, idx, $self.dtype())
    }};
}

macro_rules! get_any_value {
    ($self:ident, $index:expr) => {{
        let (chunk_idx, idx) = $self.index_to_chunked_index($index);
        let arr = &*$self.chunks[chunk_idx];
        assert!(idx < arr.len());
        // SAFETY
        // bounds are checked
        unsafe { arr_to_any_value(arr, idx, $self.dtype()) }
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
