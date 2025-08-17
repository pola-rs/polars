#![allow(unsafe_op_in_unsafe_fn)]

#[cfg(feature = "object")]
use crate::chunked_array::object::extension::polars_extension::PolarsExtension;
use crate::prelude::*;
use crate::series::implementations::null::NullChunked;
use crate::utils::index_to_chunked_index;

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
    match dtype {
        DataType::String => downcast_and_pack!(Utf8ViewArray, String),
        DataType::Binary => downcast_and_pack!(BinaryViewArray, Binary),
        DataType::Boolean => downcast_and_pack!(BooleanArray, Boolean),
        DataType::UInt8 => downcast_and_pack!(UInt8Array, UInt8),
        DataType::UInt16 => downcast_and_pack!(UInt16Array, UInt16),
        DataType::UInt32 => downcast_and_pack!(UInt32Array, UInt32),
        DataType::UInt64 => downcast_and_pack!(UInt64Array, UInt64),
        DataType::Int8 => downcast_and_pack!(Int8Array, Int8),
        DataType::Int16 => downcast_and_pack!(Int16Array, Int16),
        DataType::Int32 => downcast_and_pack!(Int32Array, Int32),
        DataType::Int64 => downcast_and_pack!(Int64Array, Int64),
        DataType::Int128 => downcast_and_pack!(Int128Array, Int128),
        DataType::Float32 => downcast_and_pack!(Float32Array, Float32),
        DataType::Float64 => downcast_and_pack!(Float64Array, Float64),
        DataType::List(dt) => {
            let v: ArrayRef = downcast!(LargeListArray);
            if dt.is_primitive() {
                let s = Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![v], dt);
                AnyValue::List(s)
            } else {
                let s = Series::from_chunks_and_dtype_unchecked(
                    PlSmallStr::EMPTY,
                    vec![v],
                    &dt.to_physical(),
                )
                .from_physical_unchecked(dt)
                .unwrap();
                AnyValue::List(s)
            }
        },
        #[cfg(feature = "dtype-array")]
        DataType::Array(dt, width) => {
            let v: ArrayRef = downcast!(FixedSizeListArray);
            if dt.is_primitive() {
                let s = Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![v], dt);
                AnyValue::Array(s, *width)
            } else {
                let s = Series::from_chunks_and_dtype_unchecked(
                    PlSmallStr::EMPTY,
                    vec![v],
                    &dt.to_physical(),
                )
                .from_physical_unchecked(dt)
                .unwrap();
                AnyValue::Array(s, *width)
            }
        },
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(cats, mapping) => {
            with_match_categorical_physical_type!(cats.physical(), |$C| {
                type A = <$C as PolarsDataType>::Array;
                let arr = &*(arr as *const dyn Array as *const A);
                let cat_id = arr.value_unchecked(idx).as_cat();
                AnyValue::Categorical(cat_id, mapping)
            })
        },
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(fcats, mapping) => {
            with_match_categorical_physical_type!(fcats.physical(), |$C| {
                type A = <$C as PolarsDataType>::Array;
                let arr = &*(arr as *const dyn Array as *const A);
                let cat_id = arr.value_unchecked(idx).as_cat();
                AnyValue::Enum(cat_id, mapping)
            })
        },
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(flds) => {
            let arr = &*(arr as *const dyn Array as *const StructArray);
            AnyValue::Struct(idx, arr, flds)
        },
        #[cfg(feature = "dtype-datetime")]
        DataType::Datetime(tu, tz) => {
            let arr = &*(arr as *const dyn Array as *const Int64Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Datetime(v, *tu, tz.as_ref())
        },
        #[cfg(feature = "dtype-date")]
        DataType::Date => {
            let arr = &*(arr as *const dyn Array as *const Int32Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Date(v)
        },
        #[cfg(feature = "dtype-duration")]
        DataType::Duration(tu) => {
            let arr = &*(arr as *const dyn Array as *const Int64Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Duration(v, *tu)
        },
        #[cfg(feature = "dtype-time")]
        DataType::Time => {
            let arr = &*(arr as *const dyn Array as *const Int64Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Time(v)
        },
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(precision, scale) => {
            let arr = &*(arr as *const dyn Array as *const Int128Array);
            let v = arr.value_unchecked(idx);
            AnyValue::Decimal(v, scale.unwrap_or_else(|| unreachable!()))
        },
        #[cfg(feature = "object")]
        DataType::Object(_) => {
            // We should almost never hit this. The only known exception is when we put objects in
            // structs. Any other hit should be considered a bug.
            let arr = arr.as_any().downcast_ref::<FixedSizeBinaryArray>().unwrap();
            PolarsExtension::arr_to_av(arr, idx)
        },
        DataType::Null => AnyValue::Null,
        DataType::BinaryOffset => downcast_and_pack!(LargeBinaryArray, Binary),
        dt => panic!("not implemented for {dt:?}"),
    }
}

#[cfg(feature = "dtype-struct")]
impl<'a> AnyValue<'a> {
    pub fn _iter_struct_av(&self) -> impl Iterator<Item = AnyValue<'_>> {
        let AnyValue::Struct(idx, arr, flds) = self else {
            unreachable!()
        };
        unsafe {
            arr.values()
                .iter()
                .zip(*flds)
                .map(move |(arr, fld)| arr_to_any_value(&**arr, *idx, fld.dtype()))
        }
    }

    pub fn _materialize_struct_av(&'a self, buf: &mut Vec<AnyValue<'a>>) {
        let iter = self._iter_struct_av();
        buf.extend(iter)
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
        if $index >= $self.len() {
            polars_bail!(oob = $index, $self.len());
        }
        // SAFETY:
        // bounds are checked
        Ok(unsafe { $self.get_any_value_unchecked($index) })
    }};
}

impl<T> ChunkAnyValue for ChunkedArray<T>
where
    T: PolarsNumericType,
{
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue<'_> {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for BooleanChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue<'_> {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for StringChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue<'_> {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for BinaryChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue<'_> {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for BinaryOffsetChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue<'_> {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for ListChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue<'_> {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        get_any_value!(self, index)
    }
}

#[cfg(feature = "dtype-array")]
impl ChunkAnyValue for ArrayChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue<'_> {
        get_any_value_unchecked!(self, index)
    }

    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        get_any_value!(self, index)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> ChunkAnyValue for ObjectChunked<T> {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, index: usize) -> AnyValue<'_> {
        match self.get_object_unchecked(index) {
            None => AnyValue::Null,
            Some(v) => AnyValue::Object(v),
        }
    }

    fn get_any_value(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        get_any_value!(self, index)
    }
}

impl ChunkAnyValue for NullChunked {
    #[inline]
    unsafe fn get_any_value_unchecked(&self, _index: usize) -> AnyValue<'_> {
        AnyValue::Null
    }

    fn get_any_value(&self, _index: usize) -> PolarsResult<AnyValue<'_>> {
        Ok(AnyValue::Null)
    }
}

#[cfg(feature = "dtype-struct")]
impl ChunkAnyValue for StructChunked {
    /// Gets AnyValue from LogicalType
    fn get_any_value(&self, i: usize) -> PolarsResult<AnyValue<'_>> {
        polars_ensure!(i < self.len(), oob = i, self.len());
        unsafe { Ok(self.get_any_value_unchecked(i)) }
    }

    unsafe fn get_any_value_unchecked(&self, i: usize) -> AnyValue<'_> {
        let (chunk_idx, idx) = index_to_chunked_index(self.chunks.iter().map(|c| c.len()), i);
        if let DataType::Struct(flds) = self.dtype() {
            // SAFETY: we already have a single chunk and we are
            // guarded by the type system.
            unsafe {
                let arr = &**self.chunks.get_unchecked(chunk_idx);
                let arr = &*(arr as *const dyn Array as *const StructArray);

                if arr.is_null_unchecked(idx) {
                    AnyValue::Null
                } else {
                    AnyValue::Struct(idx, arr, flds)
                }
            }
        } else {
            unreachable!()
        }
    }
}
