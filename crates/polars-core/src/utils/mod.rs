mod any_value;
use arrow::compute::concatenate::concatenate_validities;
use arrow::compute::utils::combine_validities_and;
pub mod flatten;
pub(crate) mod series;
mod supertype;
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};
mod schema;

pub use any_value::*;
use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::Bitmap;
pub use arrow::legacy::utils::*;
pub use arrow::trusted_len::TrustMyLength;
use flatten::*;
use num_traits::{One, Zero};
use rayon::prelude::*;
pub use schema::*;
pub use series::*;
use smartstring::alias::String as SmartString;
pub use supertype::*;
pub use {arrow, rayon};

use crate::prelude::*;
use crate::POOL;

#[repr(transparent)]
pub struct Wrap<T>(pub T);

impl<T> Deref for Wrap<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[inline(always)]
pub fn _set_partition_size() -> usize {
    POOL.current_num_threads()
}

/// Just a wrapper structure which is useful for certain impl specializations.
///
/// This is for instance use to implement
/// `impl<T> FromIterator<T::Native> for NoNull<ChunkedArray<T>>`
/// as `Option<T::Native>` was already implemented:
/// `impl<T> FromIterator<Option<T::Native>> for ChunkedArray<T>`
pub struct NoNull<T> {
    inner: T,
}

impl<T> NoNull<T> {
    pub fn new(inner: T) -> Self {
        NoNull { inner }
    }

    pub fn into_inner(self) -> T {
        self.inner
    }
}

impl<T> Deref for NoNull<T> {
    type Target = T;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<T> DerefMut for NoNull<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

pub(crate) fn get_iter_capacity<T, I: Iterator<Item = T>>(iter: &I) -> usize {
    match iter.size_hint() {
        (_lower, Some(upper)) => upper,
        (0, None) => 1024,
        (lower, None) => lower,
    }
}

// prefer this one over split_ca, as this can push the null_count into the thread pool
// returns an `(offset, length)` tuple
#[doc(hidden)]
pub fn _split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
    if n == 1 {
        vec![(0, len)]
    } else {
        let chunk_size = len / n;

        (0..n)
            .map(|partition| {
                let offset = partition * chunk_size;
                let len = if partition == (n - 1) {
                    len - offset
                } else {
                    chunk_size
                };
                (partition * chunk_size, len)
            })
            .collect_trusted()
    }
}

#[allow(clippy::len_without_is_empty)]
pub trait Container: Clone {
    fn slice(&self, offset: i64, len: usize) -> Self;

    fn split_at(&self, offset: i64) -> (Self, Self);

    fn len(&self) -> usize;

    fn iter_chunks(&self) -> impl Iterator<Item = Self>;

    fn n_chunks(&self) -> usize;

    fn chunk_lengths(&self) -> impl Iterator<Item = usize>;
}

impl Container for DataFrame {
    fn slice(&self, offset: i64, len: usize) -> Self {
        DataFrame::slice(self, offset, len)
    }

    fn split_at(&self, offset: i64) -> (Self, Self) {
        DataFrame::split_at(self, offset)
    }

    fn len(&self) -> usize {
        self.height()
    }

    fn iter_chunks(&self) -> impl Iterator<Item = Self> {
        flatten_df_iter(self)
    }

    fn n_chunks(&self) -> usize {
        DataFrame::n_chunks(self)
    }

    fn chunk_lengths(&self) -> impl Iterator<Item = usize> {
        self.get_columns()[0].chunk_lengths()
    }
}

impl<T: PolarsDataType> Container for ChunkedArray<T> {
    fn slice(&self, offset: i64, len: usize) -> Self {
        ChunkedArray::slice(self, offset, len)
    }

    fn split_at(&self, offset: i64) -> (Self, Self) {
        ChunkedArray::split_at(self, offset)
    }

    fn len(&self) -> usize {
        ChunkedArray::len(self)
    }

    fn iter_chunks(&self) -> impl Iterator<Item = Self> {
        self.downcast_iter()
            .map(|arr| Self::with_chunk(self.name(), arr.clone()))
    }

    fn n_chunks(&self) -> usize {
        self.chunks().len()
    }

    fn chunk_lengths(&self) -> impl Iterator<Item = usize> {
        ChunkedArray::chunk_lengths(self)
    }
}

impl Container for Series {
    fn slice(&self, offset: i64, len: usize) -> Self {
        self.0.slice(offset, len)
    }

    fn split_at(&self, offset: i64) -> (Self, Self) {
        self.0.split_at(offset)
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn iter_chunks(&self) -> impl Iterator<Item = Self> {
        (0..self.0.n_chunks()).map(|i| self.select_chunk(i))
    }

    fn n_chunks(&self) -> usize {
        self.chunks().len()
    }

    fn chunk_lengths(&self) -> impl Iterator<Item = usize> {
        self.0.chunk_lengths()
    }
}

fn split_impl<C: Container>(container: &C, target: usize, chunk_size: usize) -> Vec<C> {
    if target == 1 {
        return vec![container.clone()];
    }
    let mut out = Vec::with_capacity(target);
    let chunk_size = chunk_size as i64;

    // First split
    let (chunk, mut remainder) = container.split_at(chunk_size);
    out.push(chunk);

    // Take the rest of the splits of exactly chunk size, but skip the last remainder as we won't split that.
    for _ in 1..target - 1 {
        let (a, b) = remainder.split_at(chunk_size);
        out.push(a);
        remainder = b
    }
    // This can be slightly larger than `chunk_size`, but is smaller than `2 * chunk_size`.
    out.push(remainder);
    out
}

/// Splits, but doesn't flatten chunks. E.g. a container can still have multiple chunks.
pub fn split<C: Container>(container: &C, target: usize) -> Vec<C> {
    let total_len = container.len();
    if total_len == 0 {
        return vec![container.clone()];
    }

    let chunk_size = std::cmp::max(total_len / target, 1);

    if container.n_chunks() == target
        && container
            .chunk_lengths()
            .all(|len| len.abs_diff(chunk_size) < 100)
    {
        return container.iter_chunks().collect();
    }
    split_impl(container, target, chunk_size)
}

/// Split a [`Container`] in `target` elements. The target doesn't have to be respected if not
/// Deviation of the target might be done to create more equal size chunks.
pub fn split_and_flatten<C: Container>(container: &C, target: usize) -> Vec<C> {
    let total_len = container.len();
    if total_len == 0 {
        return vec![container.clone()];
    }

    let chunk_size = std::cmp::max(total_len / target, 1);

    if container.n_chunks() == target
        && container
            .chunk_lengths()
            .all(|len| len.abs_diff(chunk_size) < 100)
    {
        return container.iter_chunks().collect();
    }

    if container.n_chunks() == 1 {
        split_impl(container, target, chunk_size)
    } else {
        let mut out = Vec::with_capacity(target);
        let chunks = container.iter_chunks();

        'new_chunk: for mut chunk in chunks {
            loop {
                let h = chunk.len();
                if h < chunk_size {
                    // TODO if the chunk is much smaller than chunk size, we should try to merge it with the next one.
                    out.push(chunk);
                    continue 'new_chunk;
                }

                // If a split leads to the next chunk being smaller than 30% take the whole chunk
                if ((h - chunk_size) as f64 / chunk_size as f64) < 0.3 {
                    out.push(chunk);
                    continue 'new_chunk;
                }

                let (a, b) = chunk.split_at(chunk_size as i64);
                out.push(a);
                chunk = b;
            }
        }
        out
    }
}

/// Split a [`DataFrame`] in `target` elements. The target doesn't have to be respected if not
/// strict. Deviation of the target might be done to create more equal size chunks.
///
/// # Panics
/// if chunks are not aligned
pub fn split_df_as_ref(df: &DataFrame, target: usize, strict: bool) -> Vec<DataFrame> {
    if strict {
        split(df, target)
    } else {
        split_and_flatten(df, target)
    }
}

#[doc(hidden)]
/// Split a [`DataFrame`] into `n` parts. We take a `&mut` to be able to repartition/align chunks.
/// `strict` in that it respects `n` even if the chunks are suboptimal.
pub fn split_df(df: &mut DataFrame, target: usize, strict: bool) -> Vec<DataFrame> {
    if target == 0 || df.is_empty() {
        return vec![df.clone()];
    }
    // make sure that chunks are aligned.
    df.align_chunks();
    split_df_as_ref(df, target, strict)
}

pub fn slice_slice<T>(vals: &[T], offset: i64, len: usize) -> &[T] {
    let (raw_offset, slice_len) = slice_offsets(offset, len, vals.len());
    &vals[raw_offset..raw_offset + slice_len]
}

#[inline]
#[doc(hidden)]
pub fn slice_offsets(offset: i64, length: usize, array_len: usize) -> (usize, usize) {
    let signed_start_offset = if offset < 0 {
        offset.saturating_add_unsigned(array_len as u64)
    } else {
        offset
    };
    let signed_stop_offset = signed_start_offset.saturating_add_unsigned(length as u64);

    let signed_array_len: i64 = array_len
        .try_into()
        .expect("array length larger than i64::MAX");
    let clamped_start_offset = signed_start_offset.clamp(0, signed_array_len);
    let clamped_stop_offset = signed_stop_offset.clamp(0, signed_array_len);

    let slice_start_idx = clamped_start_offset as usize;
    let slice_len = (clamped_stop_offset - clamped_start_offset) as usize;
    (slice_start_idx, slice_len)
}

/// Apply a macro on the Series
#[macro_export]
macro_rules! match_dtype_to_physical_apply_macro {
    ($obj:expr, $macro:ident, $macro_string:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $obj {
            DataType::String => $macro_string!($($opt_args)*),
            DataType::Boolean => $macro_bool!($($opt_args)*),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $macro!(u8 $(, $opt_args)*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $macro!(u16 $(, $opt_args)*),
            DataType::UInt32 => $macro!(u32 $(, $opt_args)*),
            DataType::UInt64 => $macro!(u64 $(, $opt_args)*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $macro!(i8 $(, $opt_args)*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $macro!(i16 $(, $opt_args)*),
            DataType::Int32 => $macro!(i32 $(, $opt_args)*),
            DataType::Int64 => $macro!(i64 $(, $opt_args)*),
            DataType::Float32 => $macro!(f32 $(, $opt_args)*),
            DataType::Float64 => $macro!(f64 $(, $opt_args)*),
            dt => panic!("not implemented for dtype {:?}", dt),
        }
    }};
}

/// Apply a macro on the Series
#[macro_export]
macro_rules! match_dtype_to_logical_apply_macro {
    ($obj:expr, $macro:ident, $macro_string:ident, $macro_binary:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $obj {
            DataType::String => $macro_string!($($opt_args)*),
            DataType::Binary => $macro_binary!($($opt_args)*),
            DataType::Boolean => $macro_bool!($($opt_args)*),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $macro!(UInt8Type $(, $opt_args)*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $macro!(UInt16Type $(, $opt_args)*),
            DataType::UInt32 => $macro!(UInt32Type $(, $opt_args)*),
            DataType::UInt64 => $macro!(UInt64Type $(, $opt_args)*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $macro!(Int8Type $(, $opt_args)*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $macro!(Int16Type $(, $opt_args)*),
            DataType::Int32 => $macro!(Int32Type $(, $opt_args)*),
            DataType::Int64 => $macro!(Int64Type $(, $opt_args)*),
            DataType::Float32 => $macro!(Float32Type $(, $opt_args)*),
            DataType::Float64 => $macro!(Float64Type $(, $opt_args)*),
            dt => panic!("not implemented for dtype {:?}", dt),
        }
    }};
}

/// Apply a macro on the Downcasted ChunkedArray's
#[macro_export]
macro_rules! match_arrow_data_type_apply_macro_ca {
    ($self:expr, $macro:ident, $macro_string:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $self.dtype() {
            DataType::String => $macro_string!($self.str().unwrap() $(, $opt_args)*),
            DataType::Boolean => $macro_bool!($self.bool().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $macro!($self.u8().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $macro!($self.u16().unwrap() $(, $opt_args)*),
            DataType::UInt32 => $macro!($self.u32().unwrap() $(, $opt_args)*),
            DataType::UInt64 => $macro!($self.u64().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $macro!($self.i8().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $macro!($self.i16().unwrap() $(, $opt_args)*),
            DataType::Int32 => $macro!($self.i32().unwrap() $(, $opt_args)*),
            DataType::Int64 => $macro!($self.i64().unwrap() $(, $opt_args)*),
            DataType::Float32 => $macro!($self.f32().unwrap() $(, $opt_args)*),
            DataType::Float64 => $macro!($self.f64().unwrap() $(, $opt_args)*),
            dt => panic!("not implemented for dtype {:?}", dt),
        }
    }};
}

#[macro_export]
macro_rules! with_match_physical_numeric_type {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use $crate::datatypes::DataType::*;
    match $dtype {
        Int8 => __with_ty__! { i8 },
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        UInt8 => __with_ty__! { u8 },
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

#[macro_export]
macro_rules! with_match_physical_integer_type {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use $crate::datatypes::DataType::*;
    match $dtype {
        #[cfg(feature = "dtype-i8")]
        Int8 => __with_ty__! { i8 },
        #[cfg(feature = "dtype-i16")]
        Int16 => __with_ty__! { i16 },
        Int32 => __with_ty__! { i32 },
        Int64 => __with_ty__! { i64 },
        #[cfg(feature = "dtype-u8")]
        UInt8 => __with_ty__! { u8 },
        #[cfg(feature = "dtype-u16")]
        UInt16 => __with_ty__! { u16 },
        UInt32 => __with_ty__! { u32 },
        UInt64 => __with_ty__! { u64 },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

#[macro_export]
macro_rules! with_match_physical_float_type {(
    $dtype:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use $crate::datatypes::DataType::*;
    match $dtype {
        Float32 => __with_ty__! { f32 },
        Float64 => __with_ty__! { f64 },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

#[macro_export]
macro_rules! with_match_physical_float_polars_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use $crate::datatypes::DataType::*;
    match $key_type {
        Float32 => __with_ty__! { Float32Type },
        Float64 => __with_ty__! { Float64Type },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

#[macro_export]
macro_rules! with_match_physical_numeric_polars_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use $crate::datatypes::DataType::*;
    match $key_type {
            #[cfg(feature = "dtype-i8")]
        Int8 => __with_ty__! { Int8Type },
            #[cfg(feature = "dtype-i16")]
        Int16 => __with_ty__! { Int16Type },
        Int32 => __with_ty__! { Int32Type },
        Int64 => __with_ty__! { Int64Type },
            #[cfg(feature = "dtype-u8")]
        UInt8 => __with_ty__! { UInt8Type },
            #[cfg(feature = "dtype-u16")]
        UInt16 => __with_ty__! { UInt16Type },
        UInt32 => __with_ty__! { UInt32Type },
        UInt64 => __with_ty__! { UInt64Type },
        Float32 => __with_ty__! { Float32Type },
        Float64 => __with_ty__! { Float64Type },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

#[macro_export]
macro_rules! with_match_physical_integer_polars_type {(
    $key_type:expr, | $_:tt $T:ident | $($body:tt)*
) => ({
    macro_rules! __with_ty__ {( $_ $T:ident ) => ( $($body)* )}
    use $crate::datatypes::DataType::*;
    use $crate::datatypes::*;
    match $key_type {
            #[cfg(feature = "dtype-i8")]
        Int8 => __with_ty__! { Int8Type },
            #[cfg(feature = "dtype-i16")]
        Int16 => __with_ty__! { Int16Type },
        Int32 => __with_ty__! { Int32Type },
        Int64 => __with_ty__! { Int64Type },
            #[cfg(feature = "dtype-u8")]
        UInt8 => __with_ty__! { UInt8Type },
            #[cfg(feature = "dtype-u16")]
        UInt16 => __with_ty__! { UInt16Type },
        UInt32 => __with_ty__! { UInt32Type },
        UInt64 => __with_ty__! { UInt64Type },
        dt => panic!("not implemented for dtype {:?}", dt),
    }
})}

/// Apply a macro on the Downcasted ChunkedArray's of DataTypes that are logical numerics.
/// So no logical.
#[macro_export]
macro_rules! downcast_as_macro_arg_physical {
    ($self:expr, $macro:ident $(, $opt_args:expr)*) => {{
        match $self.dtype() {
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $macro!($self.u8().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $macro!($self.u16().unwrap() $(, $opt_args)*),
            DataType::UInt32 => $macro!($self.u32().unwrap() $(, $opt_args)*),
            DataType::UInt64 => $macro!($self.u64().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $macro!($self.i8().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $macro!($self.i16().unwrap() $(, $opt_args)*),
            DataType::Int32 => $macro!($self.i32().unwrap() $(, $opt_args)*),
            DataType::Int64 => $macro!($self.i64().unwrap() $(, $opt_args)*),
            DataType::Float32 => $macro!($self.f32().unwrap() $(, $opt_args)*),
            DataType::Float64 => $macro!($self.f64().unwrap() $(, $opt_args)*),
            dt => panic!("not implemented for {:?}", dt),
        }
    }};
}

/// Apply a macro on the Downcasted ChunkedArray's of DataTypes that are logical numerics.
/// So no logical.
#[macro_export]
macro_rules! downcast_as_macro_arg_physical_mut {
    ($self:expr, $macro:ident $(, $opt_args:expr)*) => {{
        // clone so that we do not borrow
        match $self.dtype().clone() {
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => {
                let ca: &mut UInt8Chunked = $self.as_mut();
                $macro!(UInt8Type, ca $(, $opt_args)*)
            },
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => {
                let ca: &mut UInt16Chunked = $self.as_mut();
                $macro!(UInt16Type, ca $(, $opt_args)*)
            },
            DataType::UInt32 => {
                let ca: &mut UInt32Chunked = $self.as_mut();
                $macro!(UInt32Type, ca $(, $opt_args)*)
            },
            DataType::UInt64 => {
                let ca: &mut UInt64Chunked = $self.as_mut();
                $macro!(UInt64Type, ca $(, $opt_args)*)
            },
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => {
                let ca: &mut Int8Chunked = $self.as_mut();
                $macro!(Int8Type, ca $(, $opt_args)*)
            },
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => {
                let ca: &mut Int16Chunked = $self.as_mut();
                $macro!(Int16Type, ca $(, $opt_args)*)
            },
            DataType::Int32 => {
                let ca: &mut Int32Chunked = $self.as_mut();
                $macro!(Int32Type, ca $(, $opt_args)*)
            },
            DataType::Int64 => {
                let ca: &mut Int64Chunked = $self.as_mut();
                $macro!(Int64Type, ca $(, $opt_args)*)
            },
            DataType::Float32 => {
                let ca: &mut Float32Chunked = $self.as_mut();
                $macro!(Float32Type, ca $(, $opt_args)*)
            },
            DataType::Float64 => {
                let ca: &mut Float64Chunked = $self.as_mut();
                $macro!(Float64Type, ca $(, $opt_args)*)
            },
            dt => panic!("not implemented for {:?}", dt),
        }
    }};
}

#[macro_export]
macro_rules! apply_method_all_arrow_series {
    ($self:expr, $method:ident, $($args:expr),*) => {
        match $self.dtype() {
            DataType::Boolean => $self.bool().unwrap().$method($($args),*),
            DataType::String => $self.str().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $self.u8().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $self.u16().unwrap().$method($($args),*),
            DataType::UInt32 => $self.u32().unwrap().$method($($args),*),
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            DataType::Time => $self.time().unwrap().$method($($args),*),
            DataType::Date => $self.date().unwrap().$method($($args),*),
            DataType::Datetime(_, _) => $self.datetime().unwrap().$method($($args),*),
            DataType::List(_) => $self.list().unwrap().$method($($args),*),
            DataType::Struct(_) => $self.struct_().unwrap().$method($($args),*),
            dt => panic!("dtype {:?} not supported", dt)
        }
    }
}

#[macro_export]
macro_rules! apply_method_physical_integer {
    ($self:expr, $method:ident, $($args:expr),*) => {
        match $self.dtype() {
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $self.u8().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $self.u16().unwrap().$method($($args),*),
            DataType::UInt32 => $self.u32().unwrap().$method($($args),*),
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            dt => panic!("not implemented for dtype {:?}", dt),
        }
    }
}

// doesn't include Bool and String
#[macro_export]
macro_rules! apply_method_physical_numeric {
    ($self:expr, $method:ident, $($args:expr),*) => {
        match $self.dtype() {
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            _ => apply_method_physical_integer!($self, $method, $($args),*),
        }
    }
}

#[macro_export]
macro_rules! df {
    ($($col_name:expr => $slice:expr), + $(,)?) => {
        $crate::prelude::DataFrame::new(vec![
            $(<$crate::prelude::Series as $crate::prelude::NamedFrom::<_, _>>::new($col_name, $slice),)+
        ])
    }
}

pub fn get_time_units(tu_l: &TimeUnit, tu_r: &TimeUnit) -> TimeUnit {
    use TimeUnit::*;
    match (tu_l, tu_r) {
        (Nanoseconds, Microseconds) => Microseconds,
        (_, Milliseconds) => Milliseconds,
        _ => *tu_l,
    }
}

pub fn accumulate_dataframes_vertical_unchecked_optional<I>(dfs: I) -> Option<DataFrame>
where
    I: IntoIterator<Item = DataFrame>,
{
    let mut iter = dfs.into_iter();
    let additional = iter.size_hint().0;
    let mut acc_df = iter.next()?;
    acc_df.reserve_chunks(additional);

    for df in iter {
        acc_df.vstack_mut_unchecked(&df);
    }
    Some(acc_df)
}

/// This takes ownership of the DataFrame so that drop is called earlier.
/// Does not check if schema is correct
pub fn accumulate_dataframes_vertical_unchecked<I>(dfs: I) -> DataFrame
where
    I: IntoIterator<Item = DataFrame>,
{
    let mut iter = dfs.into_iter();
    let additional = iter.size_hint().0;
    let mut acc_df = iter.next().unwrap();
    acc_df.reserve_chunks(additional);

    for df in iter {
        acc_df.vstack_mut_unchecked(&df);
    }
    acc_df
}

/// This takes ownership of the DataFrame so that drop is called earlier.
/// # Panics
/// Panics if `dfs` is empty.
pub fn accumulate_dataframes_vertical<I>(dfs: I) -> PolarsResult<DataFrame>
where
    I: IntoIterator<Item = DataFrame>,
{
    let mut iter = dfs.into_iter();
    let additional = iter.size_hint().0;
    let mut acc_df = iter.next().unwrap();
    acc_df.reserve_chunks(additional);
    for df in iter {
        acc_df.vstack_mut(&df)?;
    }

    Ok(acc_df)
}

/// Concat the DataFrames to a single DataFrame.
pub fn concat_df<'a, I>(dfs: I) -> PolarsResult<DataFrame>
where
    I: IntoIterator<Item = &'a DataFrame>,
{
    let mut iter = dfs.into_iter();
    let additional = iter.size_hint().0;
    let mut acc_df = iter.next().unwrap().clone();
    acc_df.reserve_chunks(additional);
    for df in iter {
        acc_df.vstack_mut(df)?;
    }
    Ok(acc_df)
}

/// Concat the DataFrames to a single DataFrame.
pub fn concat_df_unchecked<'a, I>(dfs: I) -> DataFrame
where
    I: IntoIterator<Item = &'a DataFrame>,
{
    let mut iter = dfs.into_iter();
    let additional = iter.size_hint().0;
    let mut acc_df = iter.next().unwrap().clone();
    acc_df.reserve_chunks(additional);
    for df in iter {
        acc_df.vstack_mut_unchecked(df);
    }
    acc_df
}

pub fn accumulate_dataframes_horizontal(dfs: Vec<DataFrame>) -> PolarsResult<DataFrame> {
    let mut iter = dfs.into_iter();
    let mut acc_df = iter.next().unwrap();
    for df in iter {
        acc_df.hstack_mut(df.get_columns())?;
    }
    Ok(acc_df)
}

/// Ensure the chunks in both ChunkedArrays have the same length.
/// # Panics
/// This will panic if `left.len() != right.len()` and array is chunked.
pub fn align_chunks_binary<'a, T, B>(
    left: &'a ChunkedArray<T>,
    right: &'a ChunkedArray<B>,
) -> (Cow<'a, ChunkedArray<T>>, Cow<'a, ChunkedArray<B>>)
where
    B: PolarsDataType,
    T: PolarsDataType,
{
    let assert = || {
        assert_eq!(
            left.len(),
            right.len(),
            "expected arrays of the same length"
        )
    };
    match (left.chunks.len(), right.chunks.len()) {
        // All chunks are equal length
        (1, 1) => (Cow::Borrowed(left), Cow::Borrowed(right)),
        // All chunks are equal length
        (a, b)
            if a == b
                && left
                    .chunk_lengths()
                    .zip(right.chunk_lengths())
                    .all(|(l, r)| l == r) =>
        {
            (Cow::Borrowed(left), Cow::Borrowed(right))
        },
        (_, 1) => {
            assert();
            (
                Cow::Borrowed(left),
                Cow::Owned(right.match_chunks(left.chunk_lengths())),
            )
        },
        (1, _) => {
            assert();
            (
                Cow::Owned(left.match_chunks(right.chunk_lengths())),
                Cow::Borrowed(right),
            )
        },
        (_, _) => {
            assert();
            // could optimize to choose to rechunk a primitive and not a string or list type
            let left = left.rechunk();
            (
                Cow::Owned(left.match_chunks(right.chunk_lengths())),
                Cow::Borrowed(right),
            )
        },
    }
}

#[cfg(feature = "performant")]
pub(crate) fn align_chunks_binary_owned_series(left: Series, right: Series) -> (Series, Series) {
    match (left.chunks().len(), right.chunks().len()) {
        (1, 1) => (left, right),
        // All chunks are equal length
        (a, b)
            if a == b
                && left
                    .chunk_lengths()
                    .zip(right.chunk_lengths())
                    .all(|(l, r)| l == r) =>
        {
            (left, right)
        },
        (_, 1) => (left.rechunk(), right),
        (1, _) => (left, right.rechunk()),
        (_, _) => (left.rechunk(), right.rechunk()),
    }
}

pub(crate) fn align_chunks_binary_owned<T, B>(
    left: ChunkedArray<T>,
    right: ChunkedArray<B>,
) -> (ChunkedArray<T>, ChunkedArray<B>)
where
    B: PolarsDataType,
    T: PolarsDataType,
{
    match (left.chunks.len(), right.chunks.len()) {
        (1, 1) => (left, right),
        // All chunks are equal length
        (a, b)
            if a == b
                && left
                    .chunk_lengths()
                    .zip(right.chunk_lengths())
                    .all(|(l, r)| l == r) =>
        {
            (left, right)
        },
        (_, 1) => (left.rechunk(), right),
        (1, _) => (left, right.rechunk()),
        (_, _) => (left.rechunk(), right.rechunk()),
    }
}

/// # Panics
/// This will panic if `a.len() != b.len() || b.len() != c.len()` and array is chunked.
#[allow(clippy::type_complexity)]
pub fn align_chunks_ternary<'a, A, B, C>(
    a: &'a ChunkedArray<A>,
    b: &'a ChunkedArray<B>,
    c: &'a ChunkedArray<C>,
) -> (
    Cow<'a, ChunkedArray<A>>,
    Cow<'a, ChunkedArray<B>>,
    Cow<'a, ChunkedArray<C>>,
)
where
    A: PolarsDataType,
    B: PolarsDataType,
    C: PolarsDataType,
{
    if a.chunks.len() == 1 && b.chunks.len() == 1 && c.chunks.len() == 1 {
        return (Cow::Borrowed(a), Cow::Borrowed(b), Cow::Borrowed(c));
    }

    assert!(
        a.len() == b.len() && b.len() == c.len(),
        "expected arrays of the same length"
    );

    match (a.chunks.len(), b.chunks.len(), c.chunks.len()) {
        (_, 1, 1) => (
            Cow::Borrowed(a),
            Cow::Owned(b.match_chunks(a.chunk_lengths())),
            Cow::Owned(c.match_chunks(a.chunk_lengths())),
        ),
        (1, 1, _) => (
            Cow::Owned(a.match_chunks(c.chunk_lengths())),
            Cow::Owned(b.match_chunks(c.chunk_lengths())),
            Cow::Borrowed(c),
        ),
        (1, _, 1) => (
            Cow::Owned(a.match_chunks(b.chunk_lengths())),
            Cow::Borrowed(b),
            Cow::Owned(c.match_chunks(b.chunk_lengths())),
        ),
        (1, _, _) => {
            let b = b.rechunk();
            (
                Cow::Owned(a.match_chunks(c.chunk_lengths())),
                Cow::Owned(b.match_chunks(c.chunk_lengths())),
                Cow::Borrowed(c),
            )
        },
        (_, 1, _) => {
            let a = a.rechunk();
            (
                Cow::Owned(a.match_chunks(c.chunk_lengths())),
                Cow::Owned(b.match_chunks(c.chunk_lengths())),
                Cow::Borrowed(c),
            )
        },
        (_, _, 1) => {
            let b = b.rechunk();
            (
                Cow::Borrowed(a),
                Cow::Owned(b.match_chunks(a.chunk_lengths())),
                Cow::Owned(c.match_chunks(a.chunk_lengths())),
            )
        },
        (len_a, len_b, len_c)
            if len_a == len_b
                && len_b == len_c
                && a.chunk_lengths()
                    .zip(b.chunk_lengths())
                    .zip(c.chunk_lengths())
                    .all(|((a, b), c)| a == b && b == c) =>
        {
            (Cow::Borrowed(a), Cow::Borrowed(b), Cow::Borrowed(c))
        },
        _ => {
            // could optimize to choose to rechunk a primitive and not a string or list type
            let a = a.rechunk();
            let b = b.rechunk();
            (
                Cow::Owned(a.match_chunks(c.chunk_lengths())),
                Cow::Owned(b.match_chunks(c.chunk_lengths())),
                Cow::Borrowed(c),
            )
        },
    }
}

pub fn binary_concatenate_validities<'a, T, B>(
    left: &'a ChunkedArray<T>,
    right: &'a ChunkedArray<B>,
) -> Option<Bitmap>
where
    B: PolarsDataType,
    T: PolarsDataType,
{
    let (left, right) = align_chunks_binary(left, right);
    let left_chunk_refs: Vec<_> = left.chunks().iter().map(|c| &**c).collect();
    let left_validity = concatenate_validities(&left_chunk_refs);
    let right_chunk_refs: Vec<_> = right.chunks().iter().map(|c| &**c).collect();
    let right_validity = concatenate_validities(&right_chunk_refs);
    combine_validities_and(left_validity.as_ref(), right_validity.as_ref())
}

pub trait IntoVec<T> {
    fn into_vec(self) -> Vec<T>;
}

pub trait Arg {}
impl Arg for bool {}

impl IntoVec<bool> for bool {
    fn into_vec(self) -> Vec<bool> {
        vec![self]
    }
}

impl<T: Arg> IntoVec<T> for Vec<T> {
    fn into_vec(self) -> Self {
        self
    }
}

impl<I, S> IntoVec<String> for I
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    fn into_vec(self) -> Vec<String> {
        self.into_iter().map(|s| s.as_ref().to_string()).collect()
    }
}

impl<I, S> IntoVec<SmartString> for I
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    fn into_vec(self) -> Vec<SmartString> {
        self.into_iter().map(|s| s.as_ref().into()).collect()
    }
}

/// This logic is same as the impl on ChunkedArray
/// The difference is that there is less indirection because the caller should preallocate
/// `chunk_lens` once. On the `ChunkedArray` we indirect through an `ArrayRef` which is an indirection
/// and a vtable.
#[inline]
pub(crate) fn index_to_chunked_index<
    I: Iterator<Item = Idx>,
    Idx: PartialOrd + std::ops::AddAssign + std::ops::SubAssign + Zero + One,
>(
    chunk_lens: I,
    index: Idx,
) -> (Idx, Idx) {
    let mut index_remainder = index;
    let mut current_chunk_idx = Zero::zero();

    for chunk_len in chunk_lens {
        if chunk_len > index_remainder {
            break;
        } else {
            index_remainder -= chunk_len;
            current_chunk_idx += One::one();
        }
    }
    (current_chunk_idx, index_remainder)
}

pub(crate) fn index_to_chunked_index_rev<
    I: Iterator<Item = Idx>,
    Idx: PartialOrd
        + std::ops::AddAssign
        + std::ops::SubAssign
        + std::ops::Sub<Output = Idx>
        + Zero
        + One
        + Copy
        + std::fmt::Debug,
>(
    chunk_lens_rev: I,
    index_from_back: Idx,
    total_chunks: Idx,
) -> (Idx, Idx) {
    debug_assert!(index_from_back > Zero::zero(), "at least -1");
    let mut index_remainder = index_from_back;
    let mut current_chunk_idx = One::one();
    let mut current_chunk_len = Zero::zero();

    for chunk_len in chunk_lens_rev {
        current_chunk_len = chunk_len;
        if chunk_len >= index_remainder {
            break;
        } else {
            index_remainder -= chunk_len;
            current_chunk_idx += One::one();
        }
    }
    (
        total_chunks - current_chunk_idx,
        current_chunk_len - index_remainder,
    )
}

pub(crate) fn first_non_null<'a, I>(iter: I) -> Option<usize>
where
    I: Iterator<Item = Option<&'a Bitmap>>,
{
    let mut offset = 0;
    for validity in iter {
        if let Some(validity) = validity {
            let mask = BitMask::from_bitmap(validity);
            if let Some(n) = mask.nth_set_bit_idx(0, 0) {
                return Some(offset + n);
            }
            offset += validity.len()
        } else {
            return Some(offset);
        }
    }
    None
}

pub(crate) fn last_non_null<'a, I>(iter: I, len: usize) -> Option<usize>
where
    I: DoubleEndedIterator<Item = Option<&'a Bitmap>>,
{
    if len == 0 {
        return None;
    }
    let mut offset = 0;
    for validity in iter.rev() {
        if let Some(validity) = validity {
            let mask = BitMask::from_bitmap(validity);
            if let Some(n) = mask.nth_set_bit_idx_rev(0, mask.len()) {
                let mask_start = len - offset - mask.len();
                return Some(mask_start + n);
            }
            offset += validity.len()
        } else {
            return Some(len - 1 - offset);
        }
    }
    None
}

/// ensure that nulls are propagated to both arrays
pub fn coalesce_nulls<'a, T: PolarsDataType>(
    a: &'a ChunkedArray<T>,
    b: &'a ChunkedArray<T>,
) -> (Cow<'a, ChunkedArray<T>>, Cow<'a, ChunkedArray<T>>) {
    if a.null_count() > 0 || b.null_count() > 0 {
        let (a, b) = align_chunks_binary(a, b);
        let mut b = b.into_owned();
        let a = a.coalesce_nulls(b.chunks());

        for arr in a.chunks().iter() {
            for arr_b in unsafe { b.chunks_mut() } {
                *arr_b = arr_b.with_validity(arr.validity().cloned())
            }
        }
        b.compute_len();
        (Cow::Owned(a), Cow::Owned(b))
    } else {
        (Cow::Borrowed(a), Cow::Borrowed(b))
    }
}

pub fn coalesce_nulls_series(a: &Series, b: &Series) -> (Series, Series) {
    if a.null_count() > 0 || b.null_count() > 0 {
        let mut a = a.rechunk();
        let mut b = b.rechunk();
        for (arr_a, arr_b) in unsafe { a.chunks_mut().iter_mut().zip(b.chunks_mut()) } {
            let validity = match (arr_a.validity(), arr_b.validity()) {
                (None, Some(b)) => Some(b.clone()),
                (Some(a), Some(b)) => Some(a & b),
                (Some(a), None) => Some(a.clone()),
                (None, None) => None,
            };
            *arr_a = arr_a.with_validity(validity.clone());
            *arr_b = arr_b.with_validity(validity);
        }
        a.compute_len();
        b.compute_len();
        (a, b)
    } else {
        (a.clone(), b.clone())
    }
}

pub fn operation_exceeded_idxsize_msg(operation: &str) -> String {
    if core::mem::size_of::<IdxSize>() == core::mem::size_of::<u32>() {
        format!(
            "{} exceeded the maximum supported limit of {} rows. Consider installing 'polars-u64-idx'.",
            operation,
            IdxSize::MAX,
        )
    } else {
        format!(
            "{} exceeded the maximum supported limit of {} rows.",
            operation,
            IdxSize::MAX,
        )
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_split() {
        let ca: Int32Chunked = (0..10).collect_ca("a");

        let out = split(&ca, 3);
        assert_eq!(out[0].len(), 3);
        assert_eq!(out[1].len(), 3);
        assert_eq!(out[2].len(), 4);
    }

    #[test]
    fn test_align_chunks() -> PolarsResult<()> {
        let a = Int32Chunked::new("", &[1, 2, 3, 4]);
        let mut b = Int32Chunked::new("", &[1]);
        let b2 = Int32Chunked::new("", &[2, 3, 4]);

        b.append(&b2)?;
        let (a, b) = align_chunks_binary(&a, &b);
        assert_eq!(
            a.chunk_lengths().collect::<Vec<_>>(),
            b.chunk_lengths().collect::<Vec<_>>()
        );

        let a = Int32Chunked::new("", &[1, 2, 3, 4]);
        let mut b = Int32Chunked::new("", &[1]);
        let b1 = b.clone();
        b.append(&b1)?;
        b.append(&b1)?;
        b.append(&b1)?;
        let (a, b) = align_chunks_binary(&a, &b);
        assert_eq!(
            a.chunk_lengths().collect::<Vec<_>>(),
            b.chunk_lengths().collect::<Vec<_>>()
        );

        Ok(())
    }
}
