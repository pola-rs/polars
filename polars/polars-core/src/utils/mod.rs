pub(crate) mod series;
mod supertype;

use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

use arrow::bitmap::Bitmap;
pub use polars_arrow::utils::{TrustMyLength, *};
use rayon::prelude::*;
pub use series::*;
pub use supertype::*;
pub use {arrow, rayon};

#[cfg(feature = "private")]
pub use crate::chunked_array::ops::sort::arg_sort_no_nulls;
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

pub fn _set_partition_size() -> usize {
    let mut n_partitions = POOL.current_num_threads();
    // set n_partitions to closes 2^n above the no of threads.
    loop {
        if n_partitions.is_power_of_two() {
            break;
        } else {
            n_partitions += 1;
        }
    }
    n_partitions
}

/// Just a wrapper structure. Useful for certain impl specializations
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

macro_rules! split_array {
    ($ca: expr, $n: expr, $ty : ty) => {{
        if $n == 1 {
            return Ok(vec![$ca.clone()]);
        }
        let total_len = $ca.len();
        let chunk_size = total_len / $n;

        let v = (0..$n)
            .map(|i| {
                let offset = i * chunk_size;
                let len = if i == ($n - 1) {
                    total_len - offset
                } else {
                    chunk_size
                };
                $ca.slice((i * chunk_size) as $ty, len)
            })
            .collect();
        Ok(v)
    }};
}

#[cfg(feature = "private")]
pub fn split_ca<T>(ca: &ChunkedArray<T>, n: usize) -> PolarsResult<Vec<ChunkedArray<T>>>
where
    T: PolarsDataType,
{
    split_array!(ca, n, i64)
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

#[cfg(feature = "private")]
#[doc(hidden)]
pub fn split_series(s: &Series, n: usize) -> PolarsResult<Vec<Series>> {
    split_array!(s, n, i64)
}

fn flatten_df(df: &DataFrame) -> impl Iterator<Item = DataFrame> + '_ {
    df.iter_chunks_physical().flat_map(|chunk| {
        let df = DataFrame::new_no_checks(
            df.iter()
                .zip(chunk.into_arrays())
                .map(|(s, arr)| {
                    // Safety:
                    // datatypes are correct
                    let mut out = unsafe {
                        Series::from_chunks_and_dtype_unchecked(s.name(), vec![arr], s.dtype())
                    };
                    out.set_sorted_flag(s.is_sorted_flag());
                    out
                })
                .collect(),
        );
        if df.height() == 0 {
            None
        } else {
            Some(df)
        }
    })
}

pub fn flatten_series(s: &Series) -> Vec<Series> {
    let name = s.name();
    let dtype = s.dtype();
    unsafe {
        s.chunks()
            .iter()
            .map(|arr| Series::from_chunks_and_dtype_unchecked(name, vec![arr.clone()], dtype))
            .collect()
    }
}

pub fn split_df_as_ref(df: &DataFrame, n: usize) -> PolarsResult<Vec<DataFrame>> {
    let total_len = df.height();
    let chunk_size = total_len / n;

    if df.n_chunks() == n
        && df.get_columns()[0]
            .chunk_lengths()
            .all(|len| len.abs_diff(chunk_size) < 100)
    {
        return Ok(flatten_df(df).collect());
    }

    let mut out = Vec::with_capacity(n);

    for i in 0..n {
        let offset = i * chunk_size;
        let len = if i == (n - 1) {
            total_len - offset
        } else {
            chunk_size
        };
        let df = df.slice((i * chunk_size) as i64, len);
        if df.n_chunks() > 1 {
            // we add every chunk as separate dataframe. This make sure that every partition
            // deals with it.
            out.extend(flatten_df(&df))
        } else {
            out.push(df)
        }
    }

    Ok(out)
}

#[cfg(feature = "private")]
#[doc(hidden)]
/// Split a [`DataFrame`] into `n` parts. We take a `&mut` to be able to repartition/align chunks.
pub fn split_df(df: &mut DataFrame, n: usize) -> PolarsResult<Vec<DataFrame>> {
    if n == 0 || df.height() == 0 {
        return Ok(vec![df.clone()]);
    }
    // make sure that chunks are aligned.
    df.rechunk();
    split_df_as_ref(df, n)
}

pub fn slice_slice<T>(vals: &[T], offset: i64, len: usize) -> &[T] {
    let (raw_offset, slice_len) = slice_offsets(offset, len, vals.len());
    &vals[raw_offset..raw_offset + slice_len]
}

#[inline]
#[cfg(feature = "private")]
#[doc(hidden)]
pub fn slice_offsets(offset: i64, length: usize, array_len: usize) -> (usize, usize) {
    let abs_offset = offset.unsigned_abs() as usize;

    // The offset counted from the start of the array
    // negative index
    if offset < 0 {
        if abs_offset <= array_len {
            (array_len - abs_offset, std::cmp::min(length, abs_offset))
            // negative index larger that array: slice from start
        } else {
            (0, std::cmp::min(length, array_len))
        }
        // positive index
    } else if abs_offset <= array_len {
        (abs_offset, std::cmp::min(length, array_len - abs_offset))
        // empty slice
    } else {
        (array_len, 0)
    }
}

/// Apply a macro on the Series
#[macro_export]
macro_rules! match_dtype_to_physical_apply_macro {
    ($obj:expr, $macro:ident, $macro_utf8:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $obj {
            DataType::Utf8 => $macro_utf8!($($opt_args)*),
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
    ($obj:expr, $macro:ident, $macro_utf8:ident, $macro_binary:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $obj {
            DataType::Utf8 => $macro_utf8!($($opt_args)*),
            #[cfg(feature = "dtype-binary")]
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
    ($self:expr, $macro:ident, $macro_utf8:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $self.dtype() {
            DataType::Utf8 => $macro_utf8!($self.utf8().unwrap() $(, $opt_args)*),
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
            _ => unimplemented!(),
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
        _ => unimplemented!()
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
        _ => unimplemented!()
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
        _ => unimplemented!()
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
            DataType::Utf8 => $self.utf8().unwrap().$method($($args),*),
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
            _ => unimplemented!(),
        }
    }
}

// doesn't include Bool and Utf8
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
        {
            $crate::prelude::DataFrame::new(vec![$($crate::prelude::Series::new($col_name, $slice),)+])
        }
    }
}

#[cfg(feature = "private")]
pub fn get_time_units(tu_l: &TimeUnit, tu_r: &TimeUnit) -> TimeUnit {
    use TimeUnit::*;
    match (tu_l, tu_r) {
        (Nanoseconds, Microseconds) => Microseconds,
        (_, Milliseconds) => Milliseconds,
        _ => *tu_l,
    }
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

/// Simple wrapper to parallelize functions that can be divided over threads aggregated and
/// finally aggregated in the main thread. This can be done for sum, min, max, etc.
#[cfg(feature = "private")]
pub fn parallel_op_series<F>(
    f: F,
    s: Series,
    n_threads: Option<usize>,
) -> PolarsResult<Option<Series>>
where
    F: Fn(Series) -> PolarsResult<Series> + Send + Sync,
{
    let n_threads = n_threads.unwrap_or_else(|| POOL.current_num_threads());
    let splits = _split_offsets(s.len(), n_threads);

    let chunks = POOL.install(|| {
        splits
            .into_par_iter()
            .map(|(offset, len)| {
                let s = s.slice(offset as i64, len);
                f(s)
            })
            .collect::<PolarsResult<Vec<_>>>()
    })?;

    let mut iter = chunks.into_iter();
    let first = iter.next().unwrap();
    let out = iter.fold(first, |mut acc, s| {
        acc.append(&s).unwrap();
        acc
    });

    f(out).map(Some)
}

pub fn align_chunks_binary<'a, T, B>(
    left: &'a ChunkedArray<T>,
    right: &'a ChunkedArray<B>,
) -> (Cow<'a, ChunkedArray<T>>, Cow<'a, ChunkedArray<B>>)
where
    B: PolarsDataType,
    T: PolarsDataType,
{
    match (left.chunks.len(), right.chunks.len()) {
        (1, 1) => (Cow::Borrowed(left), Cow::Borrowed(right)),
        (_, 1) => (
            Cow::Borrowed(left),
            Cow::Owned(right.match_chunks(left.chunk_id())),
        ),
        (1, _) => (
            Cow::Owned(left.match_chunks(right.chunk_id())),
            Cow::Borrowed(right),
        ),
        (_, _) => {
            // could optimize to choose to rechunk a primitive and not a string or list type
            let left = left.rechunk();
            (
                Cow::Owned(left.match_chunks(right.chunk_id())),
                Cow::Borrowed(right),
            )
        }
    }
}

#[cfg(feature = "performant")]
pub(crate) fn align_chunks_binary_owned_series(left: Series, right: Series) -> (Series, Series) {
    match (left.chunks().len(), right.chunks().len()) {
        (1, 1) => (left, right),
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
        (_, 1) => (left.rechunk(), right),
        (1, _) => (left, right.rechunk()),
        (_, _) => (left.rechunk(), right.rechunk()),
    }
}

#[allow(clippy::type_complexity)]
#[cfg(feature = "zip_with")]
pub(crate) fn align_chunks_ternary<'a, A, B, C>(
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
    debug_assert_eq!(a.len(), b.len());
    debug_assert_eq!(b.len(), c.len());
    match (a.chunks.len(), b.chunks.len(), c.chunks.len()) {
        (1, 1, 1) => (Cow::Borrowed(a), Cow::Borrowed(b), Cow::Borrowed(c)),
        (_, 1, 1) => (
            Cow::Borrowed(a),
            Cow::Owned(b.match_chunks(a.chunk_id())),
            Cow::Owned(c.match_chunks(a.chunk_id())),
        ),
        (1, 1, _) => (
            Cow::Owned(a.match_chunks(c.chunk_id())),
            Cow::Owned(b.match_chunks(c.chunk_id())),
            Cow::Borrowed(c),
        ),
        (1, _, 1) => (
            Cow::Owned(a.match_chunks(b.chunk_id())),
            Cow::Borrowed(b),
            Cow::Owned(c.match_chunks(b.chunk_id())),
        ),
        (1, _, _) => {
            let b = b.rechunk();
            (
                Cow::Owned(a.match_chunks(c.chunk_id())),
                Cow::Owned(b.match_chunks(c.chunk_id())),
                Cow::Borrowed(c),
            )
        }
        (_, 1, _) => {
            let a = a.rechunk();
            (
                Cow::Owned(a.match_chunks(c.chunk_id())),
                Cow::Owned(b.match_chunks(c.chunk_id())),
                Cow::Borrowed(c),
            )
        }
        (_, _, 1) => {
            let b = b.rechunk();
            (
                Cow::Borrowed(a),
                Cow::Owned(b.match_chunks(a.chunk_id())),
                Cow::Owned(c.match_chunks(a.chunk_id())),
            )
        }
        _ => {
            // could optimize to choose to rechunk a primitive and not a string or list type
            let a = a.rechunk();
            let b = b.rechunk();
            (
                Cow::Owned(a.match_chunks(c.chunk_id())),
                Cow::Owned(b.match_chunks(c.chunk_id())),
                Cow::Borrowed(c),
            )
        }
    }
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

/// This logic is same as the impl on ChunkedArray
/// The difference is that there is less indirection because the caller should preallocate
/// `chunk_lens` once. On the `ChunkedArray` we indirect through an `ArrayRef` which is an indirection
/// and a vtable.
#[inline]
pub(crate) fn index_to_chunked_index<
    I: Iterator<Item = Idx>,
    Idx: PartialOrd + std::ops::AddAssign + std::ops::SubAssign + num::Zero + num::One,
>(
    chunk_lens: I,
    index: Idx,
) -> (Idx, Idx) {
    let mut index_remainder = index;
    let mut current_chunk_idx = num::Zero::zero();

    for chunk_len in chunk_lens {
        if chunk_len > index_remainder {
            break;
        } else {
            index_remainder -= chunk_len;
            current_chunk_idx += num::One::one();
        }
    }
    (current_chunk_idx, index_remainder)
}

#[cfg(feature = "dtype-struct")]
pub(crate) fn index_to_chunked_index2(chunks: &[ArrayRef], index: usize) -> (usize, usize) {
    let mut index_remainder = index;
    let mut current_chunk_idx = 0;

    for chunk in chunks {
        if chunk.len() > index_remainder {
            break;
        } else {
            index_remainder -= chunk.len();
            current_chunk_idx += 1;
        }
    }
    (current_chunk_idx, index_remainder)
}

/// # SAFETY
/// `dst` must be valid for `dst.len()` elements, and `src` and `dst` may not overlap.
#[inline]
pub(crate) unsafe fn copy_from_slice_unchecked<T>(src: &[T], dst: &mut [T]) {
    std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), dst.len());
}

#[cfg(feature = "chunked_ids")]
pub(crate) fn create_chunked_index_mapping(chunks: &[ArrayRef], len: usize) -> Vec<ChunkId> {
    let mut vals = Vec::with_capacity(len);

    for (chunk_i, chunk) in chunks.iter().enumerate() {
        vals.extend((0..chunk.len()).map(|array_i| [chunk_i as IdxSize, array_i as IdxSize]))
    }

    vals
}

pub(crate) fn first_non_null<'a, I>(iter: I) -> Option<usize>
where
    I: Iterator<Item = Option<&'a Bitmap>>,
{
    let mut offset = 0;
    for validity in iter {
        if let Some(validity) = validity {
            for (idx, is_valid) in validity.iter().enumerate() {
                if is_valid {
                    return Some(offset + idx);
                }
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
    let len = len - 1;
    for validity in iter.rev() {
        if let Some(validity) = validity {
            for (idx, is_valid) in validity.iter().rev().enumerate() {
                if is_valid {
                    return Some(len - (offset + idx));
                }
            }
            offset += validity.len()
        } else {
            return Some(len - offset);
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
        (a, b)
    } else {
        (a.clone(), b.clone())
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_align_chunks() {
        let a = Int32Chunked::new("", &[1, 2, 3, 4]);
        let mut b = Int32Chunked::new("", &[1]);
        let b2 = Int32Chunked::new("", &[2, 3, 4]);

        b.append(&b2);
        let (a, b) = align_chunks_binary(&a, &b);
        assert_eq!(
            a.chunk_id().collect::<Vec<_>>(),
            b.chunk_id().collect::<Vec<_>>()
        );

        let a = Int32Chunked::new("", &[1, 2, 3, 4]);
        let mut b = Int32Chunked::new("", &[1]);
        let b1 = b.clone();
        b.append(&b1);
        b.append(&b1);
        b.append(&b1);
        let (a, b) = align_chunks_binary(&a, &b);
        assert_eq!(
            a.chunk_id().collect::<Vec<_>>(),
            b.chunk_id().collect::<Vec<_>>()
        );
    }
}
