use crate::prelude::*;
use crate::POOL;
pub use arrow;
#[cfg(feature = "temporal")]
pub use chrono;
pub use num_cpus;
pub use polars_arrow::utils::TrustMyLength;
pub use polars_arrow::utils::*;
pub use rayon;
use rayon::prelude::*;
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

#[repr(transparent)]
pub struct Wrap<T>(pub T);

impl<T> Deref for Wrap<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

unsafe fn index_of_unchecked<T>(slice: &[T], item: &T) -> usize {
    (item as *const _ as usize - slice.as_ptr() as usize) / std::mem::size_of::<T>()
}

fn index_of<T>(slice: &[T], item: &T) -> Option<usize> {
    debug_assert!(std::mem::size_of::<T>() > 0);
    let ptr = item as *const T;
    unsafe {
        if slice.as_ptr() < ptr && slice.as_ptr().add(slice.len()) > ptr {
            Some(index_of_unchecked(slice, item))
        } else {
            None
        }
    }
}

pub(crate) fn set_partition_size() -> usize {
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
pub fn split_ca<T>(ca: &ChunkedArray<T>, n: usize) -> Result<Vec<ChunkedArray<T>>> {
    split_array!(ca, n, i64)
}

#[cfg(feature = "private")]
pub fn split_series(s: &Series, n: usize) -> Result<Vec<Series>> {
    split_array!(s, n, i64)
}

#[cfg(feature = "private")]
pub fn split_df(df: &DataFrame, n: usize) -> Result<Vec<DataFrame>> {
    trait Len {
        fn len(&self) -> usize;
    }
    impl Len for DataFrame {
        fn len(&self) -> usize {
            self.height()
        }
    }
    split_array!(df, n, i64)
}

#[inline]
#[cfg(feature = "private")]
pub fn slice_offsets(offset: i64, length: usize, array_len: usize) -> (usize, usize) {
    let abs_offset = offset.abs() as usize;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Node(pub usize);

impl Default for Node {
    fn default() -> Self {
        Node(usize::MAX)
    }
}

#[derive(Clone)]
#[cfg(feature = "private")]
pub struct Arena<T> {
    items: Vec<T>,
}

impl<T> Default for Arena<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Simple Arena implementation
/// Allocates memory and stores item in a Vec. Only deallocates when being dropped itself.
impl<T> Arena<T> {
    pub fn add(&mut self, val: T) -> Node {
        let idx = self.items.len();
        self.items.push(val);
        Node(idx)
    }

    pub fn pop(&mut self) -> Option<T> {
        self.items.pop()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn new() -> Self {
        Arena { items: vec![] }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Arena {
            items: Vec::with_capacity(cap),
        }
    }

    pub fn get_node(&self, val: &T) -> Option<Node> {
        index_of(&self.items, val).map(Node)
    }

    #[inline]
    pub fn get(&self, idx: Node) -> &T {
        debug_assert!(idx.0 < self.items.len());
        unsafe { self.items.get_unchecked(idx.0) }
    }

    #[inline]
    pub fn get_mut(&mut self, idx: Node) -> &mut T {
        debug_assert!(idx.0 < self.items.len());
        unsafe { self.items.get_unchecked_mut(idx.0) }
    }

    #[inline]
    pub fn replace(&mut self, idx: Node, val: T) {
        let x = self.get_mut(idx);
        *x = val;
    }
}

impl<T: Default> Arena<T> {
    #[inline]
    pub fn take(&mut self, idx: Node) -> T {
        std::mem::take(self.get_mut(idx))
    }

    pub fn replace_with<F>(&mut self, idx: Node, f: F)
    where
        F: FnOnce(T) -> T,
    {
        let val = self.take(idx);
        self.replace(idx, f(val));
    }

    pub fn try_replace_with<F>(&mut self, idx: Node, mut f: F) -> Result<()>
    where
        F: FnMut(T) -> Result<T>,
    {
        let val = self.take(idx);
        self.replace(idx, f(val)?);
        Ok(())
    }
}

/// Apply a macro on the Series
#[macro_export]
macro_rules! match_arrow_data_type_apply_macro {
    ($obj:expr, $macro:ident, $macro_utf8:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $obj {
            DataType::Utf8 => $macro_utf8!($($opt_args)*),
            DataType::Boolean => $macro_bool!($($opt_args)*),
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $macro!(UInt8Type $(, $opt_args)*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $macro!(UInt16Type $(, $opt_args)*),
            DataType::UInt32 => $macro!(UInt32Type $(, $opt_args)*),
            #[cfg(feature = "dtype-u64")]
            DataType::UInt64 => $macro!(UInt64Type $(, $opt_args)*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $macro!(Int8Type $(, $opt_args)*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $macro!(Int16Type $(, $opt_args)*),
            DataType::Int32 => $macro!(Int32Type $(, $opt_args)*),
            DataType::Int64 => $macro!(Int64Type $(, $opt_args)*),
            DataType::Float32 => $macro!(Float32Type $(, $opt_args)*),
            DataType::Float64 => $macro!(Float64Type $(, $opt_args)*),
            DataType::Date32 => $macro!(Date32Type $(, $opt_args)*),
            DataType::Date64 => $macro!(Date64Type $(, $opt_args)*),
            #[cfg(feature = "dtype-time64-ns")]
            DataType::Time64(TimeUnit::Nanosecond) => $macro!(Time64NanosecondType $(, $opt_args)*),
            #[cfg(feature = "dtype-duration-ns")]
            DataType::Duration(TimeUnit::Nanosecond) => $macro!(DurationNanosecondType $(, $opt_args)*),
            #[cfg(feature = "dtype-duration-ms")]
            DataType::Duration(TimeUnit::Millisecond) => $macro!(DurationMillisecondType $(, $opt_args)*),
            _ => unimplemented!(),
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
            #[cfg(feature = "dtype-u64")]
            DataType::UInt64 => $macro!($self.u64().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $macro!($self.i8().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $macro!($self.i16().unwrap() $(, $opt_args)*),
            DataType::Int32 => $macro!($self.i32().unwrap() $(, $opt_args)*),
            DataType::Int64 => $macro!($self.i64().unwrap() $(, $opt_args)*),
            DataType::Float32 => $macro!($self.f32().unwrap() $(, $opt_args)*),
            DataType::Float64 => $macro!($self.f64().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-date32")]
            DataType::Date32 => $macro!($self.date32().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-date64")]
            DataType::Date64 => $macro!($self.date64().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-time64-ns")]
            DataType::Time64(TimeUnit::Nanosecond) => $macro!($self.time64_nanosecond().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-duration-ns")]
            DataType::Duration(TimeUnit::Nanosecond) => $macro!($self.duration_nanosecond().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-duration-ms")]
            DataType::Duration(TimeUnit::Millisecond) => $macro!($self.duration_millisecond().unwrap() $(, $opt_args)*),
            _ => unimplemented!(),
        }
    }};
}

/// Apply a macro on the Downcasted ChunkedArray's of DataTypes that are logical numerics.
/// So no dates.
#[macro_export]
macro_rules! match_arrow_data_type_apply_macro_ca_logical_num {
    ($self:expr, $macro:ident $(, $opt_args:expr)*) => {{
        match $self.dtype() {
            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $macro!($self.u8().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $macro!($self.u16().unwrap() $(, $opt_args)*),
            DataType::UInt32 => $macro!($self.u32().unwrap() $(, $opt_args)*),
            #[cfg(feature = "dtype-u64")]
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
            #[cfg(feature = "dtype-u64")]
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            DataType::Date32 => $self.date32().unwrap().$method($($args),*),
            DataType::Date64 => $self.date64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-time64-ns")]
            DataType::Time64(TimeUnit::Nanosecond) => $self.time64_nanosecond().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-duration-ns")]
            DataType::Duration(TimeUnit::Nanosecond) => $self.duration_nanosecond().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-duration-ms")]
            DataType::Duration(TimeUnit::Millisecond) => $self.duration_millisecond().unwrap().$method($($args),*),
            DataType::List(_) => $self.list().unwrap().$method($($args),*),
            dt => panic!("dtype {:?} not supported", dt)
        }
    }
}

// doesn't include Bool and Utf8
#[macro_export]
macro_rules! apply_method_numeric_series {
    ($self:ident, $method:ident, $($args:expr),*) => {
        match $self.dtype() {

            #[cfg(feature = "dtype-u8")]
            DataType::UInt8 => $self.u8().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-u16")]
            DataType::UInt16 => $self.u16().unwrap().$method($($args),*),
            DataType::UInt32 => $self.u32().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-u64")]
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-date32")]
            DataType::Date32 => $self.date32().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-date64")]
            DataType::Date64 => $self.date64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-time64-ns")]
            DataType::Time64(TimeUnit::Nanosecond) => $self.time64_nanosecond().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-duration-ns")]
            DataType::Duration(TimeUnit::Nanosecond) => $self.duration_nanosecond().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-duration-ms")]
            DataType::Duration(TimeUnit::Millisecond) => $self.duration_millisecond().unwrap().$method($($args),*),

            _ => unimplemented!(),
        }
    }
}

#[macro_export]
macro_rules! static_zip {
    ($selected_keys:ident, 0) => {
        $selected_keys[0].as_groupable_iter()?
    };
    ($selected_keys:ident, 1) => {
        static_zip!($selected_keys, 0).zip($selected_keys[1].as_groupable_iter()?)
    };
    ($selected_keys:ident, 2) => {
        static_zip!($selected_keys, 1).zip($selected_keys[2].as_groupable_iter()?)
    };
    ($selected_keys:ident, 3) => {
        static_zip!($selected_keys, 2).zip($selected_keys[3].as_groupable_iter()?)
    };
    ($selected_keys:ident, 4) => {
        static_zip!($selected_keys, 3).zip($selected_keys[4].as_groupable_iter()?)
    };
    ($selected_keys:ident, 5) => {
        static_zip!($selected_keys, 4).zip($selected_keys[5].as_groupable_iter()?)
    };
    ($selected_keys:ident, 6) => {
        static_zip!($selected_keys, 5).zip($selected_keys[6].as_groupable_iter()?)
    };
    ($selected_keys:ident, 7) => {
        static_zip!($selected_keys, 6).zip($selected_keys[7].as_groupable_iter()?)
    };
    ($selected_keys:ident, 8) => {
        static_zip!($selected_keys, 7).zip($selected_keys[8].as_groupable_iter()?)
    };
    ($selected_keys:ident, 9) => {
        static_zip!($selected_keys, 8).zip($selected_keys[9].as_groupable_iter()?)
    };
    ($selected_keys:ident, 10) => {
        static_zip!($selected_keys, 9).zip($selected_keys[10].as_groupable_iter()?)
    };
    ($selected_keys:ident, 11) => {
        static_zip!($selected_keys, 10).zip($selected_keys[11].as_groupable_iter()?)
    };
}

#[macro_export]
macro_rules! df {
    ($($col_name:expr => $slice:expr), +) => {
        {
            let mut columns = vec![];
            $(
                columns.push(Series::new($col_name, $slice));
            )+
            DataFrame::new(columns)
        }
    }
}

/// Given two datatypes, determine the supertype that both types can safely be cast to
#[cfg(feature = "private")]
pub fn get_supertype(l: &DataType, r: &DataType) -> Result<DataType> {
    match _get_supertype(l, r) {
        Some(dt) => Ok(dt),
        None => _get_supertype(r, l).ok_or_else(|| {
            PolarsError::Other(
                format!("Failed to determine supertype of {:?} and {:?}", l, r).into(),
            )
        }),
    }
}

/// Given two datatypes, determine the supertype that both types can safely be cast to
fn _get_supertype(l: &DataType, r: &DataType) -> Option<DataType> {
    use DataType::*;
    if l == r {
        return Some(l.clone());
    }

    // TODO! add list and temporal types
    match (l, r) {
        (Duration(_), Int8) => Some(Int64),
        (Duration(_), Int16) => Some(Int64),
        (Duration(_), Int32) => Some(Int64),
        (Duration(_), Int64) => Some(Int64),

        (Duration(_), UInt8) => Some(Int64),
        (Duration(_), UInt16) => Some(Int64),
        (Duration(_), UInt32) => Some(Int64),

        (Int8, Duration(_)) => Some(Int64),
        (Int16, Duration(_)) => Some(Int64),
        (Int32, Duration(_)) => Some(Int64),
        (Int64, Duration(_)) => Some(Int64),

        (UInt8, Duration(_)) => Some(Int64),
        (UInt16, Duration(_)) => Some(Int64),
        (UInt32, Duration(_)) => Some(Int64),

        (Float32, Duration(_)) => Some(Float32),
        (Float64, Duration(_)) => Some(Float64),

        (Duration(_), Float32) => Some(Float32),
        (Duration(_), Float64) => Some(Float64),

        (UInt8, Int8) => Some(Int8),
        (UInt8, Int16) => Some(Int16),
        (UInt8, Int32) => Some(Int32),
        (UInt8, Int64) => Some(Int64),

        (UInt16, Int16) => Some(Int32),
        (UInt16, Int32) => Some(Int32),
        (UInt16, Int64) => Some(Int64),

        (UInt32, Int32) => Some(Int64),
        (UInt32, Int64) => Some(Int64),

        (UInt64, Int64) => Some(Int64),

        (Int8, UInt8) => Some(Int8),

        (Int16, UInt8) => Some(Int16),
        (Int16, UInt16) => Some(Int16),

        (Int32, UInt8) => Some(Int32),
        (Int32, UInt16) => Some(Int32),
        (Int32, UInt32) => Some(Int32),

        (Int64, UInt8) => Some(Int64),
        (Int64, UInt16) => Some(Int64),
        (Int64, UInt32) => Some(Int64),
        (Int64, UInt64) => Some(Int64),

        (UInt8, UInt8) => Some(UInt8),
        (UInt8, UInt16) => Some(UInt16),
        (UInt8, UInt32) => Some(UInt32),
        (UInt8, UInt64) => Some(UInt64),
        (UInt8, Float32) => Some(Float32),
        (UInt8, Float64) => Some(Float64),

        (UInt16, UInt8) => Some(UInt16),
        (UInt16, UInt16) => Some(UInt16),
        (UInt16, UInt32) => Some(UInt32),
        (UInt16, UInt64) => Some(UInt64),
        (UInt16, Float32) => Some(Float32),
        (UInt16, Float64) => Some(Float64),

        (UInt32, UInt8) => Some(UInt32),
        (UInt32, UInt16) => Some(UInt32),
        (UInt32, UInt32) => Some(UInt32),
        (UInt32, UInt64) => Some(UInt64),
        (UInt32, Float32) => Some(Float32),
        (UInt32, Float64) => Some(Float64),
        (UInt32, Boolean) => Some(UInt32),

        (UInt64, UInt8) => Some(UInt64),
        (UInt64, UInt16) => Some(UInt64),
        (UInt64, UInt32) => Some(UInt64),
        (UInt64, UInt64) => Some(UInt64),
        (UInt64, Float32) => Some(Float32),
        (UInt64, Float64) => Some(Float64),
        (UInt64, Boolean) => Some(UInt64),

        (Int8, Int8) => Some(Int8),
        (Int8, Int16) => Some(Int16),
        (Int8, Int32) => Some(Int32),
        (Int8, Int64) => Some(Int64),
        (Int8, Float32) => Some(Float32),
        (Int8, Float64) => Some(Float64),
        (Int8, Boolean) => Some(Int8),

        (Int16, Int8) => Some(Int16),
        (Int16, Int16) => Some(Int16),
        (Int16, Int32) => Some(Int32),
        (Int16, Int64) => Some(Int64),
        (Int16, Float32) => Some(Float32),
        (Int16, Float64) => Some(Float64),
        (Int16, Boolean) => Some(Int16),

        (Int32, Int8) => Some(Int32),
        (Int32, Int16) => Some(Int32),
        (Int32, Int32) => Some(Int32),
        (Int32, Int64) => Some(Int64),
        (Int32, Float32) => Some(Float32),
        (Int32, Float64) => Some(Float64),
        (Int32, Date32) => Some(Int32),
        (Int32, Date64) => Some(Int64),
        (Int32, Boolean) => Some(Int32),

        (Int64, Int8) => Some(Int64),
        (Int64, Int16) => Some(Int64),
        (Int64, Int32) => Some(Int64),
        (Int64, Int64) => Some(Int64),
        (Int64, Float32) => Some(Float32),
        (Int64, Float64) => Some(Float64),
        (Int64, Date64) => Some(Int64),
        (Int64, Date32) => Some(Int32),
        (Int64, Boolean) => Some(Int64),

        (Float32, Float32) => Some(Float32),
        (Float32, Float64) => Some(Float64),
        (Float32, Date32) => Some(Float32),
        (Float32, Date64) => Some(Float64),
        (Float64, Float32) => Some(Float64),
        (Float64, Float64) => Some(Float64),
        (Float64, Date32) => Some(Float64),
        (Float64, Date64) => Some(Float64),
        (Float64, Boolean) => Some(Float64),

        (Date32, Int32) => Some(Int32),
        (Date32, Int64) => Some(Int64),
        (Date32, Float32) => Some(Float32),
        (Date32, Float64) => Some(Float64),
        (Date32, Date64) => Some(Date64),

        (Date64, Int32) => Some(Int64),
        (Date64, Int64) => Some(Int64),
        (Date64, Float32) => Some(Float64),
        (Date64, Float64) => Some(Float64),
        (Date64, Date32) => Some(Date64),

        (Utf8, _) => Some(Utf8),
        (_, Utf8) => Some(Utf8),

        (Boolean, Boolean) => Some(Boolean),
        (Boolean, Int8) => Some(Int8),
        (Boolean, Int16) => Some(Int16),
        (Boolean, Int32) => Some(Int32),
        (Boolean, Int64) => Some(Int64),
        (Boolean, UInt8) => Some(UInt8),
        (Boolean, UInt16) => Some(UInt16),
        (Boolean, UInt32) => Some(UInt32),
        (Boolean, UInt64) => Some(UInt64),
        (Boolean, Float32) => Some(Float32),
        (Boolean, Float64) => Some(Float64),

        (dt, Null) => Some(dt.clone()),
        (Null, dt) => Some(dt.clone()),

        _ => None,
    }
}

/// This takes ownership of the DataFrame so that drop is called earlier.
pub fn accumulate_dataframes_vertical<I>(dfs: I) -> Result<DataFrame>
where
    I: IntoIterator<Item = DataFrame>,
{
    let mut iter = dfs.into_iter();
    let mut acc_df = iter.next().unwrap();
    for df in iter {
        acc_df.vstack_mut(&df)?;
    }
    Ok(acc_df)
}

/// Concat the DataFrames to a single DataFrame.
pub fn concat_df<'a, I>(dfs: I) -> Result<DataFrame>
where
    I: IntoIterator<Item = &'a DataFrame>,
{
    let mut iter = dfs.into_iter();
    let mut acc_df = iter.next().unwrap().clone();
    for df in iter {
        acc_df.vstack_mut(df)?;
    }
    Ok(acc_df)
}

pub fn accumulate_dataframes_horizontal(dfs: Vec<DataFrame>) -> Result<DataFrame> {
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
pub fn parallel_op_series<F>(f: F, s: Series, n_threads: Option<usize>) -> Result<Series>
where
    F: Fn(Series) -> Result<Series> + Send + Sync,
{
    let n_threads = n_threads.unwrap_or_else(|| POOL.current_num_threads());
    let slices = split_series(&s, n_threads)?;

    let chunks = POOL.install(|| slices.into_par_iter().map(&f).collect::<Result<Vec<_>>>())?;

    let mut iter = chunks.into_iter();
    let first = iter.next().unwrap();
    let out = iter.fold(first, |mut acc, s| {
        acc.append(&s).unwrap();
        acc
    });

    f(out)
}

pub(crate) fn align_chunks_binary<'a, T, B>(
    left: &'a ChunkedArray<T>,
    right: &'a ChunkedArray<B>,
) -> (Cow<'a, ChunkedArray<T>>, Cow<'a, ChunkedArray<B>>)
where
    ChunkedArray<B>: ChunkOps,
    ChunkedArray<T>: ChunkOps,
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

#[allow(clippy::type_complexity)]
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
    ChunkedArray<A>: ChunkOps,
    ChunkedArray<B>: ChunkOps,
    ChunkedArray<C>: ChunkOps,
    A: PolarsDataType,
    B: PolarsDataType,
    C: PolarsDataType,
{
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

impl IntoVec<bool> for bool {
    fn into_vec(self) -> Vec<bool> {
        vec![self]
    }
}

impl<T> IntoVec<T> for Vec<T> {
    fn into_vec(self) -> Self {
        self
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
