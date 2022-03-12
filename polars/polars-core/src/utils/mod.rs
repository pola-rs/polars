pub(crate) mod series;

use crate::prelude::*;
use crate::POOL;
pub use arrow;
pub use num_cpus;
pub use polars_arrow::utils::TrustMyLength;
pub use polars_arrow::utils::*;
pub use rayon;
use rayon::prelude::*;
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

#[cfg(feature = "private")]
pub use crate::chunked_array::ops::sort::argsort_no_nulls;

#[repr(transparent)]
pub struct Wrap<T>(pub T);

impl<T> Deref for Wrap<T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
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
pub fn split_ca<T>(ca: &ChunkedArray<T>, n: usize) -> Result<Vec<ChunkedArray<T>>>
where
    ChunkedArray<T>: ChunkOps,
{
    split_array!(ca, n, i64)
}

// prefer this one over split_ca, as this can push the null_count into the thread pool
pub fn split_offsets(len: usize, n: usize) -> Vec<(usize, usize)> {
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
    ($obj:expr, $macro:ident, $macro_utf8:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $obj {
            DataType::Utf8 => $macro_utf8!($($opt_args)*),
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

/// Apply a macro on the Downcasted ChunkedArray's of DataTypes that are logical numerics.
/// So no logical.
#[macro_export]
macro_rules! match_arrow_data_type_apply_macro_ca_logical_num {
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
            DataType::Date => $self.date().unwrap().$method($($args),*),
            DataType::Datetime(_, _) => $self.datetime().unwrap().$method($($args),*),
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
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i8")]
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-i16")]
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-date")]
            DataType::Date => $self.date().unwrap().$method($($args),*),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => $self.datetime().unwrap().$method($($args),*),
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
    ($($col_name:expr => $slice:expr), + $(,)?) => {
        {
            DataFrame::new(vec![$(Series::new($col_name, $slice),)+])
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

/// Given two datatypes, determine the supertype that both types can safely be cast to
#[cfg(feature = "private")]
pub fn get_supertype(l: &DataType, r: &DataType) -> Result<DataType> {
    match _get_supertype(l, r) {
        Some(dt) => Ok(dt),
        None => _get_supertype(r, l).ok_or_else(|| {
            PolarsError::ComputeError(
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

    match (l, r) {
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
        #[cfg(feature = "dtype-date")]
        (UInt32, Date) => Some(Int64),
        #[cfg(feature = "dtype-datetime")]
        (UInt32, Datetime(_, _)) => Some(Int64),
        #[cfg(feature = "dtype-duration")]
        (UInt32, Duration(_)) => Some(Int64),

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
        #[cfg(feature = "dtype-date")]
        (Int32, Date) => Some(Int32),
        #[cfg(feature = "dtype-datetime")]
        (Int32, Datetime(_, _)) => Some(Int64),
        #[cfg(feature = "dtype-duration")]
        (Int32, Duration(_)) => Some(Int64),
        #[cfg(feature = "dtype-time")]
        (Int32, Time) => Some(Int64),
        (Int32, Boolean) => Some(Int32),

        (Int64, Int8) => Some(Int64),
        (Int64, Int16) => Some(Int64),
        (Int64, Int32) => Some(Int64),
        (Int64, Int64) => Some(Int64),
        (Int64, Float32) => Some(Float32),
        (Int64, Float64) => Some(Float64),
        #[cfg(feature = "dtype-datetime")]
        (Int64, Datetime(_, _)) => Some(Int64),
        #[cfg(feature = "dtype-duration")]
        (Int64, Duration(_)) => Some(Int64),
        #[cfg(feature = "dtype-date")]
        (Int64, Date) => Some(Int32),
        #[cfg(feature = "dtype-time")]
        (Int64, Time) => Some(Int64),
        (Int64, Boolean) => Some(Int64),

        (Float32, Float32) => Some(Float32),
        (Float32, Float64) => Some(Float64),
        #[cfg(feature = "dtype-date")]
        (Float32, Date) => Some(Float32),
        #[cfg(feature = "dtype-datetime")]
        (Float32, Datetime(_, _)) => Some(Float64),
        #[cfg(feature = "dtype-duration")]
        (Float32, Duration(_)) => Some(Float64),
        #[cfg(feature = "dtype-time")]
        (Float32, Time) => Some(Float64),
        (Float64, Float32) => Some(Float64),
        (Float64, Float64) => Some(Float64),
        #[cfg(feature = "dtype-date")]
        (Float64, Date) => Some(Float64),
        #[cfg(feature = "dtype-datetime")]
        (Float64, Datetime(_, _)) => Some(Float64),
        #[cfg(feature = "dtype-duration")]
        (Float64, Duration(_)) => Some(Float64),
        #[cfg(feature = "dtype-time")]
        (Float64, Time) => Some(Float64),
        (Float64, Boolean) => Some(Float64),

        #[cfg(feature = "dtype-datetime")]
        (Date, UInt32) => Some(Int64),
        #[cfg(feature = "dtype-datetime")]
        (Date, UInt64) => Some(Int64),
        #[cfg(feature = "dtype-datetime")]
        (Date, Int32) => Some(Int32),
        #[cfg(feature = "dtype-datetime")]
        (Date, Int64) => Some(Int64),
        #[cfg(feature = "dtype-datetime")]
        (Date, Float32) => Some(Float32),
        #[cfg(feature = "dtype-datetime")]
        (Date, Float64) => Some(Float64),
        #[cfg(feature = "dtype-datetime")]
        (Date, Datetime(tu, tz)) => Some(Datetime(*tu, tz.clone())),

        #[cfg(feature = "dtype-date")]
        (Datetime(_, _), UInt32) => Some(Int64),
        #[cfg(feature = "dtype-date")]
        (Datetime(_, _), UInt64) => Some(Int64),
        #[cfg(feature = "dtype-date")]
        (Datetime(_, _), Int32) => Some(Int64),
        #[cfg(feature = "dtype-date")]
        (Datetime(_, _), Int64) => Some(Int64),
        #[cfg(feature = "dtype-date")]
        (Datetime(_, _), Float32) => Some(Float64),
        #[cfg(feature = "dtype-date")]
        (Datetime(_, _), Float64) => Some(Float64),
        #[cfg(feature = "dtype-date")]
        (Datetime(tu, tz), Date) => Some(Datetime(*tu, tz.clone())),

        #[cfg(feature = "dtype-duration")]
        (Duration(_), UInt32) => Some(Int64),
        #[cfg(feature = "dtype-duration")]
        (Duration(_), UInt64) => Some(Int64),
        #[cfg(feature = "dtype-duration")]
        (Duration(_), Int32) => Some(Int64),
        #[cfg(feature = "dtype-duration")]
        (Duration(_), Int64) => Some(Int64),
        #[cfg(feature = "dtype-duration")]
        (Duration(_), Float32) => Some(Float64),
        #[cfg(feature = "dtype-duration")]
        (Duration(_), Float64) => Some(Float64),

        #[cfg(feature = "dtype-time")]
        (Time, Int32) => Some(Int64),
        #[cfg(feature = "dtype-time")]
        (Time, Int64) => Some(Int64),
        #[cfg(feature = "dtype-time")]
        (Time, Float32) => Some(Float64),
        #[cfg(feature = "dtype-time")]
        (Time, Float64) => Some(Float64),

        #[cfg(all(feature = "dtype-time", feature = "dtype-datetime"))]
        (Time, Datetime(_, _)) => Some(Int64),
        #[cfg(all(feature = "dtype-time", feature = "dtype-datetime"))]
        (Datetime(_, _), Time) => Some(Int64),
        #[cfg(all(feature = "dtype-time", feature = "dtype-date"))]
        (Time, Date) => Some(Int64),
        #[cfg(all(feature = "dtype-time", feature = "dtype-date"))]
        (Date, Time) => Some(Int64),

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

        (Duration(lu), Datetime(ru, Some(tz))) | (Datetime(lu, Some(tz)), Duration(ru)) => {
            if tz.is_empty() {
                Some(Datetime(get_time_units(lu, ru), None))
            } else {
                Some(Datetime(get_time_units(lu, ru), Some(tz.clone())))
            }
        }
        (Duration(lu), Datetime(ru, None)) | (Datetime(lu, None), Duration(ru)) => {
            Some(Datetime(get_time_units(lu, ru), None))
        }
        (Duration(_), Date) | (Date, Duration(_)) => Some(Datetime(TimeUnit::Milliseconds, None)),
        (Duration(lu), Duration(ru)) => Some(Duration(get_time_units(lu, ru))),

        // None and Some("") timezones
        // we cast from more precision to higher precision as that always fits with occasional loss of precision
        (Datetime(tu_l, tz_l), Datetime(tu_r, tz_r))
            if (tz_l.is_none() || tz_l.as_deref() == Some(""))
                && (tz_r.is_none() || tz_r.as_deref() == Some("")) =>
        {
            let tu = get_time_units(tu_l, tu_r);
            Some(Datetime(tu, None))
        }
        _ => None,
    }
}

/// This takes ownership of the DataFrame so that drop is called earlier.
/// Does not check if schema is correct
pub fn accumulate_dataframes_vertical_unchecked<I>(dfs: I) -> Result<DataFrame>
where
    I: IntoIterator<Item = DataFrame>,
{
    let mut iter = dfs.into_iter();
    let mut acc_df = iter.next().unwrap();
    for df in iter {
        acc_df.vstack_mut_unchecked(&df);
    }
    Ok(acc_df)
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
    let splits = split_offsets(s.len(), n_threads);

    let chunks = POOL.install(|| {
        splits
            .into_par_iter()
            .map(|(offset, len)| {
                let s = s.slice(offset as i64, len);
                f(s)
            })
            .collect::<Result<Vec<_>>>()
    })?;

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

/// # SAFETY
/// `dst` must be valid for `dst.len()` elements, and `src` and `dst` may not overlap.
#[inline]
pub(crate) unsafe fn copy_from_slice_unchecked<T>(src: &[T], dst: &mut [T]) {
    std::ptr::copy_nonoverlapping(src.as_ptr(), dst.as_mut_ptr(), dst.len());
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

    #[test]
    fn test_df_macro_trailing_commas() -> Result<()> {
        let a = df! {
            "a" => &["a one", "a two"],
            "b" => &["b one", "b two"],
            "c" => &[1, 2]
        }?;

        let b = df! {
            "a" => &["a one", "a two"],
            "b" => &["b one", "b two"],
            "c" => &[1, 2],
        }?;

        assert!(a.frame_equal(&b));
        Ok(())
    }
}
