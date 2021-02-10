use crate::prelude::*;
use crate::POOL;
pub use arrow;
#[cfg(feature = "temporal")]
pub use chrono;
pub use num_cpus;
pub use polars_arrow::utils::TrustMyLength;
pub use rayon;
use rayon::prelude::*;
use std::borrow::Cow;
use std::ops::{Deref, DerefMut};

/// Used to split the mantissa and exponent of floating point numbers
/// https://stackoverflow.com/questions/39638363/how-can-i-use-a-hashmap-with-f64-as-key-in-rust
pub(crate) fn integer_decode_f64(val: f64) -> (u64, i16, i8) {
    let bits: u64 = val.to_bits();
    let sign: i8 = if bits >> 63 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };

    exponent -= 1023 + 52;
    (mantissa, exponent, sign)
}

/// Returns the mantissa, exponent and sign as integers.
/// https://github.com/rust-lang/rust/blob/5c674a11471ec0569f616854d715941757a48a0a/src/libcore/num/f32.rs#L203-L216
pub(crate) fn integer_decode_f32(val: f32) -> (u64, i16, i8) {
    let bits: u32 = val.to_bits();
    let sign: i8 = if bits >> 31 == 0 { 1 } else { -1 };
    let mut exponent: i16 = ((bits >> 23) & 0xff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0x7fffff) << 1
    } else {
        (bits & 0x7fffff) | 0x800000
    };
    // Exponent bias + mantissa shift
    exponent -= 127 + 23;
    (mantissa as u64, exponent, sign)
}

pub(crate) fn floating_encode_f64(mantissa: u64, exponent: i16, sign: i8) -> f64 {
    sign as f64 * mantissa as f64 * (2.0f64).powf(exponent as f64)
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

pub fn get_iter_capacity<T, I: Iterator<Item = T>>(iter: &I) -> usize {
    match iter.size_hint() {
        (_lower, Some(upper)) => upper,
        (0, None) => 1024,
        (lower, None) => lower,
    }
}

macro_rules! split_array {
    ($ca: expr, $n: expr) => {{
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
                $ca.slice(i * chunk_size, len)
            })
            .collect::<Result<_>>()?;
        Ok(v)
    }};
}

pub(crate) fn split_ca<T>(ca: &ChunkedArray<T>, n: usize) -> Result<Vec<ChunkedArray<T>>> {
    split_array!(ca, n)
}

pub fn split_series(s: &Series, n: usize) -> Result<Vec<Series>> {
    split_array!(s, n)
}

pub fn split_df(df: &DataFrame, n: usize) -> Result<Vec<DataFrame>> {
    trait Len {
        fn len(&self) -> usize;
    }
    impl Len for DataFrame {
        fn len(&self) -> usize {
            self.height()
        }
    }
    split_array!(df, n)
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Node(pub usize);

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

    pub fn new() -> Self {
        Arena { items: vec![] }
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
    pub fn assign(&mut self, idx: Node, val: T) {
        let x = self.get_mut(idx);
        *x = val;
    }
}

impl<T: Default> Arena<T> {
    pub fn take(&mut self, idx: Node) -> T {
        std::mem::take(self.get_mut(idx))
    }
}

/// An iterator that iterates an unknown at compile time number
/// of iterators simultaneously.
///
/// IMPORTANT: It differs from `std::iter::Zip` in the return type
/// of `next`. It returns a `Vec` instead of a `tuple`, which implies
/// that the result is non-copiable anymore.
pub struct DynamicZip<I>
where
    I: Iterator,
{
    iterators: Vec<I>,
}

impl<I, T> Iterator for DynamicZip<I>
where
    I: Iterator<Item = T>,
{
    type Item = Vec<T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iterators.iter_mut().map(|iter| iter.next()).collect()
    }
}

/// A trait to convert a value to a `DynamicZip`.
pub trait IntoDynamicZip<I>
where
    I: Iterator,
{
    fn into_dynamic_zip(self) -> DynamicZip<I>;
}

impl<I> IntoDynamicZip<I> for Vec<I>
where
    I: Iterator,
{
    fn into_dynamic_zip(self) -> DynamicZip<I> {
        DynamicZip { iterators: self }
    }
}

#[macro_export]
macro_rules! match_arrow_data_type_apply_macro {
    ($obj:expr, $macro:ident, $macro_utf8:ident, $macro_bool:ident $(, $opt_args:expr)*) => {{
        match $obj {
            DataType::Utf8 => $macro_utf8!($($opt_args)*),
            DataType::Boolean => $macro_bool!($($opt_args)*),
            DataType::UInt8 => $macro!(UInt8Type $(, $opt_args)*),
            DataType::UInt16 => $macro!(UInt16Type $(, $opt_args)*),
            DataType::UInt32 => $macro!(UInt32Type $(, $opt_args)*),
            DataType::UInt64 => $macro!(UInt64Type $(, $opt_args)*),
            DataType::Int8 => $macro!(Int8Type $(, $opt_args)*),
            DataType::Int16 => $macro!(Int16Type $(, $opt_args)*),
            DataType::Int32 => $macro!(Int32Type $(, $opt_args)*),
            DataType::Int64 => $macro!(Int64Type $(, $opt_args)*),
            DataType::Float32 => $macro!(Float32Type $(, $opt_args)*),
            DataType::Float64 => $macro!(Float64Type $(, $opt_args)*),
            DataType::Date32 => $macro!(Date32Type $(, $opt_args)*),
            DataType::Date64 => $macro!(Date64Type $(, $opt_args)*),
            DataType::Time64(TimeUnit::Nanosecond) => $macro!(Time64NanosecondType $(, $opt_args)*),
            DataType::Duration(TimeUnit::Nanosecond) => $macro!(DurationNanosecondType $(, $opt_args)*),
            DataType::Duration(TimeUnit::Millisecond) => $macro!(DurationMillisecondType $(, $opt_args)*),
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
            DataType::UInt8 => $self.u8().unwrap().$method($($args),*),
            DataType::UInt16 => $self.u16().unwrap().$method($($args),*),
            DataType::UInt32 => $self.u32().unwrap().$method($($args),*),
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            DataType::Date32 => $self.date32().unwrap().$method($($args),*),
            DataType::Date64 => $self.date64().unwrap().$method($($args),*),
            DataType::Time64(TimeUnit::Nanosecond) => $self.time64_nanosecond().unwrap().$method($($args),*),
            DataType::Duration(TimeUnit::Nanosecond) => $self.duration_nanosecond().unwrap().$method($($args),*),
            DataType::Duration(TimeUnit::Millisecond) => $self.duration_millisecond().unwrap().$method($($args),*),
            DataType::List(_) => $self.list().unwrap().$method($($args),*),
            _ => unimplemented!()
        }
    }
}

// doesn't include Bool and Utf8
#[macro_export]
macro_rules! apply_method_numeric_series {
    ($self:ident, $method:ident, $($args:expr),*) => {
        match $self.dtype() {

            DataType::UInt8 => $self.u8().unwrap().$method($($args),*),
            DataType::UInt16 => $self.u16().unwrap().$method($($args),*),
            DataType::UInt32 => $self.u32().unwrap().$method($($args),*),
            DataType::UInt64 => $self.u64().unwrap().$method($($args),*),
            DataType::Int8 => $self.i8().unwrap().$method($($args),*),
            DataType::Int16 => $self.i16().unwrap().$method($($args),*),
            DataType::Int32 => $self.i32().unwrap().$method($($args),*),
            DataType::Int64 => $self.i64().unwrap().$method($($args),*),
            DataType::Float32 => $self.f32().unwrap().$method($($args),*),
            DataType::Float64 => $self.f64().unwrap().$method($($args),*),
            DataType::Date32 => $self.date32().unwrap().$method($($args),*),
            DataType::Date64 => $self.date64().unwrap().$method($($args),*),
            DataType::Time64(TimeUnit::Nanosecond) => $self.time64_nanosecond().unwrap().$method($($args),*),
            DataType::Duration(TimeUnit::Nanosecond) => $self.duration_nanosecond().unwrap().$method($($args),*),
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

        (UInt16, Int16) => Some(Int16),
        (UInt16, Int32) => Some(Int32),
        (UInt16, Int64) => Some(Int64),

        (UInt32, Int32) => Some(Int32),
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

        (Date64, Int32) => Some(Int64),
        (Date64, Int64) => Some(Int64),
        (Date64, Float32) => Some(Float64),
        (Date64, Float64) => Some(Float64),

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

        _ => None,
    }
}

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

pub fn accumulate_dataframes_horizontal(dfs: Vec<DataFrame>) -> Result<DataFrame> {
    let mut iter = dfs.into_iter();
    let mut acc_df = iter.next().unwrap();
    for df in iter {
        acc_df.hstack_mut(df.get_columns())?;
    }
    Ok(acc_df)
}

#[cfg(target_os = "linux")]
extern "C" {
    #[allow(dead_code)]
    pub fn malloc_trim(__pad: usize) -> std::os::raw::c_int;
}

/// Simple wrapper to parallelize functions that can be divided over threads aggregated and
/// finally aggregated in the main thread. This can be done for sum, min, max, etc.
pub fn parallel_op<F>(f: F, s: Series, n_threads: Option<usize>) -> Result<Series>
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

pub(crate) trait CustomIterTools: Iterator {
    fn fold_first_<F>(mut self, f: F) -> Option<Self::Item>
    where
        Self: Sized,
        F: FnMut(Self::Item, Self::Item) -> Self::Item,
    {
        let first = self.next()?;
        Some(self.fold(first, f))
    }
}

impl<T: ?Sized> CustomIterTools for T where T: Iterator {}

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
