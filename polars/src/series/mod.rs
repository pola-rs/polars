//! Type agnostic columnar data structure.
pub use crate::prelude::ChunkCompare;
use crate::prelude::*;
use arrow::{array::ArrayRef, buffer::Buffer};
use std::mem;
pub(crate) mod aggregate;
pub(crate) mod arithmetic;
mod comparison;
pub(crate) mod iterator;
use crate::chunked_array::builder::get_large_list_builder;
use crate::fmt::FmtLargeList;
use arrow::array::ArrayDataRef;

/// # Series
/// The columnar data type for a DataFrame. The [Series enum](enum.Series.html) consists
/// of typed [ChunkedArray](../chunked_array/struct.ChunkedArray.html)'s. To quickly cast
/// a `Series` to a `ChunkedArray` you can call the method with the name of the type:
///
/// ```
/// # use polars::prelude::*;
/// let s: Series = [1, 2, 3].iter().collect();
/// // Quickly obtain the ChunkedArray wrapped by the Series.
/// let chunked_array = s.i32().unwrap();
/// ```
///
/// ## Arithmetic
///
/// You can do standard arithmetic on series.
/// ```
/// # use polars::prelude::*;
/// let s: Series = [1, 2, 3].iter().collect();
/// let out_add = &s + &s;
/// let out_sub = &s - &s;
/// let out_div = &s / &s;
/// let out_mul = &s * &s;
/// ```
///
/// Or with series and numbers.
///
/// ```
/// # use polars::prelude::*;
/// let s: Series = (1..3).collect();
/// let out_add_one = &s + 1;
/// let out_multiply = &s * 10;
///
/// // Could not overload left hand side operator.
/// let out_divide = 1.div(&s);
/// let out_add = 1.add(&s);
/// let out_subtract = 1.sub(&s);
/// let out_multiply = 1.mul(&s);
/// ```
///
/// ## Comparison
/// You can obtain boolean mask by comparing series.
///
/// ```
/// # use polars::prelude::*;
/// use itertools::Itertools;
/// let s = Series::new("dollars", &[1, 2, 3]);
/// let mask = s.eq(1);
/// let valid = [true, false, false].iter();
/// assert!(mask
///     .into_iter()
///     .map(|opt_bool| opt_bool.unwrap()) // option, because series can be null
///     .zip(valid)
///     .all(|(a, b)| a == *b))
/// ```
///
/// See all the comparison operators in the [CmpOps trait](../chunked_array/comparison/trait.CmpOps.html)
///
/// ## Iterators
/// The Series variants contain differently typed [ChunkedArray's](../chunked_array/struct.ChunkedArray.html).
/// These structs can be turned into iterators, making it possible to use any function/ closure you want
/// on a Series.
///
/// These iterators return an `Option<T>` because the values of a series may be null.
///
/// ```
/// use polars::prelude::*;
/// let pi = 3.14;
/// let s = Series::new("angle", [2f32 * pi, pi, 1.5 * pi].as_ref());
/// let s_cos: Series = s.f32()
///                     .expect("series was not an f32 dtype")
///                     .into_iter()
///                     .map(|opt_angle| opt_angle.map(|angle| angle.cos()))
///                     .collect();
/// ```
///
/// ## Creation
/// Series can be create from different data structures. Below we'll show a few ways we can create
/// a Series object.
///
/// ```
/// # use polars::prelude::*;
/// // Series van be created from Vec's, slices and arrays
/// Series::new("boolean series", &vec![true, false, true]);
/// Series::new("int series", &[1, 2, 3]);
/// // And can be nullable
/// Series::new("got nulls", &[Some(1), None, Some(2)]);
///
/// // Series can also be collected from iterators
/// let from_iter: Series = (0..10)
///     .into_iter()
///     .collect();
///
/// ```
#[derive(Clone)]
pub enum Series {
    UInt8(ChunkedArray<UInt8Type>),
    UInt16(ChunkedArray<UInt16Type>),
    UInt32(ChunkedArray<UInt32Type>),
    UInt64(ChunkedArray<UInt64Type>),
    Int8(ChunkedArray<Int8Type>),
    Int16(ChunkedArray<Int16Type>),
    Int32(ChunkedArray<Int32Type>),
    Int64(ChunkedArray<Int64Type>),
    Float32(ChunkedArray<Float32Type>),
    Float64(ChunkedArray<Float64Type>),
    Utf8(ChunkedArray<Utf8Type>),
    Bool(ChunkedArray<BooleanType>),
    Date32(ChunkedArray<Date32Type>),
    Date64(ChunkedArray<Date64Type>),
    Time32Millisecond(Time32MillisecondChunked),
    Time32Second(Time32SecondChunked),
    Time64Nanosecond(ChunkedArray<Time64NanosecondType>),
    Time64Microsecond(ChunkedArray<Time64MicrosecondType>),
    DurationNanosecond(ChunkedArray<DurationNanosecondType>),
    DurationMicrosecond(DurationMicrosecondChunked),
    DurationMillisecond(DurationMillisecondChunked),
    DurationSecond(DurationSecondChunked),
    IntervalDayTime(IntervalDayTimeChunked),
    IntervalYearMonth(IntervalYearMonthChunked),
    TimestampNanosecond(TimestampNanosecondChunked),
    TimestampMicrosecond(TimestampMicrosecondChunked),
    TimestampMillisecond(TimestampMillisecondChunked),
    TimestampSecond(TimestampSecondChunked),
    LargeList(LargeListChunked),
}

macro_rules! unpack_series {
    ($self:ident, $variant:ident) => {
        if let Series::$variant(ca) = $self {
            Ok(ca)
        } else {
            Err(PolarsError::DataTypeMisMatch)
        }
    };
}

impl Series {
    /// Get Arrow ArrayData
    pub fn array_data(&self) -> Vec<ArrayDataRef> {
        apply_method_all_series!(self, array_data,)
    }

    pub fn from_chunked_array<T: PolarsDataType>(ca: ChunkedArray<T>) -> Self {
        pack_ca_to_series(ca)
    }

    /// Get the lengths of the underlying chunks
    pub fn chunk_lengths(&self) -> &Vec<usize> {
        apply_method_all_series!(self, chunk_id,)
    }
    /// Name of series.
    pub fn name(&self) -> &str {
        apply_method_all_series!(self, name,)
    }

    /// Rename series.
    pub fn rename(&mut self, name: &str) -> &mut Self {
        apply_method_all_series!(self, rename, name);
        self
    }

    /// Get field (used in schema)
    pub fn field(&self) -> &Field {
        apply_method_all_series!(self, ref_field,)
    }

    /// Get datatype of series.
    pub fn dtype(&self) -> &ArrowDataType {
        self.field().data_type()
    }

    /// Underlying chunks.
    pub fn chunks(&self) -> &Vec<ArrayRef> {
        apply_method_all_series!(self, chunks,)
    }

    /// No. of chunks
    pub fn n_chunks(&self) -> usize {
        self.chunks().len()
    }

    pub fn i8(&self) -> Result<&Int8Chunked> {
        unpack_series!(self, Int8)
    }

    pub fn i16(&self) -> Result<&Int16Chunked> {
        unpack_series!(self, Int16)
    }

    /// Unpack to ChunkedArray
    /// ```
    /// # use polars::prelude::*;
    /// let s: Series = [1, 2, 3].iter().collect();
    /// let s_squared: Series = s.i32()
    ///     .unwrap()
    ///     .into_iter()
    ///     .map(|opt_v| {
    ///         match opt_v {
    ///             Some(v) => Some(v * v),
    ///             None => None, // null value
    ///         }
    /// }).collect();
    /// ```
    pub fn i32(&self) -> Result<&Int32Chunked> {
        unpack_series!(self, Int32)
    }

    /// Unpack to ChunkedArray
    pub fn i64(&self) -> Result<&Int64Chunked> {
        unpack_series!(self, Int64)
    }

    /// Unpack to ChunkedArray
    pub fn f32(&self) -> Result<&Float32Chunked> {
        unpack_series!(self, Float32)
    }

    /// Unpack to ChunkedArray
    pub fn f64(&self) -> Result<&Float64Chunked> {
        unpack_series!(self, Float64)
    }

    /// Unpack to ChunkedArray
    pub fn u8(&self) -> Result<&UInt8Chunked> {
        unpack_series!(self, UInt8)
    }

    /// Unpack to ChunkedArray
    pub fn u16(&self) -> Result<&UInt16Chunked> {
        unpack_series!(self, UInt16)
    }

    /// Unpack to ChunkedArray
    pub fn u32(&self) -> Result<&UInt32Chunked> {
        unpack_series!(self, UInt32)
    }

    /// Unpack to ChunkedArray
    pub fn u64(&self) -> Result<&UInt64Chunked> {
        unpack_series!(self, UInt64)
    }

    /// Unpack to ChunkedArray
    pub fn bool(&self) -> Result<&BooleanChunked> {
        unpack_series!(self, Bool)
    }

    /// Unpack to ChunkedArray
    pub fn utf8(&self) -> Result<&Utf8Chunked> {
        unpack_series!(self, Utf8)
    }

    /// Unpack to ChunkedArray
    pub fn date32(&self) -> Result<&Date32Chunked> {
        unpack_series!(self, Date32)
    }

    /// Unpack to ChunkedArray
    pub fn date64(&self) -> Result<&Date64Chunked> {
        unpack_series!(self, Date64)
    }

    /// Unpack to ChunkedArray
    pub fn time32_millisecond(&self) -> Result<&Time32MillisecondChunked> {
        unpack_series!(self, Time32Millisecond)
    }

    /// Unpack to ChunkedArray
    pub fn time32_second(&self) -> Result<&Time32SecondChunked> {
        unpack_series!(self, Time32Second)
    }

    /// Unpack to ChunkedArray
    pub fn time64_nanosecond(&self) -> Result<&Time64NanosecondChunked> {
        unpack_series!(self, Time64Nanosecond)
    }

    /// Unpack to ChunkedArray
    pub fn time64_microsecond(&self) -> Result<&Time64MicrosecondChunked> {
        unpack_series!(self, Time64Microsecond)
    }

    /// Unpack to ChunkedArray
    pub fn duration_nanosecond(&self) -> Result<&DurationNanosecondChunked> {
        unpack_series!(self, DurationNanosecond)
    }

    /// Unpack to ChunkedArray
    pub fn duration_microsecond(&self) -> Result<&DurationMicrosecondChunked> {
        unpack_series!(self, DurationMicrosecond)
    }

    /// Unpack to ChunkedArray
    pub fn duration_millisecond(&self) -> Result<&DurationMillisecondChunked> {
        unpack_series!(self, DurationMillisecond)
    }

    /// Unpack to ChunkedArray
    pub fn duration_second(&self) -> Result<&DurationSecondChunked> {
        unpack_series!(self, DurationSecond)
    }

    /// Unpack to ChunkedArray
    pub fn timestamp_nanosecond(&self) -> Result<&TimestampNanosecondChunked> {
        unpack_series!(self, TimestampNanosecond)
    }

    /// Unpack to ChunkedArray
    pub fn timestamp_microsecond(&self) -> Result<&TimestampMicrosecondChunked> {
        unpack_series!(self, TimestampMicrosecond)
    }

    /// Unpack to ChunkedArray
    pub fn timestamp_millisecond(&self) -> Result<&TimestampMillisecondChunked> {
        unpack_series!(self, TimestampMillisecond)
    }

    /// Unpack to ChunkedArray
    pub fn timestamp_second(&self) -> Result<&TimestampSecondChunked> {
        unpack_series!(self, TimestampSecond)
    }

    /// Unpack to ChunkedArray
    pub fn interval_daytime(&self) -> Result<&IntervalDayTimeChunked> {
        unpack_series!(self, IntervalDayTime)
    }

    /// Unpack to ChunkedArray
    pub fn interval_year_month(&self) -> Result<&IntervalYearMonthChunked> {
        unpack_series!(self, IntervalYearMonth)
    }

    /// Unpack to ChunkedArray
    pub fn large_list(&self) -> Result<&LargeListChunked> {
        unpack_series!(self, LargeList)
    }

    pub fn append_array(&mut self, other: ArrayRef) -> Result<&mut Self> {
        apply_method_all_series!(self, append_array, other)?;
        Ok(self)
    }

    /// Take `num_elements` from the top as a zero copy view.
    pub fn limit(&self, num_elements: usize) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, limit, [num_elements], ?))
    }

    /// Get a zero copy view of the data.
    pub fn slice(&self, offset: usize, length: usize) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, slice, [offset, length], ?))
    }

    /// Append a Series of the same type in place.
    pub fn append(&mut self, other: &Self) -> Result<&mut Self> {
        if self.dtype() == other.dtype() {
            apply_method_all_series!(self, append, other.as_ref());
            Ok(self)
        } else {
            Err(PolarsError::DataTypeMisMatch)
        }
    }

    /// Filter by boolean mask. This operation clones data.
    pub fn filter<T: AsRef<BooleanChunked>>(&self, filter: T) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, filter, [filter.as_ref()], ?))
    }

    /// Take by index from an iterator. This operation clones the data.
    pub fn take_iter(
        &self,
        iter: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, take, [iter,  capacity], ?))
    }

    /// Take by index from an iterator. This operation clones the data.
    pub unsafe fn take_iter_unchecked(
        &self,
        iter: impl Iterator<Item = usize>,
        capacity: Option<usize>,
    ) -> Self {
        apply_method_all_series_and_return!(self, take_unchecked, [iter, capacity],)
    }

    /// Take by index from an iterator. This operation clones the data.
    pub unsafe fn take_opt_iter_unchecked(
        &self,
        iter: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Self {
        apply_method_all_series_and_return!(self, take_opt_unchecked, [iter, capacity],)
    }

    /// Take by index from an iterator. This operation clones the data.
    pub fn take_opt_iter(
        &self,
        iter: impl Iterator<Item = Option<usize>>,
        capacity: Option<usize>,
    ) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, take_opt, [iter,  capacity], ?))
    }

    /// Take by index. This operation is clone.
    pub fn take<T: AsTakeIndex>(&self, indices: &T) -> Result<Self> {
        let mut iter = indices.as_take_iter();
        let capacity = indices.take_index_len();
        self.take_iter(&mut iter, Some(capacity))
    }

    /// Get length of series.
    pub fn len(&self) -> usize {
        apply_method_all_series!(self, len,)
    }

    /// Aggregate all chunks to a contiguous array of memory.
    pub fn rechunk(&self, chunk_lengths: Option<&[usize]>) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, rechunk, [chunk_lengths], ?))
    }

    /// Get the head of the Series.
    pub fn head(&self, length: Option<usize>) -> Self {
        apply_method_all_series_and_return!(self, head, [length],)
    }

    /// Get the tail of the Series.
    pub fn tail(&self, length: Option<usize>) -> Self {
        apply_method_all_series_and_return!(self, tail, [length],)
    }

    /// Cast to some primitive type.
    pub fn cast<N>(&self) -> Result<Self>
    where
        N: PolarsDataType,
    {
        let s = match self {
            Series::Bool(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Utf8(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::UInt8(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::UInt16(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::UInt32(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::UInt64(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Int8(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Int16(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Int32(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Int64(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Float32(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Float64(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Date32(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Date64(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Time32Millisecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Time32Second(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Time64Nanosecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::Time64Microsecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::DurationNanosecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::DurationMicrosecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::DurationMillisecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::DurationSecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::TimestampNanosecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::TimestampMicrosecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::TimestampMillisecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::TimestampSecond(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::IntervalDayTime(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::IntervalYearMonth(arr) => pack_ca_to_series(arr.cast::<N>()?),
            Series::LargeList(arr) => pack_ca_to_series(arr.cast::<N>()?),
        };
        Ok(s)
    }

    /// Get the `ChunkedArray` for some `PolarsDataType`
    pub fn unpack<N>(&self) -> Result<&ChunkedArray<N>>
    where
        N: PolarsDataType,
    {
        macro_rules! unpack_if_match {
            ($ca:ident) => {{
                if *$ca.dtype() == N::get_data_type() {
                    unsafe { Ok(mem::transmute::<_, &ChunkedArray<N>>($ca)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch)
                }
            }};
        }
        match self {
            Series::Bool(arr) => unpack_if_match!(arr),
            Series::Utf8(arr) => unpack_if_match!(arr),
            Series::UInt8(arr) => unpack_if_match!(arr),
            Series::UInt16(arr) => unpack_if_match!(arr),
            Series::UInt32(arr) => unpack_if_match!(arr),
            Series::UInt64(arr) => unpack_if_match!(arr),
            Series::Int8(arr) => unpack_if_match!(arr),
            Series::Int16(arr) => unpack_if_match!(arr),
            Series::Int32(arr) => unpack_if_match!(arr),
            Series::Int64(arr) => unpack_if_match!(arr),
            Series::Float32(arr) => unpack_if_match!(arr),
            Series::Float64(arr) => unpack_if_match!(arr),
            Series::Date32(arr) => unpack_if_match!(arr),
            Series::Date64(arr) => unpack_if_match!(arr),
            Series::Time32Millisecond(arr) => unpack_if_match!(arr),
            Series::Time32Second(arr) => unpack_if_match!(arr),
            Series::Time64Nanosecond(arr) => unpack_if_match!(arr),
            Series::Time64Microsecond(arr) => unpack_if_match!(arr),
            Series::DurationNanosecond(arr) => unpack_if_match!(arr),
            Series::DurationMicrosecond(arr) => unpack_if_match!(arr),
            Series::DurationMillisecond(arr) => unpack_if_match!(arr),
            Series::DurationSecond(arr) => unpack_if_match!(arr),
            Series::TimestampNanosecond(arr) => unpack_if_match!(arr),
            Series::TimestampMicrosecond(arr) => unpack_if_match!(arr),
            Series::TimestampMillisecond(arr) => unpack_if_match!(arr),
            Series::TimestampSecond(arr) => unpack_if_match!(arr),
            Series::IntervalDayTime(arr) => unpack_if_match!(arr),
            Series::IntervalYearMonth(arr) => unpack_if_match!(arr),
            Series::LargeList(arr) => unpack_if_match!(arr),
        }
    }

    /// Get a single value by index. Don't use this operation for loops as a runtime cast is
    /// needed for every iteration.
    pub fn get(&self, index: usize) -> AnyType {
        apply_method_all_series!(self, get_any, index)
    }

    /// Sort in place.
    pub fn sort_in_place(&mut self, reverse: bool) -> &mut Self {
        apply_method_all_series!(self, sort_in_place, reverse);
        self
    }

    pub fn sort(&self, reverse: bool) -> Self {
        apply_method_all_series_and_return!(self, sort, [reverse],)
    }

    /// Retrieve the indexes needed for a sort.
    pub fn argsort(&self, reverse: bool) -> Vec<usize> {
        apply_method_all_series!(self, argsort, reverse)
    }

    /// Count the null values.
    pub fn null_count(&self) -> usize {
        apply_method_all_series!(self, null_count,)
    }

    /// Get unique values in the Series.
    pub fn unique(&self) -> Self {
        apply_method_all_series_and_return!(self, unique, [],)
    }

    /// Get first indexes of unique values.
    pub fn arg_unique(&self) -> Vec<usize> {
        apply_method_all_series!(self, arg_unique,)
    }

    /// Get a mask of the null values.
    pub fn is_null(&self) -> BooleanChunked {
        apply_method_all_series!(self, is_null,)
    }

    /// Get the bits that represent the null values of the underlying ChunkedArray
    pub fn null_bits(&self) -> Vec<(usize, Option<Buffer>)> {
        apply_method_all_series!(self, null_bits,)
    }

    /// return a Series in reversed order
    pub fn reverse(&self) -> Self {
        apply_method_all_series_and_return!(self, reverse, [],)
    }

    /// Rechunk and return a pointer to the start of the Series.
    /// Only implemented for numeric types
    pub fn as_single_ptr(&mut self) -> usize {
        apply_method_numeric_series!(self, as_single_ptr,)
    }

    /// Shift the values by a given period and fill the parts that will be empty due to this operation
    /// with `Nones`.
    ///
    /// *NOTE: If you want to fill the Nones with a value use the
    /// [`shift` operation on `ChunkedArray<T>`](../chunked_array/ops/trait.ChunkShift.html).*
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars::prelude::*;
    /// fn example() -> Result<()> {
    ///     let s = Series::new("series", &[1, 2, 3]);
    ///
    ///     let shifted = s.shift(1)?;
    ///     assert_eq!(Vec::from(shifted.i32()?), &[None, Some(1), Some(2)]);
    ///
    ///     let shifted = s.shift(-1)?;
    ///     assert_eq!(Vec::from(shifted.i32()?), &[Some(1), Some(2), None]);
    ///
    ///     let shifted = s.shift(2)?;
    ///     assert_eq!(Vec::from(shifted.i32()?), &[None, None, Some(1)]);
    ///
    ///     Ok(())
    /// }
    /// example();
    /// ```
    pub fn shift(&self, periods: i32) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, shift, [periods, &None],?))
    }

    /// Replace None values with one of the following strategies:
    /// * Forward fill (replace None with the previous value)
    /// * Backward fill (replace None with the next value)
    /// * Mean fill (replace None with the mean of the whole array)
    /// * Min fill (replace None with the minimum of the whole array)
    /// * Max fill (replace None with the maximum of the whole array)
    ///
    /// *NOTE: If you want to fill the Nones with a value use the
    /// [`fill_none` operation on `ChunkedArray<T>`](../chunked_array/ops/trait.ChunkFillNone.html)*.
    ///
    /// # Example
    ///
    /// ```rust
    /// # use polars::prelude::*;
    /// fn example() -> Result<()> {
    ///     let s = Series::new("some_missing", &[Some(1), None, Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Forward)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Backward)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Min)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Max)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(2), Some(2)]);
    ///
    ///     let filled = s.fill_none(FillNoneStrategy::Mean)?;
    ///     assert_eq!(Vec::from(filled.i32()?), &[Some(1), Some(1), Some(2)]);
    ///
    ///     Ok(())
    /// }
    /// example();
    /// ```
    pub fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Self> {
        Ok(apply_method_all_series_and_return!(self, fill_none, [strategy],?))
    }

    pub(crate) fn fmt_largelist(&self) -> String {
        apply_method_all_series!(self, fmt_largelist,)
    }
}

fn pack_ca_to_series<N: PolarsDataType>(ca: ChunkedArray<N>) -> Series {
    unsafe {
        match N::get_data_type() {
            ArrowDataType::Boolean => Series::Bool(mem::transmute(ca)),
            ArrowDataType::Utf8 => Series::Utf8(mem::transmute(ca)),
            ArrowDataType::UInt8 => Series::UInt8(mem::transmute(ca)),
            ArrowDataType::UInt16 => Series::UInt16(mem::transmute(ca)),
            ArrowDataType::UInt32 => Series::UInt32(mem::transmute(ca)),
            ArrowDataType::UInt64 => Series::UInt64(mem::transmute(ca)),
            ArrowDataType::Int8 => Series::Int8(mem::transmute(ca)),
            ArrowDataType::Int16 => Series::Int16(mem::transmute(ca)),
            ArrowDataType::Int32 => Series::Int32(mem::transmute(ca)),
            ArrowDataType::Int64 => Series::Int64(mem::transmute(ca)),
            ArrowDataType::Float32 => Series::Float32(mem::transmute(ca)),
            ArrowDataType::Float64 => Series::Float64(mem::transmute(ca)),
            ArrowDataType::Date32(DateUnit::Day) => Series::Date32(mem::transmute(ca)),
            ArrowDataType::Date64(DateUnit::Millisecond) => Series::Date64(mem::transmute(ca)),
            ArrowDataType::Time64(datatypes::TimeUnit::Microsecond) => {
                Series::Time64Microsecond(mem::transmute(ca))
            }
            ArrowDataType::Time64(datatypes::TimeUnit::Nanosecond) => {
                Series::Time64Nanosecond(mem::transmute(ca))
            }
            ArrowDataType::Time32(datatypes::TimeUnit::Millisecond) => {
                Series::Time32Millisecond(mem::transmute(ca))
            }
            ArrowDataType::Time32(datatypes::TimeUnit::Second) => {
                Series::Time32Second(mem::transmute(ca))
            }
            ArrowDataType::Duration(datatypes::TimeUnit::Nanosecond) => {
                Series::DurationNanosecond(mem::transmute(ca))
            }
            ArrowDataType::Duration(datatypes::TimeUnit::Microsecond) => {
                Series::DurationMicrosecond(mem::transmute(ca))
            }
            ArrowDataType::Duration(datatypes::TimeUnit::Millisecond) => {
                Series::DurationMillisecond(mem::transmute(ca))
            }
            ArrowDataType::Duration(datatypes::TimeUnit::Second) => {
                Series::DurationSecond(mem::transmute(ca))
            }
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => {
                Series::TimestampNanosecond(mem::transmute(ca))
            }
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => {
                Series::TimestampMicrosecond(mem::transmute(ca))
            }
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => {
                Series::TimestampMillisecond(mem::transmute(ca))
            }
            ArrowDataType::Timestamp(TimeUnit::Second, _) => {
                Series::TimestampSecond(mem::transmute(ca))
            }
            ArrowDataType::Interval(IntervalUnit::YearMonth) => {
                Series::IntervalYearMonth(mem::transmute(ca))
            }
            ArrowDataType::Interval(IntervalUnit::DayTime) => {
                Series::IntervalDayTime(mem::transmute(ca))
            }
            ArrowDataType::LargeList(_) => Series::LargeList(mem::transmute(ca)),
            _ => panic!("Not implemented: {:?}", N::get_data_type()),
        }
    }
}

pub trait NamedFrom<T, Phantom: ?Sized> {
    /// Initialize by name and values.
    fn new(name: &str, _: T) -> Self;
}

macro_rules! impl_named_from {
    ($type:ty, $series_var:ident, $method:ident) => {
        impl<T: AsRef<$type>> NamedFrom<T, $type> for Series {
            fn new(name: &str, v: T) -> Self {
                Series::$series_var(ChunkedArray::$method(name, v.as_ref()))
            }
        }
    };
}

impl<'a, T: AsRef<[&'a str]>> NamedFrom<T, [&'a str]> for Series {
    fn new(name: &str, v: T) -> Self {
        Series::Utf8(ChunkedArray::new_from_slice(name, v.as_ref()))
    }
}
impl<'a, T: AsRef<[Option<&'a str>]>> NamedFrom<T, [Option<&'a str>]> for Series {
    fn new(name: &str, v: T) -> Self {
        Series::Utf8(ChunkedArray::new_from_opt_slice(name, v.as_ref()))
    }
}

impl_named_from!([String], Utf8, new_from_slice);
impl_named_from!([bool], Bool, new_from_slice);
impl_named_from!([u8], UInt8, new_from_slice);
impl_named_from!([u16], UInt16, new_from_slice);
impl_named_from!([u32], UInt32, new_from_slice);
impl_named_from!([u64], UInt64, new_from_slice);
impl_named_from!([i8], Int8, new_from_slice);
impl_named_from!([i16], Int16, new_from_slice);
impl_named_from!([i32], Int32, new_from_slice);
impl_named_from!([i64], Int64, new_from_slice);
impl_named_from!([f32], Float32, new_from_slice);
impl_named_from!([f64], Float64, new_from_slice);
impl_named_from!([Option<String>], Utf8, new_from_opt_slice);
impl_named_from!([Option<bool>], Bool, new_from_opt_slice);
impl_named_from!([Option<u8>], UInt8, new_from_opt_slice);
impl_named_from!([Option<u16>], UInt16, new_from_opt_slice);
impl_named_from!([Option<u32>], UInt32, new_from_opt_slice);
impl_named_from!([Option<u64>], UInt64, new_from_opt_slice);
impl_named_from!([Option<i8>], Int8, new_from_opt_slice);
impl_named_from!([Option<i16>], Int16, new_from_opt_slice);
impl_named_from!([Option<i32>], Int32, new_from_opt_slice);
impl_named_from!([Option<i64>], Int64, new_from_opt_slice);
impl_named_from!([Option<f32>], Float32, new_from_opt_slice);
impl_named_from!([Option<f64>], Float64, new_from_opt_slice);

impl<T: AsRef<[Series]>> NamedFrom<T, LargeListType> for Series {
    fn new(name: &str, s: T) -> Self {
        let series_slice = s.as_ref();
        let dt = series_slice[0].dtype();
        let mut builder = get_large_list_builder(dt, series_slice.len(), name);
        for series in series_slice {
            builder.append_series(series)
        }
        builder.finish().into_series()
    }
}

macro_rules! impl_as_ref_ca {
    ($type:ident, $series_var:ident) => {
        impl AsRef<ChunkedArray<datatypes::$type>> for Series {
            fn as_ref(&self) -> &ChunkedArray<datatypes::$type> {
                match self {
                    Series::$series_var(a) => a,
                    _ => unimplemented!(),
                }
            }
        }
    };
}

impl_as_ref_ca!(UInt8Type, UInt8);
impl_as_ref_ca!(UInt16Type, UInt16);
impl_as_ref_ca!(UInt32Type, UInt32);
impl_as_ref_ca!(UInt64Type, UInt64);
impl_as_ref_ca!(Int8Type, Int8);
impl_as_ref_ca!(Int16Type, Int16);
impl_as_ref_ca!(Int32Type, Int32);
impl_as_ref_ca!(Int64Type, Int64);
impl_as_ref_ca!(Float32Type, Float32);
impl_as_ref_ca!(Float64Type, Float64);
impl_as_ref_ca!(BooleanType, Bool);
impl_as_ref_ca!(Utf8Type, Utf8);
impl_as_ref_ca!(Date32Type, Date32);
impl_as_ref_ca!(Date64Type, Date64);
impl_as_ref_ca!(Time64NanosecondType, Time64Nanosecond);
impl_as_ref_ca!(Time64MicrosecondType, Time64Microsecond);
impl_as_ref_ca!(Time32MillisecondType, Time32Millisecond);
impl_as_ref_ca!(Time32SecondType, Time32Second);
impl_as_ref_ca!(DurationNanosecondType, DurationNanosecond);
impl_as_ref_ca!(DurationMicrosecondType, DurationMicrosecond);
impl_as_ref_ca!(DurationMillisecondType, DurationMillisecond);
impl_as_ref_ca!(DurationSecondType, DurationSecond);
impl_as_ref_ca!(TimestampNanosecondType, TimestampNanosecond);
impl_as_ref_ca!(TimestampMicrosecondType, TimestampMicrosecond);
impl_as_ref_ca!(TimestampMillisecondType, TimestampMillisecond);
impl_as_ref_ca!(TimestampSecondType, TimestampSecond);
impl_as_ref_ca!(IntervalDayTimeType, IntervalDayTime);
impl_as_ref_ca!(IntervalYearMonthType, IntervalYearMonth);
impl_as_ref_ca!(LargeListType, LargeList);

macro_rules! impl_as_mut_ca {
    ($type:ident, $series_var:ident) => {
        impl AsMut<ChunkedArray<datatypes::$type>> for Series {
            fn as_mut(&mut self) -> &mut ChunkedArray<datatypes::$type> {
                match self {
                    Series::$series_var(a) => a,
                    _ => unimplemented!(),
                }
            }
        }
    };
}

impl_as_mut_ca!(UInt8Type, UInt8);
impl_as_mut_ca!(UInt16Type, UInt16);
impl_as_mut_ca!(UInt32Type, UInt32);
impl_as_mut_ca!(UInt64Type, UInt64);
impl_as_mut_ca!(Int8Type, Int8);
impl_as_mut_ca!(Int16Type, Int16);
impl_as_mut_ca!(Int32Type, Int32);
impl_as_mut_ca!(Int64Type, Int64);
impl_as_mut_ca!(Float32Type, Float32);
impl_as_mut_ca!(Float64Type, Float64);
impl_as_mut_ca!(BooleanType, Bool);
impl_as_mut_ca!(Utf8Type, Utf8);
impl_as_mut_ca!(Date32Type, Date32);
impl_as_mut_ca!(Date64Type, Date64);
impl_as_mut_ca!(Time64NanosecondType, Time64Nanosecond);
impl_as_mut_ca!(Time64MicrosecondType, Time64Microsecond);
impl_as_mut_ca!(Time32MillisecondType, Time32Millisecond);
impl_as_mut_ca!(Time32SecondType, Time32Second);
impl_as_mut_ca!(DurationNanosecondType, DurationNanosecond);
impl_as_mut_ca!(DurationMicrosecondType, DurationMicrosecond);
impl_as_mut_ca!(DurationMillisecondType, DurationMillisecond);
impl_as_mut_ca!(DurationSecondType, DurationSecond);
impl_as_mut_ca!(TimestampNanosecondType, TimestampNanosecond);
impl_as_mut_ca!(TimestampMicrosecondType, TimestampMicrosecond);
impl_as_mut_ca!(TimestampMillisecondType, TimestampMillisecond);
impl_as_mut_ca!(TimestampSecondType, TimestampSecond);
impl_as_mut_ca!(IntervalDayTimeType, IntervalDayTime);
impl_as_mut_ca!(IntervalYearMonthType, IntervalYearMonth);
impl_as_mut_ca!(LargeListType, LargeList);

macro_rules! from_series_to_ca {
    ($variant:ident, $ca:ident) => {
        impl<'a> From<&'a Series> for &'a $ca {
            fn from(s: &'a Series) -> Self {
                match s {
                    Series::$variant(ca) => ca,
                    _ => unimplemented!(),
                }
            }
        }
    };
}
from_series_to_ca!(UInt8, UInt8Chunked);
from_series_to_ca!(UInt16, UInt16Chunked);
from_series_to_ca!(UInt32, UInt32Chunked);
from_series_to_ca!(UInt64, UInt64Chunked);
from_series_to_ca!(Int8, Int8Chunked);
from_series_to_ca!(Int16, Int16Chunked);
from_series_to_ca!(Int32, Int32Chunked);
from_series_to_ca!(Int64, Int64Chunked);
from_series_to_ca!(Float32, Float32Chunked);
from_series_to_ca!(Float64, Float64Chunked);
from_series_to_ca!(Bool, BooleanChunked);
from_series_to_ca!(Utf8, Utf8Chunked);
from_series_to_ca!(Date32, Date32Chunked);
from_series_to_ca!(Date64, Date64Chunked);
from_series_to_ca!(Time32Millisecond, Time32MillisecondChunked);
from_series_to_ca!(Time32Second, Time32SecondChunked);
from_series_to_ca!(Time64Microsecond, Time64MicrosecondChunked);
from_series_to_ca!(Time64Nanosecond, Time64NanosecondChunked);
from_series_to_ca!(DurationMillisecond, DurationMillisecondChunked);
from_series_to_ca!(DurationSecond, DurationSecondChunked);
from_series_to_ca!(DurationMicrosecond, DurationMicrosecondChunked);
from_series_to_ca!(DurationNanosecond, DurationNanosecondChunked);
from_series_to_ca!(TimestampMillisecond, TimestampMillisecondChunked);
from_series_to_ca!(TimestampSecond, TimestampSecondChunked);
from_series_to_ca!(TimestampMicrosecond, TimestampMicrosecondChunked);
from_series_to_ca!(TimestampNanosecond, TimestampNanosecondChunked);
from_series_to_ca!(IntervalDayTime, IntervalDayTimeChunked);
from_series_to_ca!(IntervalYearMonth, IntervalYearMonthChunked);
from_series_to_ca!(LargeList, LargeListChunked);

// TODO: add types
impl From<(&str, ArrayRef)> for Series {
    fn from(name_arr: (&str, ArrayRef)) -> Self {
        let (name, arr) = name_arr;
        let chunk = vec![arr];
        match chunk[0].data_type() {
            ArrowDataType::Utf8 => Utf8Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::Boolean => BooleanChunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::UInt8 => UInt8Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::UInt16 => UInt16Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::UInt32 => UInt32Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::UInt64 => UInt64Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::Int8 => Int8Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::Int16 => Int16Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::Int32 => Int32Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::Int64 => Int64Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::Float32 => Float32Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::Float64 => Float64Chunked::new_from_chunks(name, chunk).into_series(),
            ArrowDataType::Date32(DateUnit::Day) => {
                Date32Chunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Date64(DateUnit::Millisecond) => {
                Date64Chunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Time32(TimeUnit::Millisecond) => {
                Time32MillisecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Time32(TimeUnit::Second) => {
                Time32SecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Time64(TimeUnit::Nanosecond) => {
                Time64NanosecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Time64(TimeUnit::Microsecond) => {
                Time64MicrosecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Interval(IntervalUnit::DayTime) => {
                IntervalDayTimeChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Interval(IntervalUnit::YearMonth) => {
                IntervalYearMonthChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Duration(TimeUnit::Nanosecond) => {
                DurationNanosecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Duration(TimeUnit::Microsecond) => {
                DurationMicrosecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Duration(TimeUnit::Millisecond) => {
                DurationMillisecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Duration(TimeUnit::Second) => {
                DurationSecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Timestamp(TimeUnit::Nanosecond, _) => {
                TimestampNanosecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Timestamp(TimeUnit::Microsecond, _) => {
                TimestampMicrosecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Timestamp(TimeUnit::Millisecond, _) => {
                TimestampMillisecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::Timestamp(TimeUnit::Second, _) => {
                TimestampSecondChunked::new_from_chunks(name, chunk).into_series()
            }
            ArrowDataType::LargeList(_) => {
                LargeListChunked::new_from_chunks(name, chunk).into_series()
            }
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn cast() {
        let ar = ChunkedArray::<Int32Type>::new_from_slice("a", &[1, 2]);
        let s = Series::Int32(ar);
        let s2 = s.cast::<Int64Type>().unwrap();
        match s2 {
            Series::Int64(_) => assert!(true),
            _ => assert!(false),
        }
        let s2 = s.cast::<Float32Type>().unwrap();
        match s2 {
            Series::Float32(_) => assert!(true),
            _ => assert!(false),
        }
    }

    #[test]
    fn new_series() {
        Series::new("boolean series", &vec![true, false, true]);
        Series::new("int series", &[1, 2, 3]);
        let ca = Int32Chunked::new_from_slice("a", &[1, 2, 3]);
        ca.into_series();
    }

    #[test]
    fn series_append() {
        let mut s1 = Series::new("a", &[1, 2]);
        let s2 = Series::new("b", &[3]);
        s1.append(&s2).unwrap();
        assert_eq!(s1.len(), 3);

        // add wrong type
        let s2 = Series::new("b", &[3.0]);
        assert!(s1.append(&s2).is_err())
    }
}
