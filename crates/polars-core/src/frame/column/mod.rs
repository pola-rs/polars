use std::borrow::Cow;

use num_traits::{Num, NumCast};
use polars_error::PolarsResult;
use polars_utils::index::check_bounds;
use polars_utils::pl_str::PlSmallStr;
pub use scalar::ScalarColumn;

use self::gather::check_bounds_ca;
use self::partitioned::PartitionedColumn;
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::metadata::{MetadataFlags, MetadataTrait};
use crate::datatypes::ReshapeDimension;
use crate::prelude::*;
use crate::series::{BitRepr, IsSorted, SeriesPhysIter};
use crate::utils::{slice_offsets, Container};
use crate::{HEAD_DEFAULT_LENGTH, TAIL_DEFAULT_LENGTH};

mod arithmetic;
mod compare;
mod partitioned;
mod scalar;

/// A column within a [`DataFrame`].
///
/// This is lazily initialized to a [`Series`] with methods like
/// [`as_materialized_series`][Column::as_materialized_series] and
/// [`take_materialized_series`][Column::take_materialized_series].
///
/// Currently, there are two ways to represent a [`Column`].
/// 1. A [`Series`] of values
/// 2. A [`ScalarColumn`] that repeats a single [`Scalar`]
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[cfg_attr(feature = "serde", serde(from = "Series"))]
#[cfg_attr(feature = "serde", serde(into = "_SerdeSeries"))]
pub enum Column {
    Series(Series),
    Partitioned(PartitionedColumn),
    Scalar(ScalarColumn),
}

/// Convert `Self` into a [`Column`]
pub trait IntoColumn: Sized {
    fn into_column(self) -> Column;
}

impl Column {
    #[inline]
    pub fn new<T, Phantom>(name: PlSmallStr, values: T) -> Self
    where
        Phantom: ?Sized,
        Series: NamedFrom<T, Phantom>,
    {
        Self::Series(NamedFrom::new(name, values))
    }

    #[inline]
    pub fn new_empty(name: PlSmallStr, dtype: &DataType) -> Self {
        Self::new_scalar(name, Scalar::new(dtype.clone(), AnyValue::Null), 0)
    }

    #[inline]
    pub fn new_scalar(name: PlSmallStr, scalar: Scalar, length: usize) -> Self {
        Self::Scalar(ScalarColumn::new(name, scalar, length))
    }

    #[inline]
    pub fn new_partitioned(name: PlSmallStr, scalar: Scalar, length: usize) -> Self {
        Self::Scalar(ScalarColumn::new(name, scalar, length))
    }

    // # Materialize
    /// Get a reference to a [`Series`] for this [`Column`]
    ///
    /// This may need to materialize the [`Series`] on the first invocation for a specific column.
    #[inline]
    pub fn as_materialized_series(&self) -> &Series {
        match self {
            Column::Series(s) => s,
            Column::Partitioned(s) => s.as_materialized_series(),
            Column::Scalar(s) => s.as_materialized_series(),
        }
    }
    /// Turn [`Column`] into a [`Column::Series`].
    ///
    /// This may need to materialize the [`Series`] on the first invocation for a specific column.
    #[inline]
    pub fn into_materialized_series(&mut self) -> &mut Series {
        match self {
            Column::Series(s) => s,
            Column::Partitioned(s) => {
                let series = std::mem::replace(
                    s,
                    PartitionedColumn::new_empty(PlSmallStr::EMPTY, DataType::Null),
                )
                .take_materialized_series();
                *self = Column::Series(series);
                let Column::Series(s) = self else {
                    unreachable!();
                };
                s
            },
            Column::Scalar(s) => {
                let series = std::mem::replace(
                    s,
                    ScalarColumn::new_empty(PlSmallStr::EMPTY, DataType::Null),
                )
                .take_materialized_series();
                *self = Column::Series(series);
                let Column::Series(s) = self else {
                    unreachable!();
                };
                s
            },
        }
    }
    /// Take [`Series`] from a [`Column`]
    ///
    /// This may need to materialize the [`Series`] on the first invocation for a specific column.
    #[inline]
    pub fn take_materialized_series(self) -> Series {
        match self {
            Column::Series(s) => s,
            Column::Partitioned(s) => s.take_materialized_series(),
            Column::Scalar(s) => s.take_materialized_series(),
        }
    }

    #[inline]
    pub fn dtype(&self) -> &DataType {
        match self {
            Column::Series(s) => s.dtype(),
            Column::Partitioned(s) => s.dtype(),
            Column::Scalar(s) => s.dtype(),
        }
    }

    #[inline]
    pub fn field(&self) -> Cow<Field> {
        match self {
            Column::Series(s) => s.field(),
            Column::Partitioned(s) => s.field(),
            Column::Scalar(s) => match s.lazy_as_materialized_series() {
                None => Cow::Owned(Field::new(s.name().clone(), s.dtype().clone())),
                Some(s) => s.field(),
            },
        }
    }

    #[inline]
    pub fn name(&self) -> &PlSmallStr {
        match self {
            Column::Series(s) => s.name(),
            Column::Partitioned(s) => s.name(),
            Column::Scalar(s) => s.name(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Column::Series(s) => s.len(),
            Column::Partitioned(s) => s.len(),
            Column::Scalar(s) => s.len(),
        }
    }

    #[inline]
    pub fn with_name(mut self, name: PlSmallStr) -> Column {
        self.rename(name);
        self
    }

    #[inline]
    pub fn rename(&mut self, name: PlSmallStr) {
        match self {
            Column::Series(s) => _ = s.rename(name),
            Column::Partitioned(s) => _ = s.rename(name),
            Column::Scalar(s) => _ = s.rename(name),
        }
    }

    // # Downcasting
    #[inline]
    pub fn as_series(&self) -> Option<&Series> {
        match self {
            Column::Series(s) => Some(s),
            _ => None,
        }
    }
    #[inline]
    pub fn as_partitioned_column(&self) -> Option<&PartitionedColumn> {
        match self {
            Column::Partitioned(s) => Some(s),
            _ => None,
        }
    }
    #[inline]
    pub fn as_scalar_column(&self) -> Option<&ScalarColumn> {
        match self {
            Column::Scalar(s) => Some(s),
            _ => None,
        }
    }

    // # Try to Chunked Arrays
    pub fn try_bool(&self) -> Option<&BooleanChunked> {
        self.as_materialized_series().try_bool()
    }
    pub fn try_i8(&self) -> Option<&Int8Chunked> {
        self.as_materialized_series().try_i8()
    }
    pub fn try_i16(&self) -> Option<&Int16Chunked> {
        self.as_materialized_series().try_i16()
    }
    pub fn try_i32(&self) -> Option<&Int32Chunked> {
        self.as_materialized_series().try_i32()
    }
    pub fn try_i64(&self) -> Option<&Int64Chunked> {
        self.as_materialized_series().try_i64()
    }
    pub fn try_u8(&self) -> Option<&UInt8Chunked> {
        self.as_materialized_series().try_u8()
    }
    pub fn try_u16(&self) -> Option<&UInt16Chunked> {
        self.as_materialized_series().try_u16()
    }
    pub fn try_u32(&self) -> Option<&UInt32Chunked> {
        self.as_materialized_series().try_u32()
    }
    pub fn try_u64(&self) -> Option<&UInt64Chunked> {
        self.as_materialized_series().try_u64()
    }
    pub fn try_f32(&self) -> Option<&Float32Chunked> {
        self.as_materialized_series().try_f32()
    }
    pub fn try_f64(&self) -> Option<&Float64Chunked> {
        self.as_materialized_series().try_f64()
    }
    pub fn try_str(&self) -> Option<&StringChunked> {
        self.as_materialized_series().try_str()
    }
    pub fn try_list(&self) -> Option<&ListChunked> {
        self.as_materialized_series().try_list()
    }
    pub fn try_binary(&self) -> Option<&BinaryChunked> {
        self.as_materialized_series().try_binary()
    }
    pub fn try_idx(&self) -> Option<&IdxCa> {
        self.as_materialized_series().try_idx()
    }
    pub fn try_binary_offset(&self) -> Option<&BinaryOffsetChunked> {
        self.as_materialized_series().try_binary_offset()
    }
    #[cfg(feature = "dtype-datetime")]
    pub fn try_datetime(&self) -> Option<&DatetimeChunked> {
        self.as_materialized_series().try_datetime()
    }
    #[cfg(feature = "dtype-struct")]
    pub fn try_struct(&self) -> Option<&StructChunked> {
        self.as_materialized_series().try_struct()
    }
    #[cfg(feature = "dtype-decimal")]
    pub fn try_decimal(&self) -> Option<&DecimalChunked> {
        self.as_materialized_series().try_decimal()
    }
    #[cfg(feature = "dtype-array")]
    pub fn try_array(&self) -> Option<&ArrayChunked> {
        self.as_materialized_series().try_array()
    }
    #[cfg(feature = "dtype-categorical")]
    pub fn try_categorical(&self) -> Option<&CategoricalChunked> {
        self.as_materialized_series().try_categorical()
    }
    #[cfg(feature = "dtype-date")]
    pub fn try_date(&self) -> Option<&DateChunked> {
        self.as_materialized_series().try_date()
    }
    #[cfg(feature = "dtype-duration")]
    pub fn try_duration(&self) -> Option<&DurationChunked> {
        self.as_materialized_series().try_duration()
    }

    // # To Chunked Arrays
    pub fn bool(&self) -> PolarsResult<&BooleanChunked> {
        self.as_materialized_series().bool()
    }
    pub fn i8(&self) -> PolarsResult<&Int8Chunked> {
        self.as_materialized_series().i8()
    }
    pub fn i16(&self) -> PolarsResult<&Int16Chunked> {
        self.as_materialized_series().i16()
    }
    pub fn i32(&self) -> PolarsResult<&Int32Chunked> {
        self.as_materialized_series().i32()
    }
    pub fn i64(&self) -> PolarsResult<&Int64Chunked> {
        self.as_materialized_series().i64()
    }
    pub fn u8(&self) -> PolarsResult<&UInt8Chunked> {
        self.as_materialized_series().u8()
    }
    pub fn u16(&self) -> PolarsResult<&UInt16Chunked> {
        self.as_materialized_series().u16()
    }
    pub fn u32(&self) -> PolarsResult<&UInt32Chunked> {
        self.as_materialized_series().u32()
    }
    pub fn u64(&self) -> PolarsResult<&UInt64Chunked> {
        self.as_materialized_series().u64()
    }
    pub fn f32(&self) -> PolarsResult<&Float32Chunked> {
        self.as_materialized_series().f32()
    }
    pub fn f64(&self) -> PolarsResult<&Float64Chunked> {
        self.as_materialized_series().f64()
    }
    pub fn str(&self) -> PolarsResult<&StringChunked> {
        self.as_materialized_series().str()
    }
    pub fn list(&self) -> PolarsResult<&ListChunked> {
        self.as_materialized_series().list()
    }
    pub fn binary(&self) -> PolarsResult<&BinaryChunked> {
        self.as_materialized_series().binary()
    }
    pub fn idx(&self) -> PolarsResult<&IdxCa> {
        self.as_materialized_series().idx()
    }
    pub fn binary_offset(&self) -> PolarsResult<&BinaryOffsetChunked> {
        self.as_materialized_series().binary_offset()
    }
    #[cfg(feature = "dtype-datetime")]
    pub fn datetime(&self) -> PolarsResult<&DatetimeChunked> {
        self.as_materialized_series().datetime()
    }
    #[cfg(feature = "dtype-struct")]
    pub fn struct_(&self) -> PolarsResult<&StructChunked> {
        self.as_materialized_series().struct_()
    }
    #[cfg(feature = "dtype-decimal")]
    pub fn decimal(&self) -> PolarsResult<&DecimalChunked> {
        self.as_materialized_series().decimal()
    }
    #[cfg(feature = "dtype-array")]
    pub fn array(&self) -> PolarsResult<&ArrayChunked> {
        self.as_materialized_series().array()
    }
    #[cfg(feature = "dtype-categorical")]
    pub fn categorical(&self) -> PolarsResult<&CategoricalChunked> {
        self.as_materialized_series().categorical()
    }
    #[cfg(feature = "dtype-date")]
    pub fn date(&self) -> PolarsResult<&DateChunked> {
        self.as_materialized_series().date()
    }
    #[cfg(feature = "dtype-duration")]
    pub fn duration(&self) -> PolarsResult<&DurationChunked> {
        self.as_materialized_series().duration()
    }

    // # Casting
    pub fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Self> {
        match self {
            Column::Series(s) => s.cast_with_options(dtype, options).map(Column::from),
            Column::Partitioned(s) => s.cast_with_options(dtype, options).map(Column::from),
            Column::Scalar(s) => s.cast_with_options(dtype, options).map(Column::from),
        }
    }
    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        match self {
            Column::Series(s) => s.strict_cast(dtype).map(Column::from),
            Column::Partitioned(s) => s.strict_cast(dtype).map(Column::from),
            Column::Scalar(s) => s.strict_cast(dtype).map(Column::from),
        }
    }
    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Column> {
        match self {
            Column::Series(s) => s.cast(dtype).map(Column::from),
            Column::Partitioned(s) => s.cast(dtype).map(Column::from),
            Column::Scalar(s) => s.cast(dtype).map(Column::from),
        }
    }
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Column> {
        match self {
            Column::Series(s) => unsafe { s.cast_unchecked(dtype) }.map(Column::from),
            Column::Partitioned(s) => unsafe { s.cast_unchecked(dtype) }.map(Column::from),
            Column::Scalar(s) => unsafe { s.cast_unchecked(dtype) }.map(Column::from),
        }
    }

    pub fn clear(&self) -> Self {
        match self {
            Column::Series(s) => s.clear().into(),
            Column::Partitioned(s) => s.clear().into(),
            Column::Scalar(s) => s.resize(0).into(),
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        match self {
            Column::Series(s) => s.shrink_to_fit(),
            // @partition-opt
            Column::Partitioned(_) => {},
            Column::Scalar(_) => {},
        }
    }

    #[inline]
    pub fn new_from_index(&self, index: usize, length: usize) -> Self {
        if index >= self.len() {
            return Self::full_null(self.name().clone(), length, self.dtype());
        }

        match self {
            Column::Series(s) => {
                // SAFETY: Bounds check done before.
                let av = unsafe { s.get_unchecked(index) };
                let scalar = Scalar::new(self.dtype().clone(), av.into_static());
                Self::new_scalar(self.name().clone(), scalar, length)
            },
            Column::Partitioned(s) => {
                // SAFETY: Bounds check done before.
                let av = unsafe { s.get_unchecked(index) };
                let scalar = Scalar::new(self.dtype().clone(), av.into_static());
                Self::new_scalar(self.name().clone(), scalar, length)
            },
            Column::Scalar(s) => s.resize(length).into(),
        }
    }

    #[inline]
    pub fn has_nulls(&self) -> bool {
        match self {
            Self::Series(s) => s.has_nulls(),
            // @partition-opt
            Self::Partitioned(s) => s.as_materialized_series().has_nulls(),
            Self::Scalar(s) => s.has_nulls(),
        }
    }

    #[inline]
    pub fn is_null(&self) -> BooleanChunked {
        match self {
            Self::Series(s) => s.is_null(),
            // @partition-opt
            Self::Partitioned(s) => s.as_materialized_series().is_null(),
            Self::Scalar(s) => {
                BooleanChunked::full(s.name().clone(), s.scalar().is_null(), s.len())
            },
        }
    }
    #[inline]
    pub fn is_not_null(&self) -> BooleanChunked {
        match self {
            Self::Series(s) => s.is_not_null(),
            // @partition-opt
            Self::Partitioned(s) => s.as_materialized_series().is_not_null(),
            Self::Scalar(s) => {
                BooleanChunked::full(s.name().clone(), !s.scalar().is_null(), s.len())
            },
        }
    }

    pub fn to_physical_repr(&self) -> Column {
        // @scalar-opt
        self.as_materialized_series()
            .to_physical_repr()
            .into_owned()
            .into()
    }

    pub fn head(&self, length: Option<usize>) -> Column {
        let len = length.unwrap_or(HEAD_DEFAULT_LENGTH);
        let len = usize::min(len, self.len());
        self.slice(0, len)
    }
    pub fn tail(&self, length: Option<usize>) -> Column {
        let len = length.unwrap_or(TAIL_DEFAULT_LENGTH);
        let len = usize::min(len, self.len());
        debug_assert!(len <= i64::MAX as usize);
        self.slice(-(len as i64), len)
    }
    pub fn slice(&self, offset: i64, length: usize) -> Column {
        match self {
            Column::Series(s) => s.slice(offset, length).into(),
            // @partition-opt
            Column::Partitioned(s) => s.as_materialized_series().slice(offset, length).into(),
            Column::Scalar(s) => {
                let (_, length) = slice_offsets(offset, length, s.len());
                s.resize(length).into()
            },
        }
    }

    pub fn split_at(&self, offset: i64) -> (Column, Column) {
        // @scalar-opt
        let (l, r) = self.as_materialized_series().split_at(offset);
        (l.into(), r.into())
    }

    #[inline]
    pub fn null_count(&self) -> usize {
        match self {
            Self::Series(s) => s.null_count(),
            Self::Partitioned(s) => s.null_count(),
            Self::Scalar(s) if s.scalar().is_null() => s.len(),
            Self::Scalar(_) => 0,
        }
    }

    pub fn take(&self, indices: &IdxCa) -> PolarsResult<Column> {
        check_bounds_ca(indices, self.len() as IdxSize)?;
        Ok(unsafe { self.take_unchecked(indices) })
    }
    pub fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Column> {
        check_bounds(indices, self.len() as IdxSize)?;
        Ok(unsafe { self.take_slice_unchecked(indices) })
    }
    /// # Safety
    ///
    /// No bounds on the indexes are performed.
    pub unsafe fn take_unchecked(&self, indices: &IdxCa) -> Column {
        debug_assert!(check_bounds_ca(indices, self.len() as IdxSize).is_ok());

        match self {
            Self::Series(s) => unsafe { s.take_unchecked(indices) }.into(),
            Self::Partitioned(s) => {
                let s = s.as_materialized_series();
                unsafe { s.take_unchecked(indices) }.into()
            },
            Self::Scalar(s) => {
                let idxs_length = indices.len();
                let idxs_null_count = indices.null_count();

                let scalar = ScalarColumn::from_single_value_series(
                    s.as_single_value_series().take_unchecked(&IdxCa::new(
                        indices.name().clone(),
                        &[0][..s.len().min(1)],
                    )),
                    idxs_length,
                );

                // We need to make sure that null values in `idx` become null values in the result
                if idxs_null_count == 0 {
                    scalar.into_column()
                } else if idxs_null_count == idxs_length {
                    scalar.into_nulls().into_column()
                } else {
                    let validity = indices.rechunk_validity();
                    let series = scalar.take_materialized_series();
                    let name = series.name().clone();
                    let dtype = series.dtype().clone();
                    let mut chunks = series.into_chunks();
                    assert_eq!(chunks.len(), 1);
                    chunks[0] = chunks[0].with_validity(validity);
                    unsafe { Series::from_chunks_and_dtype_unchecked(name, chunks, &dtype) }
                        .into_column()
                }
            },
        }
    }
    /// # Safety
    ///
    /// No bounds on the indexes are performed.
    pub unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Column {
        debug_assert!(check_bounds(indices, self.len() as IdxSize).is_ok());

        match self {
            Self::Series(s) => unsafe { s.take_slice_unchecked(indices) }.into(),
            Self::Partitioned(s) => {
                let s = s.as_materialized_series();
                unsafe { s.take_slice_unchecked(indices) }.into()
            },
            Self::Scalar(s) => ScalarColumn::from_single_value_series(
                s.as_single_value_series()
                    .take_slice_unchecked(&[0][..s.len().min(1)]),
                indices.len(),
            )
            .into(),
        }
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_min(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_min(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_max(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_max(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_mean(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_mean(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_sum(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_first(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_first(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_last(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_last(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_n_unique(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_n_unique(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_quantile(
        &self,
        groups: &GroupsProxy,
        quantile: f64,
        method: QuantileMethod,
    ) -> Self {
        // @scalar-opt
        unsafe {
            self.as_materialized_series()
                .agg_quantile(groups, quantile, method)
        }
        .into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_median(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_median(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_var(&self, groups: &GroupsProxy, ddof: u8) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_var(groups, ddof) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub(crate) unsafe fn agg_std(&self, groups: &GroupsProxy, ddof: u8) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_std(groups, ddof) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub unsafe fn agg_list(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_list(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    #[cfg(feature = "algorithm_group_by")]
    pub fn agg_valid_count(&self, groups: &GroupsProxy) -> Self {
        // @partition-opt
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_valid_count(groups) }.into()
    }

    pub fn full_null(name: PlSmallStr, size: usize, dtype: &DataType) -> Self {
        Self::new_scalar(name, Scalar::new(dtype.clone(), AnyValue::Null), size)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn reverse(&self) -> Column {
        match self {
            Column::Series(s) => s.reverse().into(),
            Column::Partitioned(s) => s.reverse().into(),
            Column::Scalar(_) => self.clone(),
        }
    }

    pub fn equals(&self, other: &Column) -> bool {
        // @scalar-opt
        self.as_materialized_series()
            .equals(other.as_materialized_series())
    }

    pub fn equals_missing(&self, other: &Column) -> bool {
        // @scalar-opt
        self.as_materialized_series()
            .equals_missing(other.as_materialized_series())
    }

    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        // @scalar-opt
        match self {
            Column::Series(s) => s.set_sorted_flag(sorted),
            Column::Partitioned(s) => s.set_sorted_flag(sorted),
            Column::Scalar(_) => {},
        }
    }

    pub fn get_flags(&self) -> MetadataFlags {
        match self {
            Column::Series(s) => s.get_flags(),
            // @partition-opt
            Column::Partitioned(_) => MetadataFlags::empty(),
            // @scalar-opt
            Column::Scalar(_) => MetadataFlags::empty(),
        }
    }

    pub fn get_metadata<'a>(&'a self) -> Option<Box<dyn MetadataTrait + 'a>> {
        match self {
            Column::Series(s) => s.boxed_metadata(),
            // @partition-opt
            Column::Partitioned(_) => None,
            // @scalar-opt
            Column::Scalar(_) => None,
        }
    }

    pub fn vec_hash(&self, build_hasher: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        // @scalar-opt?
        self.as_materialized_series().vec_hash(build_hasher, buf)
    }

    pub fn vec_hash_combine(
        &self,
        build_hasher: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        // @scalar-opt?
        self.as_materialized_series()
            .vec_hash_combine(build_hasher, hashes)
    }

    pub fn append(&mut self, other: &Column) -> PolarsResult<&mut Self> {
        // @scalar-opt
        self.into_materialized_series()
            .append(other.as_materialized_series())?;
        Ok(self)
    }

    pub fn arg_sort(&self, options: SortOptions) -> IdxCa {
        // @scalar-opt
        self.as_materialized_series().arg_sort(options)
    }

    pub fn bit_repr(&self) -> Option<BitRepr> {
        // @scalar-opt
        self.as_materialized_series().bit_repr()
    }

    pub fn into_frame(self) -> DataFrame {
        // SAFETY: A single-column dataframe cannot have length mismatches or duplicate names
        unsafe { DataFrame::new_no_checks(self.len(), vec![self]) }
    }

    pub fn extend(&mut self, other: &Column) -> PolarsResult<&mut Self> {
        // @scalar-opt
        self.into_materialized_series()
            .extend(other.as_materialized_series())?;
        Ok(self)
    }

    pub fn rechunk(&self) -> Column {
        match self {
            Column::Series(s) => s.rechunk().into(),
            Column::Partitioned(_) => self.clone(),
            Column::Scalar(_) => self.clone(),
        }
    }

    pub fn explode(&self) -> PolarsResult<Column> {
        self.as_materialized_series().explode().map(Column::from)
    }
    pub fn implode(&self) -> PolarsResult<ListChunked> {
        self.as_materialized_series().implode()
    }

    pub fn fill_null(&self, strategy: FillNullStrategy) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .fill_null(strategy)
            .map(Column::from)
    }

    pub fn divide(&self, rhs: &Column) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .divide(rhs.as_materialized_series())
            .map(Column::from)
    }

    pub fn shift(&self, periods: i64) -> Column {
        // @scalar-opt
        self.as_materialized_series().shift(periods).into()
    }

    #[cfg(feature = "zip_with")]
    pub fn zip_with(&self, mask: &BooleanChunked, other: &Self) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .zip_with(mask, other.as_materialized_series())
            .map(Self::from)
    }

    #[cfg(feature = "zip_with")]
    pub fn zip_with_same_type(
        &self,
        mask: &ChunkedArray<BooleanType>,
        other: &Column,
    ) -> PolarsResult<Column> {
        // @scalar-opt
        self.as_materialized_series()
            .zip_with_same_type(mask, other.as_materialized_series())
            .map(Column::from)
    }

    pub fn drop_nulls(&self) -> Column {
        match self {
            Column::Series(s) => s.drop_nulls().into_column(),
            // @partition-opt
            Column::Partitioned(s) => s.as_materialized_series().drop_nulls().into_column(),
            Column::Scalar(s) => s.drop_nulls().into_column(),
        }
    }

    pub fn is_sorted_flag(&self) -> IsSorted {
        // @scalar-opt
        self.as_materialized_series().is_sorted_flag()
    }

    pub fn unique(&self) -> PolarsResult<Column> {
        match self {
            Column::Series(s) => s.unique().map(Column::from),
            // @partition-opt
            Column::Partitioned(s) => s.as_materialized_series().unique().map(Column::from),
            Column::Scalar(s) => {
                _ = s.as_single_value_series().unique()?;
                if s.is_empty() {
                    return Ok(s.clone().into_column());
                }

                Ok(s.resize(1).into_column())
            },
        }
    }
    pub fn unique_stable(&self) -> PolarsResult<Column> {
        match self {
            Column::Series(s) => s.unique_stable().map(Column::from),
            // @partition-opt
            Column::Partitioned(s) => s.as_materialized_series().unique_stable().map(Column::from),
            Column::Scalar(s) => {
                _ = s.as_single_value_series().unique_stable()?;
                if s.is_empty() {
                    return Ok(s.clone().into_column());
                }

                Ok(s.resize(1).into_column())
            },
        }
    }

    pub fn reshape_list(&self, dimensions: &[ReshapeDimension]) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .reshape_list(dimensions)
            .map(Self::from)
    }

    #[cfg(feature = "dtype-array")]
    pub fn reshape_array(&self, dimensions: &[ReshapeDimension]) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .reshape_array(dimensions)
            .map(Self::from)
    }

    pub fn sort(&self, sort_options: SortOptions) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .sort(sort_options)
            .map(Self::from)
    }

    pub fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Self> {
        match self {
            Column::Series(s) => s.filter(filter).map(Column::from),
            Column::Partitioned(s) => s.as_materialized_series().filter(filter).map(Column::from),
            Column::Scalar(s) => {
                if s.is_empty() {
                    return Ok(s.clone().into_column());
                }

                // Broadcasting
                if filter.len() == 1 {
                    return match filter.get(0) {
                        Some(true) => Ok(s.clone().into_column()),
                        _ => Ok(s.resize(0).into_column()),
                    };
                }

                Ok(s.resize(filter.sum().unwrap() as usize).into_column())
            },
        }
    }

    #[cfg(feature = "random")]
    pub fn shuffle(&self, seed: Option<u64>) -> Self {
        // @scalar-opt
        self.as_materialized_series().shuffle(seed).into()
    }

    #[cfg(feature = "random")]
    pub fn sample_frac(
        &self,
        frac: f64,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        self.as_materialized_series()
            .sample_frac(frac, with_replacement, shuffle, seed)
            .map(Self::from)
    }

    #[cfg(feature = "random")]
    pub fn sample_n(
        &self,
        n: usize,
        with_replacement: bool,
        shuffle: bool,
        seed: Option<u64>,
    ) -> PolarsResult<Self> {
        self.as_materialized_series()
            .sample_n(n, with_replacement, shuffle, seed)
            .map(Self::from)
    }

    pub fn gather_every(&self, n: usize, offset: usize) -> Column {
        if self.len().saturating_sub(offset) == 0 {
            return self.clear();
        }

        match self {
            Column::Series(s) => s.gather_every(n, offset).into(),
            Column::Partitioned(s) => s.as_materialized_series().gather_every(n, offset).into(),
            Column::Scalar(s) => s.resize(s.len() - offset / n).into(),
        }
    }

    pub fn extend_constant(&self, value: AnyValue, n: usize) -> PolarsResult<Self> {
        if self.is_empty() {
            return Ok(Self::new_scalar(
                self.name().clone(),
                Scalar::new(self.dtype().clone(), value.into_static()),
                n,
            ));
        }

        match self {
            Column::Series(s) => s.extend_constant(value, n).map(Column::from),
            Column::Partitioned(s) => s.extend_constant(value, n).map(Column::from),
            Column::Scalar(s) => {
                if s.scalar().as_any_value() == value {
                    Ok(s.resize(s.len() + n).into())
                } else {
                    s.as_materialized_series()
                        .extend_constant(value, n)
                        .map(Column::from)
                }
            },
        }
    }

    pub fn is_finite(&self) -> PolarsResult<BooleanChunked> {
        self.try_map_unary_elementwise_to_bool(|s| s.is_finite())
    }
    pub fn is_infinite(&self) -> PolarsResult<BooleanChunked> {
        self.try_map_unary_elementwise_to_bool(|s| s.is_infinite())
    }
    pub fn is_nan(&self) -> PolarsResult<BooleanChunked> {
        self.try_map_unary_elementwise_to_bool(|s| s.is_nan())
    }
    pub fn is_not_nan(&self) -> PolarsResult<BooleanChunked> {
        self.try_map_unary_elementwise_to_bool(|s| s.is_not_nan())
    }

    pub fn wrapping_trunc_div_scalar<T>(&self, rhs: T) -> Self
    where
        T: Num + NumCast,
    {
        // @scalar-opt
        self.as_materialized_series()
            .wrapping_trunc_div_scalar(rhs)
            .into()
    }

    pub fn product(&self) -> PolarsResult<Scalar> {
        // @scalar-opt
        self.as_materialized_series().product()
    }

    pub fn phys_iter(&self) -> SeriesPhysIter<'_> {
        // @scalar-opt
        self.as_materialized_series().phys_iter()
    }

    #[inline]
    pub fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        polars_ensure!(index < self.len(), oob = index, self.len());

        // SAFETY: Bounds check done just before.
        Ok(unsafe { self.get_unchecked(index) })
    }
    /// # Safety
    ///
    /// Does not perform bounds check on `index`
    #[inline(always)]
    pub unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        debug_assert!(index < self.len());

        match self {
            Column::Series(s) => unsafe { s.get_unchecked(index) },
            Column::Partitioned(s) => unsafe { s.get_unchecked(index) },
            Column::Scalar(s) => s.scalar().as_any_value(),
        }
    }

    #[cfg(feature = "object")]
    pub fn get_object(
        &self,
        index: usize,
    ) -> Option<&dyn crate::chunked_array::object::PolarsObjectSafe> {
        self.as_materialized_series().get_object(index)
    }

    pub fn bitand(&self, rhs: &Self) -> PolarsResult<Self> {
        // @partition-opt
        // @scalar-opt
        (self.as_materialized_series() & rhs.as_materialized_series()).map(Column::from)
    }
    pub fn bitor(&self, rhs: &Self) -> PolarsResult<Self> {
        // @partition-opt
        // @scalar-opt
        (self.as_materialized_series() | rhs.as_materialized_series()).map(Column::from)
    }
    pub fn bitxor(&self, rhs: &Self) -> PolarsResult<Self> {
        // @partition-opt
        // @scalar-opt
        (self.as_materialized_series() ^ rhs.as_materialized_series()).map(Column::from)
    }

    pub fn try_add_owned(self, other: Self) -> PolarsResult<Self> {
        match (self, other) {
            (Column::Series(lhs), Column::Series(rhs)) => lhs.try_add_owned(rhs).map(Column::from),
            (lhs, rhs) => lhs + rhs,
        }
    }
    pub fn try_sub_owned(self, other: Self) -> PolarsResult<Self> {
        match (self, other) {
            (Column::Series(lhs), Column::Series(rhs)) => lhs.try_sub_owned(rhs).map(Column::from),
            (lhs, rhs) => lhs - rhs,
        }
    }
    pub fn try_mul_owned(self, other: Self) -> PolarsResult<Self> {
        match (self, other) {
            (Column::Series(lhs), Column::Series(rhs)) => lhs.try_mul_owned(rhs).map(Column::from),
            (lhs, rhs) => lhs * rhs,
        }
    }

    pub(crate) fn str_value(&self, index: usize) -> PolarsResult<Cow<str>> {
        Ok(self.get(index)?.str_value())
    }

    pub fn min_reduce(&self) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.min_reduce(),
            Column::Partitioned(s) => s.min_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().min_reduce()
            },
        }
    }
    pub fn max_reduce(&self) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.max_reduce(),
            Column::Partitioned(s) => s.max_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().max_reduce()
            },
        }
    }
    pub fn median_reduce(&self) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.median_reduce(),
            Column::Partitioned(s) => s.as_materialized_series().median_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().median_reduce()
            },
        }
    }
    pub fn mean_reduce(&self) -> Scalar {
        match self {
            Column::Series(s) => s.mean_reduce(),
            Column::Partitioned(s) => s.as_materialized_series().mean_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().mean_reduce()
            },
        }
    }
    pub fn std_reduce(&self, ddof: u8) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.std_reduce(ddof),
            Column::Partitioned(s) => s.as_materialized_series().std_reduce(ddof),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().std_reduce(ddof)
            },
        }
    }
    pub fn var_reduce(&self, ddof: u8) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.var_reduce(ddof),
            Column::Partitioned(s) => s.as_materialized_series().var_reduce(ddof),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().var_reduce(ddof)
            },
        }
    }
    pub fn sum_reduce(&self) -> PolarsResult<Scalar> {
        // @partition-opt
        // @scalar-opt
        self.as_materialized_series().sum_reduce()
    }
    pub fn and_reduce(&self) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.and_reduce(),
            Column::Partitioned(s) => s.and_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().and_reduce()
            },
        }
    }
    pub fn or_reduce(&self) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.or_reduce(),
            Column::Partitioned(s) => s.or_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().or_reduce()
            },
        }
    }
    pub fn xor_reduce(&self) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.xor_reduce(),
            // @partition-opt
            Column::Partitioned(s) => s.as_materialized_series().xor_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().xor_reduce()
            },
        }
    }
    pub fn n_unique(&self) -> PolarsResult<usize> {
        match self {
            Column::Series(s) => s.n_unique(),
            Column::Partitioned(s) => s.partitions().n_unique(),
            // @scalar-opt
            Column::Scalar(s) => s.as_single_value_series().n_unique(),
        }
    }
    pub fn quantile_reduce(&self, quantile: f64, method: QuantileMethod) -> PolarsResult<Scalar> {
        self.as_materialized_series()
            .quantile_reduce(quantile, method)
    }

    pub(crate) fn estimated_size(&self) -> usize {
        // @scalar-opt
        self.as_materialized_series().estimated_size()
    }

    pub fn sort_with(&self, options: SortOptions) -> PolarsResult<Self> {
        match self {
            Column::Series(s) => s.sort_with(options).map(Self::from),
            // @partition-opt
            Column::Partitioned(s) => s
                .as_materialized_series()
                .sort_with(options)
                .map(Self::from),
            Column::Scalar(s) => {
                // This makes this function throw the same errors as Series::sort_with
                _ = s.as_single_value_series().sort_with(options)?;

                Ok(self.clone())
            },
        }
    }

    pub fn map_unary_elementwise_to_bool(
        &self,
        f: impl Fn(&Series) -> BooleanChunked,
    ) -> BooleanChunked {
        self.try_map_unary_elementwise_to_bool(|s| Ok(f(s)))
            .unwrap()
    }
    pub fn try_map_unary_elementwise_to_bool(
        &self,
        f: impl Fn(&Series) -> PolarsResult<BooleanChunked>,
    ) -> PolarsResult<BooleanChunked> {
        match self {
            Column::Series(s) => f(s),
            Column::Partitioned(s) => f(s.as_materialized_series()),
            Column::Scalar(s) => Ok(f(&s.as_single_value_series())?.new_from_index(0, s.len())),
        }
    }

    pub fn apply_unary_elementwise(&self, f: impl Fn(&Series) -> Series) -> Column {
        self.try_apply_unary_elementwise(|s| Ok(f(s))).unwrap()
    }
    pub fn try_apply_unary_elementwise(
        &self,
        f: impl Fn(&Series) -> PolarsResult<Series>,
    ) -> PolarsResult<Column> {
        match self {
            Column::Series(s) => f(s).map(Column::from),
            Column::Partitioned(s) => s.try_apply_unary_elementwise(f).map(Self::from),
            Column::Scalar(s) => Ok(ScalarColumn::from_single_value_series(
                f(&s.as_single_value_series())?,
                s.len(),
            )
            .into()),
        }
    }

    pub fn apply_broadcasting_binary_elementwise(
        &self,
        other: &Self,
        op: impl Fn(&Series, &Series) -> Series,
    ) -> PolarsResult<Column> {
        self.try_apply_broadcasting_binary_elementwise(other, |lhs, rhs| Ok(op(lhs, rhs)))
    }
    pub fn try_apply_broadcasting_binary_elementwise(
        &self,
        other: &Self,
        op: impl Fn(&Series, &Series) -> PolarsResult<Series>,
    ) -> PolarsResult<Column> {
        fn output_length(a: &Column, b: &Column) -> PolarsResult<usize> {
            match (a.len(), b.len()) {
                // broadcasting
                (1, o) | (o, 1) => Ok(o),
                // equal
                (a, b) if a == b => Ok(a),
                // unequal
                (a, b) => {
                    polars_bail!(InvalidOperation: "cannot do a binary operation on columns of different lengths: got {} and {}", a, b)
                },
            }
        }

        // Here we rely on the underlying broadcast operations.
        let length = output_length(self, other)?;
        match (self, other) {
            (Column::Series(lhs), Column::Series(rhs)) => op(lhs, rhs).map(Column::from),
            (Column::Series(lhs), Column::Scalar(rhs)) => {
                op(lhs, &rhs.as_single_value_series()).map(Column::from)
            },
            (Column::Scalar(lhs), Column::Series(rhs)) => {
                op(&lhs.as_single_value_series(), rhs).map(Column::from)
            },
            (Column::Scalar(lhs), Column::Scalar(rhs)) => {
                let lhs = lhs.as_single_value_series();
                let rhs = rhs.as_single_value_series();

                Ok(ScalarColumn::from_single_value_series(op(&lhs, &rhs)?, length).into_column())
            },
            // @partition-opt
            (lhs, rhs) => {
                op(lhs.as_materialized_series(), rhs.as_materialized_series()).map(Column::from)
            },
        }
    }

    pub fn apply_binary_elementwise(
        &self,
        other: &Self,
        f: impl Fn(&Series, &Series) -> Series,
        f_lb: impl Fn(&Scalar, &Series) -> Series,
        f_rb: impl Fn(&Series, &Scalar) -> Series,
    ) -> Column {
        self.try_apply_binary_elementwise(
            other,
            |lhs, rhs| Ok(f(lhs, rhs)),
            |lhs, rhs| Ok(f_lb(lhs, rhs)),
            |lhs, rhs| Ok(f_rb(lhs, rhs)),
        )
        .unwrap()
    }
    pub fn try_apply_binary_elementwise(
        &self,
        other: &Self,
        f: impl Fn(&Series, &Series) -> PolarsResult<Series>,
        f_lb: impl Fn(&Scalar, &Series) -> PolarsResult<Series>,
        f_rb: impl Fn(&Series, &Scalar) -> PolarsResult<Series>,
    ) -> PolarsResult<Column> {
        debug_assert_eq!(self.len(), other.len());

        match (self, other) {
            (Column::Series(lhs), Column::Series(rhs)) => f(lhs, rhs).map(Column::from),
            (Column::Series(lhs), Column::Scalar(rhs)) => f_rb(lhs, rhs.scalar()).map(Column::from),
            (Column::Scalar(lhs), Column::Series(rhs)) => f_lb(lhs.scalar(), rhs).map(Column::from),
            (Column::Scalar(lhs), Column::Scalar(rhs)) => {
                let lhs = lhs.as_single_value_series();
                let rhs = rhs.as_single_value_series();

                Ok(
                    ScalarColumn::from_single_value_series(f(&lhs, &rhs)?, self.len())
                        .into_column(),
                )
            },
            // @partition-opt
            (lhs, rhs) => {
                f(lhs.as_materialized_series(), rhs.as_materialized_series()).map(Column::from)
            },
        }
    }

    #[cfg(feature = "approx_unique")]
    pub fn approx_n_unique(&self) -> PolarsResult<IdxSize> {
        match self {
            Column::Series(s) => s.approx_n_unique(),
            // @partition-opt
            Column::Partitioned(s) => s.as_materialized_series().approx_n_unique(),
            Column::Scalar(s) => {
                // @NOTE: We do this for the error handling.
                s.as_single_value_series().approx_n_unique()?;
                Ok(1)
            },
        }
    }

    pub fn n_chunks(&self) -> usize {
        match self {
            Column::Series(s) => s.n_chunks(),
            Column::Scalar(_) | Column::Partitioned(_) => 1,
        }
    }
}

impl Default for Column {
    fn default() -> Self {
        Self::new_scalar(
            PlSmallStr::EMPTY,
            Scalar::new(DataType::Int64, AnyValue::Null),
            0,
        )
    }
}

impl PartialEq for Column {
    fn eq(&self, other: &Self) -> bool {
        // @scalar-opt
        self.as_materialized_series()
            .eq(other.as_materialized_series())
    }
}

impl From<Series> for Column {
    #[inline]
    fn from(series: Series) -> Self {
        // We instantiate a Scalar Column if the Series is length is 1. This makes it possible for
        // future operations to be faster.
        if series.len() == 1 {
            return Self::Scalar(ScalarColumn::unit_scalar_from_series(series));
        }

        Self::Series(series)
    }
}

impl<T: IntoSeries> IntoColumn for T {
    #[inline]
    fn into_column(self) -> Column {
        self.into_series().into()
    }
}

impl IntoColumn for Column {
    #[inline(always)]
    fn into_column(self) -> Column {
        self
    }
}

/// We don't want to serialize the scalar columns. So this helps pretend that columns are always
/// initialized without implementing From<Column> for Series.
///
/// Those casts should be explicit.
#[derive(Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize))]
#[cfg_attr(feature = "serde", serde(into = "Series"))]
struct _SerdeSeries(Series);

impl From<Column> for _SerdeSeries {
    #[inline]
    fn from(value: Column) -> Self {
        Self(value.take_materialized_series())
    }
}

impl From<_SerdeSeries> for Series {
    #[inline]
    fn from(value: _SerdeSeries) -> Self {
        value.0
    }
}
