use std::borrow::Cow;

use num_traits::{Num, NumCast};
use polars_error::PolarsResult;
use polars_utils::index::check_bounds;
use polars_utils::pl_str::PlSmallStr;
pub use scalar::ScalarColumn;

use self::gather::check_bounds_ca;
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::metadata::{MetadataFlags, MetadataTrait};
use crate::datatypes::ReshapeDimension;
use crate::prelude::*;
use crate::series::{BitRepr, IsSorted, SeriesPhysIter};
use crate::utils::{slice_offsets, Container};
use crate::{HEAD_DEFAULT_LENGTH, TAIL_DEFAULT_LENGTH};

mod arithmetic;
mod compare;
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

    // # Materialize
    /// Get a reference to a [`Series`] for this [`Column`]
    ///
    /// This may need to materialize the [`Series`] on the first invocation for a specific column.
    #[inline]
    pub fn as_materialized_series(&self) -> &Series {
        match self {
            Column::Series(s) => s,
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
            Column::Scalar(s) => s.take_materialized_series(),
        }
    }

    #[inline]
    pub fn dtype(&self) -> &DataType {
        match self {
            Column::Series(s) => s.dtype(),
            Column::Scalar(s) => s.dtype(),
        }
    }

    #[inline]
    pub fn field(&self) -> Cow<Field> {
        match self {
            Column::Series(s) => s.field(),
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
            Column::Scalar(s) => s.name(),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Column::Series(s) => s.len(),
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
            Column::Scalar(s) => _ = s.rename(name),
        }
    }

    // # Downcasting
    #[inline]
    pub fn as_series(&self) -> Option<&Series> {
        match self {
            Column::Series(s) => Some(s),
            Column::Scalar(_) => None,
        }
    }
    #[inline]
    pub fn as_scalar_column(&self) -> Option<&ScalarColumn> {
        match self {
            Column::Series(_) => None,
            Column::Scalar(s) => Some(s),
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
            Column::Scalar(s) => s.cast_with_options(dtype, options).map(Column::from),
        }
    }
    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        match self {
            Column::Series(s) => s.strict_cast(dtype).map(Column::from),
            Column::Scalar(s) => s.strict_cast(dtype).map(Column::from),
        }
    }
    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Column> {
        match self {
            Column::Series(s) => s.cast(dtype).map(Column::from),
            Column::Scalar(s) => s.cast(dtype).map(Column::from),
        }
    }
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Column> {
        match self {
            Column::Series(s) => unsafe { s.cast_unchecked(dtype) }.map(Column::from),
            Column::Scalar(s) => unsafe { s.cast_unchecked(dtype) }.map(Column::from),
        }
    }

    pub fn clear(&self) -> Self {
        match self {
            Column::Series(s) => s.clear().into(),
            Column::Scalar(s) => s.resize(0).into(),
        }
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        match self {
            Column::Series(s) => s.shrink_to_fit(),
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
            Column::Scalar(s) => s.resize(length).into(),
        }
    }

    #[inline]
    pub fn has_nulls(&self) -> bool {
        match self {
            Self::Series(s) => s.has_nulls(),
            Self::Scalar(s) => s.has_nulls(),
        }
    }

    #[inline]
    pub fn is_null(&self) -> BooleanChunked {
        match self {
            Self::Series(s) => s.is_null(),
            Self::Scalar(s) => {
                BooleanChunked::full(s.name().clone(), s.scalar().is_null(), s.len())
            },
        }
    }
    #[inline]
    pub fn is_not_null(&self) -> BooleanChunked {
        match self {
            Self::Series(s) => s.is_not_null(),
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
            Self::Scalar(s) => s.resize(indices.len()).into(),
        }
    }
    /// # Safety
    ///
    /// No bounds on the indexes are performed.
    pub unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Column {
        debug_assert!(check_bounds(indices, self.len() as IdxSize).is_ok());

        match self {
            Self::Series(s) => unsafe { s.take_unchecked_from_slice(indices) }.into(),
            Self::Scalar(s) => s.resize(indices.len()).into(),
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
        interpol: QuantileInterpolOptions,
    ) -> Self {
        // @scalar-opt
        unsafe {
            self.as_materialized_series()
                .agg_quantile(groups, quantile, interpol)
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

    pub fn full_null(name: PlSmallStr, size: usize, dtype: &DataType) -> Self {
        Series::full_null(name, size, dtype).into()
        // @TODO: This causes failures
        // Self::new_scalar(name, Scalar::new(dtype.clone(), AnyValue::Null), size)
    }

    pub fn is_empty(&self) -> bool {
        // @scalar-opt
        self.as_materialized_series().is_empty()
    }

    pub fn reverse(&self) -> Column {
        match self {
            Column::Series(s) => s.reverse().into(),
            Column::Scalar(_) => self.clone(),
        }
    }

    pub fn equals(&self, right: &Column) -> bool {
        // @scalar-opt
        self.as_materialized_series()
            .equals(right.as_materialized_series())
    }

    pub fn equals_missing(&self, right: &Column) -> bool {
        // @scalar-opt
        self.as_materialized_series()
            .equals_missing(right.as_materialized_series())
    }

    pub fn set_sorted_flag(&mut self, sorted: IsSorted) {
        // @scalar-opt
        match self {
            Column::Series(s) => s.set_sorted_flag(sorted),
            Column::Scalar(_) => {},
        }
    }

    pub fn get_flags(&self) -> MetadataFlags {
        match self {
            Column::Series(s) => s.get_flags(),
            // @scalar-opt
            Column::Scalar(_) => MetadataFlags::empty(),
        }
    }

    pub fn get_metadata<'a>(&'a self) -> Option<Box<dyn MetadataTrait + 'a>> {
        match self {
            Column::Series(s) => s.boxed_metadata(),
            // @scalar-opt
            Column::Scalar(_) => None,
        }
    }

    pub fn get_data_ptr(&self) -> usize {
        // @scalar-opt
        self.as_materialized_series().get_data_ptr()
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
        unsafe { DataFrame::new_no_checks(vec![self]) }
    }

    pub fn unique_stable(&self) -> PolarsResult<Column> {
        // @scalar-opt?
        self.as_materialized_series()
            .unique_stable()
            .map(Column::from)
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
            Column::Scalar(_) => self.clone(),
        }
    }

    pub fn explode(&self) -> PolarsResult<Column> {
        // @scalar-opt
        self.as_materialized_series().explode().map(Column::from)
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
        // @scalar-opt
        self.as_materialized_series().drop_nulls().into()
    }

    pub fn is_sorted_flag(&self) -> IsSorted {
        // @scalar-opt
        self.as_materialized_series().is_sorted_flag()
    }

    pub fn unique(&self) -> PolarsResult<Column> {
        // @scalar-opt
        self.as_materialized_series().unique().map(Column::from)
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

    pub fn filter(&self, filter: &ChunkedArray<BooleanType>) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series().filter(filter).map(Self::from)
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
        // @scalar-opt
        self.as_materialized_series().is_finite()
    }

    pub fn is_infinite(&self) -> PolarsResult<BooleanChunked> {
        // @scalar-opt
        self.as_materialized_series().is_infinite()
    }

    pub fn is_nan(&self) -> PolarsResult<BooleanChunked> {
        // @scalar-opt
        self.as_materialized_series().is_nan()
    }

    pub fn is_not_nan(&self) -> PolarsResult<BooleanChunked> {
        // @scalar-opt
        self.as_materialized_series().is_not_nan()
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
            Column::Series(s) => s.get_unchecked(index),
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
        self.as_materialized_series()
            .bitand(rhs.as_materialized_series())
            .map(Column::from)
    }

    pub(crate) fn str_value(&self, index: usize) -> PolarsResult<Cow<str>> {
        Ok(self.get(index)?.str_value())
    }

    pub fn max_reduce(&self) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.max_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().max_reduce()
            },
        }
    }

    pub fn min_reduce(&self) -> PolarsResult<Scalar> {
        match self {
            Column::Series(s) => s.min_reduce(),
            Column::Scalar(s) => {
                // We don't really want to deal with handling the full semantics here so we just
                // cast to a single value series. This is a tiny bit wasteful, but probably fine.
                s.as_single_value_series().min_reduce()
            },
        }
    }

    pub(crate) fn estimated_size(&self) -> usize {
        // @scalar-opt
        self.as_materialized_series().estimated_size()
    }

    pub(crate) fn sort_with(&self, options: SortOptions) -> PolarsResult<Self> {
        match self {
            Column::Series(s) => s.sort_with(options).map(Self::from),
            Column::Scalar(s) => {
                // This makes this function throw the same errors as Series::sort_with
                _ = s.as_single_value_series().sort_with(options)?;

                Ok(self.clone())
            },
        }
    }

    pub fn apply_unary_elementwise(&self, f: impl Fn(&Series) -> Series) -> Column {
        match self {
            Column::Series(s) => f(s).into(),
            Column::Scalar(s) => {
                ScalarColumn::from_single_value_series(f(&s.as_single_value_series()), s.len())
                    .into()
            },
        }
    }

    pub fn try_apply_unary_elementwise(
        &self,
        f: impl Fn(&Series) -> PolarsResult<Series>,
    ) -> PolarsResult<Column> {
        match self {
            Column::Series(s) => f(s).map(Column::from),
            Column::Scalar(s) => Ok(ScalarColumn::from_single_value_series(
                f(&s.as_single_value_series())?,
                s.len(),
            )
            .into()),
        }
    }

    #[cfg(feature = "approx_unique")]
    pub fn approx_n_unique(&self) -> PolarsResult<IdxSize> {
        match self {
            Column::Series(s) => s.approx_n_unique(),
            Column::Scalar(s) => {
                // @NOTE: We do this for the error handling.
                s.as_single_value_series().approx_n_unique()?;
                Ok(1)
            },
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
