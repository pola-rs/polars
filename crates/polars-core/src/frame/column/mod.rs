use std::borrow::Cow;
use std::sync::OnceLock;

use num_traits::{Num, NumCast};
use polars_error::PolarsResult;
use polars_utils::index::check_bounds;
use polars_utils::pl_str::PlSmallStr;

use self::gather::check_bounds_ca;
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::metadata::{MetadataFlags, MetadataTrait};
use crate::prelude::*;
use crate::series::{BitRepr, IsSorted, SeriesPhysIter};
use crate::utils::{slice_offsets, Container};
use crate::{HEAD_DEFAULT_LENGTH, TAIL_DEFAULT_LENGTH};

mod arithmetic;

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

/// A [`Column`] that consists of a repeated [`Scalar`]
///
/// This is lazily materialized into a [`Series`].
#[derive(Debug, Clone)]
pub struct ScalarColumn {
    name: PlSmallStr,
    // The value of this scalar may be incoherent when `length == 0`.
    scalar: Scalar,
    length: usize,

    // invariants:
    // materialized.name() == name
    // materialized.len() == length
    // materialized.dtype() == value.dtype
    // materialized[i] == value, for all 0 <= i < length
    /// A lazily materialized [`Series`] variant of this [`ScalarColumn`]
    materialized: OnceLock<Series>,
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
                let series = s.materialized.take().unwrap_or_else(|| s.to_series());
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
            Column::Scalar(s) => s.scalar.dtype(),
        }
    }

    #[inline]
    pub fn field(&self) -> Cow<Field> {
        match self {
            Column::Series(s) => s.field(),
            Column::Scalar(s) => match s.materialized.get() {
                None => Cow::Owned(Field::new(s.name.clone(), s.scalar.dtype().clone())),
                Some(s) => s.field(),
            },
        }
    }

    #[inline]
    pub fn name(&self) -> &PlSmallStr {
        match self {
            Column::Series(s) => s.name(),
            Column::Scalar(s) => &s.name,
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Column::Series(s) => s.len(),
            Column::Scalar(s) => s.length,
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
            Column::Scalar(s) => {
                if let Some(series) = s.materialized.get_mut() {
                    series.rename(name.clone());
                }

                s.name = name;
            },
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
            Column::Scalar(s) => Self::new_scalar(s.name.clone(), s.scalar.clone(), 0),
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
        match self {
            Column::Series(s) => s.new_from_index(index, length).into(),
            Column::Scalar(s) => {
                if index >= s.length {
                    Self::full_null(s.name.clone(), length, s.scalar.dtype())
                } else {
                    s.resize(length).into()
                }
            },
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
            Self::Scalar(s) => BooleanChunked::full(s.name.clone(), s.scalar.is_null(), s.length),
        }
    }
    #[inline]
    pub fn is_not_null(&self) -> BooleanChunked {
        match self {
            Self::Series(s) => s.is_not_null(),
            Self::Scalar(s) => BooleanChunked::full(s.name.clone(), !s.scalar.is_null(), s.length),
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
                let (_, length) = slice_offsets(offset, length, s.length);
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
            Self::Scalar(s) if s.scalar.is_null() => s.length,
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
    pub unsafe fn agg_first(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_first(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    pub unsafe fn agg_last(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_last(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
    pub unsafe fn agg_n_unique(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_n_unique(groups) }.into()
    }

    /// # Safety
    ///
    /// Does no bounds checks, groups must be correct.
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

    /// # Safety
    ///
    /// Indexes need to be in bounds.
    pub(crate) unsafe fn equal_element(
        &self,
        idx_self: usize,
        idx_other: usize,
        other: &Column,
    ) -> bool {
        // @scalar-opt
        unsafe {
            self.as_materialized_series().equal_element(
                idx_self,
                idx_other,
                other.as_materialized_series(),
            )
        }
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

    pub fn reshape_list(&self, dimensions: &[i64]) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .reshape_list(dimensions)
            .map(Self::from)
    }

    #[cfg(feature = "dtype-array")]
    pub fn reshape_array(&self, dimensions: &[i64]) -> PolarsResult<Self> {
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
            Column::Scalar(s) => s.resize(s.length - offset / n).into(),
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
                if s.scalar.as_any_value() == value {
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
            Column::Scalar(s) => s.scalar.as_any_value(),
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
}

impl ChunkCompare<&Column> for Column {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking for equality.
    #[inline]
    fn equal(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        self.as_materialized_series()
            .equal(rhs.as_materialized_series())
    }

    /// Create a boolean mask by checking for equality.
    #[inline]
    fn equal_missing(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        self.as_materialized_series()
            .equal_missing(rhs.as_materialized_series())
    }

    /// Create a boolean mask by checking for inequality.
    #[inline]
    fn not_equal(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        self.as_materialized_series()
            .not_equal(rhs.as_materialized_series())
    }

    /// Create a boolean mask by checking for inequality.
    #[inline]
    fn not_equal_missing(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        self.as_materialized_series()
            .not_equal_missing(rhs.as_materialized_series())
    }

    /// Create a boolean mask by checking if self > rhs.
    #[inline]
    fn gt(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        self.as_materialized_series()
            .gt(rhs.as_materialized_series())
    }

    /// Create a boolean mask by checking if self >= rhs.
    #[inline]
    fn gt_eq(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        self.as_materialized_series()
            .gt_eq(rhs.as_materialized_series())
    }

    /// Create a boolean mask by checking if self < rhs.
    #[inline]
    fn lt(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        self.as_materialized_series()
            .lt(rhs.as_materialized_series())
    }

    /// Create a boolean mask by checking if self <= rhs.
    #[inline]
    fn lt_eq(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        self.as_materialized_series()
            .lt_eq(rhs.as_materialized_series())
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
        if series.len() == 1 {
            // SAFETY: We just did the bounds check
            let value = unsafe { series.get_unchecked(0) }.into_static();
            let value = Scalar::new(series.dtype().clone(), value);
            let mut col = ScalarColumn::new(series.name().clone(), value, 1);
            col.materialized = OnceLock::from(series);
            return Self::Scalar(col);
        }

        Self::Series(series)
    }
}

impl From<ScalarColumn> for Column {
    #[inline]
    fn from(value: ScalarColumn) -> Self {
        Self::Scalar(value)
    }
}

impl ScalarColumn {
    #[inline]
    pub fn new(name: PlSmallStr, scalar: Scalar, length: usize) -> Self {
        Self {
            name,
            scalar,
            length,

            materialized: OnceLock::new(),
        }
    }

    #[inline]
    pub fn new_empty(name: PlSmallStr, dtype: DataType) -> Self {
        Self {
            name,
            scalar: Scalar::new(dtype, AnyValue::Null),
            length: 0,

            materialized: OnceLock::new(),
        }
    }

    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    pub fn dtype(&self) -> &DataType {
        self.scalar.dtype()
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    fn _to_series(name: PlSmallStr, value: Scalar, length: usize) -> Series {
        let series = if length == 0 {
            Series::new_empty(name, value.dtype())
        } else {
            value.into_series(name).new_from_index(0, length)
        };

        debug_assert_eq!(series.len(), length);

        series
    }

    /// Materialize the [`ScalarColumn`] into a [`Series`].
    pub fn to_series(&self) -> Series {
        Self::_to_series(self.name.clone(), self.scalar.clone(), self.length)
    }

    /// Get the [`ScalarColumn`] as [`Series`]
    ///
    /// This needs to materialize upon the first call. Afterwards, this is cached.
    pub fn as_materialized_series(&self) -> &Series {
        self.materialized.get_or_init(|| self.to_series())
    }

    /// Take the [`ScalarColumn`] and materialize as a [`Series`] if not already done.
    pub fn take_materialized_series(self) -> Series {
        self.materialized
            .into_inner()
            .unwrap_or_else(|| Self::_to_series(self.name, self.scalar, self.length))
    }

    /// Take the [`ScalarColumn`] as a series with a single value.
    ///
    /// If the [`ScalarColumn`] has `length=0` the resulting `Series` will also have `length=0`.
    pub fn as_single_value_series(&self) -> Series {
        match self.materialized.get() {
            Some(s) => s.head(Some(1)),
            None => Self::_to_series(
                self.name.clone(),
                self.scalar.clone(),
                usize::min(1, self.length),
            ),
        }
    }

    /// Create a new [`ScalarColumn`] from a `length=1` Series and expand it `length`.
    ///
    /// This will panic if the value cannot be made static or if the series has length `0`.
    pub fn from_single_value_series(series: Series, length: usize) -> PolarsResult<Self> {
        debug_assert_eq!(series.len(), 1);
        let value = series.get(0)?;
        let value = value.into_static();
        let value = Scalar::new(series.dtype().clone(), value);
        Ok(ScalarColumn::new(series.name().clone(), value, length))
    }

    /// Resize the [`ScalarColumn`] to new `length`.
    ///
    /// This reuses the materialized [`Series`], if `length <= self.length`.
    pub fn resize(&self, length: usize) -> ScalarColumn {
        if self.length == length {
            return self.clone();
        }

        // This is violates an invariant if this triggers, the scalar value is undefined if the
        // self.length == 0 so therefore we should never resize using that value.
        debug_assert_ne!(self.length, 0);

        let mut resized = Self {
            name: self.name.clone(),
            scalar: self.scalar.clone(),
            length,
            materialized: OnceLock::new(),
        };

        if self.length >= length {
            if let Some(materialized) = self.materialized.get() {
                resized.materialized = OnceLock::from(materialized.head(Some(length)));
                debug_assert_eq!(resized.materialized.get().unwrap().len(), length);
            }
        }

        resized
    }

    pub fn cast_with_options(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Self> {
        // @NOTE: We expect that when casting the materialized series mostly does not need change
        // the physical array. Therefore, we try to cast the entire materialized array if it is
        // available.

        match self.materialized.get() {
            Some(s) => {
                let materialized = s.cast_with_options(dtype, options)?;
                assert_eq!(self.length, materialized.len());

                let mut casted = if materialized.len() == 0 {
                    Self::new_empty(materialized.name().clone(), materialized.dtype().clone())
                } else {
                    // SAFETY: Just did bounds check
                    let scalar = unsafe { materialized.get_unchecked(0) }.into_static();
                    Self::new(
                        materialized.name().clone(),
                        Scalar::new(materialized.dtype().clone(), scalar),
                        self.length,
                    )
                };
                casted.materialized = OnceLock::from(materialized);
                Ok(casted)
            },
            None => {
                let s = self
                    .as_single_value_series()
                    .cast_with_options(dtype, options)?;
                assert_eq!(1, s.len());

                if self.length == 0 {
                    Ok(Self::new_empty(s.name().clone(), s.dtype().clone()))
                } else {
                    Self::from_single_value_series(s, self.length)
                }
            },
        }
    }

    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        self.cast_with_options(dtype, CastOptions::Strict)
    }
    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        self.cast_with_options(dtype, CastOptions::NonStrict)
    }
    /// # Safety
    ///
    /// This can lead to invalid memory access in downstream code.
    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Self> {
        // @NOTE: We expect that when casting the materialized series mostly does not need change
        // the physical array. Therefore, we try to cast the entire materialized array if it is
        // available.

        match self.materialized.get() {
            Some(s) => {
                let materialized = s.cast_unchecked(dtype)?;
                assert_eq!(self.length, materialized.len());

                let mut casted = if materialized.len() == 0 {
                    Self::new_empty(materialized.name().clone(), materialized.dtype().clone())
                } else {
                    // SAFETY: Just did bounds check
                    let scalar = unsafe { materialized.get_unchecked(0) }.into_static();
                    Self::new(
                        materialized.name().clone(),
                        Scalar::new(materialized.dtype().clone(), scalar),
                        self.length,
                    )
                };
                casted.materialized = OnceLock::from(materialized);
                Ok(casted)
            },
            None => {
                let s = self.as_single_value_series().cast_unchecked(dtype)?;
                assert_eq!(1, s.len());

                if self.length == 0 {
                    Ok(Self::new_empty(s.name().clone(), s.dtype().clone()))
                } else {
                    Self::from_single_value_series(s, self.length)
                }
            },
        }
    }

    pub fn has_nulls(&self) -> bool {
        self.length != 0 && self.scalar.is_null()
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

impl IntoColumn for ScalarColumn {
    #[inline(always)]
    fn into_column(self) -> Column {
        self.into()
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
