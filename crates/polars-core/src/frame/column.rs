use std::borrow::Cow;
use std::ops::{Add, Div, Mul, Rem, Sub};
use std::sync::OnceLock;

use num_traits::{Num, NumCast};
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use crate::chunked_array::metadata::MetadataFlags;
use crate::prelude::*;
use crate::series::{BitRepr, IsSorted};

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub enum Column {
    Series(Series),
    Scalar(ScalarColumn),
}

#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct ScalarColumn {
    name: PlSmallStr,
    value: AnyValue<'static>,
    // invariant: Series.len() == length
    #[cfg_attr(feature = "serde", serde(skip))]
    materialized: OnceLock<Series>,
    length: usize,
}

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
        // @scalar-opt
        Self::Series(Series::new_empty(name, &dtype))
    }

    #[inline]
    pub fn new_scalar(name: PlSmallStr, value: AnyValue<'static>, length: usize) -> Self {
        Self::Scalar(ScalarColumn::new(name, value, length))
    }

    #[inline]
    pub fn as_materialized_series(&self) -> &Series {
        match self {
            Column::Series(s) => s,
            Column::Scalar(s) => s.as_materialized_series(),
        }
    }

    #[inline]
    pub fn as_materialized_series_mut(&mut self) -> &mut Series {
        match self {
            Column::Series(s) => s,
            Column::Scalar(s) => {
                *self = Column::Series(s.to_series());
                let Column::Series(s) = self else {
                    unreachable!();
                };
                s
            },
        }
    }

    #[inline]
    pub fn dtype(&self) -> &DataType {
        // @scalar-opt
        self.as_materialized_series().dtype()
    }

    #[inline]
    pub fn field(&self) -> Cow<Field> {
        // @scalar-opt
        self.as_materialized_series().field()
    }

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

    pub fn i8(&self) -> PolarsResult<&Int8Chunked> {
        // @scalar-opt
        self.as_materialized_series().i8()
    }

    pub fn i16(&self) -> PolarsResult<&Int16Chunked> {
        // @scalar-opt
        self.as_materialized_series().i16()
    }

    pub fn i32(&self) -> PolarsResult<&Int32Chunked> {
        // @scalar-opt
        self.as_materialized_series().i32()
    }

    pub fn i64(&self) -> PolarsResult<&Int64Chunked> {
        // @scalar-opt
        self.as_materialized_series().i64()
    }

    pub fn u8(&self) -> PolarsResult<&UInt8Chunked> {
        // @scalar-opt
        self.as_materialized_series().u8()
    }

    pub fn u16(&self) -> PolarsResult<&UInt16Chunked> {
        // @scalar-opt
        self.as_materialized_series().u16()
    }

    pub fn u32(&self) -> PolarsResult<&UInt32Chunked> {
        // @scalar-opt
        self.as_materialized_series().u32()
    }

    pub fn u64(&self) -> PolarsResult<&UInt64Chunked> {
        // @scalar-opt
        self.as_materialized_series().u64()
    }

    pub fn f32(&self) -> PolarsResult<&Float32Chunked> {
        // @scalar-opt
        self.as_materialized_series().f32()
    }

    pub fn f64(&self) -> PolarsResult<&Float64Chunked> {
        // @scalar-opt
        self.as_materialized_series().f64()
    }

    pub fn str(&self) -> PolarsResult<&StringChunked> {
        // @scalar-opt
        self.as_materialized_series().str()
    }

    pub fn datetime(&self) -> PolarsResult<&DatetimeChunked> {
        // @scalar-opt
        self.as_materialized_series().datetime()
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

    pub fn clear(&self) -> Self {
        match self {
            Column::Series(s) => s.clear().into(),
            Column::Scalar(s) => Self::new_scalar(s.name.clone(), s.value.clone(), 0),
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
        // @scalar-opt
        Self::Series(self.as_materialized_series().new_from_index(index, length))
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Column::Series(s) => s.len(),
            Column::Scalar(s) => s.length,
        }
    }

    #[inline]
    pub fn name(&self) -> &PlSmallStr {
        match self {
            Column::Series(s) => s.name(),
            Column::Scalar(s) => &s.name,
        }
    }

    pub fn has_nulls(&self) -> bool {
        // @scalar-opt
        self.as_materialized_series().has_nulls()
    }

    pub fn is_not_null(&self) -> ChunkedArray<BooleanType> {
        // @scalar-opt
        self.as_materialized_series().is_not_null()
    }

    pub fn to_physical_repr(&self) -> Column {
        // @scalar-opt
        self.as_materialized_series()
            .to_physical_repr()
            .into_owned()
            .into()
    }

    pub fn head(&self, length: Option<usize>) -> Column {
        // @scalar-opt
        self.as_materialized_series().head(length).into()
    }

    pub fn tail(&self, length: Option<usize>) -> Column {
        // @scalar-opt
        self.as_materialized_series().tail(length).into()
    }

    pub fn slice(&self, offset: i64, length: usize) -> Column {
        // @scalar-opt
        self.as_materialized_series().slice(offset, length).into()
    }

    pub fn split_at(&self, offset: i64) -> (Column, Column) {
        // @scalar-opt
        let (l, r) = self.as_materialized_series().split_at(offset);
        (l.into(), r.into())
    }

    pub fn null_count(&self) -> usize {
        // @scalar-opt
        self.as_materialized_series().null_count()
    }

    pub unsafe fn agg_min(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_min(groups) }.into()
    }

    pub unsafe fn agg_max(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_max(groups) }.into()
    }

    pub unsafe fn agg_mean(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_mean(groups) }.into()
    }

    pub unsafe fn agg_sum(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_sum(groups) }.into()
    }

    pub unsafe fn agg_first(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_first(groups) }.into()
    }

    pub unsafe fn agg_last(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_last(groups) }.into()
    }

    pub unsafe fn agg_n_unique(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_n_unique(groups) }.into()
    }

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

    pub unsafe fn agg_median(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_median(groups) }.into()
    }

    pub unsafe fn agg_var(&self, groups: &GroupsProxy, ddof: u8) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_var(groups, ddof) }.into()
    }

    pub unsafe fn agg_std(&self, groups: &GroupsProxy, ddof: u8) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_std(groups, ddof) }.into()
    }

    pub unsafe fn agg_list(&self, groups: &GroupsProxy) -> Self {
        // @scalar-opt
        unsafe { self.as_materialized_series().agg_list(groups) }.into()
    }

    pub fn full_null(name: PlSmallStr, size: usize, dtype: &DataType) -> Column {
        // @scalar-opt
        Series::full_null(name, size, dtype).into()
    }

    pub fn is_empty(&self) -> bool {
        // @scalar-opt
        self.as_materialized_series().is_empty()
    }

    pub fn reverse(&self) -> Column {
        // @scalar-opt
        self.as_materialized_series().reverse().into()
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

    pub fn categorical(&self) -> PolarsResult<&CategoricalChunked> {
        self.as_materialized_series().categorical()
    }

    pub fn with_name(self, name: PlSmallStr) -> Column {
        match self {
            Column::Series(s) => s.with_name(name).into(),
            Column::Scalar(s) => s.with_name(name).into(),
        }
    }

    pub fn append(&mut self, other: &Column) -> PolarsResult<&mut Self> {
        // @scalar-opt
        self.as_materialized_series_mut()
            .append(other.as_materialized_series())?;
        Ok(self)
    }

    pub fn arg_sort(&self, options: SortOptions) -> IdxCa {
        // @scalar-opt
        self.as_materialized_series().arg_sort(options)
    }

    pub fn cast(&self, dtype: &DataType) -> PolarsResult<Column> {
        // @scalar-opt
        self.as_materialized_series().cast(dtype).map(Column::from)
    }

    pub fn idx(&self) -> PolarsResult<&IdxCa> {
        // @scalar-opt
        self.as_materialized_series().idx()
    }

    pub fn binary(&self) -> PolarsResult<&BinaryChunked> {
        // @scalar-opt
        self.as_materialized_series().binary()
    }

    pub fn bit_repr(&self) -> Option<BitRepr> {
        // @scalar-opt
        self.as_materialized_series().bit_repr()
    }

    pub fn bool(&self) -> PolarsResult<&BooleanChunked> {
        // @scalar-opt
        self.as_materialized_series().bool()
    }

    pub fn struct_(&self) -> PolarsResult<&StructChunked> {
        // @scalar-opt
        self.as_materialized_series().struct_()
    }

    pub fn into_frame(&self) -> DataFrame {
        // @scalar-opt
        self.as_materialized_series().clone().into_frame()
    }

    pub fn unique_stable(&self) -> PolarsResult<Column> {
        // @scalar-opt?
        self.as_materialized_series()
            .unique_stable()
            .map(Column::from)
    }

    pub fn extend(&mut self, other: &Column) -> PolarsResult<&mut Self> {
        // @scalar-opt
        self.as_materialized_series_mut()
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

    pub fn strict_cast(&self, dtype: &DataType) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .strict_cast(dtype)
            .map(Column::from)
    }

    pub unsafe fn cast_unchecked(&self, dtype: &DataType) -> PolarsResult<Column> {
        // @scalar-opt
        unsafe { self.as_materialized_series().cast_unchecked(dtype) }.map(Column::from)
    }

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

    pub fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        // @scalar-opt
        self.as_materialized_series().get(index)
    }

    pub fn decimal(&self) -> PolarsResult<&DecimalChunked> {
        // @scalar-opt
        self.as_materialized_series().decimal()
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

    pub fn shuffle(&self, seed: Option<u64>) -> Self {
        // @scalar-opt
        self.as_materialized_series().shuffle(seed).into()
    }

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
        // @scalar-opt
        self.as_materialized_series().gather_every(n, offset).into()
    }

    pub fn extend_constant(&self, value: AnyValue, n: usize) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .extend_constant(value, n)
            .map(Self::from)
    }

    pub fn array(&self) -> PolarsResult<&ArrayChunked> {
        // @scalar-opt
        self.as_materialized_series().array()
    }

    pub fn list(&self) -> PolarsResult<&ListChunked> {
        // @scalar-opt
        self.as_materialized_series().list()
    }

    pub fn is_null(&self) -> BooleanChunked {
        // @scalar-opt
        self.as_materialized_series().is_null()
    }

    pub fn zip_with(&self, mask: &BooleanChunked, other: &Self) -> PolarsResult<Self> {
        // @scalar-opt
        self.as_materialized_series()
            .zip_with(mask, other.as_materialized_series())
            .map(Self::from)
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

    pub fn date(&self) -> PolarsResult<&DateChunked> {
        // @scalar-opt
        self.as_materialized_series().date()
    }

    pub fn duration(&self) -> PolarsResult<&DurationChunked> {
        // @scalar-opt
        self.as_materialized_series().duration()
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
}

impl Default for Column {
    fn default() -> Self {
        // @scalar-opt
        Column::Series(Series::default())
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
    fn from(value: Series) -> Self {
        Self::Series(value)
    }
}

impl From<ScalarColumn> for Column {
    #[inline]
    fn from(value: ScalarColumn) -> Self {
        Self::Scalar(value)
    }
}

impl Add for Column {
    type Output = PolarsResult<Column>;

    fn add(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .add(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl Add for &Column {
    type Output = PolarsResult<Column>;

    fn add(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .add(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl Sub for Column {
    type Output = PolarsResult<Column>;

    fn sub(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .sub(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl Sub for &Column {
    type Output = PolarsResult<Column>;

    fn sub(self, rhs: Self) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series()
            .sub(rhs.as_materialized_series())
            .map(Column::from)
    }
}

impl<T> Sub<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn sub(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().sub(rhs).into()
    }
}

impl<T> Sub<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn sub(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().sub(rhs).into()
    }
}

impl<T> Add<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn add(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().add(rhs).into()
    }
}

impl<T> Add<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn add(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().add(rhs).into()
    }
}

impl<T> Div<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn div(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().div(rhs).into()
    }
}

impl<T> Div<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn div(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().div(rhs).into()
    }
}

impl<T> Mul<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn mul(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().mul(rhs).into()
    }
}

impl<T> Mul<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn mul(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().mul(rhs).into()
    }
}

impl<T> Rem<T> for &Column
where
    T: Num + NumCast,
{
    type Output = Column;

    fn rem(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().rem(rhs).into()
    }
}

impl<T> Rem<T> for Column
where
    T: Num + NumCast,
{
    type Output = Self;

    fn rem(self, rhs: T) -> Self::Output {
        // @scalar-opt
        self.as_materialized_series().rem(rhs).into()
    }
}

impl ScalarColumn {
    #[inline]
    pub fn new(name: PlSmallStr, value: AnyValue<'static>, length: usize) -> Self {
        Self {
            name,
            value,
            materialized: OnceLock::new(),
            length,
        }
    }

    fn _to_series(name: PlSmallStr, value: AnyValue<'static>, length: usize) -> Series {
        // @TODO: There is probably a better way to do this.
        Scalar::new(value.dtype(), value)
            .into_series(name)
            .new_from_index(0, length)
    }

    pub fn to_series(&self) -> Series {
        Self::_to_series(self.name.clone(), self.value.clone(), self.length)
    }

    pub fn as_materialized_series(&self) -> &Series {
        self.materialized.get_or_init(|| self.to_series())
    }

    pub fn select_chunk(&self, _: usize) -> Series {
        // @scalar-opt
        // @scalar-correctness?
        todo!()
    }

    fn with_name(self, name: PlSmallStr) -> Self {
        // @TODO: Keep materialized somehow?
        Self::new(name, self.value, self.length)
    }
}

impl<T: IntoSeries> IntoColumn for T {
    #[inline]
    fn into_column(self) -> Column {
        Column::from(self.into_series())
    }
}

impl IntoColumn for Column {
    #[inline(always)]
    fn into_column(self) -> Column {
        self
    }
}
