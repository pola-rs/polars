use super::private;
use super::IntoSeries;
use super::SeriesTrait;
use crate::chunked_array::{
    ops::aggregate::{ChunkAggSeries, VarAggSeries},
    AsSinglePtr,
};
use crate::fmt::FmtList;
use crate::frame::group_by::*;
use crate::frame::hash_join::{HashJoin, ZipOuterJoinColumn};
use crate::prelude::*;
#[cfg(feature = "object")]
use crate::series::private::PrivateSeries;
use ahash::RandomState;
use arrow::array::{ArrayDataRef, ArrayRef};
use arrow::buffer::Buffer;
#[cfg(feature = "object")]
use std::any::Any;
use std::borrow::Cow;
#[cfg(feature = "object")]
use std::fmt::Debug;
use std::ops::Deref;

impl IntoSeries for Arc<dyn SeriesTrait> {
    fn into_series(self) -> Series {
        Series(self)
    }
}

impl IntoSeries for Series {
    fn into_series(self) -> Series {
        self
    }
}

pub(crate) struct Wrap<T>(pub T);

impl<T> From<ChunkedArray<T>> for Wrap<ChunkedArray<T>> {
    fn from(ca: ChunkedArray<T>) -> Self {
        Wrap(ca)
    }
}

impl<T> Deref for Wrap<ChunkedArray<T>> {
    type Target = ChunkedArray<T>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<'a, T> AsRef<ChunkedArray<T>> for dyn SeriesTrait + 'a
where
    T: 'static + PolarsDataType,
{
    fn as_ref(&self) -> &ChunkedArray<T> {
        if &T::get_dtype() == self.dtype() ||
            // needed because we want to get ref of List no matter what the inner type is.
            (matches!(T::get_dtype(), DataType::List(_)) && matches!(self.dtype(), DataType::List(_)) )
        {
            unsafe { &*(self as *const dyn SeriesTrait as *const ChunkedArray<T>) }
        } else {
            panic!(
                "implementation error, cannot get ref {:?} from {:?}",
                T::get_dtype(),
                self.dtype()
            )
        }
    }
}

macro_rules! impl_dyn_series {
    ($ca: ident) => {
        impl IntoSeries for $ca {
            fn into_series(self) -> Series {
                Series(Arc::new(Wrap(self)))
            }
        }

        impl private::PrivateSeries for Wrap<$ca> {
            fn vec_hash(&self, random_state: RandomState) -> UInt64Chunked {
                self.0.vec_hash(random_state)
            }

            fn agg_mean(&self, groups: &[Vec<u32>]) -> Option<Series> {
                self.0.agg_mean(groups)
            }

            fn agg_min(&self, groups: &[Vec<u32>]) -> Option<Series> {
                self.0.agg_min(groups)
            }

            fn agg_max(&self, groups: &[Vec<u32>]) -> Option<Series> {
                self.0.agg_max(groups)
            }

            fn agg_sum(&self, groups: &[Vec<u32>]) -> Option<Series> {
                self.0.agg_sum(groups)
            }

            fn agg_first(&self, groups: &[Vec<u32>]) -> Series {
                self.0.agg_first(groups)
            }

            fn agg_last(&self, groups: &[Vec<u32>]) -> Series {
                self.0.agg_last(groups)
            }

            fn agg_std(&self, groups: &[Vec<u32>]) -> Option<Series> {
                self.0.agg_std(groups)
            }

            fn agg_var(&self, groups: &[Vec<u32>]) -> Option<Series> {
                self.0.agg_var(groups)
            }

            fn agg_n_unique(&self, groups: &[Vec<u32>]) -> Option<UInt32Chunked> {
                self.0.agg_n_unique(groups)
            }

            fn agg_list(&self, groups: &[Vec<u32>]) -> Option<Series> {
                self.0.agg_list(groups)
            }

            fn agg_quantile(&self, groups: &[Vec<u32>], quantile: f64) -> Option<Series> {
                self.0.agg_quantile(groups, quantile)
            }

            fn agg_median(&self, groups: &[Vec<u32>]) -> Option<Series> {
                self.0.agg_median(groups)
            }

            fn pivot<'a>(
                &self,
                pivot_series: &'a (dyn SeriesTrait + 'a),
                keys: Vec<Series>,
                groups: &[Vec<u32>],
                agg_type: PivotAgg,
            ) -> Result<DataFrame> {
                self.0.pivot(pivot_series, keys, groups, agg_type)
            }

            fn pivot_count<'a>(
                &self,
                pivot_series: &'a (dyn SeriesTrait + 'a),
                keys: Vec<Series>,
                groups: &[Vec<u32>],
            ) -> Result<DataFrame> {
                self.0.pivot_count(pivot_series, keys, groups)
            }
            fn hash_join_inner(&self, other: &Series) -> Vec<(u32, u32)> {
                HashJoin::hash_join_inner(&self.0, other.as_ref().as_ref())
            }
            fn hash_join_left(&self, other: &Series) -> Vec<(u32, Option<u32>)> {
                HashJoin::hash_join_left(&self.0, other.as_ref().as_ref())
            }
            fn hash_join_outer(&self, other: &Series) -> Vec<(Option<u32>, Option<u32>)> {
                HashJoin::hash_join_outer(&self.0, other.as_ref().as_ref())
            }
            fn zip_outer_join_column(
                &self,
                right_column: &Series,
                opt_join_tuples: &[(Option<u32>, Option<u32>)],
            ) -> Series {
                ZipOuterJoinColumn::zip_outer_join_column(&self.0, right_column, opt_join_tuples)
            }
            fn subtract(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::subtract(&self.0, rhs)
            }
            fn add_to(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::add_to(&self.0, rhs)
            }
            fn multiply(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::multiply(&self.0, rhs)
            }
            fn divide(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::divide(&self.0, rhs)
            }
            fn remainder(&self, rhs: &Series) -> Result<Series> {
                NumOpsDispatch::remainder(&self.0, rhs)
            }
            fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
                IntoGroupTuples::group_tuples(&self.0, multithreaded)
            }
        }

        impl SeriesTrait for Wrap<$ca> {
            fn cum_max(&self, reverse: bool) -> Series {
                self.0.cum_max(reverse).into_series()
            }

            fn cum_min(&self, reverse: bool) -> Series {
                self.0.cum_min(reverse).into_series()
            }

            fn cum_sum(&self, reverse: bool) -> Series {
                self.0.cum_sum(reverse).into_series()
            }

            fn rename(&mut self, name: &str) {
                self.0.rename(name);
            }

            fn array_data(&self) -> Vec<ArrayDataRef> {
                self.0.array_data()
            }

            fn chunk_lengths(&self) -> &Vec<usize> {
                self.0.chunk_id()
            }
            fn name(&self) -> &str {
                self.0.name()
            }

            fn field(&self) -> &Field {
                self.0.ref_field()
            }

            fn chunks(&self) -> &Vec<ArrayRef> {
                self.0.chunks()
            }

            fn i8(&self) -> Result<&Int8Chunked> {
                if matches!(self.0.dtype(), DataType::Int8) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Int8Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into i8",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            // For each column create a series
            fn i16(&self) -> Result<&Int16Chunked> {
                if matches!(self.0.dtype(), DataType::Int16) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Int16Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into i16",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn i32(&self) -> Result<&Int32Chunked> {
                if matches!(self.0.dtype(), DataType::Int32) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Int32Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into i32",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn i64(&self) -> Result<&Int64Chunked> {
                if matches!(self.0.dtype(), DataType::Int64) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Int64Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into i64",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn f32(&self) -> Result<&Float32Chunked> {
                if matches!(self.0.dtype(), DataType::Float32) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Float32Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into f32",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn f64(&self) -> Result<&Float64Chunked> {
                if matches!(self.0.dtype(), DataType::Float64) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Float64Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into f64",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn u8(&self) -> Result<&UInt8Chunked> {
                if matches!(self.0.dtype(), DataType::UInt8) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const UInt8Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into u8",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn u16(&self) -> Result<&UInt16Chunked> {
                if matches!(self.0.dtype(), DataType::UInt16) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const UInt16Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into u16",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn u32(&self) -> Result<&UInt32Chunked> {
                if matches!(self.0.dtype(), DataType::UInt32) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const UInt32Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into u32",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn u64(&self) -> Result<&UInt64Chunked> {
                if matches!(self.0.dtype(), DataType::UInt64) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const UInt64Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into u64",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn bool(&self) -> Result<&BooleanChunked> {
                if matches!(self.0.dtype(), DataType::Boolean) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const BooleanChunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into bool",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn utf8(&self) -> Result<&Utf8Chunked> {
                if matches!(self.0.dtype(), DataType::Utf8) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Utf8Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into utf8",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn date32(&self) -> Result<&Date32Chunked> {
                if matches!(self.0.dtype(), DataType::Date32) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Date32Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into date32",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn date64(&self) -> Result<&Date64Chunked> {
                if matches!(self.0.dtype(), DataType::Date64) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const Date64Chunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into date64",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn time64_nanosecond(&self) -> Result<&Time64NanosecondChunked> {
                if matches!(self.0.dtype(), DataType::Time64(TimeUnit::Nanosecond)) {
                    unsafe {
                        Ok(&*(self as *const dyn SeriesTrait as *const Time64NanosecondChunked))
                    }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into time64",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn duration_nanosecond(&self) -> Result<&DurationNanosecondChunked> {
                if matches!(self.0.dtype(), DataType::Duration(TimeUnit::Nanosecond)) {
                    unsafe {
                        Ok(&*(self as *const dyn SeriesTrait as *const DurationNanosecondChunked))
                    }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into duration_nanosecond",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn duration_millisecond(&self) -> Result<&DurationMillisecondChunked> {
                if matches!(self.0.dtype(), DataType::Duration(TimeUnit::Millisecond)) {
                    unsafe {
                        Ok(&*(self as *const dyn SeriesTrait as *const DurationMillisecondChunked))
                    }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into duration_millisecond",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn list(&self) -> Result<&ListChunked> {
                if matches!(self.0.dtype(), DataType::List(_)) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const ListChunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into list",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn categorical(&self) -> Result<&CategoricalChunked> {
                if matches!(self.0.dtype(), DataType::Categorical) {
                    unsafe { Ok(&*(self as *const dyn SeriesTrait as *const CategoricalChunked)) }
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        format!(
                            "cannot unpack Series: {:?} of type {:?} into categorical",
                            self.name(),
                            self.dtype(),
                        )
                        .into(),
                    ))
                }
            }

            fn append_array(&mut self, other: ArrayRef) -> Result<()> {
                self.0.append_array(other)
            }

            fn slice(&self, offset: usize, length: usize) -> Result<Series> {
                self.0.slice(offset, length).map(|ca| ca.into_series())
            }

            fn append(&mut self, other: &Series) -> Result<()> {
                if self.0.dtype() == other.dtype() {
                    // todo! add object
                    self.0.append(other.as_ref().as_ref());
                    Ok(())
                } else {
                    Err(PolarsError::DataTypeMisMatch(
                        "cannot append Series; data types don't match".into(),
                    ))
                }
            }

            fn filter(&self, filter: &BooleanChunked) -> Result<Series> {
                ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
            }

            fn take(&self, indices: &UInt32Chunked) -> Series {
                let indices = if indices.chunks.len() > 1 {
                    Cow::Owned(indices.rechunk())
                } else {
                    Cow::Borrowed(indices)
                };
                ChunkTake::take(&self.0, (&*indices).into()).into_series()
            }

            fn take_iter(&self, iter: &mut dyn Iterator<Item = usize>) -> Series {
                ChunkTake::take(&self.0, iter.into()).into_series()
            }

            fn take_every(&self, n: usize) -> Series {
                self.0.take_every(n).into_series()
            }

            unsafe fn take_iter_unchecked(&self, iter: &mut dyn Iterator<Item = usize>) -> Series {
                ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
            }

            unsafe fn take_unchecked(&self, idx: &UInt32Chunked) -> Result<Series> {
                let idx = if idx.chunks.len() > 1 {
                    Cow::Owned(idx.rechunk())
                } else {
                    Cow::Borrowed(idx)
                };
                Ok(ChunkTake::take_unchecked(&self.0, (&*idx).into()).into_series())
            }

            unsafe fn take_opt_iter_unchecked(
                &self,
                iter: &mut dyn Iterator<Item = Option<usize>>,
            ) -> Series {
                ChunkTake::take_unchecked(&self.0, Wrap(iter).into()).into_series()
            }

            fn take_opt_iter(&self, iter: &mut dyn Iterator<Item = Option<usize>>) -> Series {
                ChunkTake::take(&self.0, Wrap(iter).into()).into_series()
            }

            fn len(&self) -> usize {
                self.0.len()
            }

            fn rechunk(&self) -> Series {
                ChunkOps::rechunk(&self.0).into_series()
            }

            fn head(&self, length: Option<usize>) -> Series {
                self.0.head(length).into_series()
            }

            fn tail(&self, length: Option<usize>) -> Series {
                self.0.tail(length).into_series()
            }

            fn expand_at_index(&self, index: usize, length: usize) -> Series {
                ChunkExpandAtIndex::expand_at_index(&self.0, index, length).into_series()
            }

            fn cast_with_datatype(&self, data_type: &DataType) -> Result<Series> {
                use DataType::*;
                match data_type {
                    Boolean => ChunkCast::cast::<BooleanType>(&self.0).map(|ca| ca.into_series()),
                    Utf8 => ChunkCast::cast::<Utf8Type>(&self.0).map(|ca| ca.into_series()),
                    UInt8 => ChunkCast::cast::<UInt8Type>(&self.0).map(|ca| ca.into_series()),
                    UInt16 => ChunkCast::cast::<UInt16Type>(&self.0).map(|ca| ca.into_series()),
                    UInt32 => ChunkCast::cast::<UInt32Type>(&self.0).map(|ca| ca.into_series()),
                    UInt64 => ChunkCast::cast::<UInt64Type>(&self.0).map(|ca| ca.into_series()),
                    Int8 => ChunkCast::cast::<Int8Type>(&self.0).map(|ca| ca.into_series()),
                    Int16 => ChunkCast::cast::<Int16Type>(&self.0).map(|ca| ca.into_series()),
                    Int32 => ChunkCast::cast::<Int32Type>(&self.0).map(|ca| ca.into_series()),
                    Int64 => ChunkCast::cast::<Int64Type>(&self.0).map(|ca| ca.into_series()),
                    Float32 => ChunkCast::cast::<Float32Type>(&self.0).map(|ca| ca.into_series()),
                    Float64 => ChunkCast::cast::<Float64Type>(&self.0).map(|ca| ca.into_series()),
                    Date32 => ChunkCast::cast::<Date32Type>(&self.0).map(|ca| ca.into_series()),
                    Date64 => ChunkCast::cast::<Date64Type>(&self.0).map(|ca| ca.into_series()),
                    Time64(TimeUnit::Nanosecond) => {
                        ChunkCast::cast::<Time64NanosecondType>(&self.0).map(|ca| ca.into_series())
                    }
                    Duration(TimeUnit::Nanosecond) => {
                        ChunkCast::cast::<DurationNanosecondType>(&self.0)
                            .map(|ca| ca.into_series())
                    }
                    Duration(TimeUnit::Millisecond) => {
                        ChunkCast::cast::<DurationMillisecondType>(&self.0)
                            .map(|ca| ca.into_series())
                    }
                    List(_) => ChunkCast::cast::<ListType>(&self.0).map(|ca| ca.into_series()),
                    Categorical => {
                        ChunkCast::cast::<CategoricalType>(&self.0).map(|ca| ca.into_series())
                    }
                    dt => Err(PolarsError::Other(
                        format!("Casting to {:?} is not supported", dt).into(),
                    )),
                }
            }

            fn to_dummies(&self) -> Result<DataFrame> {
                ToDummies::to_dummies(&self.0)
            }

            fn value_counts(&self) -> Result<DataFrame> {
                ChunkUnique::value_counts(&self.0)
            }

            fn get(&self, index: usize) -> AnyValue {
                self.0.get_any_value(index)
            }

            #[inline]
            unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
                self.0.get_any_value_unchecked(index)
            }

            fn sort_in_place(&mut self, reverse: bool) {
                ChunkSort::sort_in_place(&mut self.0, reverse);
            }

            fn sort(&self, reverse: bool) -> Series {
                ChunkSort::sort(&self.0, reverse).into_series()
            }

            fn argsort(&self, reverse: bool) -> UInt32Chunked {
                ChunkSort::argsort(&self.0, reverse)
            }

            fn null_count(&self) -> usize {
                self.0.null_count()
            }

            fn unique(&self) -> Result<Series> {
                ChunkUnique::unique(&self.0).map(|ca| ca.into_series())
            }

            fn n_unique(&self) -> Result<usize> {
                ChunkUnique::n_unique(&self.0)
            }

            fn arg_unique(&self) -> Result<Vec<u32>> {
                ChunkUnique::arg_unique(&self.0)
            }

            fn arg_true(&self) -> Result<UInt32Chunked> {
                let ca: &BooleanChunked = self.bool()?;
                Ok(ca.arg_true())
            }

            fn is_null(&self) -> BooleanChunked {
                self.0.is_null()
            }

            fn is_not_null(&self) -> BooleanChunked {
                self.0.is_not_null()
            }

            fn is_unique(&self) -> Result<BooleanChunked> {
                ChunkUnique::is_unique(&self.0)
            }

            fn is_duplicated(&self) -> Result<BooleanChunked> {
                ChunkUnique::is_duplicated(&self.0)
            }

            fn null_bits(&self) -> Vec<(usize, Option<Buffer>)> {
                self.0.null_bits()
            }

            fn reverse(&self) -> Series {
                ChunkReverse::reverse(&self.0).into_series()
            }

            fn as_single_ptr(&mut self) -> Result<usize> {
                self.0.as_single_ptr()
            }

            fn shift(&self, periods: i64) -> Series {
                ChunkShift::shift(&self.0, periods).into_series()
            }

            fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Series> {
                ChunkFillNone::fill_none(&self.0, strategy).map(|ca| ca.into_series())
            }

            fn zip_with(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
                ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref())
                    .map(|ca| ca.into_series())
            }

            fn sum_as_series(&self) -> Series {
                ChunkAggSeries::sum_as_series(&self.0)
            }
            fn max_as_series(&self) -> Series {
                ChunkAggSeries::max_as_series(&self.0)
            }
            fn min_as_series(&self) -> Series {
                ChunkAggSeries::min_as_series(&self.0)
            }
            fn mean_as_series(&self) -> Series {
                ChunkAggSeries::mean_as_series(&self.0)
            }
            fn median_as_series(&self) -> Series {
                ChunkAggSeries::median_as_series(&self.0)
            }
            fn var_as_series(&self) -> Series {
                VarAggSeries::var_as_series(&self.0)
            }
            fn std_as_series(&self) -> Series {
                VarAggSeries::std_as_series(&self.0)
            }
            fn quantile_as_series(&self, quantile: f64) -> Result<Series> {
                ChunkAggSeries::quantile_as_series(&self.0, quantile)
            }
            fn rolling_mean(
                &self,
                window_size: usize,
                weight: Option<&[f64]>,
                ignore_null: bool,
            ) -> Result<Series> {
                ChunkWindow::rolling_mean(&self.0, window_size, weight, ignore_null)
                    .map(|ca| ca.into_series())
            }
            fn rolling_sum(
                &self,
                window_size: usize,
                weight: Option<&[f64]>,
                ignore_null: bool,
            ) -> Result<Series> {
                ChunkWindow::rolling_sum(&self.0, window_size, weight, ignore_null)
                    .map(|ca| ca.into_series())
            }
            fn rolling_min(
                &self,
                window_size: usize,
                weight: Option<&[f64]>,
                ignore_null: bool,
            ) -> Result<Series> {
                ChunkWindow::rolling_min(&self.0, window_size, weight, ignore_null)
                    .map(|ca| ca.into_series())
            }
            fn rolling_max(
                &self,
                window_size: usize,
                weight: Option<&[f64]>,
                ignore_null: bool,
            ) -> Result<Series> {
                ChunkWindow::rolling_max(&self.0, window_size, weight, ignore_null)
                    .map(|ca| ca.into_series())
            }

            fn fmt_list(&self) -> String {
                FmtList::fmt_list(&self.0)
            }

            #[cfg(feature = "temporal")]
            #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
            fn hour(&self) -> Result<Series> {
                self.date64().map(|ca| ca.hour().into_series())
            }

            #[cfg(feature = "temporal")]
            #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
            fn minute(&self) -> Result<Series> {
                self.date64().map(|ca| ca.minute().into_series())
            }

            #[cfg(feature = "temporal")]
            #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
            fn second(&self) -> Result<Series> {
                self.date64().map(|ca| ca.second().into_series())
            }

            #[cfg(feature = "temporal")]
            #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
            fn nanosecond(&self) -> Result<Series> {
                self.date64().map(|ca| ca.nanosecond().into_series())
            }

            #[cfg(feature = "temporal")]
            #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
            fn day(&self) -> Result<Series> {
                match self.0.dtype() {
                    DataType::Date32 => self.date32().map(|ca| ca.day().into_series()),
                    DataType::Date64 => self.date64().map(|ca| ca.day().into_series()),
                    _ => Err(PolarsError::InvalidOperation(
                        format!("operation not supported on dtype {:?}", self.dtype()).into(),
                    )),
                }
            }

            #[cfg(feature = "temporal")]
            #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
            fn ordinal_day(&self) -> Result<Series> {
                match self.0.dtype() {
                    DataType::Date32 => self.date32().map(|ca| ca.ordinal().into_series()),
                    DataType::Date64 => self.date64().map(|ca| ca.ordinal().into_series()),
                    _ => Err(PolarsError::InvalidOperation(
                        format!("operation not supported on dtype {:?}", self.dtype()).into(),
                    )),
                }
            }

            #[cfg(feature = "temporal")]
            #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
            fn month(&self) -> Result<Series> {
                match self.0.dtype() {
                    DataType::Date32 => self.date32().map(|ca| ca.month().into_series()),
                    DataType::Date64 => self.date64().map(|ca| ca.month().into_series()),
                    _ => Err(PolarsError::InvalidOperation(
                        format!("operation not supported on dtype {:?}", self.dtype()).into(),
                    )),
                }
            }

            #[cfg(feature = "temporal")]
            #[cfg_attr(docsrs, doc(cfg(feature = "temporal")))]
            fn year(&self) -> Result<Series> {
                match self.0.dtype() {
                    DataType::Date32 => self.date32().map(|ca| ca.year().into_series()),
                    DataType::Date64 => self.date64().map(|ca| ca.year().into_series()),
                    _ => Err(PolarsError::InvalidOperation(
                        format!("operation not supported on dtype {:?}", self.dtype()).into(),
                    )),
                }
            }
            fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
                Arc::new(Wrap(Clone::clone(&self.0)))
            }

            #[cfg(feature = "random")]
            #[cfg_attr(docsrs, doc(cfg(feature = "random")))]
            fn sample_n(&self, n: usize, with_replacement: bool) -> Result<Series> {
                self.0
                    .sample_n(n, with_replacement)
                    .map(|ca| ca.into_series())
            }

            #[cfg(feature = "random")]
            #[cfg_attr(docsrs, doc(cfg(feature = "random")))]
            fn sample_frac(&self, frac: f64, with_replacement: bool) -> Result<Series> {
                self.0
                    .sample_frac(frac, with_replacement)
                    .map(|ca| ca.into_series())
            }

            fn pow(&self, exponent: f64) -> Result<Series> {
                let f_err = || {
                    Err(PolarsError::InvalidOperation(
                        format!("power operation not supported on dtype {:?}", self.dtype()).into(),
                    ))
                };

                match self.dtype() {
                    DataType::Utf8 | DataType::List(_) | DataType::Boolean => f_err(),
                    DataType::Float32 => Ok(self.0.pow_f32(exponent as f32).into_series()),
                    _ => Ok(self.0.pow_f64(exponent).into_series()),
                }
            }

            fn peak_max(&self) -> BooleanChunked {
                self.0.peak_max()
            }

            fn peak_min(&self) -> BooleanChunked {
                self.0.peak_min()
            }
        }
    };
}

impl_dyn_series!(Float32Chunked);
impl_dyn_series!(Float64Chunked);
impl_dyn_series!(Utf8Chunked);
impl_dyn_series!(ListChunked);
impl_dyn_series!(BooleanChunked);
impl_dyn_series!(UInt8Chunked);
impl_dyn_series!(UInt16Chunked);
impl_dyn_series!(UInt32Chunked);
impl_dyn_series!(UInt64Chunked);
impl_dyn_series!(Int8Chunked);
impl_dyn_series!(Int16Chunked);
impl_dyn_series!(Int32Chunked);
impl_dyn_series!(Int64Chunked);
impl_dyn_series!(DurationNanosecondChunked);
impl_dyn_series!(DurationMillisecondChunked);
impl_dyn_series!(Date32Chunked);
impl_dyn_series!(Date64Chunked);
impl_dyn_series!(Time64NanosecondChunked);
impl_dyn_series!(CategoricalChunked);

#[cfg(feature = "object")]
impl<T> IntoSeries for ObjectChunked<T>
where
    T: 'static + std::fmt::Debug + Clone + Send + Sync + Default,
{
    fn into_series(self) -> Series {
        Series(Arc::new(Wrap(self)))
    }
}

#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
impl<T> PrivateSeries for Wrap<ObjectChunked<T>> where
    T: 'static + Debug + Clone + Send + Sync + Default
{
}
#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
impl<T> SeriesTrait for Wrap<ObjectChunked<T>>
where
    T: 'static + Debug + Clone + Send + Sync + Default,
{
    fn rename(&mut self, name: &str) {
        ObjectChunked::rename(&mut self.0, name)
    }

    fn array_data(&self) -> Vec<ArrayDataRef> {
        ObjectChunked::array_data(&self.0)
    }

    fn chunk_lengths(&self) -> &Vec<usize> {
        ObjectChunked::chunk_id(&self.0)
    }

    fn name(&self) -> &str {
        ObjectChunked::name(&self.0)
    }

    fn field(&self) -> &Field {
        ObjectChunked::ref_field(&self.0)
    }

    fn dtype(&self) -> &DataType {
        ObjectChunked::dtype(&self.0)
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        ObjectChunked::chunks(&self.0)
    }

    fn append_array(&mut self, other: ArrayRef) -> Result<()> {
        ObjectChunked::append_array(&mut self.0, other)
    }

    fn slice(&self, offset: usize, length: usize) -> Result<Series> {
        ObjectChunked::slice(&self.0, offset, length).map(|ca| ca.into_series())
    }

    fn append(&mut self, other: &Series) -> Result<()> {
        if self.dtype() == other.dtype() {
            ObjectChunked::append(&mut self.0, other.as_ref().as_ref());
            Ok(())
        } else {
            Err(PolarsError::DataTypeMisMatch(
                "cannot append Series; data types don't match".into(),
            ))
        }
    }

    fn filter(&self, filter: &BooleanChunked) -> Result<Series> {
        ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
    }

    fn take_iter(&self, _iter: &mut dyn Iterator<Item = usize>) -> Series {
        todo!()
    }

    unsafe fn take_iter_unchecked(&self, _iter: &mut dyn Iterator<Item = usize>) -> Series {
        todo!()
    }

    unsafe fn take_unchecked(&self, _idx: &UInt32Chunked) -> Result<Series> {
        todo!()
    }

    unsafe fn take_opt_iter_unchecked(
        &self,
        _iter: &mut dyn Iterator<Item = Option<usize>>,
    ) -> Series {
        todo!()
    }

    fn take_opt_iter(&self, _iter: &mut dyn Iterator<Item = Option<usize>>) -> Series {
        todo!()
    }

    fn len(&self) -> usize {
        ObjectChunked::len(&self.0)
    }

    fn rechunk(&self) -> Series {
        ChunkOps::rechunk(&self.0).into_series()
    }

    fn head(&self, length: Option<usize>) -> Series {
        ObjectChunked::head(&self.0, length).into_series()
    }

    fn tail(&self, length: Option<usize>) -> Series {
        ObjectChunked::tail(&self.0, length).into_series()
    }

    fn take_every(&self, n: usize) -> Series {
        self.0.take_every(n).into_series()
    }

    fn expand_at_index(&self, index: usize, length: usize) -> Series {
        ChunkExpandAtIndex::expand_at_index(&self.0, index, length).into_series()
    }

    fn cast_with_datatype(&self, _data_type: &DataType) -> Result<Series> {
        Err(PolarsError::InvalidOperation(
            "cannot cast array of type ObjectChunked to arrow datatype".into(),
        ))
    }

    fn to_dummies(&self) -> Result<DataFrame> {
        ToDummies::to_dummies(&self.0)
    }

    fn value_counts(&self) -> Result<DataFrame> {
        ChunkUnique::value_counts(&self.0)
    }

    fn get(&self, index: usize) -> AnyValue {
        ObjectChunked::get_any_value(&self.0, index)
    }

    fn sort_in_place(&mut self, reverse: bool) {
        ChunkSort::sort_in_place(&mut self.0, reverse)
    }

    fn sort(&self, reverse: bool) -> Series {
        ChunkSort::sort(&self.0, reverse).into_series()
    }

    fn argsort(&self, reverse: bool) -> UInt32Chunked {
        ChunkSort::argsort(&self.0, reverse)
    }

    fn null_count(&self) -> usize {
        ObjectChunked::null_count(&self.0)
    }

    fn unique(&self) -> Result<Series> {
        ChunkUnique::unique(&self.0).map(|ca| ca.into_series())
    }

    fn n_unique(&self) -> Result<usize> {
        ChunkUnique::n_unique(&self.0)
    }

    fn arg_unique(&self) -> Result<Vec<u32>> {
        ChunkUnique::arg_unique(&self.0)
    }

    fn is_null(&self) -> BooleanChunked {
        ObjectChunked::is_null(&self.0)
    }

    fn is_not_null(&self) -> BooleanChunked {
        ObjectChunked::is_not_null(&self.0)
    }

    fn is_unique(&self) -> Result<BooleanChunked> {
        ChunkUnique::is_unique(&self.0)
    }

    fn is_duplicated(&self) -> Result<BooleanChunked> {
        ChunkUnique::is_duplicated(&self.0)
    }

    fn null_bits(&self) -> Vec<(usize, Option<Buffer>)> {
        ObjectChunked::null_bits(&self.0)
    }

    fn reverse(&self) -> Series {
        ChunkReverse::reverse(&self.0).into_series()
    }

    fn shift(&self, periods: i64) -> Series {
        ChunkShift::shift(&self.0, periods).into_series()
    }

    fn fill_none(&self, strategy: FillNoneStrategy) -> Result<Series> {
        ChunkFillNone::fill_none(&self.0, strategy).map(|ca| ca.into_series())
    }

    fn zip_with(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
        ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref()).map(|ca| ca.into_series())
    }

    fn fmt_list(&self) -> String {
        FmtList::fmt_list(&self.0)
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(Wrap(Clone::clone(&self.0)))
    }

    #[cfg(feature = "random")]
    #[cfg_attr(docsrs, doc(cfg(feature = "random")))]
    fn sample_n(&self, n: usize, with_replacement: bool) -> Result<Series> {
        ObjectChunked::sample_n(&self.0, n, with_replacement).map(|ca| ca.into_series())
    }

    #[cfg(feature = "random")]
    #[cfg_attr(docsrs, doc(cfg(feature = "random")))]
    fn sample_frac(&self, frac: f64, with_replacement: bool) -> Result<Series> {
        ObjectChunked::sample_frac(&self.0, frac, with_replacement).map(|ca| ca.into_series())
    }

    fn get_as_any(&self, index: usize) -> &dyn Any {
        ObjectChunked::get_as_any(&self.0, index)
    }
}
