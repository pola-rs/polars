//! This module exists to reduce compilation times.
//!
//! All the data types are backed by a physical type in memory e.g. Date -> i32, Datetime-> i64.
//!
//! Series lead to code implementations of all traits. Whereas there are a lot of duplicates due to
//! data types being backed by the same physical type. In this module we reduce compile times by
//! opting for a little more run time cost. We cast to the physical type -> apply the operation and
//! (depending on the result) cast back to the original type
//!
use super::*;
#[cfg(feature = "algorithm_group_by")]
use crate::frame::group_by::*;
use crate::prelude::*;

unsafe impl IntoSeries for DateChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl private::PrivateSeries for SeriesWrap<DateChunked> {
    fn compute_len(&mut self) {
        self.0.compute_len()
    }

    fn _field(&self) -> Cow<Field> {
        Cow::Owned(self.0.field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn _get_flags(&self) -> MetadataFlags {
        self.0.get_flags()
    }

    fn _set_flags(&mut self, flags: MetadataFlags) {
        self.0.set_flags(flags)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        let other = other.to_physical_repr().into_owned();
        self.0
            .zip_with(mask, other.as_ref().as_ref())
            .map(|ca| ca.into_date().into_series())
    }

    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.0.vec_hash(random_state, buf)?;
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        build_hasher: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        self.0.vec_hash_combine(build_hasher, hashes)?;
        Ok(())
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_min(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_min(groups).into_date().into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_max(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_max(groups).into_date().into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        // we cannot cast and dispatch as the inner type of the list would be incorrect
        self.0
            .agg_list(groups)
            .cast(&DataType::List(Box::new(self.dtype().clone())))
            .unwrap()
    }

    fn subtract(&self, rhs: &Series) -> PolarsResult<Series> {
        match rhs.dtype() {
            DataType::Date => {
                let dt = DataType::Datetime(TimeUnit::Milliseconds, None);
                let lhs = self.cast(&dt, CastOptions::NonStrict)?;
                let rhs = rhs.cast(&dt)?;
                lhs.subtract(&rhs)
            },
            DataType::Duration(_) => std::ops::Sub::sub(
                &self.cast(
                    &DataType::Datetime(TimeUnit::Milliseconds, None),
                    CastOptions::NonStrict,
                )?,
                rhs,
            )?
            .cast(&DataType::Date),
            dtr => polars_bail!(opq = sub, DataType::Date, dtr),
        }
    }

    fn add_to(&self, rhs: &Series) -> PolarsResult<Series> {
        match rhs.dtype() {
            DataType::Duration(_) => std::ops::Add::add(
                &self.cast(
                    &DataType::Datetime(TimeUnit::Milliseconds, None),
                    CastOptions::NonStrict,
                )?,
                rhs,
            )?
            .cast(&DataType::Date),
            dtr => polars_bail!(opq = add, DataType::Date, dtr),
        }
    }

    fn multiply(&self, rhs: &Series) -> PolarsResult<Series> {
        polars_bail!(opq = mul, self.0.dtype(), rhs.dtype());
    }

    fn divide(&self, rhs: &Series) -> PolarsResult<Series> {
        polars_bail!(opq = div, self.0.dtype(), rhs.dtype());
    }

    fn remainder(&self, rhs: &Series) -> PolarsResult<Series> {
        polars_bail!(opq = rem, self.0.dtype(), rhs.dtype());
    }
    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        self.0.group_tuples(multithreaded, sorted)
    }

    fn arg_sort_multiple(
        &self,
        by: &[Series],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        self.0.deref().arg_sort_multiple(by, options)
    }
}

impl SeriesTrait for SeriesWrap<DateChunked> {
    fn rename(&mut self, name: &str) {
        self.0.rename(name);
    }

    fn chunk_lengths(&self) -> ChunkLenIter {
        self.0.chunk_lengths()
    }
    fn name(&self) -> &str {
        self.0.name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        self.0.chunks()
    }
    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.chunks_mut()
    }

    fn shrink_to_fit(&mut self) {
        self.0.shrink_to_fit()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_date().into_series()
    }
    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (a, b) = self.0.split_at(offset);
        (a.into_date().into_series(), b.into_date().into_series())
    }

    fn mean(&self) -> Option<f64> {
        self.0.mean()
    }

    fn median(&self) -> Option<f64> {
        self.0.median()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        let other = other.to_physical_repr();
        // 3 refs
        // ref Cow
        // ref SeriesTrait
        // ref ChunkedArray
        self.0.append(other.as_ref().as_ref().as_ref())?;
        Ok(())
    }
    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        // 3 refs
        // ref Cow
        // ref SeriesTrait
        // ref ChunkedArray
        let other = other.to_physical_repr();
        self.0.extend(other.as_ref().as_ref().as_ref())?;
        Ok(())
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        self.0.filter(filter).map(|ca| ca.into_date().into_series())
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(self.0.take(indices)?.into_date().into_series())
    }

    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
        self.0.take_unchecked(indices).into_date().into_series()
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        Ok(self.0.take(indices)?.into_date().into_series())
    }

    unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
        self.0.take_unchecked(indices).into_date().into_series()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.0.rechunk().into_date().into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.0
            .new_from_index(index, length)
            .into_date()
            .into_series()
    }

    fn cast(&self, data_type: &DataType, cast_options: CastOptions) -> PolarsResult<Series> {
        match data_type {
            DataType::String => Ok(self
                .0
                .clone()
                .into_series()
                .date()
                .unwrap()
                .to_string("%Y-%m-%d")?
                .into_series()),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                let mut out = self
                    .0
                    .cast_with_options(data_type, CastOptions::NonStrict)?;
                out.set_sorted_flag(self.0.is_sorted_flag());
                Ok(out)
            },
            _ => self.0.cast_with_options(data_type, cast_options),
        }
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        self.0.get_any_value(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
        Ok(self.0.sort_with(options).into_date().into_series())
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_nulls(&self) -> bool {
        self.0.has_nulls()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn unique(&self) -> PolarsResult<Series> {
        self.0.unique().map(|ca| ca.into_date().into_series())
    }

    #[cfg(feature = "algorithm_group_by")]
    fn n_unique(&self) -> PolarsResult<usize> {
        self.0.n_unique()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.0.arg_unique()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn reverse(&self) -> Series {
        self.0.reverse().into_date().into_series()
    }

    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self.0.as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        self.0.shift(periods).into_date().into_series()
    }

    fn max_reduce(&self) -> PolarsResult<Scalar> {
        let sc = self.0.max_reduce();
        let av = sc.value().cast(self.dtype()).into_static().unwrap();
        Ok(Scalar::new(self.dtype().clone(), av))
    }

    fn min_reduce(&self) -> PolarsResult<Scalar> {
        let sc = self.0.min_reduce();
        let av = sc.value().cast(self.dtype()).into_static().unwrap();
        Ok(Scalar::new(self.dtype().clone(), av))
    }

    fn median_reduce(&self) -> PolarsResult<Scalar> {
        let av: AnyValue = self
            .median()
            .map(|v| (v * (MS_IN_DAY as f64)) as i64)
            .into();
        Ok(Scalar::new(
            DataType::Datetime(TimeUnit::Milliseconds, None),
            av,
        ))
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<DateChunked> {
    fn bit_repr(&self) -> Option<BitRepr> {
        Some(self.0.to_bit_repr())
    }
}
