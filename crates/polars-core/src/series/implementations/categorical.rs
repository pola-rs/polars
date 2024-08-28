use super::*;
use crate::chunked_array::comparison::*;
use crate::prelude::*;

unsafe impl IntoSeries for CategoricalChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl SeriesWrap<CategoricalChunked> {
    fn finish_with_state(&self, keep_fast_unique: bool, cats: UInt32Chunked) -> CategoricalChunked {
        let mut out = unsafe {
            CategoricalChunked::from_cats_and_rev_map_unchecked(
                cats,
                self.0.get_rev_map().clone(),
                self.0.is_enum(),
                self.0.get_ordering(),
            )
        };
        if keep_fast_unique && self.0._can_fast_unique() {
            out.set_fast_unique(true)
        }
        out
    }

    fn with_state<F>(&self, keep_fast_unique: bool, apply: F) -> CategoricalChunked
    where
        F: Fn(&UInt32Chunked) -> UInt32Chunked,
    {
        let cats = apply(self.0.physical());
        self.finish_with_state(keep_fast_unique, cats)
    }

    fn try_with_state<'a, F>(
        &'a self,
        keep_fast_unique: bool,
        apply: F,
    ) -> PolarsResult<CategoricalChunked>
    where
        F: for<'b> Fn(&'a UInt32Chunked) -> PolarsResult<UInt32Chunked>,
    {
        let cats = apply(self.0.physical())?;
        Ok(self.finish_with_state(keep_fast_unique, cats))
    }
}

impl private::PrivateSeries for SeriesWrap<CategoricalChunked> {
    fn compute_len(&mut self) {
        self.0.physical_mut().compute_len()
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

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.physical().equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        self.0
            .zip_with(mask, other.categorical()?)
            .map(|ca| ca.into_series())
    }
    fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
        if self.0.uses_lexical_ordering() {
            (&self.0).into_total_ord_inner()
        } else {
            self.0.physical().into_total_ord_inner()
        }
    }

    fn vec_hash(&self, random_state: PlRandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.0.physical().vec_hash(random_state, buf)?;
        Ok(())
    }

    fn vec_hash_combine(
        &self,
        build_hasher: PlRandomState,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        self.0.physical().vec_hash_combine(build_hasher, hashes)?;
        Ok(())
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        // we cannot cast and dispatch as the inner type of the list would be incorrect
        let list = self.0.physical().agg_list(groups);
        let mut list = list.list().unwrap().clone();
        unsafe { list.to_logical(self.dtype().clone()) };
        list.into_series()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        #[cfg(feature = "performant")]
        {
            Ok(self.0.group_tuples_perfect(multithreaded, sorted))
        }
        #[cfg(not(feature = "performant"))]
        {
            self.0.physical().group_tuples(multithreaded, sorted)
        }
    }

    fn arg_sort_multiple(
        &self,
        by: &[Series],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        self.0.arg_sort_multiple(by, options)
    }
}

impl SeriesTrait for SeriesWrap<CategoricalChunked> {
    fn rename(&mut self, name: &str) {
        self.0.physical_mut().rename(name);
    }

    fn chunk_lengths(&self) -> ChunkLenIter {
        self.0.physical().chunk_lengths()
    }
    fn name(&self) -> &str {
        self.0.physical().name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        self.0.physical().chunks()
    }
    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.physical_mut().chunks_mut()
    }
    fn shrink_to_fit(&mut self) {
        self.0.physical_mut().shrink_to_fit()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.with_state(false, |cats| cats.slice(offset, length))
            .into_series()
    }
    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (a, b) = self.0.physical().split_at(offset);
        let a = self.finish_with_state(false, a).into_series();
        let b = self.finish_with_state(false, b).into_series();
        (a, b)
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        self.0.append(other.categorical().unwrap())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        let other_ca = other.categorical().unwrap();
        // Fast path for globals of the same source
        let rev_map_self = self.0.get_rev_map();
        let rev_map_other = other_ca.get_rev_map();
        match (&**rev_map_self, &**rev_map_other) {
            (RevMapping::Global(_, _, idl), RevMapping::Global(_, _, idr)) if idl == idr => {
                let mut rev_map_merger = GlobalRevMapMerger::new(rev_map_self.clone());
                rev_map_merger.merge_map(rev_map_other)?;
                self.0.physical_mut().extend(other_ca.physical())?;
                // SAFETY: rev_maps are merged
                unsafe { self.0.set_rev_map(rev_map_merger.finish(), false) };
                Ok(())
            },
            _ => self.0.append(other_ca),
        }
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        self.try_with_state(false, |cats| cats.filter(filter))
            .map(|ca| ca.into_series())
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        self.try_with_state(false, |cats| cats.take(indices))
            .map(|ca| ca.into_series())
    }

    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
        self.with_state(false, |cats| cats.take_unchecked(indices))
            .into_series()
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        self.try_with_state(false, |cats| cats.take(indices))
            .map(|ca| ca.into_series())
    }

    unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
        self.with_state(false, |cats| cats.take_unchecked(indices))
            .into_series()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.with_state(true, |ca| ca.rechunk()).into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.with_state(true, |cats| cats.new_from_index(index, length))
            .into_series()
    }

    fn cast(&self, data_type: &DataType, options: CastOptions) -> PolarsResult<Series> {
        self.0.cast_with_options(data_type, options)
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        self.0.get_any_value(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
        Ok(self.0.sort_with(options).into_series())
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }

    fn null_count(&self) -> usize {
        self.0.physical().null_count()
    }

    fn has_nulls(&self) -> bool {
        self.0.physical().has_nulls()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn unique(&self) -> PolarsResult<Series> {
        self.0.unique().map(|ca| ca.into_series())
    }

    #[cfg(feature = "algorithm_group_by")]
    fn n_unique(&self) -> PolarsResult<usize> {
        self.0.n_unique()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.0.physical().arg_unique()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.physical().is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.physical().is_not_null()
    }

    fn reverse(&self) -> Series {
        self.with_state(true, |cats| cats.reverse()).into_series()
    }

    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self.0.physical_mut().as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        self.with_state(false, |ca| ca.shift(periods)).into_series()
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn min_reduce(&self) -> PolarsResult<Scalar> {
        Ok(ChunkAggSeries::min_reduce(&self.0))
    }

    fn max_reduce(&self) -> PolarsResult<Scalar> {
        Ok(ChunkAggSeries::max_reduce(&self.0))
    }
    fn as_any(&self) -> &dyn Any {
        &self.0
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<CategoricalChunked> {
    fn bit_repr(&self) -> Option<BitRepr> {
        Some(BitRepr::Small(self.0.physical().clone()))
    }
}
