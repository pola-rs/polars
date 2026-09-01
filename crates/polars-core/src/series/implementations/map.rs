use super::*;
use crate::prelude::*;

unsafe impl IntoSeries for MapChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl SeriesWrap<MapChunked> {
    /// Re-wrap a storage series produced by an operation that preserves the key
    /// uniqueness invariant, i.e. one that only adds, removes or reorders whole rows.
    fn rewrap(&self, storage: Series) -> Series {
        unsafe { MapChunked::from_storage_unchecked(self.0.dtype().clone(), storage) }.into_series()
    }

    fn apply_on_storage<F>(&self, apply: F) -> Series
    where
        F: FnOnce(&Series) -> Series,
    {
        self.rewrap(apply(self.0.storage()))
    }

    fn try_apply_on_storage<F>(&self, apply: F) -> PolarsResult<Series>
    where
        F: Fn(&Series) -> PolarsResult<Series>,
    {
        Ok(self.rewrap(apply(self.0.storage())?))
    }
}

impl private::PrivateSeries for SeriesWrap<MapChunked> {
    fn _field(&self) -> Cow<'_, Field> {
        Cow::Owned(self.0.field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn compute_len(&mut self) {
        self.0.storage_mut().compute_len();
    }

    fn _get_flags(&self) -> StatisticsFlags {
        self.0.storage().get_flags()
    }

    fn _set_flags(&mut self, flags: StatisticsFlags) {
        self.0.storage_mut().set_flags(flags)
    }

    fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
        self.0.storage().into_total_eq_inner()
    }

    fn into_total_ord_inner<'a>(&'a self) -> Box<dyn TotalOrdInner + 'a> {
        self.0.storage().into_total_ord_inner()
    }

    fn vec_hash(
        &self,
        build_hasher: PlSeedableRandomStateQuality,
        buf: &mut Vec<u64>,
    ) -> PolarsResult<()> {
        self.0.storage().vec_hash(build_hasher, buf)
    }

    fn vec_hash_combine(
        &self,
        build_hasher: PlSeedableRandomStateQuality,
        hashes: &mut [u64],
    ) -> PolarsResult<()> {
        self.0.storage().vec_hash_combine(build_hasher, hashes)
    }

    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsType> {
        self.0.storage().group_tuples(multithreaded, sorted)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        assert!(self._dtype() == other.dtype());
        self.try_apply_on_storage(|s| s.zip_with_same_type(mask, other.map()?.storage()))
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsType) -> Series {
        let list = self.0.storage().agg_list(groups);
        let mut list = list.list().unwrap().clone();

        list.set_inner_dtype(self.dtype().clone());
        list.into_series()
    }

    fn arg_sort_multiple(
        &self,
        by: &[Column],
        options: &SortMultipleOptions,
    ) -> PolarsResult<IdxCa> {
        self.0.storage().arg_sort_multiple(by, options)
    }
}

impl private::PrivateSeriesNumeric for SeriesWrap<MapChunked> {
    fn bit_repr(&self) -> Option<BitRepr> {
        None
    }
}

impl SeriesTrait for SeriesWrap<MapChunked> {
    fn rename(&mut self, name: PlSmallStr) {
        self.0.rename(name);
    }

    fn chunk_lengths(&self) -> ChunkLenIter<'_> {
        self.0.storage().chunk_lengths()
    }

    fn name(&self) -> &PlSmallStr {
        self.0.name()
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        self.0.storage().chunks()
    }

    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.storage_mut().chunks_mut()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.apply_on_storage(|s| s.slice(offset, length))
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (left, right) = self.0.storage().split_at(offset);
        (self.rewrap(left), self.rewrap(right))
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        assert!(self.0.dtype() == other.dtype());
        self.0.storage_mut().append(other.map()?.storage())?;
        Ok(())
    }

    fn append_owned(&mut self, mut other: Series) -> PolarsResult<()> {
        assert!(self.0.dtype() == other.dtype());
        self.0.storage_mut().append_owned(std::mem::take(
            other
                ._get_inner_mut()
                .as_any_mut()
                .downcast_mut::<MapChunked>()
                .unwrap()
                .storage_mut(),
        ))?;
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        assert!(self.0.dtype() == other.dtype());
        self.0.storage_mut().extend(other.map()?.storage())?;
        Ok(())
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        self.try_apply_on_storage(|s| s.filter(filter))
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        self.try_apply_on_storage(|s| s.take(indices))
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> Series {
        self.apply_on_storage(|s| s.take_unchecked(idx))
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        self.try_apply_on_storage(|s| s.take_slice(indices))
    }

    unsafe fn take_slice_unchecked(&self, idx: &[IdxSize]) -> Series {
        self.apply_on_storage(|s| s.take_slice_unchecked(idx))
    }

    fn len(&self) -> usize {
        self.0.storage().len()
    }

    fn rechunk(&self) -> Series {
        self.apply_on_storage(|s| s.rechunk())
    }

    fn with_validity(&self, validity: Option<Bitmap>) -> Series {
        self.apply_on_storage(move |s| s.with_validity(validity))
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.apply_on_storage(|s| s.new_from_index(index, length))
    }

    fn deposit(&self, validity: &Bitmap) -> Series {
        self.apply_on_storage(|s| s.deposit(validity))
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        // `handle_casting_failures` compares a cast's input against its output, so `other`
        // is same-length but not necessarily the same dtype, or even a Map.
        let other = other.try_map().map_or(other, |map| map.storage());
        self.0.storage().find_validity_mismatch(other, idxs)
    }

    fn cast(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        self.0.cast_with_options(dtype, options)
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue<'_>> {
        self.0.get_any_value(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> AnyValue<'_> {
        unsafe { self.0.get_any_value_unchecked(index) }
    }

    fn null_count(&self) -> usize {
        self.0.storage().null_count()
    }

    fn has_nulls(&self) -> bool {
        self.0.storage().has_nulls()
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.storage().is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.storage().is_not_null()
    }

    fn reverse(&self) -> Series {
        self.apply_on_storage(|s| s.reverse())
    }

    fn shift(&self, periods: i64) -> Series {
        self.apply_on_storage(|s| s.shift(periods))
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }

    fn as_phys_any(&self) -> &dyn Any {
        self.0.storage().as_phys_any()
    }

    fn as_arc_any(self: Arc<Self>) -> Arc<dyn Any + Send + Sync> {
        self
    }

    fn field(&self) -> Cow<'_, Field> {
        Cow::Owned(self.0.field())
    }

    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn n_chunks(&self) -> usize {
        self.0.storage().n_chunks()
    }

    fn shrink_to_fit(&mut self) {
        self.0.storage_mut().shrink_to_fit();
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Series> {
        let trimmed = self.0.storage().trim_lists_to_normalized_offsets()?;
        Some(self.rewrap(trimmed))
    }

    fn propagate_nulls(&self) -> Option<Series> {
        let propagated = self.0.storage().propagate_nulls()?;
        Some(self.rewrap(propagated))
    }

    fn sort_with(&self, options: SortOptions) -> PolarsResult<Series> {
        self.try_apply_on_storage(|s| s.sort_with(options))
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.storage().arg_sort(options)
    }

    fn unique(&self) -> PolarsResult<Series> {
        self.try_apply_on_storage(|s| s.unique())
    }

    #[cfg(feature = "algorithm_group_by")]
    fn n_unique(&self) -> PolarsResult<usize> {
        self.0.storage().n_unique()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.0.storage().arg_unique()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn unique_id(&self) -> PolarsResult<(IdxSize, Vec<IdxSize>)> {
        self.0.storage().unique_id()
    }

    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self.0.storage_mut().as_single_ptr()
    }

    #[cfg(feature = "approx_unique")]
    fn approx_n_unique(&self) -> PolarsResult<IdxSize> {
        self.0.storage().approx_n_unique()
    }
}
