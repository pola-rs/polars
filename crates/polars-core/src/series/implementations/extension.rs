use super::*;
use crate::prelude::*;

unsafe impl IntoSeries for ExtensionChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl SeriesWrap<ExtensionChunked> {
    fn apply_on_storage<F>(&self, apply: F) -> Series
    where
        F: Fn(&Series) -> Series,
    {
        apply(self.0.storage()).into_extension(self.0.extension_type().clone())
    }

    fn try_apply_on_storage<F>(&self, apply: F) -> PolarsResult<Series>
    where
        F: Fn(&Series) -> PolarsResult<Series>,
    {
        Ok(apply(self.0.storage())?.into_extension(self.0.extension_type().clone()))
    }
}

impl private::PrivateSeries for SeriesWrap<ExtensionChunked> {
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

    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsType> {
        self.0.storage().group_tuples(multithreaded, sorted)
    }

    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        assert!(self._dtype() == other.dtype());
        self.try_apply_on_storage(|s| s.zip_with_same_type(mask, other.ext()?.storage()))
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsType) -> Series {
        let list = self.0.storage().agg_list(groups);
        let mut list = list.list().unwrap().clone();
        unsafe { list.to_logical(self.dtype().clone()) };
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

impl private::PrivateSeriesNumeric for SeriesWrap<ExtensionChunked> {
    fn bit_repr(&self) -> Option<BitRepr> {
        self.0.storage().bit_repr()
    }
}

impl SeriesTrait for SeriesWrap<ExtensionChunked> {
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
        self.0
            .storage()
            .slice(offset, length)
            .into_extension(self.0.extension_type().clone())
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (left, right) = self.0.storage().split_at(offset);
        (
            left.into_extension(self.0.extension_type().clone()),
            right.into_extension(self.0.extension_type().clone()),
        )
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        assert!(self.0.dtype() == other.dtype());
        self.0.storage_mut().append(other.ext()?.storage())?;
        Ok(())
    }

    fn append_owned(&mut self, mut other: Series) -> PolarsResult<()> {
        assert!(self.0.dtype() == other.dtype());
        self.0.storage_mut().append_owned(std::mem::take(
            other
                ._get_inner_mut()
                .as_any_mut()
                .downcast_mut::<ExtensionChunked>()
                .unwrap()
                .storage_mut(),
        ))?;
        Ok(())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        assert!(self.0.dtype() == other.dtype());
        self.0.storage_mut().extend(other.ext()?.storage())?;
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

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.apply_on_storage(|s| s.new_from_index(index, length))
    }

    fn deposit(&self, validity: &Bitmap) -> Series {
        self.apply_on_storage(|s| s.deposit(validity))
    }

    fn find_validity_mismatch(&self, other: &Series, idxs: &mut Vec<IdxSize>) {
        assert!(self.0.dtype() == other.dtype());
        self.0
            .storage()
            .find_validity_mismatch(other.ext().unwrap().storage(), idxs)
    }

    fn cast(&self, dtype: &DataType, options: CastOptions) -> PolarsResult<Series> {
        self.0.cast_with_options(dtype, options)
    }

    unsafe fn get_unchecked(&self, index: usize) -> AnyValue<'_> {
        self.0.storage().get_unchecked(index)
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
        // no-op
    }

    fn trim_lists_to_normalized_offsets(&self) -> Option<Series> {
        let trimmed = self.0.storage().trim_lists_to_normalized_offsets()?;
        Some(trimmed.into_extension(self.0.extension_type().clone()))
    }

    fn propagate_nulls(&self) -> Option<Series> {
        let propagated = self.0.storage().propagate_nulls()?;
        Some(propagated.into_extension(self.0.extension_type().clone()))
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

    fn n_unique(&self) -> PolarsResult<usize> {
        self.0.storage().n_unique()
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        self.0.storage().arg_unique()
    }

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
