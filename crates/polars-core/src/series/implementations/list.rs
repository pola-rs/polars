use super::*;
use crate::chunked_array::comparison::*;
#[cfg(feature = "algorithm_group_by")]
use crate::frame::group_by::*;
use crate::prelude::*;

impl private::PrivateSeries for SeriesWrap<ListChunked> {
    fn compute_len(&mut self) {
        self.0.compute_len()
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.ref_field().data_type()
    }
    fn _get_flags(&self) -> MetadataFlags {
        self.0.get_flags()
    }
    fn _set_flags(&mut self, flags: MetadataFlags) {
        self.0.set_flags(flags)
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        self.0.equal_element(idx_self, idx_other, other)
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        ChunkZip::zip_with(&self.0, mask, other.as_ref().as_ref()).map(|ca| ca.into_series())
    }

    #[cfg(feature = "algorithm_group_by")]
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_list(groups)
    }

    #[cfg(feature = "algorithm_group_by")]
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        IntoGroupsProxy::group_tuples(&self.0, multithreaded, sorted)
    }

    fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
        (&self.0).into_total_eq_inner()
    }
}

impl SeriesTrait for SeriesWrap<ListChunked> {
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

    fn sum_reduce(&self) -> PolarsResult<Scalar> {
        polars_bail!(
            op = "`sum`",
            self.dtype(),
            hint = "you may mean to call `concat_list`"
        );
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        self.0.slice(offset, length).into_series()
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (a, b) = self.0.split_at(offset);
        (a.into_series(), b.into_series())
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), append);
        self.0.append(other.as_ref().as_ref())
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        polars_ensure!(self.0.dtype() == other.dtype(), extend);
        self.0.extend(other.as_ref().as_ref())
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(self.0.take(indices)?.into_series())
    }

    unsafe fn take_unchecked(&self, indices: &IdxCa) -> Series {
        self.0.take_unchecked(indices).into_series()
    }

    fn take_slice(&self, indices: &[IdxSize]) -> PolarsResult<Series> {
        Ok(self.0.take(indices)?.into_series())
    }

    unsafe fn take_slice_unchecked(&self, indices: &[IdxSize]) -> Series {
        self.0.take_unchecked(indices).into_series()
    }

    fn len(&self) -> usize {
        self.0.len()
    }

    fn rechunk(&self) -> Series {
        self.0.rechunk().into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        ChunkExpandAtIndex::new_from_index(&self.0, index, length).into_series()
    }

    fn cast(&self, data_type: &DataType, cast_options: CastOptions) -> PolarsResult<Series> {
        self.0.cast_with_options(data_type, cast_options)
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        self.0.get_any_value(index)
    }

    #[inline]
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    fn has_nulls(&self) -> bool {
        self.0.has_nulls()
    }

    #[cfg(feature = "algorithm_group_by")]
    fn unique(&self) -> PolarsResult<Series> {
        if !self.inner_dtype().is_numeric() {
            polars_bail!(opq = unique, self.dtype());
        }
        // this can be called in aggregation, so this fast path can be worth a lot
        if self.len() < 2 {
            return Ok(self.0.clone().into_series());
        }
        let main_thread = POOL.current_thread_index().is_none();
        let groups = self.group_tuples(main_thread, false);
        // SAFETY:
        // groups are in bounds
        Ok(unsafe { self.0.clone().into_series().agg_first(&groups?) })
    }

    #[cfg(feature = "algorithm_group_by")]
    fn n_unique(&self) -> PolarsResult<usize> {
        if !self.inner_dtype().is_numeric() {
            polars_bail!(opq = n_unique, self.dtype());
        }
        // this can be called in aggregation, so this fast path can be worth a lot
        match self.len() {
            0 => Ok(0),
            1 => Ok(1),
            _ => {
                let main_thread = POOL.current_thread_index().is_none();
                let groups = self.group_tuples(main_thread, false)?;
                Ok(groups.len())
            },
        }
    }

    #[cfg(feature = "algorithm_group_by")]
    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        if !self.inner_dtype().is_numeric() {
            polars_bail!(opq = arg_unique, self.dtype());
        }
        // this can be called in aggregation, so this fast path can be worth a lot
        if self.len() == 1 {
            return Ok(IdxCa::new_vec(self.name(), vec![0 as IdxSize]));
        }
        let main_thread = POOL.current_thread_index().is_none();
        // arg_unique requires a stable order
        let groups = self.group_tuples(main_thread, true)?;
        let first = groups.take_group_firsts();
        Ok(IdxCa::from_vec(self.name(), first))
    }

    fn is_null(&self) -> BooleanChunked {
        self.0.is_null()
    }

    fn is_not_null(&self) -> BooleanChunked {
        self.0.is_not_null()
    }

    fn reverse(&self) -> Series {
        ChunkReverse::reverse(&self.0).into_series()
    }

    fn as_single_ptr(&mut self) -> PolarsResult<usize> {
        self.0.as_single_ptr()
    }

    fn shift(&self, periods: i64) -> Series {
        ChunkShift::shift(&self.0, periods).into_series()
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }
    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    /// Get a hold to self as `Any` trait reference.
    /// Only implemented for ObjectType
    fn as_any_mut(&mut self) -> &mut dyn Any {
        &mut self.0
    }
}
