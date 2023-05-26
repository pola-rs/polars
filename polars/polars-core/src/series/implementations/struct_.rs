use std::any::Any;

use super::*;
use crate::hashing::series_to_hashes;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};

unsafe impl IntoSeries for StructChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl PrivateSeriesNumeric for SeriesWrap<StructChunked> {}

impl private::PrivateSeries for SeriesWrap<StructChunked> {
    fn compute_len(&mut self) {
        for s in self.0.fields_mut() {
            s._get_inner_mut().compute_len();
        }
    }
    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }
    fn _dtype(&self) -> &DataType {
        self.0.ref_field().data_type()
    }
    fn explode_by_offsets(&self, offsets: &[i64]) -> Series {
        self.0
            .apply_fields(|s| s.explode_by_offsets(offsets))
            .into_series()
    }

    unsafe fn equal_element(&self, idx_self: usize, idx_other: usize, other: &Series) -> bool {
        let other = other.struct_().unwrap();
        self.0
            .fields()
            .iter()
            .zip(other.fields())
            .all(|(s, other)| s.equal_element(idx_self, idx_other, other))
    }

    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        let other = other.struct_()?;
        let fields = self
            .0
            .fields()
            .iter()
            .zip(other.fields())
            .map(|(lhs, rhs)| lhs.zip_with_same_type(mask, rhs))
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(StructChunked::new_unchecked(self.0.name(), &fields).into_series())
    }

    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_list(groups)
    }

    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        let df = DataFrame::new_no_checks(vec![]);
        let gb = df
            .groupby_with_series(self.0.fields().to_vec(), multithreaded, sorted)
            .unwrap();
        Ok(gb.take_groups())
    }

    fn vec_hash(&self, random_state: RandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        series_to_hashes(self.0.fields(), Some(random_state), buf)?;
        Ok(())
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) -> PolarsResult<()> {
        for field in self.0.fields() {
            field.vec_hash_combine(build_hasher.clone(), hashes)?;
        }
        Ok(())
    }
}

impl SeriesTrait for SeriesWrap<StructChunked> {
    fn rename(&mut self, name: &str) {
        self.0.rename(name)
    }

    fn has_validity(&self) -> bool {
        self.0.fields().iter().any(|s| s.has_validity())
    }

    /// Name of series.
    fn name(&self) -> &str {
        self.0.name()
    }

    fn chunk_lengths(&self) -> ChunkIdIter {
        let s = self.0.fields().first().unwrap();
        s.chunk_lengths()
    }

    /// Underlying chunks.
    fn chunks(&self) -> &Vec<ArrayRef> {
        self.0.chunks()
    }

    /// Number of chunks in this Series
    fn n_chunks(&self) -> usize {
        let s = self.0.fields().first().unwrap();
        s.n_chunks()
    }

    /// Get a zero copy view of the data.
    ///
    /// When offset is negative the offset is counted from the
    /// end of the array
    fn slice(&self, offset: i64, length: usize) -> Series {
        let mut out = self.0.apply_fields(|s| s.slice(offset, length));
        out.update_chunks(0);
        out.into_series()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        let other = other.struct_()?;
        if self.is_empty() {
            self.0 = other.clone();
            Ok(())
        } else if other.is_empty() {
            Ok(())
        } else {
            let offset = self.chunks().len();
            for (lhs, rhs) in self.0.fields_mut().iter_mut().zip(other.fields()) {
                polars_ensure!(
                    lhs.name() == rhs.name(), SchemaMismatch:
                    "cannot append field with name {:?} to struct with field name {:?}",
                    rhs.name(), lhs.name(),
                );
                lhs.append(rhs)?;
            }
            self.0.update_chunks(offset);
            Ok(())
        }
    }

    fn extend(&mut self, other: &Series) -> PolarsResult<()> {
        let other = other.struct_()?;
        if self.is_empty() {
            self.0 = other.clone();
            Ok(())
        } else if other.is_empty() {
            Ok(())
        } else {
            for (lhs, rhs) in self.0.fields_mut().iter_mut().zip(other.fields()) {
                polars_ensure!(
                    lhs.name() == rhs.name(), SchemaMismatch:
                    "cannot extend field with name {:?} to struct with field name {:?}",
                    rhs.name(), lhs.name(),
                );
                lhs.extend(rhs)?;
            }
            self.0.update_chunks(0);
            Ok(())
        }
    }

    /// Filter by boolean mask. This operation clones data.
    fn filter(&self, _filter: &BooleanChunked) -> PolarsResult<Series> {
        self.0
            .try_apply_fields(|s| s.filter(_filter))
            .map(|ca| ca.into_series())
    }

    /// Take by index from an iterator. This operation clones the data.
    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        self.0
            .try_apply_fields(|s| {
                let mut iter = iter.boxed_clone();
                s.take_iter(&mut *iter)
            })
            .map(|ca| ca.into_series())
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        self.0
            .apply_fields(|s| s._take_chunked_unchecked(by, sorted))
            .into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        self.0
            .apply_fields(|s| s._take_opt_chunked_unchecked(by))
            .into_series()
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// - This doesn't check any bounds.
    /// - Iterator must be TrustedLen
    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        self.0
            .apply_fields(|s| {
                let mut iter = iter.boxed_clone();
                s.take_iter_unchecked(&mut *iter)
            })
            .into_series()
    }

    /// Take by index if ChunkedArray contains a single chunk.
    ///
    /// # Safety
    /// This doesn't check any bounds.
    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        self.0
            .try_apply_fields(|s| s.take_unchecked(idx))
            .map(|ca| ca.into_series())
    }

    /// Take by index from an iterator. This operation clones the data.
    ///
    /// # Safety
    ///
    /// - This doesn't check any bounds.
    /// - Iterator must be TrustedLen
    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        self.0
            .apply_fields(|s| {
                let mut iter = iter.boxed_clone();
                s.take_opt_iter_unchecked(&mut *iter)
            })
            .into_series()
    }

    /// Take by index. This operation is clone.
    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        self.0
            .try_apply_fields(|s| s.take(indices))
            .map(|ca| ca.into_series())
    }

    /// Get length of series.
    fn len(&self) -> usize {
        self.0.len()
    }

    /// Aggregate all chunks to a contiguous array of memory.
    fn rechunk(&self) -> Series {
        let mut out = self.0.clone();
        out.rechunk();
        out.into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        self.0
            .apply_fields(|s| s.new_from_index(index, length))
            .into_series()
    }

    fn cast(&self, dtype: &DataType) -> PolarsResult<Series> {
        self.0.cast(dtype)
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        self.0.get_any_value(index)
    }

    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        self.0.get_any_value_unchecked(index)
    }

    /// Count the null values.
    fn null_count(&self) -> usize {
        self.0.null_count()
    }

    /// Get unique values in the Series.
    fn unique(&self) -> PolarsResult<Series> {
        // this can called in aggregation, so this fast path can be worth a lot
        if self.len() < 2 {
            return Ok(self.0.clone().into_series());
        }
        let main_thread = POOL.current_thread_index().is_none();
        let groups = self.group_tuples(main_thread, false);
        // safety:
        // groups are in bounds
        Ok(unsafe { self.0.clone().into_series().agg_first(&groups?) })
    }

    /// Get unique values in the Series.
    fn n_unique(&self) -> PolarsResult<usize> {
        // this can called in aggregation, so this fast path can be worth a lot
        match self.len() {
            0 => Ok(0),
            1 => Ok(1),
            _ => {
                // TODO! try row encoding
                let main_thread = POOL.current_thread_index().is_none();
                let groups = self.group_tuples(main_thread, false)?;
                Ok(groups.len())
            }
        }
    }

    /// Get first indexes of unique values.
    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        // this can called in aggregation, so this fast path can be worth a lot
        if self.len() == 1 {
            return Ok(IdxCa::new_vec(self.name(), vec![0 as IdxSize]));
        }
        // TODO! try row encoding
        let main_thread = POOL.current_thread_index().is_none();
        let groups = self.group_tuples(main_thread, false)?;
        let first = groups.take_group_firsts();
        Ok(IdxCa::from_vec(self.name(), first))
    }

    /// Get a mask of the null values.
    fn is_null(&self) -> BooleanChunked {
        let is_null = self.0.fields().iter().map(|s| s.is_null());
        is_null.reduce(|lhs, rhs| lhs.bitand(rhs)).unwrap()
    }

    /// Get a mask of the non-null values.
    fn is_not_null(&self) -> BooleanChunked {
        let is_not_null = self.0.fields().iter().map(|s| s.is_not_null());
        is_not_null.reduce(|lhs, rhs| lhs.bitand(rhs)).unwrap()
    }

    fn shrink_to_fit(&mut self) {
        self.0.fields_mut().iter_mut().for_each(|s| {
            s.shrink_to_fit();
        });
    }

    fn reverse(&self) -> Series {
        self.0.apply_fields(|s| s.reverse()).into_series()
    }

    fn shift(&self, periods: i64) -> Series {
        self.0.apply_fields(|s| s.shift(periods)).into_series()
    }

    #[cfg(feature = "is_in")]
    fn is_in(&self, other: &Series) -> PolarsResult<BooleanChunked> {
        self.0.is_in(other)
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn sort_with(&self, options: SortOptions) -> Series {
        let df = self.0.clone().unnest();

        let desc = if options.descending {
            vec![true; df.width()]
        } else {
            vec![false; df.width()]
        };
        let out = df
            .sort_impl(
                df.columns.clone(),
                desc,
                options.nulls_last,
                None,
                options.multithreaded,
            )
            .unwrap();
        StructChunked::new_unchecked(self.name(), &out.columns).into_series()
    }

    fn arg_sort(&self, options: SortOptions) -> IdxCa {
        self.0.arg_sort(options)
    }
}
