use super::*;
use crate::prelude::*;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};

impl IntoSeries for StructChunked {
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

impl PrivateSeriesNumeric for SeriesWrap<StructChunked> {}

impl private::PrivateSeries for SeriesWrap<StructChunked> {
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
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> Result<Series> {
        let other = other.struct_()?;
        let fields = self
            .0
            .fields()
            .iter()
            .zip(other.fields())
            .map(|(lhs, rhs)| lhs.zip_with_same_type(mask, rhs))
            .collect::<Result<Vec<_>>>()?;
        Ok(StructChunked::new_unchecked(self.0.name(), &fields).into_series())
    }

    fn agg_list(&self, groups: &GroupsProxy) -> Option<Series> {
        let fields = self
            .0
            .fields()
            .iter()
            .map(|s| s.agg_list(groups))
            .collect::<Option<Vec<_>>>()?;
        Some(StructChunked::new_unchecked(self.name(), &fields).into_series())
    }

    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> GroupsProxy {
        let df = DataFrame::new_no_checks(vec![]);
        let gb = df
            .groupby_with_series(self.0.fields().to_vec(), multithreaded, sorted)
            .unwrap();
        gb.groups
    }
}

impl SeriesTrait for SeriesWrap<StructChunked> {
    #[cfg(feature = "interpolate")]
    fn interpolate(&self) -> Series {
        self.0.apply_fields(|s| s.interpolate()).into_series()
    }

    fn rename(&mut self, name: &str) {
        self.0.rename(name)
    }

    fn take_every(&self, n: usize) -> Series {
        self.0.apply_fields(|s| s.take_every(n)).into_series()
    }

    fn has_validity(&self) -> bool {
        self.0.fields().iter().any(|s| s.has_validity())
    }

    /// Name of series.
    fn name(&self) -> &str {
        self.0.name()
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
        self.0
            .apply_fields(|s| s.slice(offset, length))
            .into_series()
    }

    #[doc(hidden)]
    fn append(&mut self, other: &Series) -> Result<()> {
        let other = other.struct_()?;

        for (lhs, rhs) in self.0.fields_mut().iter_mut().zip(other.fields()) {
            lhs.append(rhs)?;
        }
        Ok(())
    }

    #[doc(hidden)]
    fn extend(&mut self, other: &Series) -> Result<()> {
        let other = other.struct_()?;

        for (lhs, rhs) in self.0.fields_mut().iter_mut().zip(other.fields()) {
            lhs.extend(rhs)?;
        }
        Ok(())
    }

    /// Filter by boolean mask. This operation clones data.
    fn filter(&self, _filter: &BooleanChunked) -> Result<Series> {
        self.0
            .try_apply_fields(|s| s.filter(_filter))
            .map(|ca| ca.into_series())
    }

    /// Take by index from an iterator. This operation clones the data.
    fn take_iter(&self, iter: &mut dyn TakeIterator) -> Result<Series> {
        self.0
            .try_apply_fields(|s| {
                let mut iter = iter.boxed_clone();
                s.take_iter(&mut *iter)
            })
            .map(|ca| ca.into_series())
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
    unsafe fn take_unchecked(&self, idx: &IdxCa) -> Result<Series> {
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
    fn take(&self, indices: &IdxCa) -> Result<Series> {
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
        self.0.apply_fields(|s| s.rechunk()).into_series()
    }

    fn expand_at_index(&self, index: usize, length: usize) -> Series {
        self.0
            .apply_fields(|s| s.expand_at_index(index, length))
            .into_series()
    }

    fn cast(&self, dtype: &DataType) -> Result<Series> {
        self.0.cast(dtype)
    }

    fn get(&self, index: usize) -> AnyValue {
        self.0.get_any_value(index)
    }

    /// Count the null values.
    fn null_count(&self) -> usize {
        if self
            .0
            .fields()
            .iter()
            .map(|s| s.null_count())
            .sum::<usize>()
            > 0
        {
            let mut null_count = 0;

            let chunks_lens = self.0.fields()[0].chunks().len();

            for i in 0..chunks_lens {
                // If all fields are null we count it as null
                // so we bitand every chunk
                let mut validity_agg = None;

                for s in self.0.fields() {
                    let arr = &s.chunks()[i];

                    match (&validity_agg, arr.validity()) {
                        (Some(agg), Some(validity)) => validity_agg = Some(validity.bitand(agg)),
                        (None, Some(validity)) => validity_agg = Some(validity.clone()),
                        _ => {}
                    }
                    if let Some(validity) = &validity_agg {
                        null_count += validity.null_count()
                    }
                }
            }

            null_count
        } else {
            0
        }
    }

    /// Get unique values in the Series.
    fn unique(&self) -> Result<Series> {
        let groups = self.group_tuples(true, false);
        Ok(self.0.clone().into_series().agg_first(&groups))
    }

    /// Get unique values in the Series.
    fn n_unique(&self) -> Result<usize> {
        let groups = self.group_tuples(true, false);
        Ok(groups.len())
    }

    /// Get first indexes of unique values.
    fn arg_unique(&self) -> Result<IdxCa> {
        let groups = self.group_tuples(true, false);
        let first = std::mem::take(groups.into_idx().first_mut());
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

    fn shift(&self, periods: i64) -> Series {
        self.0.apply_fields(|s| s.shift(periods)).into_series()
    }

    fn fill_null(&self, strategy: FillNullStrategy) -> Result<Series> {
        self.0
            .try_apply_fields(|s| s.fill_null(strategy))
            .map(|ca| ca.into_series())
    }

    fn fmt_list(&self) -> String {
        self.0.fmt_list()
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }
}
