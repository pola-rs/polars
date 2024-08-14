use std::any::Any;
use std::borrow::Cow;

use super::{BitRepr, MetadataFlags};
use crate::chunked_array::cast::CastOptions;
use crate::chunked_array::object::PolarsObjectSafe;
use crate::chunked_array::ops::compare_inner::{IntoTotalEqInner, TotalEqInner};
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};

impl<T: PolarsObject> PrivateSeriesNumeric for SeriesWrap<ObjectChunked<T>> {
    fn bit_repr(&self) -> Option<BitRepr> {
        None
    }
}

impl<T> PrivateSeries for SeriesWrap<ObjectChunked<T>>
where
    T: PolarsObject,
{
    fn get_list_builder(
        &self,
        _name: &str,
        _values_capacity: usize,
        _list_capacity: usize,
    ) -> Box<dyn ListBuilderTrait> {
        ObjectChunked::<T>::get_list_builder(_name, _values_capacity, _list_capacity)
    }

    fn compute_len(&mut self) {
        self.0.compute_len()
    }

    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    fn _set_flags(&mut self, flags: MetadataFlags) {
        self.0.set_flags(flags)
    }
    fn _get_flags(&self) -> MetadataFlags {
        self.0.get_flags()
    }
    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_list(groups)
    }

    fn into_total_eq_inner<'a>(&'a self) -> Box<dyn TotalEqInner + 'a> {
        (&self.0).into_total_eq_inner()
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
    fn group_tuples(&self, multithreaded: bool, sorted: bool) -> PolarsResult<GroupsProxy> {
        IntoGroupsProxy::group_tuples(&self.0, multithreaded, sorted)
    }
    #[cfg(feature = "zip_with")]
    fn zip_with_same_type(&self, mask: &BooleanChunked, other: &Series) -> PolarsResult<Series> {
        self.0
            .zip_with(mask, other.as_ref().as_ref())
            .map(|ca| ca.into_series())
    }
}
impl<T> SeriesTrait for SeriesWrap<ObjectChunked<T>>
where
    T: PolarsObject,
{
    fn rename(&mut self, name: &str) {
        ObjectChunked::rename(&mut self.0, name)
    }

    fn chunk_lengths(&self) -> ChunkLenIter {
        ObjectChunked::chunk_lengths(&self.0)
    }

    fn name(&self) -> &str {
        ObjectChunked::name(&self.0)
    }

    fn dtype(&self) -> &DataType {
        ObjectChunked::dtype(&self.0)
    }

    fn chunks(&self) -> &Vec<ArrayRef> {
        ObjectChunked::chunks(&self.0)
    }
    unsafe fn chunks_mut(&mut self) -> &mut Vec<ArrayRef> {
        self.0.chunks_mut()
    }

    fn slice(&self, offset: i64, length: usize) -> Series {
        ObjectChunked::slice(&self.0, offset, length).into_series()
    }

    fn split_at(&self, offset: i64) -> (Series, Series) {
        let (a, b) = ObjectChunked::split_at(&self.0, offset);
        (a.into_series(), b.into_series())
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        if self.dtype() != other.dtype() {
            polars_bail!(append);
        }
        ObjectChunked::append(&mut self.0, other.as_ref().as_ref())?;
        Ok(())
    }

    fn extend(&mut self, _other: &Series) -> PolarsResult<()> {
        polars_bail!(opq = extend, self.dtype());
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
        ObjectChunked::len(&self.0)
    }

    fn rechunk(&self) -> Series {
        // do not call normal rechunk
        self.rechunk_object().into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        ChunkExpandAtIndex::new_from_index(&self.0, index, length).into_series()
    }

    fn cast(&self, data_type: &DataType, _cast_options: CastOptions) -> PolarsResult<Series> {
        if matches!(data_type, DataType::Object(_, None)) {
            Ok(self.0.clone().into_series())
        } else {
            Err(PolarsError::ComputeError(
                "cannot cast 'Object' type".into(),
            ))
        }
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        ObjectChunked::get_any_value(&self.0, index)
    }
    unsafe fn get_unchecked(&self, index: usize) -> AnyValue {
        ObjectChunked::get_any_value_unchecked(&self.0, index)
    }
    fn null_count(&self) -> usize {
        ObjectChunked::null_count(&self.0)
    }

    fn has_nulls(&self) -> bool {
        ObjectChunked::has_nulls(&self.0)
    }

    fn unique(&self) -> PolarsResult<Series> {
        ChunkUnique::unique(&self.0).map(|ca| ca.into_series())
    }

    fn n_unique(&self) -> PolarsResult<usize> {
        ChunkUnique::n_unique(&self.0)
    }

    fn arg_unique(&self) -> PolarsResult<IdxCa> {
        ChunkUnique::arg_unique(&self.0)
    }

    fn is_null(&self) -> BooleanChunked {
        ObjectChunked::is_null(&self.0)
    }

    fn is_not_null(&self) -> BooleanChunked {
        ObjectChunked::is_not_null(&self.0)
    }

    fn reverse(&self) -> Series {
        ChunkReverse::reverse(&self.0).into_series()
    }

    fn shift(&self, periods: i64) -> Series {
        ChunkShift::shift(&self.0, periods).into_series()
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn get_object(&self, index: usize) -> Option<&dyn PolarsObjectSafe> {
        ObjectChunked::<T>::get_object(&self.0, index)
    }

    unsafe fn get_object_chunked_unchecked(
        &self,
        chunk: usize,
        index: usize,
    ) -> Option<&dyn PolarsObjectSafe> {
        ObjectChunked::<T>::get_object_chunked_unchecked(&self.0, chunk, index)
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_downcast_object() -> PolarsResult<()> {
        #[allow(non_local_definitions)]
        impl PolarsObject for i32 {
            fn type_name() -> &'static str {
                "i32"
            }
        }

        let ca = ObjectChunked::new_from_vec("a", vec![0i32, 1, 2]);
        let s = ca.into_series();

        let ca = s.as_any().downcast_ref::<ObjectChunked<i32>>().unwrap();
        assert_eq!(*ca.get(0).unwrap(), 0);
        assert_eq!(*ca.get(1).unwrap(), 1);
        assert_eq!(*ca.get(2).unwrap(), 2);

        Ok(())
    }
}
