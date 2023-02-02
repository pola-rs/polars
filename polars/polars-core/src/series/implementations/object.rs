use std::any::Any;
use std::borrow::Cow;

use ahash::RandomState;

use crate::chunked_array::object::compare_inner::{IntoPartialEqInner, PartialEqInner};
use crate::chunked_array::object::PolarsObjectSafe;
use crate::fmt::FmtList;
use crate::frame::groupby::{GroupsProxy, IntoGroupsProxy};
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};
use crate::series::IsSorted;

impl<T: PolarsObject> PrivateSeriesNumeric for SeriesWrap<ObjectChunked<T>> {}

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

    fn _field(&self) -> Cow<Field> {
        Cow::Borrowed(self.0.ref_field())
    }

    fn _dtype(&self) -> &DataType {
        self.0.dtype()
    }

    unsafe fn agg_list(&self, groups: &GroupsProxy) -> Series {
        self.0.agg_list(groups)
    }

    fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
        (&self.0).into_partial_eq_inner()
    }

    fn vec_hash(&self, random_state: RandomState, buf: &mut Vec<u64>) -> PolarsResult<()> {
        self.0.vec_hash(random_state, buf);
        Ok(())
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) -> PolarsResult<()> {
        self.0.vec_hash_combine(build_hasher, hashes);
        Ok(())
    }

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

    fn chunk_lengths(&self) -> ChunkIdIter {
        ObjectChunked::chunk_id(&self.0)
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

    fn slice(&self, offset: i64, length: usize) -> Series {
        ObjectChunked::slice(&self.0, offset, length).into_series()
    }

    fn append(&mut self, other: &Series) -> PolarsResult<()> {
        if self.dtype() == other.dtype() {
            ObjectChunked::append(&mut self.0, other.as_ref().as_ref());
            Ok(())
        } else {
            Err(PolarsError::SchemaMisMatch(
                "cannot append Series; data types don't match".into(),
            ))
        }
    }

    fn extend(&mut self, _other: &Series) -> PolarsResult<()> {
        panic!("extend not implemented for Object dtypes")
    }

    fn filter(&self, filter: &BooleanChunked) -> PolarsResult<Series> {
        ChunkFilter::filter(&self.0, filter).map(|ca| ca.into_series())
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_chunked_unchecked(&self, by: &[ChunkId], sorted: IsSorted) -> Series {
        self.0.take_chunked_unchecked(by, sorted).into_series()
    }

    #[cfg(feature = "chunked_ids")]
    unsafe fn _take_opt_chunked_unchecked(&self, by: &[Option<ChunkId>]) -> Series {
        self.0.take_opt_chunked_unchecked(by).into_series()
    }

    fn take(&self, indices: &IdxCa) -> PolarsResult<Series> {
        Ok(ChunkTake::take(&self.0, indices.into())?.into_series())
    }

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> PolarsResult<Series> {
        Ok(ChunkTake::take(&self.0, iter.into())?.into_series())
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
        ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
    }

    unsafe fn take_unchecked(&self, idx: &IdxCa) -> PolarsResult<Series> {
        let idx = if idx.chunks.len() > 1 {
            Cow::Owned(idx.rechunk())
        } else {
            Cow::Borrowed(idx)
        };
        Ok(ChunkTake::take_unchecked(&self.0, (&*idx).into()).into_series())
    }

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
    }

    #[cfg(feature = "take_opt_iter")]
    fn take_opt_iter(&self, _iter: &mut dyn TakeIteratorNulls) -> PolarsResult<Series> {
        todo!()
    }

    fn len(&self) -> usize {
        ObjectChunked::len(&self.0)
    }

    fn rechunk(&self) -> Series {
        // do not call normal rechunk
        self.rechunk_object().into_series()
    }

    fn take_every(&self, n: usize) -> Series {
        self.0.take_every(n).into_series()
    }

    fn new_from_index(&self, index: usize, length: usize) -> Series {
        ChunkExpandAtIndex::new_from_index(&self.0, index, length).into_series()
    }

    fn cast(&self, _data_type: &DataType) -> PolarsResult<Series> {
        Err(PolarsError::InvalidOperation(
            "cannot cast array of type ObjectChunked to arrow datatype".into(),
        ))
    }

    fn get(&self, index: usize) -> PolarsResult<AnyValue> {
        ObjectChunked::get_any_value(&self.0, index)
    }
    fn null_count(&self) -> usize {
        ObjectChunked::null_count(&self.0)
    }

    fn has_validity(&self) -> bool {
        ObjectChunked::has_validity(&self.0)
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

    fn is_unique(&self) -> PolarsResult<BooleanChunked> {
        ChunkUnique::is_unique(&self.0)
    }

    fn is_duplicated(&self) -> PolarsResult<BooleanChunked> {
        ChunkUnique::is_duplicated(&self.0)
    }

    fn reverse(&self) -> Series {
        ChunkReverse::reverse(&self.0).into_series()
    }

    fn shift(&self, periods: i64) -> Series {
        ChunkShift::shift(&self.0, periods).into_series()
    }

    fn fmt_list(&self) -> String {
        FmtList::fmt_list(&self.0)
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
    }

    fn get_object(&self, index: usize) -> Option<&dyn PolarsObjectSafe> {
        ObjectChunked::<T>::get_object(&self.0, index)
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn _sum_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn max_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn min_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn median_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn var_as_series(&self, _ddof: u8) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn std_as_series(&self, _ddof: u8) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_downcast_object() -> PolarsResult<()> {
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
