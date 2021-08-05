use crate::chunked_array::object::compare_inner::{IntoPartialEqInner, PartialEqInner};
use crate::chunked_array::object::PolarsObjectSafe;
use crate::chunked_array::ChunkIdIter;
use crate::fmt::FmtList;
use crate::frame::groupby::{GroupTuples, IntoGroupTuples};
use crate::prelude::*;
use crate::series::implementations::SeriesWrap;
use crate::series::private::{PrivateSeries, PrivateSeriesNumeric};
use ahash::RandomState;
use arrow::array::{ArrayData, ArrayRef};
use arrow::buffer::Buffer;
use std::any::Any;
use std::borrow::Cow;

#[cfg(feature = "object")]
impl<T> IntoSeries for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn into_series(self) -> Series {
        Series(Arc::new(SeriesWrap(self)))
    }
}

#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
impl<T> PrivateSeriesNumeric for SeriesWrap<ObjectChunked<T>> {}

#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
impl<T> PrivateSeries for SeriesWrap<ObjectChunked<T>>
where
    T: PolarsObject,
{
    fn str_value(&self, index: usize) -> Cow<str> {
        match (&self.0).get(index) {
            None => Cow::Borrowed("null"),
            Some(val) => Cow::Owned(format!("{}", val)),
        }
    }
    fn into_partial_eq_inner<'a>(&'a self) -> Box<dyn PartialEqInner + 'a> {
        (&self.0).into_partial_eq_inner()
    }

    fn vec_hash(&self, random_state: RandomState) -> AlignedVec<u64> {
        self.0.vec_hash(random_state)
    }

    fn vec_hash_combine(&self, build_hasher: RandomState, hashes: &mut [u64]) {
        self.0.vec_hash_combine(build_hasher, hashes)
    }

    fn group_tuples(&self, multithreaded: bool) -> GroupTuples {
        IntoGroupTuples::group_tuples(&self.0, multithreaded)
    }
}
#[cfg(feature = "object")]
#[cfg_attr(docsrs, doc(cfg(feature = "object")))]
impl<T> SeriesTrait for SeriesWrap<ObjectChunked<T>>
where
    T: PolarsObject,
{
    fn rename(&mut self, name: &str) {
        ObjectChunked::rename(&mut self.0, name)
    }

    fn array_data(&self) -> Vec<&ArrayData> {
        ObjectChunked::array_data(&self.0)
    }

    fn chunk_lengths(&self) -> ChunkIdIter {
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

    fn slice(&self, offset: i64, length: usize) -> Series {
        ObjectChunked::slice(&self.0, offset, length).into_series()
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

    fn take_iter(&self, iter: &mut dyn TakeIterator) -> Result<Series> {
        Ok(ChunkTake::take(&self.0, iter.into())?.into_series())
    }

    unsafe fn take_iter_unchecked(&self, iter: &mut dyn TakeIterator) -> Series {
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

    unsafe fn take_opt_iter_unchecked(&self, iter: &mut dyn TakeIteratorNulls) -> Series {
        ChunkTake::take_unchecked(&self.0, iter.into()).into_series()
    }

    #[cfg(feature = "take_opt_iter")]
    fn take_opt_iter(&self, _iter: &mut dyn TakeIteratorNulls) -> Result<Series> {
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

    fn cast_with_dtype(&self, _data_type: &DataType) -> Result<Series> {
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

    fn arg_unique(&self) -> Result<UInt32Chunked> {
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

    fn null_bits(&self) -> Vec<(usize, Option<&Buffer>)> {
        ObjectChunked::null_bits(&self.0).collect()
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

    fn fmt_list(&self) -> String {
        FmtList::fmt_list(&self.0)
    }

    fn clone_inner(&self) -> Arc<dyn SeriesTrait> {
        Arc::new(SeriesWrap(Clone::clone(&self.0)))
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

    fn get_object(&self, index: usize) -> Option<&dyn PolarsObjectSafe> {
        ObjectChunked::<T>::get_object(&self.0, index)
    }

    fn as_any(&self) -> &dyn Any {
        &self.0
    }

    fn sum_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn max_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn min_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn mean_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn median_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn var_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
    fn std_as_series(&self) -> Series {
        ObjectChunked::<T>::full_null(self.name(), 1).into_series()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_downcast_object() -> Result<()> {
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
