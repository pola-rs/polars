use arrow::array::ArrayFromIter;
use arrow::bitmap::BitmapBuilder;

use crate::chunked_array::object::{ObjectArray, PolarsObject};

impl<'a, T: PolarsObject> ArrayFromIter<&'a T> for ObjectArray<T> {
    fn arr_from_iter<I: IntoIterator<Item = &'a T>>(iter: I) -> Self {
        ObjectArray::from(Vec::from_iter(iter.into_iter().cloned()))
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<&'a T, E>>>(iter: I) -> Result<Self, E> {
        let values: Vec<T> = iter
            .into_iter()
            .map(|r| r.cloned())
            .collect::<Result<_, E>>()?;
        Ok(ObjectArray::from(values))
    }
}

impl<'a, T: PolarsObject> ArrayFromIter<Option<&'a T>> for ObjectArray<T> {
    fn arr_from_iter<I: IntoIterator<Item = Option<&'a T>>>(iter: I) -> Self {
        let iter = iter.into_iter();
        let size = iter.size_hint().0;

        let mut null_mask_builder = BitmapBuilder::with_capacity(size);
        let mut values: Vec<T> = Vec::with_capacity(size);

        for val in iter {
            null_mask_builder.push(val.is_some());
            values.push(val.cloned().unwrap_or_default());
        }

        ObjectArray::from(values).with_validity(null_mask_builder.into_opt_validity())
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<Option<&'a T>, E>>>(
        iter: I,
    ) -> Result<Self, E> {
        let iter = iter.into_iter();
        let size = iter.size_hint().0;

        let mut null_mask_builder = BitmapBuilder::with_capacity(size);
        let mut values: Vec<T> = Vec::with_capacity(size);

        for val in iter {
            let val = val?;
            null_mask_builder.push(val.is_some());
            values.push(val.cloned().unwrap_or_default());
        }

        Ok(ObjectArray::from(values).with_validity(null_mask_builder.into_opt_validity()))
    }
}
