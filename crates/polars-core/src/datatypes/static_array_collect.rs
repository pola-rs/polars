use arrow::array::ArrayFromIter;
use arrow::bitmap::BitmapBuilder;
use arrow::trusted_len::TrustedLen;

use crate::chunked_array::object::{ObjectArray, PolarsObject};

impl<'a, T: PolarsObject> ArrayFromIter<&'a T> for ObjectArray<T> {
    fn arr_from_iter<I: IntoIterator<Item = &'a T>>(iter: I) -> Self {
        let values: Vec<T> = iter.into_iter().cloned().collect();
        ObjectArray::from(values)
    }

    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = &'a T>,
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let size = iter
            .size_hint()
            .1
            .expect("TrustedLen must have upper bound");
        let mut values: Vec<T> = Vec::with_capacity(size);
        values.extend(iter.cloned());
        ObjectArray::from(values)
    }

    fn try_arr_from_iter<E, I: IntoIterator<Item = Result<&'a T, E>>>(iter: I) -> Result<Self, E> {
        let values: Vec<T> = iter
            .into_iter()
            .map(|r| r.cloned())
            .collect::<Result<_, E>>()?;
        Ok(ObjectArray::from(values))
    }

    fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<&'a T, E>>,
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let size = iter
            .size_hint()
            .1
            .expect("TrustedLen must have upper bound");
        let mut values: Vec<T> = Vec::with_capacity(size);
        for r in iter {
            values.push(r?.clone());
        }
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
            match val {
                Some(value) => {
                    null_mask_builder.push(true);
                    values.push(value.clone());
                },
                None => {
                    null_mask_builder.push(false);
                    values.push(T::default());
                },
            }
        }

        ObjectArray::from(values).with_validity(null_mask_builder.into_opt_validity())
    }

    fn arr_from_iter_trusted<I>(iter: I) -> Self
    where
        I: IntoIterator<Item = Option<&'a T>>,
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let size = iter
            .size_hint()
            .1
            .expect("TrustedLen must have upper bound");

        let mut null_mask_builder = BitmapBuilder::with_capacity(size);
        let mut values: Vec<T> = Vec::with_capacity(size);

        for val in iter {
            match val {
                Some(value) => {
                    null_mask_builder.push(true);
                    values.push(value.clone());
                },
                None => {
                    null_mask_builder.push(false);
                    values.push(T::default());
                },
            }
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
            match val? {
                Some(value) => {
                    null_mask_builder.push(true);
                    values.push(value.clone());
                },
                None => {
                    null_mask_builder.push(false);
                    values.push(T::default());
                },
            }
        }

        Ok(ObjectArray::from(values).with_validity(null_mask_builder.into_opt_validity()))
    }

    fn try_arr_from_iter_trusted<E, I>(iter: I) -> Result<Self, E>
    where
        I: IntoIterator<Item = Result<Option<&'a T>, E>>,
        I::IntoIter: TrustedLen,
    {
        let iter = iter.into_iter();
        let size = iter
            .size_hint()
            .1
            .expect("TrustedLen must have upper bound");

        let mut null_mask_builder = BitmapBuilder::with_capacity(size);
        let mut values: Vec<T> = Vec::with_capacity(size);

        for val in iter {
            match val? {
                Some(value) => {
                    null_mask_builder.push(true);
                    values.push(value.clone());
                },
                None => {
                    null_mask_builder.push(false);
                    values.push(T::default());
                },
            }
        }

        Ok(ObjectArray::from(values).with_validity(null_mask_builder.into_opt_validity()))
    }
}
