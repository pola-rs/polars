use crate::prelude::*;
use arrow::array::DictionaryArray;
use polars_arrow::compute::cast::cast;
mod builder;
mod merge;

pub use builder::*;
use std::ops::{Deref, DerefMut};

impl From<&CategoricalChunked> for DictionaryArray<u32> {
    fn from(ca: &CategoricalChunked) -> Self {
        let ca = ca.rechunk();
        let keys = ca.downcast_iter().next().unwrap();
        let map = &**ca.categorical_map.as_ref().unwrap();
        match map {
            RevMapping::Local(arr) => {
                DictionaryArray::from_data(keys.clone(), Arc::new(arr.clone()))
            }
            RevMapping::Global(reverse_map, values, _uuid) => {
                let iter = keys
                    .into_iter()
                    .map(|opt_k| opt_k.map(|k| *reverse_map.get(k).unwrap()));
                let keys = PrimitiveArray::from_trusted_len_iter(iter);

                DictionaryArray::from_data(keys, Arc::new(values.clone()))
            }
        }
    }
}
impl From<&CategoricalChunked> for DictionaryArray<i64> {
    fn from(ca: &CategoricalChunked) -> Self {
        let ca = ca.rechunk();
        let keys = ca.downcast_iter().next().unwrap();
        let map = &**ca.categorical_map.as_ref().unwrap();
        match map {
            RevMapping::Local(arr) => DictionaryArray::from_data(
                cast(keys, &ArrowDataType::Int64)
                    .unwrap()
                    .as_any()
                    .downcast_ref::<PrimitiveArray<i64>>()
                    .unwrap()
                    .clone(),
                Arc::new(arr.clone()),
            ),
            RevMapping::Global(reverse_map, values, _uuid) => {
                let iter = keys
                    .into_iter()
                    .map(|opt_k| opt_k.map(|k| *reverse_map.get(k).unwrap() as i64));
                let keys = PrimitiveArray::from_trusted_len_iter(iter);

                DictionaryArray::from_data(keys, Arc::new(values.clone()))
            }
        }
    }
}

impl CategoricalChunked {
    /// Get a reference to the mapping of categorical types to the string values.
    pub fn get_categorical_map(&self) -> Option<&Arc<RevMapping>> {
        self.categorical_map.as_ref()
    }

    pub(crate) fn set_categorical_map(&mut self, categorical_map: Arc<RevMapping>) {
        self.categorical_map = Some(categorical_map)
    }

    pub(crate) fn set_state<T>(mut self, other: &ChunkedArray<T>) -> Self {
        self.categorical_map = other.categorical_map.clone();
        self
    }

    pub(crate) fn can_fast_unique(&self) -> bool {
        self.bit_settings & 1 << 3 != 0 && self.chunks.len() == 1
    }

    pub(crate) fn set_fast_unique(&mut self, can: bool) {
        if can {
            self.bit_settings |= 1u8 << 3;
        } else {
            self.bit_settings &= !(1u8 << 3);
        }
    }

    /// Create an `[Iterator]` that iterates over the `&str` values of the `[CategoricalChunked]`.
    pub fn iter_str(&self) -> CatIter<'_> {
        let iter = self.deref().into_iter();
        CatIter {
            rev: self.categorical_map.as_ref().unwrap(),
            iter,
        }
    }
}

pub struct CatIter<'a> {
    rev: &'a RevMapping,
    iter: Box<dyn PolarsIterator<Item = Option<u32>> + 'a>,
}

impl<'a> Iterator for CatIter<'a> {
    type Item = Option<&'a str>;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next().map(|item| {
            item.map(|idx| {
                // Safety:
                // all categories are in bound
                unsafe { self.rev.get_unchecked(idx) }
            })
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

impl<'a> ExactSizeIterator for CatIter<'a> {}

impl Deref for CategoricalChunked {
    type Target = UInt32Chunked;

    fn deref(&self) -> &Self::Target {
        // TODO: update the Field, dtype still points to Categorical
        let ptr = self as *const CategoricalChunked;
        let ptr = ptr as *const UInt32Chunked;
        unsafe { &*ptr }
    }
}

impl DerefMut for CategoricalChunked {
    fn deref_mut(&mut self) -> &mut Self::Target {
        let ptr = self as *mut CategoricalChunked;
        let ptr = ptr as *mut UInt32Chunked;
        unsafe { &mut *ptr }
    }
}

impl From<UInt32Chunked> for CategoricalChunked {
    fn from(ca: UInt32Chunked) -> Self {
        ca.cast(&DataType::Categorical)
            .unwrap()
            .categorical()
            .unwrap()
            .clone()
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{reset_string_cache, toggle_string_cache, SINGLE_LOCK};
    use std::convert::TryFrom;

    #[test]
    fn test_categorical_round_trip() -> Result<()> {
        let _lock = SINGLE_LOCK.lock();
        reset_string_cache();
        let slice = &[
            Some("foo"),
            None,
            Some("bar"),
            Some("foo"),
            Some("foo"),
            Some("bar"),
        ];
        let ca = Utf8Chunked::new("a", slice);
        let ca = ca.cast(&DataType::Categorical)?;
        let ca = ca.categorical().unwrap();

        let arr: DictionaryArray<u32> = (ca).into();
        let s = Series::try_from(("foo", Arc::new(arr) as ArrayRef))?;
        assert_eq!(s.dtype(), &DataType::Categorical);
        assert_eq!(s.null_count(), 1);
        assert_eq!(s.len(), 6);

        Ok(())
    }

    #[test]
    fn test_append_categorical() {
        let _lock = SINGLE_LOCK.lock();
        reset_string_cache();
        toggle_string_cache(true);

        let mut s1 = Series::new("1", vec!["a", "b", "c"])
            .cast(&DataType::Categorical)
            .unwrap();
        let s2 = Series::new("2", vec!["a", "x", "y"])
            .cast(&DataType::Categorical)
            .unwrap();
        let appended = s1.append(&s2).unwrap();
        assert_eq!(appended.str_value(0), "\"a\"");
        assert_eq!(appended.str_value(1), "\"b\"");
        assert_eq!(appended.str_value(4), "\"x\"");
        assert_eq!(appended.str_value(5), "\"y\"");
    }

    #[test]
    fn test_fast_unique() {
        let s = Series::new("1", vec!["a", "b", "c"])
            .cast(&DataType::Categorical)
            .unwrap();

        assert_eq!(s.n_unique().unwrap(), 3);
        // make sure that it does not take the fast path after take/ slice
        let out = s.take(&([1, 2].as_ref()).into()).unwrap();
        assert_eq!(out.n_unique().unwrap(), 2);
        let out = s.slice(1, 2);
        assert_eq!(out.n_unique().unwrap(), 2);
    }
}
