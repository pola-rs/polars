use crate::chunked_array::RevMapping;
use crate::prelude::*;
use arrow::array::DictionaryArray;
use arrow::compute::cast::cast;

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

#[cfg(test)]
mod test {
    use super::*;
    use crate::{reset_string_cache, SINGLE_LOCK};
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
        let ca = Utf8Chunked::new_from_opt_slice("a", slice);
        let ca = ca.cast::<CategoricalType>()?;

        let arr: DictionaryArray<u32> = (&ca).into();
        let s = Series::try_from(("foo", Arc::new(arr) as ArrayRef))?;
        assert_eq!(s.dtype(), &DataType::Categorical);
        assert_eq!(s.null_count(), 1);
        assert_eq!(s.len(), 6);

        Ok(())
    }
}
