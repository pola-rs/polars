use super::*;
use arrow::array::DictionaryArray;
use polars_arrow::compute::cast::cast;

impl From<&CategoricalChunked> for DictionaryArray<u32> {
    fn from(ca: &CategoricalChunked) -> Self {
        let keys = ca.logical().rechunk();
        let keys = keys.downcast_iter().next().unwrap();
        let map = &**ca.get_rev_map();
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
        let keys = ca.logical().rechunk();
        let keys = keys.downcast_iter().next().unwrap();
        let map = &**ca.get_rev_map();
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
