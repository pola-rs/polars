use arrow::array::DictionaryArray;
use arrow::datatypes::IntegerType;
use polars_arrow::compute::cast::cast;

use super::*;
use crate::using_string_cache;

impl From<&CategoricalChunked> for DictionaryArray<u32> {
    fn from(ca: &CategoricalChunked) -> Self {
        let keys = ca.logical().rechunk();
        let keys = keys.downcast_iter().next().unwrap();
        let map = &**ca.get_rev_map();
        let dtype = ArrowDataType::Dictionary(
            IntegerType::UInt32,
            Box::new(ArrowDataType::LargeUtf8),
            false,
        );
        match map {
            RevMapping::Local(arr) => {
                // Safety:
                // the keys are in bounds
                unsafe {
                    DictionaryArray::try_new_unchecked(dtype, keys.clone(), Box::new(arr.clone()))
                        .unwrap()
                }
            }
            RevMapping::Global(reverse_map, values, _uuid) => {
                let iter = keys
                    .into_iter()
                    .map(|opt_k| opt_k.map(|k| *reverse_map.get(k).unwrap()));
                let keys = PrimitiveArray::from_trusted_len_iter(iter);

                // Safety:
                // the keys are in bounds
                unsafe {
                    DictionaryArray::try_new_unchecked(dtype, keys, Box::new(values.clone()))
                        .unwrap()
                }
            }
        }
    }
}
impl From<&CategoricalChunked> for DictionaryArray<i64> {
    fn from(ca: &CategoricalChunked) -> Self {
        let keys = ca.logical().rechunk();
        let keys = keys.downcast_iter().next().unwrap();
        let map = &**ca.get_rev_map();
        let dtype = ArrowDataType::Dictionary(
            IntegerType::UInt32,
            Box::new(ArrowDataType::LargeUtf8),
            false,
        );
        match map {
            // Safety:
            // the keys are in bounds
            RevMapping::Local(arr) => unsafe {
                DictionaryArray::try_new_unchecked(
                    dtype,
                    cast(keys, &ArrowDataType::Int64)
                        .unwrap()
                        .as_any()
                        .downcast_ref::<PrimitiveArray<i64>>()
                        .unwrap()
                        .clone(),
                    Box::new(arr.clone()),
                )
                .unwrap()
            },
            RevMapping::Global(reverse_map, values, _uuid) => {
                let iter = keys
                    .into_iter()
                    .map(|opt_k| opt_k.map(|k| *reverse_map.get(k).unwrap() as i64));
                let keys = PrimitiveArray::from_trusted_len_iter(iter);

                // Safety:
                // the keys are in bounds
                unsafe {
                    DictionaryArray::try_new_unchecked(dtype, keys, Box::new(values.clone()))
                        .unwrap()
                }
            }
        }
    }
}

impl CategoricalChunked {
    /// # Safety
    /// The caller must ensure that index values in the `keys` are in within bounds of the `values` length.
    pub(crate) unsafe fn from_keys_and_values(
        name: &str,
        keys: &PrimitiveArray<u32>,
        values: &Utf8Array<i64>,
    ) -> Self {
        if using_string_cache() {
            let mut builder = CategoricalChunkedBuilder::new(name, keys.len());
            builder.global_map_from_local(keys, values.clone());
            builder.finish()
        } else {
            CategoricalChunked::from_chunks_original(
                name,
                vec![Box::new(keys.clone())],
                RevMapping::Local(values.clone()),
            )
        }
    }
}
