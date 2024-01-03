use arrow::array::DictionaryArray;
use arrow::datatypes::IntegerType;
use arrow::legacy::compute::cast::cast;

use super::*;

impl From<&CategoricalChunked> for DictionaryArray<u32> {
    fn from(ca: &CategoricalChunked) -> Self {
        let keys = ca.physical().rechunk();
        let keys = keys.downcast_iter().next().unwrap();
        let map = &**ca.get_rev_map();
        let dtype = ArrowDataType::Dictionary(
            IntegerType::UInt32,
            Box::new(ArrowDataType::LargeUtf8),
            false,
        );
        match map {
            RevMapping::Enum(arr, _) => {
                let dtype =
                    ArrowDataType::Extension("POLARS_ENUM_TYPE".to_string(), Box::new(dtype), None);
                // Safety:
                // the keys are in bounds
                unsafe {
                    DictionaryArray::try_new_unchecked(dtype, keys.clone(), Box::new(arr.clone()))
                        .unwrap()
                }
            },
            RevMapping::Local(arr, _) => {
                // Safety:
                // the keys are in bounds
                unsafe {
                    DictionaryArray::try_new_unchecked(dtype, keys.clone(), Box::new(arr.clone()))
                        .unwrap()
                }
            },
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
            },
        }
    }
}
impl From<&CategoricalChunked> for DictionaryArray<i64> {
    fn from(ca: &CategoricalChunked) -> Self {
        let keys = ca.physical().rechunk();
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
            RevMapping::Local(arr, _) | RevMapping::Enum(arr, _) => unsafe {
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
            },
        }
    }
}
