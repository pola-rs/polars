use arrow::array::DictionaryArray;
use arrow::compute::cast::{cast, utf8view_to_utf8, CastOptions};
use arrow::datatypes::IntegerType;

use super::*;

fn convert_values(arr: &Utf8ViewArray, pl_flavor: bool) -> ArrayRef {
    if pl_flavor {
        arr.clone().boxed()
    } else {
        utf8view_to_utf8::<i64>(arr).boxed()
    }
}

impl CategoricalChunked {
    pub fn to_arrow(&self, pl_flavor: bool, as_i64: bool) -> ArrayRef {
        if as_i64 {
            self.to_i64(pl_flavor).boxed()
        } else {
            self.to_u32(pl_flavor).boxed()
        }
    }

    fn to_u32(&self, pl_flavor: bool) -> DictionaryArray<u32> {
        let values_dtype = if pl_flavor {
            ArrowDataType::Utf8View
        } else {
            ArrowDataType::LargeUtf8
        };
        let keys = self.physical().rechunk();
        let keys = keys.downcast_iter().next().unwrap();
        let map = &**self.get_rev_map();
        let dtype = ArrowDataType::Dictionary(IntegerType::UInt32, Box::new(values_dtype), false);
        match map {
            RevMapping::Local(arr, _) => {
                let values = convert_values(arr, pl_flavor);

                // SAFETY:
                // the keys are in bounds
                unsafe { DictionaryArray::try_new_unchecked(dtype, keys.clone(), values).unwrap() }
            },
            RevMapping::Global(reverse_map, values, _uuid) => {
                let iter = keys
                    .into_iter()
                    .map(|opt_k| opt_k.map(|k| *reverse_map.get(k).unwrap()));
                let keys = PrimitiveArray::from_trusted_len_iter(iter);

                let values = convert_values(values, pl_flavor);

                // SAFETY:
                // the keys are in bounds
                unsafe { DictionaryArray::try_new_unchecked(dtype, keys, values).unwrap() }
            },
        }
    }

    fn to_i64(&self, pl_flavor: bool) -> DictionaryArray<i64> {
        let values_dtype = if pl_flavor {
            ArrowDataType::Utf8View
        } else {
            ArrowDataType::LargeUtf8
        };
        let keys = self.physical().rechunk();
        let keys = keys.downcast_iter().next().unwrap();
        let map = &**self.get_rev_map();
        let dtype = ArrowDataType::Dictionary(IntegerType::Int64, Box::new(values_dtype), false);
        match map {
            RevMapping::Local(arr, _) => {
                let values = convert_values(arr, pl_flavor);

                // SAFETY:
                // the keys are in bounds
                unsafe {
                    DictionaryArray::try_new_unchecked(
                        dtype,
                        cast(keys, &ArrowDataType::Int64, CastOptions::unchecked())
                            .unwrap()
                            .as_any()
                            .downcast_ref::<PrimitiveArray<i64>>()
                            .unwrap()
                            .clone(),
                        values,
                    )
                    .unwrap()
                }
            },
            RevMapping::Global(reverse_map, values, _uuid) => {
                let iter = keys
                    .into_iter()
                    .map(|opt_k| opt_k.map(|k| *reverse_map.get(k).unwrap() as i64));
                let keys = PrimitiveArray::from_trusted_len_iter(iter);

                let values = convert_values(values, pl_flavor);

                // SAFETY:
                // the keys are in bounds
                unsafe { DictionaryArray::try_new_unchecked(dtype, keys, values).unwrap() }
            },
        }
    }
}
