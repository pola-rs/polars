use simd_json::value::BorrowedValue;

use super::*;

pub(crate) fn json_values_to_supertype(
    values: &[BorrowedValue],
    infer_schema_len: usize,
) -> PolarsResult<DataType> {
    // struct types may have missing fields so find supertype
    values
        .iter()
        .take(infer_schema_len)
        .map(|value| polars_json::json::infer(value).map(|dt| DataType::from(&dt)))
        .reduce(|l, r| {
            let l = l?;
            let r = r?;
            try_get_supertype(&l, &r)
        })
        .unwrap()
}

pub(crate) fn data_types_to_supertype<I: Iterator<Item = DataType>>(
    datatypes: I,
) -> PolarsResult<DataType> {
    datatypes
        .map(Ok)
        .reduce(|l, r| {
            let l = l?;
            let r = r?;
            try_get_supertype(&l, &r)
        })
        .unwrap()
}
