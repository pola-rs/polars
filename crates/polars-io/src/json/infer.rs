use std::num::NonZeroUsize;

use polars_core::prelude::DataType;
use polars_core::utils::try_get_supertype;
use polars_error::{polars_bail, PolarsError, PolarsResult};
use simd_json::BorrowedValue;

pub(crate) fn json_values_to_supertype(
    values: &[BorrowedValue],
    infer_schema_len: NonZeroUsize,
) -> PolarsResult<DataType> {
    // struct types may have missing fields so find supertype
    let out_opt: Option<Result<DataType, PolarsError>> = values
        .iter()
        .take(infer_schema_len.into())
        .map(|value| polars_json::json::infer(value).map(|dt| DataType::from(&dt)))
        .reduce(|l, r| {
            let l = l?;
            let r = r?;
            try_get_supertype(&l, &r)
        });
    match (out_opt, values.len() == 0) {
        (Some(out), true) => match out {
            Ok(out) => Ok(out),
            _ => Err(PolarsError::NoData("no data".into())),
        },
        (Some(out), false) => out,
        (None, true) => Err(PolarsError::NoData("no data".into())),
        (None, false) => polars_bail!(ComputeError: "could not infer data-type"),
    }
}

pub(crate) fn dtypes_to_supertype<I: Iterator<Item = DataType>>(
    datatypes: I,
) -> PolarsResult<DataType> {
    datatypes
        .map(Ok)
        .reduce(|l, r| {
            let l = l?;
            let r = r?;
            try_get_supertype(&l, &r)
        })
        .unwrap_or_else(|| polars_bail!(ComputeError: "could not infer data-type"))
}
