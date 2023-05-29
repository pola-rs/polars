use polars_core::prelude::*;

pub(super) fn utf8_to_decimal(ca: &Utf8Chunked, infer_length: usize) -> PolarsResult<Series> {
    let mut params = PlHashMap::new();
    let mut iter = ca.into_iter().take(infer_length);
    while let Some(Some(v)) = iter.next() {
        if let Some(p) = polars_arrow::compute::decimal::infer_params(v.as_bytes()) {
            let count = params.entry(p).or_insert(0usize);
            *count += 1;
        }
    }
    polars_ensure!(!params.is_empty(), ComputeError: "could not infer decimal parameters");
    let mut params = params.into_iter().collect::<Vec<_>>();
    params.sort_unstable_by_key(|k| k.1);
    let ((precision, scale), _count) = params.last().unwrap();
    ca.cast(&DataType::Decimal(
        Some(*precision as usize),
        Some(*scale as usize),
    ))
}
