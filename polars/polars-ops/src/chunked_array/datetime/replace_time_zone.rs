use polars_arrow::kernels::replace_time_zone as replace_time_zone_kernel;
use polars_core::prelude::*;

pub fn replace_time_zone(
    series: &DatetimeChunked,
    time_zone: Option<&str>,
    use_earliest: Option<bool>,
) -> PolarsResult<DatetimeChunked> {
    let out: PolarsResult<_> = {
        let from = series.time_zone().as_deref().unwrap_or("UTC");
        let to = time_zone.unwrap_or("UTC");
        let chunks = series
            .downcast_iter()
            .map(|arr| {
                replace_time_zone_kernel(arr, series.time_unit().to_arrow(), from, to, use_earliest)
            })
            .collect::<PolarsResult<_>>()?;
        let out = unsafe { ChunkedArray::from_chunks(series.name(), chunks) };
        Ok(out.into_datetime(series.time_unit(), time_zone.map(|x| x.to_string())))
    };
    let mut out = out?;
    out.set_sorted_flag(series.is_sorted_flag());
    Ok(out)
}
