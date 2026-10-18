use polars_error::PolarsResult;
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;

pub fn compute_cut_labels(breaks: &[f64], left_closed: bool) -> PolarsResult<Vec<PlSmallStr>> {
    let lo = std::iter::once(&f64::NEG_INFINITY).chain(breaks.iter());
    let hi = breaks.iter().chain(std::iter::once(&f64::INFINITY));

    let ret = lo
        .zip(hi)
        .map(|(l, h)| {
            if left_closed {
                format_pl_smallstr!("[{}, {})", l, h)
            } else {
                format_pl_smallstr!("({}, {}]", l, h)
            }
        })
        .collect();
    Ok(ret)
}
