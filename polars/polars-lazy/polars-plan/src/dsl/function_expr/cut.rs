use super::*;

pub(super) fn cut(
    s: &Series,
    breaks: Vec<f64>,
    labels: Option<Vec<String>>,
    left_closed: bool,
) -> PolarsResult<Series> {
    cut(s, breaks, labels, left_closed)
}

pub(super) fn qcut(
    s: &Series,
    probs: Vec<f64>,
    labels: Option<Vec<String>>,
    left_closed: bool,
    allow_duplicates: bool,
) -> PolarsResult<Series> {
    qcut(s, probs, labels, left_closed, allow_duplicates)
}
