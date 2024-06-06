use super::*;

macro_rules! prepare_binary_function {
    ($f:ident) => {
        move |s: &mut [Series]| {
            let s0 = std::mem::take(&mut s[0]);
            let s1 = std::mem::take(&mut s[1]);

            $f(s0, s1)
        }
    };
}

/// Apply a closure on the two columns that are evaluated from [`Expr`] a and [`Expr`] b.
///
/// The closure takes two arguments, each a [`Series`]. `output_type` must be the output dtype of the resulting [`Series`].
pub fn map_binary<F>(a: Expr, b: Expr, f: F, output_type: GetOutput) -> Expr
where
    F: 'static + Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync,
{
    let function = prepare_binary_function!(f);
    a.map_many(function, &[b], output_type)
}

/// Like [`map_binary`], but used in a group_by-aggregation context.
///
/// See [`Expr::apply`] for the difference between [`map`](Expr::map) and [`apply`](Expr::apply).
pub fn apply_binary<F>(a: Expr, b: Expr, f: F, output_type: GetOutput) -> Expr
where
    F: 'static + Fn(Series, Series) -> PolarsResult<Option<Series>> + Send + Sync,
{
    let function = prepare_binary_function!(f);
    a.apply_many(function, &[b], output_type)
}
