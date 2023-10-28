//! # Functions
//!
//! Functions that might be useful.
//!
use std::ops::Add;

#[cfg(feature = "diagonal_concat")]
use ahash::AHashSet;
use arrow::compute;
use arrow::types::simd::Simd;
use num_traits::ToPrimitive;

use crate::prelude::*;
use crate::utils::coalesce_nulls;
#[cfg(feature = "diagonal_concat")]
use crate::utils::concat_df;

/// Compute the covariance between two columns.
pub fn cov<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: ToPrimitive,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    if a.len() != b.len() {
        None
    } else {
        let a_mean = a.mean()?;
        let b_mean = b.mean()?;
        let a: Float64Chunked = a.apply_values_generic(|a| a.to_f64().unwrap() - a_mean);
        let b: Float64Chunked = b.apply_values_generic(|b| b.to_f64().unwrap() - b_mean);

        let tmp = a * b;
        let n = tmp.len() - tmp.null_count();
        Some(tmp.sum()? / (n - 1) as f64)
    }
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>, ddof: u8) -> Option<f64>
where
    T: PolarsNumericType,
    T::Native: ToPrimitive,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
    ChunkedArray<T>: ChunkVar,
{
    let (a, b) = coalesce_nulls(a, b);
    let a = a.as_ref();
    let b = b.as_ref();

    Some(cov(a, b)? / (a.std(ddof)? * b.std(ddof)?))
}

/// Concat [`DataFrame`]s horizontally.
#[cfg(feature = "horizontal_concat")]
/// Concat horizontally and extend with null values if lengths don't match
pub fn concat_df_horizontal(dfs: &[DataFrame]) -> PolarsResult<DataFrame> {
    let max_len = dfs
        .iter()
        .map(|df| df.height())
        .max()
        .ok_or_else(|| polars_err!(ComputeError: "cannot concat empty dataframes"))?;

    let owned_df;

    // if not all equal length, extend the DataFrame with nulls
    let dfs = if !dfs.iter().all(|df| df.height() == max_len) {
        owned_df = dfs
            .iter()
            .cloned()
            .map(|mut df| {
                if df.height() != max_len {
                    let diff = max_len - df.height();
                    df.columns
                        .iter_mut()
                        .for_each(|s| *s = s.extend_constant(AnyValue::Null, diff).unwrap());
                }
                df
            })
            .collect::<Vec<_>>();
        owned_df.as_slice()
    } else {
        dfs
    };

    let mut first_df = dfs[0].clone();

    for df in &dfs[1..] {
        first_df.hstack_mut(df.get_columns())?;
    }
    Ok(first_df)
}

/// Concat [`DataFrame`]s diagonally.
#[cfg(feature = "diagonal_concat")]
/// Concat diagonally thereby combining different schemas.
pub fn concat_df_diagonal(dfs: &[DataFrame]) -> PolarsResult<DataFrame> {
    // TODO! replace with lazy only?
    let upper_bound_width = dfs.iter().map(|df| df.width()).sum();
    let mut column_names = AHashSet::with_capacity(upper_bound_width);
    let mut schema = Vec::with_capacity(upper_bound_width);

    for df in dfs {
        df.get_columns().iter().for_each(|s| {
            let name = s.name();
            if column_names.insert(name) {
                schema.push((name, s.dtype()))
            }
        });
    }

    let dfs = dfs
        .iter()
        .map(|df| {
            let height = df.height();
            let mut columns = Vec::with_capacity(schema.len());

            for (name, dtype) in &schema {
                match df.column(name).ok() {
                    Some(s) => columns.push(s.clone()),
                    None => columns.push(Series::full_null(name, height, dtype)),
                }
            }
            DataFrame::new_no_checks(columns)
        })
        .collect::<Vec<_>>();

    concat_df(&dfs)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_cov() {
        let a = Series::new("a", &[1.0f32, 2.0, 5.0]);
        let b = Series::new("b", &[1.0f32, 2.0, -3.0]);
        let out = cov(a.f32().unwrap(), b.f32().unwrap());
        assert_eq!(out, Some(-5.0));
        let a = a.cast(&DataType::Int32).unwrap();
        let b = b.cast(&DataType::Int32).unwrap();
        let out = cov(a.i32().unwrap(), b.i32().unwrap());
        assert_eq!(out, Some(-5.0));
    }

    #[test]
    fn test_pearson_corr() {
        let a = Series::new("a", &[1.0f32, 2.0]);
        let b = Series::new("b", &[1.0f32, 2.0]);
        assert!((cov(a.f32().unwrap(), b.f32().unwrap()).unwrap() - 0.5).abs() < 0.001);
        assert!((pearson_corr(a.f32().unwrap(), b.f32().unwrap(), 1).unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "diagonal_concat")]
    fn test_diag_concat() -> PolarsResult<()> {
        let a = df![
            "a" => [1, 2],
            "b" => ["a", "b"]
        ]?;

        let b = df![
            "b" => ["a", "b"],
            "c" => [1, 2]
        ]?;

        let c = df![
            "a" => [5, 7],
            "c" => [1, 2],
            "d" => [1, 2]
        ]?;

        let out = concat_df_diagonal(&[a, b, c])?;

        let expected = df![
            "a" => [Some(1), Some(2), None, None, Some(5), Some(7)],
            "b" => [Some("a"), Some("b"), Some("a"), Some("b"), None, None],
            "c" => [None, None, Some(1), Some(2), Some(1), Some(2)],
            "d" => [None, None, None, None, Some(1), Some(2)]
        ]?;

        assert!(out.frame_equal_missing(&expected));

        Ok(())
    }
}
