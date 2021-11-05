//! # Functions
//!
//! Functions that might be useful.
//!
#[cfg(feature = "sort_multiple")]
use crate::chunked_array::ops::sort::prepare_argsort;
use crate::prelude::*;
#[cfg(feature = "diagonal_concat")]
use crate::utils::concat_df;
#[cfg(feature = "diagonal_concat")]
use ahash::AHashSet;
use arrow::compute;
use arrow::types::simd::Simd;
use num::{Float, NumCast};
#[cfg(feature = "concat_str")]
use polars_arrow::prelude::ValueSize;
use std::ops::Add;

/// Compute the covariance between two columns.
pub fn cov<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<T::Native>
where
    T: PolarsFloatType,
    T::Native: Float,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
{
    if a.len() != b.len() {
        None
    } else {
        let tmp = (a - a.mean()?) * (b - b.mean()?);
        let n = tmp.len() - tmp.null_count();
        Some(tmp.sum()? / NumCast::from(n - 1).unwrap())
    }
}

/// Compute the pearson correlation between two columns.
pub fn pearson_corr<T>(a: &ChunkedArray<T>, b: &ChunkedArray<T>) -> Option<T::Native>
where
    T: PolarsFloatType,
    T::Native: Float,
    <T::Native as Simd>::Simd: Add<Output = <T::Native as Simd>::Simd>
        + compute::aggregate::Sum<T::Native>
        + compute::aggregate::SimdOrd<T::Native>,
    ChunkedArray<T>: ChunkVar<T::Native>,
{
    Some(cov(a, b)? / (a.std()? * b.std()?))
}

#[cfg(feature = "sort_multiple")]
/// Find the indexes that would sort these series in order of appearance.
/// That means that the first `Series` will be used to determine the ordering
/// until duplicates are found. Once duplicates are found, the next `Series` will
/// be used and so on.
pub fn argsort_by(by: &[Series], reverse: &[bool]) -> Result<UInt32Chunked> {
    if by.len() != reverse.len() {
        return Err(PolarsError::ValueError(
            format!(
                "The amount of ordering booleans: {} does not match amount of Series: {}",
                reverse.len(),
                by.len()
            )
            .into(),
        ));
    }
    let (first, by, reverse) =
        prepare_argsort(by.to_vec(), reverse.iter().copied().collect()).unwrap();
    first.argsort_multiple(&by, &reverse)
}

// utility to be able to also add literals ot concat_str function
#[cfg(feature = "concat_str")]
enum IterBroadCast<'a> {
    Column(Box<dyn PolarsIterator<Item = Option<&'a str>> + 'a>),
    Value(Option<&'a str>),
}

#[cfg(feature = "concat_str")]
impl<'a> IterBroadCast<'a> {
    fn next(&mut self) -> Option<Option<&'a str>> {
        use IterBroadCast::*;
        match self {
            Column(iter) => iter.next(),
            Value(val) => Some(*val),
        }
    }
}

/// Casts all series to string data and will concat them in linear time.
/// The concatenated strings are separated by a `delimiter`.
/// If no `delimiter` is needed, an empty &str should be passed as argument.
#[cfg(feature = "concat_str")]
#[cfg_attr(docsrs, doc(cfg(feature = "concat_str")))]
pub fn concat_str(s: &[Series], delimiter: &str) -> Result<Utf8Chunked> {
    if s.is_empty() {
        return Err(PolarsError::NoData(
            "expected multiple series in concat_str function".into(),
        ));
    }
    let len = s.iter().map(|s| s.len()).max().unwrap();

    let cas = s
        .iter()
        .map(|s| {
            let s = s.cast(&DataType::Utf8)?;
            let mut ca = s.utf8()?.clone();
            // broadcast
            if ca.len() == 1 && len > 1 {
                ca = ca.expand_at_index(0, len)
            }

            Ok(ca)
        })
        .collect::<Result<Vec<_>>>()?;

    if !s.iter().all(|s| s.len() == 1 || s.len() == len) {
        return Err(PolarsError::ValueError(
            "all series in concat_str function should have equal length or unit length".into(),
        ));
    }
    let mut iters = cas
        .iter()
        .map(|ca| match ca.len() {
            1 => IterBroadCast::Value(ca.get(0)),
            _ => IterBroadCast::Column(ca.into_iter()),
        })
        .collect::<Vec<_>>();

    let bytes_cap = cas.iter().map(|ca| ca.get_values_size()).sum();
    let mut builder = Utf8ChunkedBuilder::new(s[0].name(), len, bytes_cap);

    // use a string buffer, to amortize alloc
    let mut buf = String::with_capacity(128);

    for _ in 0..len {
        let mut has_null = false;

        iters.iter_mut().enumerate().for_each(|(i, it)| {
            if i > 0 {
                buf.push_str(delimiter);
            }

            match it.next() {
                Some(Some(s)) => buf.push_str(s),
                Some(None) => has_null = true,
                None => {
                    // should not happen as the out loop counts to length
                    unreachable!()
                }
            }
        });

        if has_null {
            builder.append_null();
        } else {
            builder.append_value(&buf)
        }
        buf.truncate(0)
    }
    Ok(builder.finish())
}

/// Concat `[DataFrame]`s diagonally.
#[cfg(feature = "diagonal_concat")]
#[cfg_attr(docsrs, doc(cfg(feature = "diagonal_concat")))]
pub fn diag_concat_df(dfs: &[DataFrame]) -> Result<DataFrame> {
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
    fn test_pearson_corr() {
        let a = Series::new("a", &[1.0f32, 2.0]);
        let b = Series::new("b", &[1.0f32, 2.0]);
        assert!((cov(a.f32().unwrap(), b.f32().unwrap()).unwrap() - 0.5).abs() < 0.001);
        assert!((pearson_corr(a.f32().unwrap(), b.f32().unwrap()).unwrap() - 1.0).abs() < 0.001);
    }

    #[test]
    #[cfg(feature = "concat_str")]
    fn test_concat_str() {
        let a = Series::new("a", &["foo", "bar"]);
        let b = Series::new("b", &["spam", "ham"]);

        let out = concat_str(&[a.clone(), b.clone()], "_").unwrap();
        assert_eq!(Vec::from(&out), &[Some("foo_spam"), Some("bar_ham")]);

        let c = Series::new("b", &["literal"]);
        let out = concat_str(&[a, b, c], "_").unwrap();
        assert_eq!(
            Vec::from(&out),
            &[Some("foo_spam_literal"), Some("bar_ham_literal")]
        );
    }

    #[test]
    #[cfg(feature = "diagonal_concat")]
    fn test_diag_concat() -> Result<()> {
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

        let out = diag_concat_df(&[a, b, c])?;

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
