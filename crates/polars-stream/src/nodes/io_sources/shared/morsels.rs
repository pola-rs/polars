use polars_core::frame::DataFrame;

pub(crate) fn split_to_morsels(
    df: &DataFrame,
    ideal_morsel_size: usize,
    last_morsel: bool,
    last_morsel_pipelines: usize,
) -> impl Iterator<Item = DataFrame> + '_ {
    let mut n_morsels = if df.height() > 3 * ideal_morsel_size / 2 {
        // num_rows > (1.5 * ideal_morsel_size)
        (df.height() / ideal_morsel_size).max(2)
    } else {
        1
    };

    if last_morsel {
        n_morsels = n_morsels.max(last_morsel_pipelines);
    }

    let rows_per_morsel = df.height().div_ceil(n_morsels).max(1);

    (0..i64::try_from(df.height()).unwrap())
        .step_by(rows_per_morsel)
        .map(move |offset| df.slice(offset, rows_per_morsel))
        .filter(|df| df.height() > 0)
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::Column;

    use super::{DataFrame, split_to_morsels};

    fn heights(height: usize, ideal: usize, last: bool, pipelines: usize) -> Vec<usize> {
        let values: Vec<u32> = (0..height as u32).collect();
        let df = DataFrame::new(height, vec![Column::new("a".into(), values)]).unwrap();
        split_to_morsels(&df, ideal, last, pipelines)
            .map(|df| df.height())
            .collect()
    }

    #[test]
    fn test_split_to_morsels() {
        // up to 1.5 * ideal stays whole; above it, at least 2 morsels
        assert_eq!(heights(6, 4, false, 1), [6]);
        assert_eq!(heights(7, 4, false, 1), [4, 3]);
        assert_eq!(heights(100, 10, false, 1), [10; 10]);
        // the last morsel is spread over the pipelines, but never into fewer morsels
        assert_eq!(heights(6, 4, true, 3), [2, 2, 2]);
        assert_eq!(heights(10, 4, true, 1), [5, 5]);
        assert!(heights(0, 4, false, 1).is_empty());
    }
}
