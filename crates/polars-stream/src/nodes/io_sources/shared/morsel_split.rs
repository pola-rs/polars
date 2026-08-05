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
    use polars_core::prelude::{Column, IntoColumn, UInt32Chunked};

    use super::{DataFrame, split_to_morsels};

    fn df_with_height(height: usize) -> DataFrame {
        let values: Vec<u32> = (0..height as u32).collect();
        let column: Column = UInt32Chunked::from_vec("a".into(), values).into_column();
        unsafe { DataFrame::new_unchecked(height, vec![column]) }
    }

    fn heights(df: &DataFrame, ideal: usize, last: bool, pipelines: usize) -> Vec<usize> {
        split_to_morsels(df, ideal, last, pipelines)
            .map(|df| df.height())
            .collect()
    }

    #[test]
    fn stays_whole_below_the_split_threshold() {
        // height <= 1.5 * ideal_morsel_size, so n_morsels is 1
        let df = df_with_height(6);
        assert_eq!(heights(&df, 4, false, 1), vec![6]);
    }

    #[test]
    fn splits_above_the_threshold() {
        let df = df_with_height(10);
        let out = heights(&df, 4, false, 1);
        assert_eq!(out.iter().sum::<usize>(), 10);
        assert!(out.len() > 1);
    }

    #[test]
    fn last_morsel_respects_pipeline_count() {
        // A small frame is still spread across pipelines for the last morsel.
        let df = df_with_height(6);
        assert_eq!(heights(&df, 4, true, 3), vec![2, 2, 2]);
    }

    #[test]
    fn empty_frame_yields_nothing() {
        let df = df_with_height(0);
        assert!(heights(&df, 4, false, 1).is_empty());
    }
}
