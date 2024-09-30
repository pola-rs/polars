use super::*;

pub(super) fn rename_impl(
    mut df: DataFrame,
    existing: &[PlSmallStr],
    new: &[PlSmallStr],
) -> PolarsResult<DataFrame> {
    let positions = if existing.len() > 1 && df.get_columns().len() > 10 {
        let schema = df.schema();
        existing
            .iter()
            .map(|old| schema.get_full(old).map(|(idx, _, _)| idx))
            .collect::<Vec<_>>()
    } else {
        existing
            .iter()
            .map(|old| df.get_column_index(old))
            .collect::<Vec<_>>()
    };

    for (pos, name) in positions.iter().zip(new.iter()) {
        // the column might be removed due to projection pushdown
        // so we only update if we can find it.
        if let Some(pos) = pos {
            unsafe { df.get_columns_mut()[*pos].rename(name.clone()) };
        }
    }
    // recreate dataframe so we check duplicates
    let columns = unsafe { std::mem::take(df.get_columns_mut()) };
    DataFrame::new(columns)
}
