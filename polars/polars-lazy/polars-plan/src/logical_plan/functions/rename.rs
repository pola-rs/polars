use super::*;

pub(super) fn rename_impl(
    mut df: DataFrame,
    existing: &[String],
    new: &[String],
) -> PolarsResult<DataFrame> {
    let positions = existing
        .iter()
        .map(|old| df.find_idx_by_name(old))
        .collect::<Vec<_>>();

    for (pos, name) in positions.iter().zip(new.iter()) {
        // the column might be removed due to projection pushdown
        // so we only update if we can find it.
        if let Some(pos) = pos {
            df.get_columns_mut()[*pos].rename(name);
        }
    }
    // recreate dataframe so we check duplicates
    let columns = std::mem::take(df.get_columns_mut());
    DataFrame::new(columns)
}

pub(super) fn rename_schema<'a>(
    input_schema: &'a SchemaRef,
    existing: &[String],
    new: &[String],
) -> PolarsResult<Cow<'a, SchemaRef>> {
    let mut new_schema = (**input_schema).clone();
    for (old, new) in existing.iter().zip(new.iter()) {
        // the column might be removed due to projection pushdown
        // so we only update if we can find it.
        if let Some(dtype) = input_schema.get(old) {
            if new_schema.with_column(new.clone(), dtype.clone()).is_none() {
                new_schema.remove(old);
            }
        }
    }
    Ok(Cow::Owned(Arc::new(new_schema)))
}
