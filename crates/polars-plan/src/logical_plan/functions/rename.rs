use super::*;

pub(super) fn rename_impl(
    mut df: DataFrame,
    existing: &[SmartString],
    new: &[SmartString],
) -> PolarsResult<DataFrame> {
    let positions = existing
        .iter()
        .map(|old| df.get_column_index(old))
        .collect::<Vec<_>>();

    for (pos, name) in positions.iter().zip(new.iter()) {
        // the column might be removed due to projection pushdown
        // so we only update if we can find it.
        if let Some(pos) = pos {
            unsafe { df.get_columns_mut()[*pos].rename(name) };
        }
    }
    // recreate dataframe so we check duplicates
    let columns = unsafe { std::mem::take(df.get_columns_mut()) };
    DataFrame::new(columns)
}

pub(super) fn rename_schema<'a>(
    input_schema: &'a SchemaRef,
    existing: &[SmartString],
    new: &[SmartString],
) -> PolarsResult<Cow<'a, SchemaRef>> {
    let mut new_schema = input_schema.iter_fields().collect::<Vec<_>>();

    for (old, new) in existing.iter().zip(new.iter()) {
        // The column might be removed due to projection pushdown
        // so we only update if we can find it.
        if let Some((idx, _, _)) = input_schema.get_full(old) {
            new_schema[idx].name = new.as_str().into();
        }
    }
    Ok(Cow::Owned(Arc::new(new_schema.into_iter().collect())))
}
