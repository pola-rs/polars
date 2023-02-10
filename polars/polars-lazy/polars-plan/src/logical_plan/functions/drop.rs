use super::*;

pub(super) fn drop_impl(mut df: DataFrame, names: &[String]) -> PolarsResult<DataFrame> {
    for name in names {
        // ignore names that are not in there
        // they might already be removed by projection pushdown
        if let Some(idx) = df.find_idx_by_name(name) {
            let _ = df.get_columns_mut().remove(idx);
        }
    }

    Ok(df)
}

pub(super) fn drop_schema<'a>(
    input_schema: &'a SchemaRef,
    names: &[String],
) -> PolarsResult<Cow<'a, SchemaRef>> {
    let to_drop = PlHashSet::from_iter(names);

    let new_schema = input_schema
        .iter()
        .flat_map(|(name, dtype)| {
            if to_drop.contains(name) {
                None
            } else {
                Some(Field::new(name, dtype.clone()))
            }
        })
        .collect::<Schema>();

    Ok(Cow::Owned(Arc::new(new_schema)))
}
