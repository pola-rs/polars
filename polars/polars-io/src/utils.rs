use polars_core::prelude::*;

pub(crate) fn to_arrow_compatible_df(df: &DataFrame) -> DataFrame {
    // Our categorical type is not known to arrow/ parquet, so we coerce to large-utf8.
    let cols = df
        .get_columns()
        .iter()
        .map(|s| match s.dtype() {
            DataType::Categorical => s.cast::<Utf8Type>().unwrap(),
            _ => s.clone(),
        })
        .collect::<Vec<_>>();
    DataFrame::new_no_checks(cols)
}
