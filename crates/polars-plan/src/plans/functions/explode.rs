use super::*;

pub(super) fn explode_impl(df: DataFrame, columns: &[PlSmallStr]) -> PolarsResult<DataFrame> {
    df.explode(columns)
}
