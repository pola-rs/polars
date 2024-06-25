use super::*;

pub(super) fn explode_impl(df: DataFrame, columns: &[SmartString]) -> PolarsResult<DataFrame> {
    df.explode(columns)
}
