use super::*;

/// If Categorical types are created without a global string cache or under
/// a different global string cache the mapping will be incorrect.
pub(crate) fn _check_categorical_src(l: &DataType, r: &DataType) -> PolarsResult<()> {
    if let (DataType::Categorical(Some(l), _), DataType::Categorical(Some(r), _)) = (l, r) {
        polars_ensure!(l.same_src(r), string_cache_mismatch);
    }
    Ok(())
}
