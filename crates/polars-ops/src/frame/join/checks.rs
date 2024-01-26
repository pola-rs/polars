use super::*;

/// If Categorical types are created without a global string cache or under
/// a different global string cache the mapping will be incorrect.
pub(crate) fn _check_categorical_src(l: &DataType, r: &DataType) -> PolarsResult<()> {
    match (l, r) {
        (DataType::Categorical(Some(l), _), DataType::Categorical(Some(r), _))
        | (DataType::Enum(Some(l), _), DataType::Enum(Some(r), _)) => {
            polars_ensure!(l.same_src(r), string_cache_mismatch);
        },
        (DataType::Categorical(_, _), DataType::Enum(_, _))
        | (DataType::Enum(_, _), DataType::Categorical(_, _)) => {
            polars_bail!(ComputeError: "enum and categorical are not from the same source")
        },
        _ => (),
    };
    Ok(())
}
