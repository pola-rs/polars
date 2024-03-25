use crate::prelude::*;
use crate::utils::try_get_supertype;

/// Determine the supertype of a collection of [`AnyValue`].
///
/// [`AnyValue`]: crate::datatypes::AnyValue
pub fn any_values_to_supertype<'a, I>(values: I) -> PolarsResult<DataType>
where
    I: IntoIterator<Item = &'a AnyValue<'a>>,
{
    let mut supertype = DataType::Null;
    let mut dtypes = PlHashSet::<DataType>::new();
    for av in values {
        if dtypes.insert(av.dtype()) {
            supertype = try_get_supertype(&supertype, &av.dtype()).map_err(|_| {
                polars_err!(
                    SchemaMismatch:
                    "failed to infer supertype of values; partial supertype is {:?}, found value of type {:?}: {}",
                    supertype, av.dtype(), av
                )
            })?;
        }
    }
    Ok(supertype)
}
