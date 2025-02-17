use crate::prelude::*;
use crate::utils::dtypes_to_supertype;

/// Determine the supertype of a collection of [`AnyValue`].
///
/// [`AnyValue`]: crate::datatypes::AnyValue
pub fn any_values_to_supertype<'a, I>(values: I) -> PolarsResult<DataType>
where
    I: IntoIterator<Item = &'a AnyValue<'a>>,
{
    let dtypes = any_values_to_dtype_set(values);
    dtypes_to_supertype(&dtypes)
}

/// Determine the supertype and the number of unique data types of a collection of [`AnyValue`].
///
/// [`AnyValue`]: crate::datatypes::AnyValue
pub fn any_values_to_supertype_and_n_dtypes<'a, I>(values: I) -> PolarsResult<(DataType, usize)>
where
    I: IntoIterator<Item = &'a AnyValue<'a>>,
{
    let dtypes = any_values_to_dtype_set(values);
    let supertype = dtypes_to_supertype(&dtypes)?;
    let n_dtypes = dtypes.len();
    Ok((supertype, n_dtypes))
}

/// Extract the ordered set of data types from a collection of AnyValues
///
/// Retaining the order is important if the set is used to determine a supertype,
/// as this can influence how Struct fields are constructed.
fn any_values_to_dtype_set<'a, I>(values: I) -> PlIndexSet<DataType>
where
    I: IntoIterator<Item = &'a AnyValue<'a>>,
{
    values.into_iter().map(|av| av.into()).collect()
}
