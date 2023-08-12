#[cfg(feature = "dtype-struct")]
use super::*;

/// Take several expressions and collect them into a [`StructChunked`].
#[cfg(feature = "dtype-struct")]
pub fn as_struct(exprs: &[Expr]) -> Expr {
    map_multiple(
        |s| StructChunked::new(s[0].name(), s).map(|ca| Some(ca.into_series())),
        exprs,
        GetOutput::map_fields(|fld| Field::new(fld[0].name(), DataType::Struct(fld.to_vec()))),
    )
    .with_function_options(|mut options| {
        options.input_wildcard_expansion = true;
        options.fmt_str = "as_struct";
        options.pass_name_to_apply = true;
        options
    })
}
