use std::sync::Arc;

use polars_core::datatypes::DataType;
use polars_core::error::{PolarsResult, polars_ensure};
use polars_core::prelude::*;
use polars_core::series::Series;
use polars_ops::prelude::BinaryNameSpaceImpl;
#[cfg(feature = "strings")]
use polars_ops::prelude::StringNameSpaceImpl;
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRCategoricalFunction;

pub fn function_expr_to_udf(func: IRCategoricalFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRCategoricalFunction::*;
    match func {
        GetCategories => map!(get_categories),
        #[cfg(feature = "strings")]
        LenBytes => map!(len_bytes),
        #[cfg(feature = "strings")]
        LenChars => map!(len_chars),
        #[cfg(feature = "strings")]
        StartsWith(prefix) => map!(starts_with, prefix.as_str()),
        #[cfg(feature = "strings")]
        EndsWith(suffix) => map!(ends_with, suffix.as_str()),
        #[cfg(feature = "strings")]
        Slice(offset, length) => map!(slice, offset, length),
        To(dtype, strict) => map!(cat_to, &dtype, strict),
        Physical => map!(cat_physical),
    }
}

fn get_categories(s: &Column) -> PolarsResult<Column> {
    let mapping = s.dtype().cat_mapping()?;
    let ca = unsafe { StringChunked::from_chunks(s.name().clone(), vec![mapping.to_arrow(true)]) };
    Ok(ca.into_column())
}

// Determine mapping between categories and underlying physical. For local, this is just 0..n.
// For global, this is the global indexes.
fn _get_cat_phys_map(col: &Column) -> (StringChunked, Series) {
    let mapping = col.dtype().cat_mapping().unwrap();
    let cats =
        unsafe { StringChunked::from_chunks(col.name().clone(), vec![mapping.to_arrow(true)]) };
    let mut phys = col.to_physical_repr();
    if phys.dtype() != &IDX_DTYPE {
        phys = phys.cast(&IDX_DTYPE).unwrap();
    }
    let phys = phys.as_materialized_series().clone();
    (cats, phys)
}

/// Fast path: apply a string function to the categories of a categorical column and broadcast the
/// result back to the array.
fn apply_to_cats<F, T>(c: &Column, mut op: F) -> PolarsResult<Column>
where
    F: FnMut(StringChunked) -> ChunkedArray<T>,
    T: PolarsPhysicalType<HasViews = FalseT, IsStruct = FalseT, IsNested = FalseT>,
{
    let (categories, phys) = _get_cat_phys_map(c);
    let result = op(categories);
    // SAFETY: physical idx array is valid.
    let out = unsafe { result.take_unchecked(phys.idx().unwrap()) };
    Ok(out.into_column())
}

#[cfg(feature = "strings")]
fn len_bytes(c: &Column) -> PolarsResult<Column> {
    apply_to_cats(c, |s| s.str_len_bytes())
}

#[cfg(feature = "strings")]
fn len_chars(c: &Column) -> PolarsResult<Column> {
    apply_to_cats(c, |s| s.str_len_chars())
}

#[cfg(feature = "strings")]
fn starts_with(c: &Column, prefix: &str) -> PolarsResult<Column> {
    apply_to_cats(c, |s| s.as_binary().starts_with(prefix.as_bytes()))
}

#[cfg(feature = "strings")]
fn ends_with(c: &Column, suffix: &str) -> PolarsResult<Column> {
    apply_to_cats(c, |s| s.as_binary().ends_with(suffix.as_bytes()))
}

#[cfg(feature = "strings")]
fn slice(c: &Column, offset: i64, length: Option<usize>) -> PolarsResult<Column> {
    let length = length.unwrap_or(usize::MAX) as u64;
    let (categories, phys) = _get_cat_phys_map(c);

    let result = unsafe {
        categories.apply_views(|view, val| {
            let (start, end) =
                polars_ops::prelude::substring_ternary_offsets_value(val, offset, length);
            polars_ops::prelude::update_view(view, start, end, val)
        })
    };
    // SAFETY: physical idx array is valid.
    let out = unsafe { result.take_unchecked(phys.idx().unwrap()) };
    Ok(out.into_column())
}

fn cat_to(s: &Column, dtype: &DataType, strict: bool) -> PolarsResult<Column> {
    s.try_apply_unary_elementwise(|s| Series::from_cats_and_dtype(s, dtype, strict))
}

fn cat_physical(s: &Column) -> PolarsResult<Column> {
    polars_ensure!(s.dtype().is_categorical() || s.dtype().is_enum(), SchemaMismatch: "cannot call `.cat.physical` on a column which isn't a Categorical or Enum type ({})", s.dtype());
    Ok(s.apply_unary_elementwise(|s| s.to_physical_repr().into_owned()))
}
