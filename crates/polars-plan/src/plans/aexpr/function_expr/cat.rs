use super::*;
use crate::map;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum IRCategoricalFunction {
    GetCategories,
    #[cfg(feature = "strings")]
    LenBytes,
    #[cfg(feature = "strings")]
    LenChars,
    #[cfg(feature = "strings")]
    StartsWith(String),
    #[cfg(feature = "strings")]
    EndsWith(String),
    #[cfg(feature = "strings")]
    Slice(i64, Option<usize>),
}

impl IRCategoricalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use IRCategoricalFunction::*;
        match self {
            GetCategories => mapper.with_dtype(DataType::String),
            #[cfg(feature = "strings")]
            LenBytes => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "strings")]
            LenChars => mapper.with_dtype(DataType::UInt32),
            #[cfg(feature = "strings")]
            StartsWith(_) => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "strings")]
            EndsWith(_) => mapper.with_dtype(DataType::Boolean),
            #[cfg(feature = "strings")]
            Slice(_, _) => mapper.with_dtype(DataType::String),
        }
    }

    pub fn function_options(&self) -> FunctionOptions {
        use IRCategoricalFunction as C;
        match self {
            C::GetCategories => FunctionOptions::groupwise(),
            #[cfg(feature = "strings")]
            C::LenBytes | C::LenChars | C::StartsWith(_) | C::EndsWith(_) | C::Slice(_, _) => {
                FunctionOptions::elementwise()
            },
        }
    }
}

impl Display for IRCategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use IRCategoricalFunction::*;
        let s = match self {
            GetCategories => "get_categories",
            #[cfg(feature = "strings")]
            LenBytes => "len_bytes",
            #[cfg(feature = "strings")]
            LenChars => "len_chars",
            #[cfg(feature = "strings")]
            StartsWith(_) => "starts_with",
            #[cfg(feature = "strings")]
            EndsWith(_) => "ends_with",
            #[cfg(feature = "strings")]
            Slice(_, _) => "slice",
        };
        write!(f, "cat.{s}")
    }
}

impl From<IRCategoricalFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: IRCategoricalFunction) -> Self {
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
        }
    }
}

impl From<IRCategoricalFunction> for IRFunctionExpr {
    fn from(func: IRCategoricalFunction) -> Self {
        IRFunctionExpr::Categorical(func)
    }
}

fn get_categories(s: &Column) -> PolarsResult<Column> {
    let mapping = s.dtype().cat_mapping()?;
    let ca = unsafe { StringChunked::from_chunks(s.name().clone(), vec![mapping.to_arrow(true)]) };
    Ok(Column::from(ca.into_series()))
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
            let (start, end) = substring_ternary_offsets_value(val, offset, length);
            update_view(view, start, end, val)
        })
    };
    // SAFETY: physical idx array is valid.
    let out = unsafe { result.take_unchecked(phys.idx().unwrap()) };
    Ok(out.into_column())
}
