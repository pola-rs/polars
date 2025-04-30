use super::*;
use crate::map;

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[derive(Clone, PartialEq, Debug, Eq, Hash)]
pub enum CategoricalFunction {
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

impl CategoricalFunction {
    pub(super) fn get_field(&self, mapper: FieldsMapper) -> PolarsResult<Field> {
        use CategoricalFunction::*;
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
        use CategoricalFunction as C;
        match self {
            C::GetCategories => FunctionOptions::groupwise(),
            #[cfg(feature = "strings")]
            C::LenBytes | C::LenChars | C::StartsWith(_) | C::EndsWith(_) | C::Slice(_, _) => {
                FunctionOptions::elementwise()
            },
        }
    }
}

impl Display for CategoricalFunction {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use CategoricalFunction::*;
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

impl From<CategoricalFunction> for SpecialEq<Arc<dyn ColumnsUdf>> {
    fn from(func: CategoricalFunction) -> Self {
        use CategoricalFunction::*;
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

impl From<CategoricalFunction> for FunctionExpr {
    fn from(func: CategoricalFunction) -> Self {
        FunctionExpr::Categorical(func)
    }
}

fn get_categories(s: &Column) -> PolarsResult<Column> {
    // categorical check
    let ca = s.categorical()?;
    let rev_map = ca.get_rev_map();
    let arr = rev_map.get_categories().clone().boxed();
    Series::try_from((ca.name().clone(), arr)).map(Column::from)
}

// Determine mapping between categories and underlying physical. For local, this is just 0..n.
// For global, this is the global indexes.
fn _get_cat_phys_map(ca: &CategoricalChunked) -> (StringChunked, Series) {
    let (categories, phys) = match &**ca.get_rev_map() {
        RevMapping::Local(c, _) => (c, ca.physical().cast(&IDX_DTYPE).unwrap()),
        RevMapping::Global(physical_map, c, _) => {
            // Map physical to its local representation for use with take() later.
            let phys = ca
                .physical()
                .apply(|opt_v| opt_v.map(|v| *physical_map.get(&v).unwrap()));
            let out = phys.cast(&IDX_DTYPE).unwrap();
            (c, out)
        },
    };
    let categories = StringChunked::with_chunk(ca.name().clone(), categories.clone());
    (categories, phys)
}

/// Fast path: apply a string function to the categories of a categorical column and broadcast the
/// result back to the array.
// fn apply_to_cats<F, T>(ca: &CategoricalChunked, mut op: F) -> PolarsResult<Column>
fn apply_to_cats<F, T>(c: &Column, mut op: F) -> PolarsResult<Column>
where
    F: FnMut(&StringChunked) -> ChunkedArray<T>,
    ChunkedArray<T>: IntoSeries,
    T: PolarsDataType<HasViews = FalseT, IsStruct = FalseT, IsNested = FalseT>,
{
    let ca = c.categorical()?;
    let (categories, phys) = _get_cat_phys_map(ca);
    let result = op(&categories);
    // SAFETY: physical idx array is valid.
    let out = unsafe { result.take_unchecked(phys.idx().unwrap()) };
    Ok(out.into_column())
}

/// Fast path: apply a binary function to the categories of a categorical column and broadcast the
/// result back to the array.
fn apply_to_cats_binary<F, T>(c: &Column, mut op: F) -> PolarsResult<Column>
where
    F: FnMut(&BinaryChunked) -> ChunkedArray<T>,
    ChunkedArray<T>: IntoSeries,
    T: PolarsDataType<HasViews = FalseT, IsStruct = FalseT, IsNested = FalseT>,
{
    let ca = c.categorical()?;
    let (categories, phys) = _get_cat_phys_map(ca);
    let result = op(&categories.as_binary());
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
    apply_to_cats_binary(c, |s| s.starts_with(prefix.as_bytes()))
}

#[cfg(feature = "strings")]
fn ends_with(c: &Column, suffix: &str) -> PolarsResult<Column> {
    apply_to_cats_binary(c, |s| s.ends_with(suffix.as_bytes()))
}

#[cfg(feature = "strings")]
fn slice(c: &Column, offset: i64, length: Option<usize>) -> PolarsResult<Column> {
    let length = length.unwrap_or(usize::MAX) as u64;
    let ca = c.categorical()?;
    let (categories, phys) = _get_cat_phys_map(ca);

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
