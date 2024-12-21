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
fn apply_to_cats<F, T>(ca: &CategoricalChunked, mut op: F) -> PolarsResult<Column>
where
    F: FnMut(&StringChunked) -> ChunkedArray<T>,
    ChunkedArray<T>: IntoSeries,
    T: PolarsDataType<HasViews = FalseT, IsStruct = FalseT, IsNested = FalseT>,
{
    let (categories, phys) = _get_cat_phys_map(ca);
    let result = op(&categories);
    // SAFETY: physical idx array is valid.
    let out = unsafe { result.take_unchecked(phys.idx().unwrap()) };
    Ok(out.into_column())
}

/// Fast path: apply a binary function to the categories of a categorical column and broadcast the
/// result back to the array.
fn apply_to_cats_binary<F, T>(ca: &CategoricalChunked, mut op: F) -> PolarsResult<Column>
where
    F: FnMut(&BinaryChunked) -> ChunkedArray<T>,
    ChunkedArray<T>: IntoSeries,
    T: PolarsDataType<HasViews = FalseT, IsStruct = FalseT, IsNested = FalseT>,
{
    let (categories, phys) = _get_cat_phys_map(ca);
    let result = op(&categories.as_binary());
    // SAFETY: physical idx array is valid.
    let out = unsafe { result.take_unchecked(phys.idx().unwrap()) };
    Ok(out.into_column())
}

#[cfg(feature = "strings")]
fn len_bytes(s: &Column) -> PolarsResult<Column> {
    let ca = s.categorical()?;
    apply_to_cats(ca, |s| s.str_len_bytes())
}

#[cfg(feature = "strings")]
fn len_chars(s: &Column) -> PolarsResult<Column> {
    let ca = s.categorical()?;
    apply_to_cats(ca, |s| s.str_len_chars())
}

#[cfg(feature = "strings")]
fn starts_with(s: &Column, prefix: &str) -> PolarsResult<Column> {
    let ca = s.categorical()?;
    apply_to_cats(ca, |s| s.starts_with(prefix))
}

#[cfg(feature = "strings")]
fn ends_with(s: &Column, suffix: &str) -> PolarsResult<Column> {
    let ca = s.categorical()?;
    apply_to_cats_binary(ca, |s| s.as_binary().ends_with(suffix.as_bytes()))
}
