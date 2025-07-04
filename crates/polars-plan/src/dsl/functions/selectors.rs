use super::*;

/// Create a Column Expression based on a column name.
///
/// # Arguments
///
/// * `name` - A string slice that holds the name of the column. If a column with this name does not exist when the
///   LazyFrame is collected, an error is returned.
///
/// # Examples
///
/// ```ignore
/// // select a column name
/// col("foo")
/// ```
///
/// ```ignore
/// // select all columns by using a wildcard
/// col("*")
/// ```
///
/// ```ignore
/// // select specific columns by writing a regular expression that starts with `^` and ends with `$`
/// // only if regex features is activated
/// col("^foo.*$")
/// ```
pub fn col<S>(name: S) -> Expr
where
    S: Into<PlSmallStr>,
{
    let name = name.into();
    match name.as_str() {
        "*" => all().as_expr(),
        n if is_regex_projection(n) => Expr::Selector(Selector::Matches(name)),
        _ => Expr::Column(name),
    }
}

/// Selects no columns.
pub fn empty() -> Selector {
    Selector::Empty
}

/// Selects all columns.
pub fn all() -> Selector {
    Selector::Wildcard
}

/// Select multiple columns by name.
pub fn cols<I, S>(names: I) -> Selector
where
    I: IntoIterator<Item = S>,
    S: Into<PlSmallStr>,
{
    let names = names.into_iter().map(|x| x.into()).collect();
    Selector::ByName { names, strict: true }
}

/// Select multiple columns by dtype.
pub fn dtype_col(dtype: &DataType) -> Selector {
    Selector::ByDType([dtype.clone()].into())
}

/// Select multiple columns by dtype.
pub fn dtype_cols<DT: AsRef<[DataType]>>(dtype: DT) -> Selector {
    let dtypes = dtype.as_ref();
    Selector::ByDType(dtypes.into())
}

/// Select multiple columns by index.
pub fn index_cols<N: AsRef<[i64]>>(indices: N) -> Selector {
    let indices = indices.as_ref().into();
    Selector::ByIndex { indices, strict: true }
}
