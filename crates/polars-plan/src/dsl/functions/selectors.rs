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
        "*" => Expr::Wildcard,
        _ => Expr::Column(name),
    }
}

/// Selects all columns. Shorthand for `col("*")`.
pub fn all() -> Expr {
    Expr::Wildcard
}

/// Select multiple columns by name.
pub fn cols<I, S>(names: I) -> Expr
where
    I: IntoIterator<Item = S>,
    S: Into<PlSmallStr>,
{
    let names = names.into_iter().map(|x| x.into()).collect();
    Expr::Columns(names)
}

/// Select multiple columns by dtype.
pub fn dtype_col(dtype: &DataType) -> Expr {
    Expr::DtypeColumn(vec![dtype.clone()])
}

/// Select multiple columns by dtype.
pub fn dtype_cols<DT: AsRef<[DataType]>>(dtype: DT) -> Expr {
    let dtypes = dtype.as_ref().to_vec();
    Expr::DtypeColumn(dtypes)
}

/// Select multiple columns by index.
pub fn index_cols<N: AsRef<[i64]>>(indices: N) -> Expr {
    let indices = indices.as_ref().to_vec();
    Expr::IndexColumn(Arc::from(indices))
}
