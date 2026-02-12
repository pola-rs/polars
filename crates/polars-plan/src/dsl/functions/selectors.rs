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

pub fn element() -> Expr {
    Expr::Element
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
    by_name(names, true, true)
}

/// Select multiple columns by dtype.
pub fn dtype_col(dtype: &DataType) -> DataTypeSelector {
    DataTypeSelector::AnyOf([dtype.clone()].into())
}

/// Select multiple columns by dtype.
pub fn dtype_cols<DT: AsRef<[DataType]>>(dtype: DT) -> DataTypeSelector {
    let dtypes = dtype.as_ref();
    DataTypeSelector::AnyOf(dtypes.into())
}

/// Select multiple columns by name.
///
/// When `expand_patterns` is `true`, a single wildcard `"*"` and anchored regex patterns
/// (e.g. `"^...$"`) are expanded to their matching columns. When `false`, names are
/// treated as literals.
pub fn by_name<S: Into<PlSmallStr>, I: IntoIterator<Item = S>>(
    names: I,
    strict: bool,
    expand_patterns: bool,
) -> Selector {
    if !expand_patterns {
        let names = names.into_iter().map(Into::into).collect::<Arc<[_]>>();
        return Selector::ByName { names, strict };
    }

    // When expand_patterns is true, handle wildcards and regex patterns
    let mut selector = None;
    let _s = &mut selector;
    let names = names
        .into_iter()
        .map(Into::into)
        .filter_map(|name| match name.as_str() {
            "*" => {
                *_s = Some(std::mem::take(_s).map_or(all(), |s| s | all()));
                None
            },
            n if is_regex_projection(n) => {
                let m = Selector::Matches(name);
                *_s = Some(std::mem::take(_s).map_or_else(|| m.clone(), |s| s | m.clone()));
                None
            },
            _ => Some(name),
        })
        .collect::<Arc<[_]>>();

    let no_names = names.is_empty();
    let names = Selector::ByName { names, strict };
    if let Some(selector) = selector {
        if no_names { selector } else { selector | names }
    } else {
        names
    }
}

/// Select multiple columns by index.
pub fn index_cols<N: AsRef<[i64]>>(indices: N) -> Selector {
    let indices = indices.as_ref().into();
    Selector::ByIndex {
        indices,
        strict: true,
    }
}
