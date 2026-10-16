//! Arguments of SQL table functions: a positional path followed by named options, as in
//! `read_csv('data.csv', has_header => false, separator => ';')`.

use std::num::NonZeroUsize;
use std::str::FromStr;

use polars_core::prelude::*;
#[cfg(feature = "csv")]
use sqlparser::ast::Array as SQLArray;
use sqlparser::ast::{
    DictionaryField, Expr as SQLExpr, FunctionArg as SQLFunctionArg,
    FunctionArgExpr as SQLFunctionArgExpr, Value as SQLValue, ValueWithSpan as SQLValueWithSpan,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::types::map_sql_dtype_to_polars;

/// The options of one table function call, validated against the options it accepts.
pub(crate) struct TableFunctionOptions<'a> {
    function: &'static str,
    options: PlHashMap<String, &'a SQLExpr>,
}

/// A `schema_overrides` value: by column name, or by position.
#[cfg(feature = "csv")]
pub(crate) enum DtypeOverrides {
    ByName(Schema),
    ByPosition(Vec<DataType>),
}

impl<'a> TableFunctionOptions<'a> {
    /// Split `args` into the leading positional argument and the named options that follow it.
    ///
    /// `valid` lists the accepted option names. `excluded` pairs options of the equivalent
    /// Python function that SQL does not accept with what to do instead (possibly empty), so the
    /// error can say more than "unknown option".
    pub(crate) fn parse(
        function: &'static str,
        args: &'a [SQLFunctionArg],
        valid: &[&str],
        excluded: &[(&str, &str)],
    ) -> PolarsResult<(&'a SQLFunctionArg, Self)> {
        let Some((first, rest)) = args.split_first() else {
            polars_bail!(SQLSyntax: "`{}` expects a file path as its first argument", function);
        };
        polars_ensure!(
            matches!(first, SQLFunctionArg::Unnamed(_)),
            SQLSyntax: "`{}` expects a file path as its first argument; found: {}", function, first
        );

        let mut options = PlHashMap::with_capacity(rest.len());
        for arg in rest {
            let (name, value) = match arg {
                SQLFunctionArg::Named {
                    name,
                    arg: SQLFunctionArgExpr::Expr(value),
                    ..
                } => (name, value),
                SQLFunctionArg::Named { name, arg, .. } => polars_bail!(
                    SQLSyntax: "`{}` option '{}' expects a value; found: {}",
                    function, name.value.to_lowercase(), arg
                ),
                _ => polars_bail!(
                    SQLSyntax: "`{}` options must be named with '=>' (found: {})", function, arg
                ),
            };
            let key = name.value.to_lowercase();
            if !valid.contains(&key.as_str()) {
                if let Some((_, reason)) = excluded.iter().find(|(name, _)| *name == key) {
                    let reason = if reason.is_empty() {
                        String::new()
                    } else {
                        format!(": {reason}")
                    };
                    polars_bail!(
                        SQLInterface: "`{}` option '{}' is not supported in SQL{}", function, key, reason
                    );
                }
                polars_bail!(
                    SQLInterface: "unknown `{}` option '{}'; valid options are: {}",
                    function, key, valid.join(", ")
                );
            }
            polars_ensure!(
                options.insert(key.clone(), value).is_none(),
                SQLSyntax: "`{}` option '{}' is given more than once", function, key
            );
        }
        Ok((first, Self { function, options }))
    }

    fn get(&self, name: &str) -> Option<&'a SQLExpr> {
        self.options.get(name).copied()
    }

    fn type_error(&self, name: &str, expected: &str, found: &SQLExpr) -> PolarsError {
        polars_err!(
            SQLSyntax: "`{}` option '{}' expects {}; found: {}", self.function, name, expected, found
        )
    }

    pub(crate) fn bool(&self, name: &str) -> PolarsResult<Option<bool>> {
        self.get(name)
            .map(|expr| match literal(expr) {
                Some(SQLValue::Boolean(b)) => Ok(*b),
                _ => Err(self.type_error(name, "a boolean", expr)),
            })
            .transpose()
    }

    #[cfg(feature = "csv")]
    pub(crate) fn usize(&self, name: &str) -> PolarsResult<Option<usize>> {
        self.get(name)
            .map(|expr| self.expect_usize(name, expr))
            .transpose()
    }

    /// An integer, or NULL for "no limit".
    #[cfg(feature = "csv")]
    pub(crate) fn nullable_usize(&self, name: &str) -> PolarsResult<Option<Option<usize>>> {
        self.get(name)
            .map(|expr| match literal(expr) {
                Some(SQLValue::Null) => Ok(None),
                _ => self.expect_usize(name, expr).map(Some),
            })
            .transpose()
    }

    #[cfg(feature = "csv")]
    pub(crate) fn positive_usize(&self, name: &str) -> PolarsResult<Option<NonZeroUsize>> {
        self.get(name)
            .map(|expr| self.expect_number(name, expr, "a positive integer"))
            .transpose()
    }

    /// A positive integer, or NULL for "no limit".
    #[cfg(feature = "json")]
    pub(crate) fn nullable_positive_usize(
        &self,
        name: &str,
    ) -> PolarsResult<Option<Option<NonZeroUsize>>> {
        self.get(name)
            .map(|expr| match literal(expr) {
                Some(SQLValue::Null) => Ok(None),
                _ => self
                    .expect_number(name, expr, "a positive integer or NULL")
                    .map(Some),
            })
            .transpose()
    }

    #[cfg(feature = "csv")]
    fn expect_usize(&self, name: &str, expr: &SQLExpr) -> PolarsResult<usize> {
        self.expect_number(name, expr, "a non-negative integer")
    }

    fn expect_number<T: FromStr>(
        &self,
        name: &str,
        expr: &SQLExpr,
        expected: &str,
    ) -> PolarsResult<T> {
        match literal(expr) {
            Some(SQLValue::Number(n, _)) => n.parse::<T>().ok(),
            _ => None,
        }
        .ok_or_else(|| self.type_error(name, expected, expr))
    }

    /// One of the string `values`, matched case-insensitively, as the value paired with it.
    #[cfg(feature = "csv")]
    pub(crate) fn one_of<T: Copy>(
        &self,
        name: &str,
        values: &[(&str, T)],
    ) -> PolarsResult<Option<T>> {
        self.get(name)
            .map(|expr| {
                string_literal(expr)
                    .and_then(|s| {
                        values
                            .iter()
                            .find(|(value, _)| value.eq_ignore_ascii_case(&s))
                    })
                    .map(|(_, out)| *out)
                    .ok_or_else(|| {
                        let expected: Vec<String> = values
                            .iter()
                            .map(|(value, _)| format!("'{value}'"))
                            .collect();
                        self.type_error(name, &expected.join(" or "), expr)
                    })
            })
            .transpose()
    }

    pub(crate) fn string(&self, name: &str) -> PolarsResult<Option<String>> {
        self.get(name)
            .map(|expr| self.expect_string(name, expr))
            .transpose()
    }

    /// A string, or NULL to unset the option.
    #[cfg(feature = "csv")]
    pub(crate) fn nullable_string(&self, name: &str) -> PolarsResult<Option<Option<String>>> {
        self.get(name)
            .map(|expr| match literal(expr) {
                Some(SQLValue::Null) => Ok(None),
                _ => self.expect_string(name, expr).map(Some),
            })
            .transpose()
    }

    fn expect_string(&self, name: &str, expr: &SQLExpr) -> PolarsResult<String> {
        string_literal(expr).ok_or_else(|| self.type_error(name, "a string", expr))
    }

    /// A single-byte character, or NULL or an empty string (as in Python) to unset the option.
    #[cfg(feature = "csv")]
    pub(crate) fn nullable_byte(&self, name: &str) -> PolarsResult<Option<Option<u8>>> {
        self.nullable_string(name)?
            .map(|s| {
                s.filter(|s| !s.is_empty())
                    .map(|s| self.expect_byte(name, &s))
                    .transpose()
            })
            .transpose()
    }

    #[cfg(feature = "csv")]
    pub(crate) fn byte(&self, name: &str) -> PolarsResult<Option<u8>> {
        self.string(name)?
            .map(|s| self.expect_byte(name, &s))
            .transpose()
    }

    #[cfg(feature = "csv")]
    fn expect_byte(&self, name: &str, s: &str) -> PolarsResult<u8> {
        match s.as_bytes() {
            [b] => Ok(*b),
            _ => {
                let hint = if s.starts_with('\\') {
                    "; escape sequences need an E'' string, such as E'\\t'"
                } else {
                    ""
                };
                polars_bail!(
                    SQLSyntax: "`{}` option '{}' expects a single-byte character; found: '{}'{}",
                    self.function, name, s, hint
                )
            },
        }
    }

    #[cfg(feature = "csv")]
    pub(crate) fn strings(&self, name: &str) -> PolarsResult<Option<Vec<String>>> {
        self.get(name)
            .map(|expr| {
                array_elements(expr)
                    .and_then(|elems| elems.iter().map(string_literal).collect())
                    .ok_or_else(|| self.type_error(name, "a list of strings", expr))
            })
            .transpose()
    }

    /// A `{'key': 'value'}` dictionary of strings, in the order given.
    fn string_dict(&self, name: &str, expr: &SQLExpr) -> PolarsResult<Vec<(String, String)>> {
        let SQLExpr::Dictionary(fields) = expr else {
            return Err(self.type_error(name, "a dictionary such as {'a': 'INT'}", expr));
        };
        let mut seen = PlHashSet::with_capacity(fields.len());
        fields
            .iter()
            .map(|DictionaryField { key, value }| {
                polars_ensure!(
                    seen.insert(key.value.as_str()),
                    SQLSyntax: "`{}` option '{}' has key '{}' more than once", self.function, name, key.value
                );
                let value = string_literal(value).ok_or_else(|| {
                    self.type_error(name, "string values in its dictionary", value)
                })?;
                Ok((key.value.clone(), value))
            })
            .collect()
    }

    /// A dictionary of column name to SQL type name, such as `{'a': 'INT', 'b': 'VARCHAR'}`.
    pub(crate) fn schema(&self, name: &str) -> PolarsResult<Option<Schema>> {
        self.get(name)
            .map(|expr| {
                self.string_dict(name, expr)?
                    .into_iter()
                    .map(|(column, sql_type)| {
                        let dtype = self.parse_dtype(name, &sql_type)?;
                        Ok(Field::new(column.into(), dtype))
                    })
                    .collect()
            })
            .transpose()
    }

    /// A dictionary of column name to SQL type name, or a list of SQL type names by position.
    #[cfg(feature = "csv")]
    pub(crate) fn dtype_overrides(&self, name: &str) -> PolarsResult<Option<DtypeOverrides>> {
        let Some(expr) = self.get(name) else {
            return Ok(None);
        };
        if let Some(elems) = array_elements(expr) {
            let dtypes: Vec<DataType> = elems
                .iter()
                .map(|elem| {
                    let sql_type = string_literal(elem)
                        .ok_or_else(|| self.type_error(name, "a list of type names", expr))?;
                    self.parse_dtype(name, &sql_type)
                })
                .collect::<PolarsResult<_>>()?;
            return Ok(Some(DtypeOverrides::ByPosition(dtypes)));
        }
        Ok(self.schema(name)?.map(DtypeOverrides::ByName))
    }

    /// One string, a list of strings, or a `{'key': 'value'}` dictionary of strings.
    #[cfg(feature = "csv")]
    pub(crate) fn string_values(&self, name: &str) -> PolarsResult<Option<StringValues>> {
        self.get(name)
            .map(|expr| {
                if matches!(expr, SQLExpr::Dictionary(_)) {
                    return self.string_dict(name, expr).map(StringValues::Dict);
                }
                let values = match array_elements(expr) {
                    Some(elems) => elems
                        .iter()
                        .map(string_literal)
                        .collect::<Option<_>>()
                        .map(StringValues::List),
                    None => string_literal(expr).map(StringValues::One),
                };
                values.ok_or_else(|| {
                    self.type_error(name, "a string, a list of strings, or a dictionary", expr)
                })
            })
            .transpose()
    }

    /// Parse a SQL type name such as `'INT'` or `'DECIMAL(10,2)'`, mapping it the way `CAST` does.
    fn parse_dtype(&self, name: &str, sql_type: &str) -> PolarsResult<DataType> {
        let parsed = Parser::new(&GenericDialect)
            .try_with_sql(sql_type)
            .and_then(|mut parser| {
                let dtype = parser.parse_data_type()?;
                parser.expect_token(&sqlparser::tokenizer::Token::EOF)?;
                Ok(dtype)
            })
            .map_err(|_| {
                polars_err!(
                    SQLSyntax: "`{}` option '{}' has an invalid type name: '{}'", self.function, name, sql_type
                )
            })?;
        map_sql_dtype_to_polars(&parsed)
    }
}

/// The value of an option that accepts one string, a list of strings, or a dictionary.
#[cfg(feature = "csv")]
pub(crate) enum StringValues {
    One(String),
    List(Vec<String>),
    Dict(Vec<(String, String)>),
}

fn literal(expr: &SQLExpr) -> Option<&SQLValue> {
    match expr {
        SQLExpr::Value(SQLValueWithSpan { value, .. }) => Some(value),
        _ => None,
    }
}

fn string_literal(expr: &SQLExpr) -> Option<String> {
    match literal(expr)? {
        SQLValue::SingleQuotedString(s) | SQLValue::EscapedStringLiteral(s) => Some(s.clone()),
        _ => None,
    }
}

#[cfg(feature = "csv")]
fn array_elements(expr: &SQLExpr) -> Option<&[SQLExpr]> {
    match expr {
        SQLExpr::Array(SQLArray { elem, .. }) => Some(elem),
        _ => None,
    }
}
