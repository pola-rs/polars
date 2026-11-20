//! Query parameters: the placeholders `$1`, `?`, `$name` and `:name`, whose values the
//! caller gives with the query.

use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_utils::format_pl_smallstr;
use sqlparser::ast::{
    Array, DictionaryField, Expr as SQLExpr, GroupByExpr, Ident, OrderByKind, Query, Select,
    Statement, UnaryOperator, Value as SQLValue, ValueWithSpan, Visit, Visitor,
};
use sqlparser::tokenizer::Location;

use crate::dyn_visit;

/// The values of the parameters of a query.
#[derive(Clone, Debug)]
pub enum QueryParams {
    /// The values of `$1`, `$2`, ... (also written `?1` or `:1`), or of each `?` in order.
    Positional(Vec<Scalar>),
    /// The values of `$name` and `:name`, by name.
    Named(PlHashMap<PlSmallStr, Scalar>),
}

impl Default for QueryParams {
    fn default() -> Self {
        Self::Positional(Vec::new())
    }
}

/// The parameter a placeholder refers to.
#[derive(Clone, Copy)]
enum ParamRef<'a> {
    /// `?`: the next parameter.
    Next,
    /// `$1`, `?1` or `:1`.
    Number(usize),
    /// `$name` or `:name`.
    Name(&'a str),
}

fn param_ref(placeholder: &str) -> Option<ParamRef<'_>> {
    let rest = placeholder.strip_prefix(['$', '?', ':'])?;
    let question = placeholder.starts_with('?');
    if rest.is_empty() {
        return question.then_some(ParamRef::Next);
    }
    if rest.bytes().all(|b| b.is_ascii_digit()) {
        return Some(ParamRef::Number(rest.parse().unwrap_or(usize::MAX)));
    }
    (!question).then_some(ParamRef::Name(rest))
}

fn placeholder_ref(expr: &SQLExpr) -> Option<&str> {
    match expr {
        SQLExpr::Value(ValueWithSpan {
            value: SQLValue::Placeholder(p),
            ..
        }) if param_ref(p).is_some() => Some(p),
        _ => None,
    }
}

/// The first query parameter in `node`, if any.
pub(crate) fn first_param(node: &(impl Visit + ?Sized)) -> Option<String> {
    match dyn_visit::visit_expressions(node, |expr| match placeholder_ref(expr) {
        Some(p) => ControlFlow::Break(p.to_string()),
        None => ControlFlow::Continue(()),
    }) {
        ControlFlow::Break(p) => Some(p),
        ControlFlow::Continue(()) => None,
    }
}

/// The SQL literal of `value`, if it has one: as the parser reads it, so a negative number
/// is a negation and a list or struct is built from literals.
fn sql_literal(value: &AnyValue) -> Option<SQLExpr> {
    let number = |s: String| SQLExpr::value(SQLValue::Number(s, false));
    let negate = |expr: SQLExpr| SQLExpr::UnaryOp {
        op: UnaryOperator::Minus,
        expr: Box::new(expr),
    };
    Some(match value {
        AnyValue::Null => SQLExpr::value(SQLValue::Null),
        AnyValue::Boolean(b) => SQLExpr::value(SQLValue::Boolean(*b)),
        AnyValue::String(_) | AnyValue::StringOwned(_) => {
            SQLExpr::value(SQLValue::SingleQuotedString(value.get_str()?.to_string()))
        },
        // the magnitude of i64::MIN is not an Int64 literal
        AnyValue::Int64(v) if *v >= 0 => number(v.to_string()),
        AnyValue::Int64(v) if *v > i64::MIN => negate(number((-v).to_string())),
        AnyValue::Float64(v) if v.is_finite() => match v.is_sign_negative() {
            false => number(format!("{v:e}")),
            true => negate(number(format!("{:e}", -v))),
        },
        AnyValue::List(s) => SQLExpr::Array(Array {
            elem: s.iter().map(|v| sql_literal(&v)).collect::<Option<_>>()?,
            named: false,
        }),
        AnyValue::StructOwned(payload) => {
            let (values, fields) = payload.as_ref();
            SQLExpr::Dictionary(
                values
                    .iter()
                    .zip(fields)
                    .map(|(v, field)| {
                        Some(DictionaryField {
                            key: Ident::with_quote('\'', field.name.as_str()),
                            value: Box::new(sql_literal(v)?),
                        })
                    })
                    .collect::<Option<_>>()?,
            )
        },
        _ => return None,
    })
}

/// The bare parameter in a GROUP BY item, which would be read as a column position.
fn grouping_param(expr: &SQLExpr) -> Option<&str> {
    match expr {
        SQLExpr::Rollup(sets) | SQLExpr::Cube(sets) | SQLExpr::GroupingSets(sets) => {
            sets.iter().flatten().find_map(placeholder_ref)
        },
        _ => placeholder_ref(expr),
    }
}

/// The placeholders of a statement.
#[derive(Default)]
struct Placeholders {
    /// Where each `?` is.
    next: Vec<Location>,
    numbered: bool,
    named: bool,
}

impl Visitor for Placeholders {
    type Break = PolarsError;

    fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<PolarsError> {
        if let Some(order_by) = &query.order_by
            && let OrderByKind::Expressions(items) = &order_by.kind
            && let Some(p) = items.iter().find_map(|item| placeholder_ref(&item.expr))
        {
            return ControlFlow::Break(
                polars_err!(SQLSyntax: "query parameter {} cannot be an ORDER BY item", p),
            );
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_select(&mut self, select: &Select) -> ControlFlow<PolarsError> {
        if let GroupByExpr::Expressions(items, _) = &select.group_by
            && let Some(p) = items.iter().find_map(grouping_param)
        {
            return ControlFlow::Break(
                polars_err!(SQLSyntax: "query parameter {} cannot be a GROUP BY item", p),
            );
        }
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<PolarsError> {
        if let SQLExpr::Value(ValueWithSpan {
            value: SQLValue::Placeholder(p),
            span,
        }) = expr
        {
            match param_ref(p) {
                Some(ParamRef::Next) => self.next.push(span.start),
                Some(ParamRef::Number(_)) => self.numbered = true,
                Some(ParamRef::Name(_)) => self.named = true,
                None => {},
            }
        }
        ControlFlow::Continue(())
    }
}

/// Bind the parameters of `stmt` to `params`. A value with an SQL literal form replaces its
/// placeholder. The other placeholders become `$<number>` or `$<name>`, and are returned with
/// their values. Only placeholders in expressions are parameters. Returns `None` if `stmt`
/// has no parameters.
pub(crate) fn bind_params(
    stmt: &mut Statement,
    params: &QueryParams,
) -> PolarsResult<Option<PlHashMap<PlSmallStr, Scalar>>> {
    let mut found = Placeholders::default();
    if let ControlFlow::Break(err) = dyn_visit::visit(stmt, &mut found) {
        return Err(err);
    }
    if found.next.is_empty() && !found.numbered && !found.named {
        check_no_params(params)?;
        return Ok(None);
    }
    polars_ensure!(
        !matches!(stmt, Statement::CreateView(_)),
        SQLInterface: "query parameters are not supported in CREATE VIEW"
    );
    polars_ensure!(
        found.next.is_empty() || !found.numbered,
        SQLSyntax: "cannot mix '?' with numbered query parameters"
    );
    found.next.sort_unstable();
    found.next.dedup();
    if let QueryParams::Positional(values) = params
        && !found.next.is_empty()
    {
        polars_ensure!(
            found.next.len() == values.len(),
            SQLInterface: "the query has {} '?' parameters, but {} values were given",
            found.next.len(), values.len()
        );
    }

    let n_numbers = match params {
        QueryParams::Positional(values) => values.len(),
        QueryParams::Named(_) => 0,
    };
    let mut binder = Binder {
        params,
        next: &found.next,
        used_numbers: vec![false; n_numbers],
        used_names: PlHashSet::new(),
        bound: PlHashMap::new(),
    };
    if let ControlFlow::Break(err) =
        dyn_visit::visit_expressions_mut(stmt, |expr| match binder.bind(expr) {
            Ok(()) => ControlFlow::Continue(()),
            Err(err) => ControlFlow::Break(err),
        })
    {
        return Err(err);
    }
    check_all_used(params, &binder.used_numbers, &binder.used_names)?;
    Ok(Some(binder.bound))
}

/// An error if `params` has a value, for a statement that has no parameters.
pub(crate) fn check_no_params(params: &QueryParams) -> PolarsResult<()> {
    check_all_used(params, &[], &PlHashSet::new())
}

fn check_all_used(
    params: &QueryParams,
    used_numbers: &[bool],
    used_names: &PlHashSet<&str>,
) -> PolarsResult<()> {
    match params {
        QueryParams::Positional(values) => {
            if let Some(i) =
                (0..values.len()).find(|&i| !used_numbers.get(i).copied().unwrap_or(false))
            {
                polars_bail!(SQLInterface: "query parameter ${} is not used in the query", i + 1);
            }
        },
        QueryParams::Named(values) => {
            let mut unused: Vec<&str> = values
                .keys()
                .map(|k| k.as_str())
                .filter(|k| !used_names.contains(k))
                .collect();
            unused.sort_unstable();
            if let Some(name) = unused.first() {
                polars_bail!(SQLInterface: "query parameter ${} is not used in the query", name);
            }
        },
    }
    Ok(())
}

struct Binder<'a> {
    params: &'a QueryParams,
    next: &'a [Location],
    used_numbers: Vec<bool>,
    used_names: PlHashSet<&'a str>,
    bound: PlHashMap<PlSmallStr, Scalar>,
}

impl<'a> Binder<'a> {
    /// Replace the placeholder `expr` by the literal of its value, or by `$<number>` or
    /// `$<name>` if the value has no literal.
    fn bind(&mut self, expr: &mut SQLExpr) -> PolarsResult<()> {
        let SQLExpr::Value(ValueWithSpan {
            value: SQLValue::Placeholder(p),
            span,
        }) = expr
        else {
            return Ok(());
        };
        let Some(param) = param_ref(p) else {
            return Ok(());
        };
        let (name, value) = self.resolve(p, param, span.start)?;
        *expr = match sql_literal(value.value()) {
            Some(literal) => literal,
            None => {
                let placeholder = SQLValue::Placeholder(name.to_string()).with_span(*span);
                self.bound.insert(name, value.clone());
                SQLExpr::Value(placeholder)
            },
        };
        Ok(())
    }

    /// The name the placeholder `p` at `location` gets, and its value.
    fn resolve(
        &mut self,
        p: &str,
        param: ParamRef,
        location: Location,
    ) -> PolarsResult<(PlSmallStr, &'a Scalar)> {
        match (self.params, param) {
            (QueryParams::Positional(values), ParamRef::Next) => {
                let i = self.next.binary_search(&location).unwrap();
                self.used_numbers[i] = true;
                Ok((format_pl_smallstr!("${}", i + 1), &values[i]))
            },
            (QueryParams::Positional(values), ParamRef::Number(n)) => {
                polars_ensure!(n >= 1, SQLSyntax: "query parameter {} is invalid; numbering starts at 1", p);
                let value = values.get(n - 1).ok_or_else(
                    || polars_err!(SQLInterface: "no value given for query parameter {}", p),
                )?;
                self.used_numbers[n - 1] = true;
                Ok((format_pl_smallstr!("${}", n), value))
            },
            (QueryParams::Named(values), ParamRef::Name(name)) => {
                let (key, value) = values.get_key_value(name).ok_or_else(
                    || polars_err!(SQLInterface: "no value given for query parameter {}", p),
                )?;
                self.used_names.insert(key.as_str());
                Ok((format_pl_smallstr!("${}", name), value))
            },
            (QueryParams::Positional(_), ParamRef::Name(_)) => {
                polars_bail!(SQLInterface: "query parameter {} is named, but positional values were given", p)
            },
            (QueryParams::Named(_), _) => {
                polars_bail!(SQLInterface: "query parameter {} is positional, but named values were given", p)
            },
        }
    }
}
