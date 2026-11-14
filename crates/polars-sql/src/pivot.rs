//! The `PIVOT` and `UNPIVOT` FROM items.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_utils::aliases::PlHashSet;
#[cfg(feature = "pivot")]
use polars_utils::format_pl_smallstr;
#[cfg(feature = "pivot")]
use sqlparser::ast::NullInclusion;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Expr as SQLExpr, ExprWithAlias, GroupByExpr, Ident,
    PivotValueSource, Query, SelectItem, SetExpr, Statement, TableFactor, VisitorMut,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::context::relation_name;
use crate::subquery_group_by::is_aggregate_call;
use crate::subquery_pull_up::has_aggregate;
use crate::{SQLContext, dyn_visit};

/// The name the input of a `PIVOT` is registered under while it is aggregated.
const PIVOT_INPUT: &str = "__POLARS_PIVOT_INPUT";

impl SQLContext {
    /// `<table> PIVOT (<aggregates> FOR <column> IN (<values>))`: a row per group of the input
    /// columns the PIVOT doesn't read, and per value and aggregate a column with the aggregate
    /// over the rows of the group whose `<column>` is that value.
    pub(crate) fn execute_pivot(
        &mut self,
        table: &TableFactor,
        aggregates: &[ExprWithAlias],
        value_column: &[SQLExpr],
        value_source: &PivotValueSource,
    ) -> PolarsResult<LazyFrame> {
        let [column] = value_column else {
            polars_bail!(SQLInterface: "PIVOT supports a single column after FOR")
        };
        let PivotValueSource::List(values) = value_source else {
            polars_bail!(SQLInterface: "PIVOT requires a list of values: `IN (<value>, ...)`")
        };
        polars_ensure!(
            !has_aggregate(self, column),
            SQLInterface: "PIVOT does not support an aggregate after FOR, found `{}`", column
        );
        if let Some(value) = values.iter().find(|value| !is_constant(&value.expr)) {
            polars_bail!(SQLInterface: "PIVOT values must be constants, found `{}`", value.expr)
        }
        // The input is read without its name, so its qualifier is taken off column references.
        let relation = relation_name(table);
        let mut input = InputColumns {
            relation: relation.as_ref().map_or(&[], |name| name.parts()),
            read: PlHashSet::new(),
            window: false,
            query_depth: 0,
        };
        let mut column = column.clone();
        let _ = dyn_visit::visit_mut(&mut column, &mut input);
        let mut exprs: Vec<SQLExpr> = aggregates.iter().map(|a| a.expr.clone()).collect();
        for expr in &mut exprs {
            let _ = dyn_visit::visit_mut(expr, &mut input);
        }
        polars_ensure!(!input.window, SQLInterface: "PIVOT does not support window functions");
        let read = input.read;

        self.execute_isolated(|ctx| {
            let (_, mut lf) = ctx.get_table(table)?;
            let schema = ctx.get_frame_schema(&mut lf)?;
            let group_by: Vec<SQLExpr> = schema
                .iter_names()
                .filter(|name| !read.contains(name.as_str()))
                .map(|name| SQLExpr::Identifier(Ident::with_quote('"', name.as_str())))
                .collect();

            let mut projection: Vec<SelectItem> = group_by
                .iter()
                .cloned()
                .map(SelectItem::UnnamedExpr)
                .collect();
            for value in values {
                let value_name = match &value.alias {
                    Some(alias) => alias.value.clone(),
                    None => pivot_value_name(&value.expr),
                };
                let matches =
                    SQLExpr::IsNotDistinctFrom(Box::new(column.clone()), Box::new(value.expr.clone()));
                for (aggregate, expr) in aggregates.iter().zip(&exprs) {
                    let mut expr = expr.clone();
                    let mut filter = FilterAggregates {
                        ctx,
                        matches: &matches,
                        found: false,
                        query_depth: 0,
                    };
                    let _ = dyn_visit::visit_mut(&mut expr, &mut filter);
                    polars_ensure!(
                        filter.found,
                        SQLInterface: "PIVOT expects aggregate functions, found `{}`", aggregate.expr
                    );
                    let name = match (&aggregate.alias, aggregates.len()) {
                        (None, 1) => value_name.clone(),
                        (Some(alias), _) => format!("{value_name}_{}", alias.value),
                        (None, _) => format!("{value_name}_{}", aggregate.expr),
                    };
                    projection.push(SelectItem::ExprWithAlias {
                        expr,
                        alias: Ident::with_quote('"', name),
                    });
                }
            }

            let mut query = select_from_pivot_input();
            let SetExpr::Select(select) = query.body.as_mut() else {
                unreachable!()
            };
            select.projection = projection;
            select.group_by = GroupByExpr::Expressions(group_by, vec![]);
            ctx.register_cte(PIVOT_INPUT, lf);
            ctx.execute_query(&query)
        })
    }

    /// `<table> UNPIVOT [INCLUDE | EXCLUDE NULLS] (<value> FOR <name> IN (<columns>))`: per input
    /// row and listed column, a row with the other columns, the column's name (or alias) as
    /// `<name>` and its value as `<value>`. Rows with a NULL value are left out unless
    /// `INCLUDE NULLS` is given.
    #[cfg(feature = "pivot")]
    pub(crate) fn execute_unpivot(
        &mut self,
        table: &TableFactor,
        value: &SQLExpr,
        name: &Ident,
        columns: &[ExprWithAlias],
        null_inclusion: Option<&NullInclusion>,
    ) -> PolarsResult<LazyFrame> {
        let SQLExpr::Identifier(value_name) = value else {
            polars_bail!(SQLInterface: "UNPIVOT supports a single value column, found `{}`", value)
        };
        let mut lf = self.execute_isolated(|ctx| Ok(ctx.get_table(table)?.1))?;
        let schema = self.get_frame_schema(&mut lf)?;
        let relation = relation_name(table);
        let relation = relation.as_ref().map_or(&[][..], |name| name.parts());

        let mut sources: Vec<PlSmallStr> = Vec::with_capacity(columns.len());
        let mut labels: Vec<PlSmallStr> = Vec::with_capacity(columns.len());
        for column in columns {
            let source = match &column.expr {
                SQLExpr::Identifier(ident) => Some(ident),
                SQLExpr::CompoundIdentifier(parts) => {
                    match &parts[qualifier_len(parts, relation)..] {
                        [ident] => Some(ident),
                        _ => None,
                    }
                },
                expr => {
                    polars_bail!(SQLInterface: "UNPIVOT expects column names, found `{}`", expr)
                },
            };
            let source = source
                .map(|ident| ident.value.as_str())
                .filter(|name| schema.contains(name))
                .ok_or_else(
                    || polars_err!(ColumnNotFound: "UNPIVOT column '{}' was not found", column.expr),
                )?;
            sources.push(source.into());
            labels.push(column.alias.as_ref().map_or(source, |a| &a.value).into());
        }
        // A column can be listed more than once and a label can be any name, so the listed
        // columns are unpivoted as copies with names of their own.
        let copies: Vec<PlSmallStr> = (0..sources.len())
            .map(|i| format_pl_smallstr!("__POLARS_UNPIVOT_{i}"))
            .collect();
        let index: Vec<PlSmallStr> = schema
            .iter_names()
            .filter(|name| !sources.contains(name))
            .cloned()
            .collect();
        let select = index.iter().map(|name| col(name.clone())).chain(
            sources
                .iter()
                .zip(&copies)
                .map(|(source, copy)| col(source.clone()).alias(copy.clone())),
        );
        let strings = |names: &[PlSmallStr]| {
            let names: Vec<&str> = names.iter().map(PlSmallStr::as_str).collect();
            lit(Series::new(PlSmallStr::EMPTY, names))
        };
        let variable = col(name.value.as_str()).replace(strings(&copies), strings(&labels));
        let lf = lf
            .select(select.collect::<Vec<_>>())
            .unpivot(UnpivotArgsDSL {
                on: Some(Selector::ByName {
                    names: copies.into(),
                    strict: true,
                }),
                index: Selector::ByName {
                    names: index.into(),
                    strict: true,
                },
                variable_name: Some(name.value.as_str().into()),
                value_name: Some(value_name.value.as_str().into()),
            })
            .with_column(variable);
        Ok(match null_inclusion {
            Some(NullInclusion::IncludeNulls) => lf,
            _ => lf.filter(col(value_name.value.as_str()).is_not_null()),
        })
    }
}

/// Adds the condition `matches` to the FILTER of each aggregate call outside of subqueries.
struct FilterAggregates<'a> {
    ctx: &'a SQLContext,
    matches: &'a SQLExpr,
    found: bool,
    query_depth: usize,
}

impl VisitorMut for FilterAggregates<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
        self.query_depth += 1;
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
        self.query_depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
        if self.query_depth == 0
            && is_aggregate_call(self.ctx, expr)
            && let SQLExpr::Function(function) = expr
        {
            self.found = true;
            function.filter = Some(Box::new(match function.filter.take() {
                Some(filter) => SQLExpr::BinaryOp {
                    left: filter,
                    op: SQLBinaryOperator::And,
                    right: Box::new(self.matches.clone()),
                },
                None => self.matches.clone(),
            }));
        }
        ControlFlow::Continue(())
    }
}

/// Takes the qualifier naming the PIVOT input off column references outside of subqueries,
/// and collects the columns they read.
struct InputColumns<'a> {
    relation: &'a [String],
    read: PlHashSet<String>,
    window: bool,
    query_depth: usize,
}

impl VisitorMut for InputColumns<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
        self.query_depth += 1;
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
        self.query_depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
        if self.query_depth > 0 {
            return ControlFlow::Continue(());
        }
        if let SQLExpr::CompoundIdentifier(parts) = expr {
            let n = qualifier_len(parts, self.relation);
            if n > 0 {
                let mut rest = parts.split_off(n);
                *expr = match rest.len() {
                    1 => SQLExpr::Identifier(rest.pop().unwrap()),
                    _ => SQLExpr::CompoundIdentifier(rest),
                };
            }
        }
        match expr {
            SQLExpr::Identifier(ident) => {
                self.read.insert(ident.value.clone());
            },
            SQLExpr::CompoundIdentifier(parts) => {
                self.read.insert(parts[0].value.clone());
            },
            SQLExpr::Function(function) if function.over.is_some() => self.window = true,
            _ => {},
        }
        ControlFlow::Continue(())
    }
}

/// The number of leading `parts` that name `relation`: `t` or `ns.t` for a relation `ns.t`.
fn qualifier_len(parts: &[Ident], relation: &[String]) -> usize {
    (1..=relation.len())
        .rev()
        .find(|&n| {
            parts.len() > n
                && parts[..n]
                    .iter()
                    .zip(&relation[relation.len() - n..])
                    .all(|(part, name)| part.value.eq_ignore_ascii_case(name))
        })
        .unwrap_or(0)
}

/// Whether `expr` is a literal, possibly negated or cast.
fn is_constant(expr: &SQLExpr) -> bool {
    match expr {
        SQLExpr::Value(_) | SQLExpr::TypedString(_) | SQLExpr::Interval(_) => true,
        SQLExpr::UnaryOp { expr, .. } | SQLExpr::Nested(expr) | SQLExpr::Cast { expr, .. } => {
            is_constant(expr)
        },
        _ => false,
    }
}

/// The column name a PIVOT value gives: a string as is, else the value as written.
fn pivot_value_name(expr: &SQLExpr) -> String {
    match expr {
        SQLExpr::Value(value) => value
            .value
            .clone()
            .into_string()
            .unwrap_or_else(|| value.to_string()),
        SQLExpr::TypedString(typed) => typed
            .value
            .value
            .clone()
            .into_string()
            .unwrap_or_else(|| expr.to_string()),
        expr => expr.to_string(),
    }
}

/// `SELECT * FROM <PIVOT_INPUT>`.
fn select_from_pivot_input() -> Query {
    let sql = format!("SELECT * FROM {PIVOT_INPUT}");
    let Statement::Query(query) = Parser::parse_sql(&GenericDialect, &sql).unwrap().remove(0)
    else {
        unreachable!()
    };
    *query
}
