//! The `PIVOT` and `UNPIVOT` FROM items.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_utils::aliases::PlHashSet;
#[cfg(feature = "pivot")]
use sqlparser::ast::NullInclusion;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Expr as SQLExpr, ExprWithAlias, GroupByExpr, Ident,
    PivotValueSource, Query, SelectItem, SetExpr, Statement, TableAlias, TableFactor, VisitorMut,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

use crate::SQLContext;
use crate::dyn_visit::{self, visit_expressions};
use crate::subquery_group_by::is_aggregate_call;

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
        self.execute_isolated(|ctx| {
            let (name, mut lf) = ctx.get_table(table)?;
            let schema = ctx.get_frame_schema(&mut lf)?;

            let mut read = PlHashSet::new();
            for expr in aggregates.iter().map(|a| &a.expr).chain([column]) {
                let _ = visit_expressions(expr, |e| {
                    match e {
                        SQLExpr::Identifier(ident) => {
                            read.insert(ident.value.clone());
                        },
                        SQLExpr::CompoundIdentifier(parts) => {
                            read.extend(parts.iter().map(|ident| ident.value.clone()));
                        },
                        _ => {},
                    }
                    ControlFlow::<()>::Continue(())
                });
            }
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
                for aggregate in aggregates {
                    let mut expr = aggregate.expr.clone();
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
            if let TableFactor::Table { alias, .. } = &mut select.from[0].relation
                && !name.is_empty()
            {
                *alias = Some(TableAlias {
                    explicit: true,
                    name: Ident::new(name),
                    columns: vec![],
                    at: None,
                });
            }
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

        let mut on: Vec<PlSmallStr> = Vec::with_capacity(columns.len());
        let mut labels: Vec<PlSmallStr> = Vec::with_capacity(columns.len());
        for column in columns {
            let column_name = match &column.expr {
                SQLExpr::Identifier(ident) => &ident.value,
                SQLExpr::CompoundIdentifier(parts) => &parts.last().unwrap().value,
                expr => {
                    polars_bail!(SQLInterface: "UNPIVOT expects column names, found `{}`", expr)
                },
            };
            polars_ensure!(
                schema.contains(column_name),
                ColumnNotFound: "UNPIVOT column '{}' was not found", column_name
            );
            on.push(column_name.into());
            labels.push(
                column
                    .alias
                    .as_ref()
                    .map_or(column_name, |a| &a.value)
                    .into(),
            );
        }
        let index: Vec<PlSmallStr> = schema
            .iter_names()
            .filter(|name| !on.contains(name))
            .cloned()
            .collect();
        let lf = lf.rename(&on, &labels, true).unpivot(UnpivotArgsDSL {
            on: Some(Selector::ByName {
                names: labels.into(),
                strict: true,
            }),
            index: Selector::ByName {
                names: index.into(),
                strict: true,
            },
            variable_name: Some(name.value.as_str().into()),
            value_name: Some(value_name.value.as_str().into()),
        });
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
