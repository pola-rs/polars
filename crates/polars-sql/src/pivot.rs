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
    BinaryOperator as SQLBinaryOperator, CastKind, DataType as SQLDataType, Expr as SQLExpr,
    ExprWithAlias, GroupByExpr, Ident, PivotValueSource, Query, SelectItem, SetExpr, Statement,
    TableFactor, Value as SQLValue, Visitor, VisitorMut,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;

#[cfg(feature = "pivot")]
use crate::col;
use crate::context::{RelationName, factor_relations, match_qualifier};
use crate::functions::reads_no_input;
use crate::sql_expr::parse_sql_expr;
use crate::subquery_group_by::is_aggregate_call;
use crate::subquery_pull_up::has_aggregate;
use crate::{SQLContext, dyn_visit};

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
        let relations = factor_relations(std::iter::once(table), self.catalog_spellings());
        let mut input = InputColumns {
            relations: &relations,
            read: PlHashSet::new(),
            window: false,
            subquery: false,
            query_depth: 0,
        };
        for aggregate in aggregates {
            let _ = dyn_visit::visit(&aggregate.expr, &mut input);
        }
        polars_ensure!(!input.subquery, SQLInterface: "PIVOT aggregates cannot contain subqueries");
        let _ = dyn_visit::visit(column, &mut input);
        polars_ensure!(!input.window, SQLInterface: "PIVOT does not support window functions");
        let read = input.read;

        self.execute_isolated(|ctx| {
            let (name, mut lf) = ctx.get_table(table)?;
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
            let mut filters = BooleanFilters {
                ctx: ctx.isolated(),
                schema: &schema,
                query_depth: 0,
            };
            let exprs: Vec<SQLExpr> = aggregates
                .iter()
                .map(|aggregate| {
                    let mut expr = aggregate.expr.clone();
                    let _ = dyn_visit::visit_mut(&mut expr, &mut filters);
                    expr
                })
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

            ctx.planned_from_item = Some((table.clone(), (name, lf)));
            ctx.execute_query(&aggregate_query(table, projection, group_by))
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
        let relations = factor_relations(std::iter::once(table), self.catalog_spellings());

        let mut sources: Vec<PlSmallStr> = Vec::with_capacity(columns.len());
        let mut labels: Vec<PlSmallStr> = Vec::with_capacity(columns.len());
        for column in columns {
            let source = match &column.expr {
                SQLExpr::Identifier(ident) => Some(ident),
                SQLExpr::CompoundIdentifier(parts) => {
                    match &parts[qualifier_len(parts, &relations)..] {
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

/// Adds the condition `matches` to the FILTER of each aggregate call outside of subqueries. It
/// does so after visiting the call, so the added condition itself is not visited.
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

    fn post_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
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

/// Casts the FILTER of each aggregate call outside of subqueries to boolean when it reads no
/// input, as FILTER does on its own; the PIVOT adds a condition to it.
struct BooleanFilters<'a> {
    ctx: SQLContext,
    schema: &'a Schema,
    query_depth: usize,
}

impl VisitorMut for BooleanFilters<'_> {
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
            && is_aggregate_call(&self.ctx, expr)
            && let SQLExpr::Function(function) = expr
            && let Some(filter) = &mut function.filter
            && parse_sql_expr(filter, &mut self.ctx, Some(self.schema))
                .and_then(|pred| self.ctx.with_typed_subqueries(&pred))
                .is_ok_and(|pred| reads_no_input(&pred))
        {
            let filter = filter.as_mut();
            *filter = SQLExpr::Cast {
                kind: CastKind::Cast,
                expr: Box::new(std::mem::replace(filter, SQLExpr::value(SQLValue::Null))),
                data_type: SQLDataType::Boolean,
                format: None,
            };
        }
        ControlFlow::Continue(())
    }
}

/// Collects the input columns read outside of subqueries, and whether a window function or a
/// subquery is used.
struct InputColumns<'a> {
    relations: &'a [RelationName],
    read: PlHashSet<String>,
    window: bool,
    subquery: bool,
    query_depth: usize,
}

impl Visitor for InputColumns<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
        self.query_depth += 1;
        self.subquery = true;
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
        self.query_depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
        if self.query_depth > 0 {
            return ControlFlow::Continue(());
        }
        match expr {
            SQLExpr::Identifier(ident) => {
                self.read.insert(ident.value.clone());
            },
            SQLExpr::CompoundIdentifier(parts) => {
                let column = &parts[qualifier_len(parts, self.relations)];
                self.read.insert(column.value.clone());
            },
            SQLExpr::Function(function) if function.over.is_some() => self.window = true,
            _ => {},
        }
        ControlFlow::Continue(())
    }
}

/// The number of leading `parts` that name one of `relations`.
fn qualifier_len(parts: &[Ident], relations: &[RelationName]) -> usize {
    match_qualifier(parts, relations.iter()).map_or(0, |(_, len)| len)
}

/// Whether `expr` is a literal, possibly negated or cast.
fn is_constant(expr: &SQLExpr) -> bool {
    match expr {
        SQLExpr::Value(_) | SQLExpr::TypedString(_) => true,
        SQLExpr::Interval(interval) => is_constant(&interval.value),
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

/// `SELECT <projection> FROM <table> GROUP BY <group_by>`.
fn aggregate_query(
    table: &TableFactor,
    projection: Vec<SelectItem>,
    group_by: Vec<SQLExpr>,
) -> Query {
    let Statement::Query(mut query) = Parser::parse_sql(&GenericDialect, "SELECT * FROM t")
        .unwrap()
        .remove(0)
    else {
        unreachable!()
    };
    let SetExpr::Select(select) = query.body.as_mut() else {
        unreachable!()
    };
    select.from[0].relation = table.clone();
    select.projection = projection;
    select.group_by = GroupByExpr::Expressions(group_by, vec![]);
    *query
}
