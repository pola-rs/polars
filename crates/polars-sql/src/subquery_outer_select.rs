//! A subquery whose only SELECT item reads columns of the queries around it, and none
//! of its own relations, has that value on every row. The item is moved out of the
//! subquery into the query around it, which can read those columns:
//!
//! ```sql
//! lhs IN (SELECT f FROM r WHERE c)
//! -- CASE WHEN EXISTS (SELECT 1 FROM r WHERE c) THEN lhs = f ELSE FALSE END
//! (SELECT f FROM r WHERE c)
//! -- CASE WHEN (SELECT 1 FROM r WHERE c) IS NOT NULL THEN f END
//! ```
//!
//! An `IN` over no rows is false; over rows that all hold `f` it is `lhs = f`. The
//! scalar `(SELECT 1 ...)` has no row, one row or more rows exactly when the subquery
//! has, so it is NULL, `f` or raises in the same cases.
//!
//! A scalar subquery whose only FROM item is a derived table that reads columns of the
//! queries around it is first merged with that table. It reads the table's FROM and
//! WHERE itself, with the table's SELECT items in place of its columns:
//!
//! ```sql
//! (SELECT MAX(d.v) FROM (SELECT x AS v FROM r WHERE r.k = o.k) AS d WHERE c)
//! -- (SELECT MAX(x) FROM r WHERE r.k = o.k AND c')
//! ```
use std::ops::ControlFlow;

use polars_core::prelude::*;
use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, CaseWhen, Distinct, Expr as SQLExpr, GroupByExpr, Query,
    Select, SelectItem, SetExpr, TableFactor, TableWithJoins, UnaryOperator as SQLUnaryOperator,
    Value as SQLValue, VisitorMut,
};

use crate::SQLContext;
use crate::context::{RelationName, combine_conditions, match_qualifier};
use crate::dyn_visit::{self, visit_expressions};
use crate::functions::PolarsSQLFunctions;
use crate::sql_visitors::{
    contains_random, expr_contains_subquery, expr_has_window_functions, expr_references_any_column,
    has_column_selector, is_column_selector,
};
use crate::subquery::{eligible_query_block, strip_nested};
use crate::subquery_group_by::is_count_star;
use crate::subquery_pull_up::has_aggregate;

impl SQLContext {
    /// `select` with the subqueries of its clauses whose SELECT item reads only columns of
    /// queries around them rewritten to read those columns here, and its scalar subqueries
    /// merged with a derived table as in the module docs, or `None` if there are none. The
    /// subqueries of its derived tables are left to those tables.
    pub(crate) fn hoist_outer_select_items(&self, select: &Select) -> Option<Select> {
        let has_candidate = visit_expressions(select, |e| match e {
            SQLExpr::Subquery(_) | SQLExpr::InSubquery { .. } => ControlFlow::Break(()),
            _ => ControlFlow::Continue(()),
        });
        if has_candidate.is_continue() {
            return None;
        }
        struct Hoist<'a> {
            ctx: &'a SQLContext,
            query_depth: usize,
            // Per visited expression: whether it hid its quantified subquery.
            skips: Vec<bool>,
            changed: bool,
        }

        impl VisitorMut for Hoist<'_> {
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
                // The subquery of `x > ANY (subquery)` is a set of values, not a scalar.
                let skip = self.query_depth == 0
                    && matches!(
                        expr,
                        SQLExpr::AnyOp { right, .. } | SQLExpr::AllOp { right, .. }
                            if matches!(strip_nested(right), SQLExpr::Subquery(_))
                    );
                if skip && let SQLExpr::AnyOp { left, .. } | SQLExpr::AllOp { left, .. } = expr {
                    dyn_visit::visit_mut(left.as_mut(), self)?;
                    self.query_depth += 1;
                }
                self.skips.push(skip);
                if self.query_depth == 0
                    && let SQLExpr::Subquery(subquery) = expr
                    && let Some(merged) = self.ctx.merge_outer_derived_table(subquery)
                {
                    **subquery = merged;
                    self.changed = true;
                }
                if self.query_depth == 0
                    && let Some(hoisted) = self.ctx.hoist_outer_select_item(expr)
                {
                    *expr = hoisted;
                    self.changed = true;
                }
                ControlFlow::Continue(())
            }

            fn post_visit_expr(&mut self, _expr: &mut SQLExpr) -> ControlFlow<()> {
                if self.skips.pop() == Some(true) {
                    self.query_depth -= 1;
                }
                ControlFlow::Continue(())
            }

            // The expressions of a PIVOT read its input, not the relations of this select.
            fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
                if matches!(factor, TableFactor::Pivot { .. }) {
                    self.query_depth += 1;
                }
                ControlFlow::Continue(())
            }

            fn post_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<()> {
                if matches!(factor, TableFactor::Pivot { .. }) {
                    self.query_depth -= 1;
                }
                ControlFlow::Continue(())
            }
        }

        let mut select = select.clone();
        let mut hoist = Hoist {
            ctx: self,
            query_depth: 0,
            skips: Vec::new(),
            changed: false,
        };
        let _ = dyn_visit::visit_mut(&mut select, &mut hoist);
        hoist.changed.then_some(select)
    }

    /// `expr` rewritten as in the module docs when it is `[NOT] IN` or a scalar subquery
    /// whose SELECT item reads only columns of queries around it.
    fn hoist_outer_select_item(&self, expr: &SQLExpr) -> Option<SQLExpr> {
        match expr {
            SQLExpr::InSubquery {
                expr: lhs,
                subquery,
                negated,
            } => {
                let (item, exists_query) = self.outer_select_item(subquery)?;
                // The item is only evaluated where the subquery has rows.
                let in_set = when_then(
                    SQLExpr::Exists {
                        subquery: Box::new(exists_query),
                        negated: false,
                    },
                    SQLExpr::BinaryOp {
                        left: lhs.clone(),
                        op: SQLBinaryOperator::Eq,
                        right: Box::new(SQLExpr::Nested(Box::new(item))),
                    },
                    Some(SQLExpr::value(SQLValue::Boolean(false))),
                );
                Some(match negated {
                    true => SQLExpr::UnaryOp {
                        op: SQLUnaryOperator::Not,
                        expr: Box::new(in_set),
                    },
                    false => in_set,
                })
            },
            SQLExpr::Subquery(subquery) => {
                let (item, row_query) = self.outer_select_item(subquery)?;
                Some(when_then(
                    SQLExpr::IsNotNull(Box::new(SQLExpr::Subquery(Box::new(row_query)))),
                    SQLExpr::Nested(Box::new(item)),
                    None,
                ))
            },
            _ => None,
        }
    }

    /// The SELECT item of `subquery` if it reads only columns of queries around it, with
    /// the subquery that selects `1` in its place.
    fn outer_select_item(&self, subquery: &Query) -> Option<(SQLExpr, Query)> {
        if subquery.with.is_some() {
            return None;
        }
        let SetExpr::Select(select) = subquery.body.as_ref() else {
            return None;
        };
        let [SelectItem::UnnamedExpr(item) | SelectItem::ExprWithAlias { expr: item, .. }] =
            select.projection.as_slice()
        else {
            return None;
        };
        if !expr_references_any_column(item)
            || expr_contains_subquery(item)
            || expr_has_window_functions(item)
            || has_aggregate(self, item)
            || has_column_selector(item)
            || calls_query_function(self, item)
        {
            return None;
        }
        // A grouping set null-extends the item on some rows.
        let has_grouping_sets = match &select.group_by {
            GroupByExpr::All(modifiers) => !modifiers.is_empty(),
            GroupByExpr::Expressions(exprs, modifiers) => {
                !modifiers.is_empty()
                    || exprs.iter().any(|e| {
                        matches!(
                            strip_nested(e),
                            SQLExpr::Rollup(_) | SQLExpr::Cube(_) | SQLExpr::GroupingSets(_)
                        )
                    })
            },
        };
        if has_grouping_sets {
            return None;
        }

        let (inner_names, inner_schema) = if select.from.is_empty() {
            // Without FROM only a plain `SELECT f` is moved: it has exactly one row.
            let filters_rows = select.selection.is_some()
                || select.having.is_some()
                || select.qualify.is_some()
                || subquery.limit_clause.is_some()
                || subquery.fetch.is_some();
            if filters_rows {
                return None;
            }
            (Vec::new(), SchemaRef::default())
        } else {
            // An error leaves the subquery to fail as it did.
            let (names, _, schema) = self.isolated().resolve_subquery_from(select).ok()??;
            (names, schema)
        };
        if reads_inner_column(item, &inner_names, &inner_schema) {
            return None;
        }

        let mut row_query = subquery.clone();
        let SetExpr::Select(row_select) = row_query.body.as_mut() else {
            unreachable!()
        };
        let (SelectItem::UnnamedExpr(slot) | SelectItem::ExprWithAlias { expr: slot, .. }) =
            &mut row_select.projection[0]
        else {
            unreachable!()
        };
        *slot = SQLExpr::value(SQLValue::Number("1".into(), false));
        // Another clause reading the alias would read `1` instead.
        if let SelectItem::ExprWithAlias { alias, .. } = &select.projection[0]
            && visit_expressions(&row_query, |e| match e {
                SQLExpr::Identifier(ident) if ident.value.eq_ignore_ascii_case(&alias.value) => {
                    ControlFlow::Break(())
                },
                _ => ControlFlow::Continue(()),
            })
            .is_break()
        {
            return None;
        }
        Some((item.clone(), row_query))
    }

    /// `subquery` merged with its only FROM item as in the module docs, or `None` if that
    /// item is not a derived table that reads a relation of a query further out, or either
    /// of them has another shape. The subquery may only read the derived table's columns,
    /// and each of those must be a plain expression of a column.
    fn merge_outer_derived_table(&self, subquery: &Query) -> Option<Query> {
        let select = eligible_query_block(subquery)?;
        let [
            TableWithJoins {
                relation:
                    TableFactor::Derived {
                        subquery: derived,
                        alias,
                        sample: None,
                        ..
                    },
                joins,
            },
        ] = select.from.as_slice()
        else {
            return None;
        };
        // ORDER BY and DISTINCT ON may read a SELECT item rather than a column.
        if !joins.is_empty()
            || subquery.order_by.is_some()
            || matches!(select.distinct, Some(Distinct::On(_)))
            || select.exclude.is_some()
            || !select.named_window.is_empty()
            || alias.as_ref().is_some_and(|a| !a.columns.is_empty())
            || contains_random(subquery, true)
            || has_unchecked_clauses(subquery, select)
            || !select.projection.iter().all(|item| {
                matches!(
                    item,
                    SelectItem::UnnamedExpr(_) | SelectItem::ExprWithAlias { .. }
                )
            })
        {
            return None;
        }
        let derived_select = eligible_query_block(derived)?;
        if derived.order_by.is_some()
            || derived.limit_clause.is_some()
            || derived.fetch.is_some()
            || derived_select.distinct.is_some()
            || derived_select.exclude.is_some()
            || !derived_select.named_window.is_empty()
            || has_unchecked_clauses(derived, derived_select)
        {
            return None;
        }
        let mut items: Vec<(&str, &SQLExpr)> = Vec::new();
        let mut aliases: Vec<&str> = Vec::new();
        for item in &derived_select.projection {
            let (name, expr) = match item {
                SelectItem::ExprWithAlias { expr, alias } => {
                    aliases.push(alias.value.as_str());
                    (alias.value.as_str(), expr)
                },
                SelectItem::UnnamedExpr(expr @ SQLExpr::Identifier(ident)) => {
                    (ident.value.as_str(), expr)
                },
                SelectItem::UnnamedExpr(expr @ SQLExpr::CompoundIdentifier(parts)) => {
                    (parts.last()?.value.as_str(), expr)
                },
                _ => return None,
            };
            // A column's type, not a literal's, decides the type of the item.
            if items.iter().any(|(n, _)| *n == name)
                || !is_plain_column_expr(expr)
                || !expr_references_any_column(expr)
            {
                return None;
            }
            items.push((name, expr));
        }
        // An item may read the alias of another one, which is not kept.
        let reads_alias = items.iter().any(|(own_name, expr)| {
            visit_expressions(*expr, |e| {
                let name = match e {
                    SQLExpr::Identifier(ident) => ident,
                    SQLExpr::CompoundIdentifier(parts) => &parts[0],
                    _ => return ControlFlow::Continue(()),
                };
                if name.value != *own_name && aliases.contains(&name.value.as_str()) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .is_break()
        });
        if reads_alias {
            return None;
        }

        // Only a derived table that reads a relation it does not declare, rather than a
        // field of one of its struct columns, could not be planned on its own.
        if derived_select
            .selection
            .as_ref()
            .is_some_and(expr_contains_subquery)
        {
            return None;
        }
        let (names, schema) = if derived_select.from.is_empty() {
            (Vec::new(), SchemaRef::default())
        } else {
            let mut from_only = derived_select.clone();
            from_only.projection.clear();
            from_only.selection = None;
            let (names, _, schema) = self.isolated().resolve_subquery_from(&from_only).ok()??;
            (names, schema)
        };
        let reads_outer = |expr: &SQLExpr| {
            visit_expressions(expr, |e| match e {
                SQLExpr::CompoundIdentifier(parts)
                    if match_qualifier(parts, names.iter()).is_none()
                        && !schema.contains(&parts[0].value) =>
                {
                    ControlFlow::Break(())
                },
                _ => ControlFlow::Continue(()),
            })
            .is_break()
        };
        if !items.iter().any(|(_, expr)| reads_outer(expr))
            && !derived_select.selection.as_ref().is_some_and(reads_outer)
        {
            return None;
        }

        // The clauses of the query block read the derived table, and LIMIT is left as is.
        let mut merged = subquery.clone();
        let SetExpr::Select(merged_select) = merged.body.as_mut() else {
            unreachable!()
        };
        merged_select.from.clear();
        let alias = alias
            .as_ref()
            .map(|a| RelationName::new(vec![a.name.value.clone()]));
        let mut read_items = ReadItems {
            items: &items,
            alias: alias.as_ref(),
            read: vec![false; items.len()],
        };
        // An item that is not read would not be checked.
        if dyn_visit::visit_mut(merged_select.as_mut(), &mut read_items).is_break()
            || read_items.read.contains(&false)
        {
            return None;
        }
        // An aggregate of an item that reads no column of the derived table's relations
        // would aggregate a query further out.
        if has_aggregate(self, merged_select.as_ref())
            && items
                .iter()
                .any(|(_, expr)| !reads_inner_column(expr, &names, &schema))
        {
            return None;
        }
        merged_select.from = derived_select.from.clone();
        merged_select.selection = combine_conditions(
            derived_select
                .selection
                .iter()
                .cloned()
                .chain(merged_select.selection.take())
                .collect(),
            SQLBinaryOperator::And,
        );
        Some(merged)
    }
}

pub(crate) fn when_then(
    condition: SQLExpr,
    result: SQLExpr,
    else_result: Option<SQLExpr>,
) -> SQLExpr {
    SQLExpr::Case {
        case_token: AttachedToken::empty(),
        end_token: AttachedToken::empty(),
        operand: None,
        conditions: vec![CaseWhen { condition, result }],
        else_result: else_result.map(Box::new),
    }
}

/// Whether `expr` calls a function whose value depends on the query it is in: `UNNEST`,
/// which can return several rows for one, `GROUPING` / `GROUPING_ID`, or a ranking or
/// offset function, which may be called without OVER.
pub(crate) fn calls_query_function(ctx: &SQLContext, expr: &SQLExpr) -> bool {
    visit_expressions(expr, |e| match e {
        SQLExpr::Function(function)
            if matches!(
                PolarsSQLFunctions::try_from_sql(function, ctx),
                Ok(PolarsSQLFunctions::Explode
                    | PolarsSQLFunctions::Grouping
                    | PolarsSQLFunctions::GroupingId
                    | PolarsSQLFunctions::RowNumber
                    | PolarsSQLFunctions::Rank
                    | PolarsSQLFunctions::DenseRank
                    | PolarsSQLFunctions::PercentRank
                    | PolarsSQLFunctions::CumeDist
                    | PolarsSQLFunctions::Ntile
                    | PolarsSQLFunctions::Lag
                    | PolarsSQLFunctions::Lead)
            ) =>
        {
            ControlFlow::Break(())
        },
        _ => ControlFlow::Continue(()),
    })
    .is_break()
}

/// Whether `expr` is built from columns and literals by operators and casts only.
fn is_plain_column_expr(expr: &SQLExpr) -> bool {
    match expr {
        SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_) | SQLExpr::Value(_) => true,
        SQLExpr::Nested(expr)
        | SQLExpr::UnaryOp { expr, .. }
        | SQLExpr::Cast { expr, .. }
        | SQLExpr::IsNull(expr)
        | SQLExpr::IsNotNull(expr) => is_plain_column_expr(expr),
        SQLExpr::BinaryOp { left, right, .. } => {
            is_plain_column_expr(left) && is_plain_column_expr(right)
        },
        _ => false,
    }
}

/// Whether `query` has a clause that is not checked once `select` has been decorrelated.
fn has_unchecked_clauses(query: &Query, select: &Select) -> bool {
    !query.locks.is_empty()
        || query.format_clause.is_some()
        || !select.cluster_by.is_empty()
        || !select.distribute_by.is_empty()
        || !select.sort_by.is_empty()
}

/// Replaces the columns of a derived table in the query that reads it by the items that
/// define them, and marks in `read` which items. Breaks on any other name, a column
/// selector, a window function, a nested query or a lambda.
struct ReadItems<'a> {
    items: &'a [(&'a str, &'a SQLExpr)],
    alias: Option<&'a RelationName>,
    read: Vec<bool>,
}

impl VisitorMut for ReadItems<'_> {
    type Break = ();

    fn pre_visit_query(&mut self, _query: &mut Query) -> ControlFlow<()> {
        ControlFlow::Break(())
    }

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
        let window = matches!(expr, SQLExpr::Function(f) if f.over.is_some());
        if window
            || matches!(expr, SQLExpr::Lambda(_))
            || (is_column_selector(expr) && !is_count_star(expr))
        {
            return ControlFlow::Break(());
        }
        ControlFlow::Continue(())
    }

    fn post_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
        let column = match expr {
            SQLExpr::Identifier(column) => column,
            SQLExpr::CompoundIdentifier(parts) => match parts.as_slice() {
                [relation, column]
                    if self
                        .alias
                        .is_some_and(|a| a.named_by(std::slice::from_ref(relation), true)) =>
                {
                    column
                },
                _ => return ControlFlow::Break(()),
            },
            _ => return ControlFlow::Continue(()),
        };
        let Some(index) = self.items.iter().position(|(n, _)| *n == column.value) else {
            return ControlFlow::Break(());
        };
        let item = self.items[index].1;
        *expr = match item {
            SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_) => item.clone(),
            _ => SQLExpr::Nested(Box::new(item.clone())),
        };
        self.read[index] = true;
        ControlFlow::Continue(())
    }
}

/// Whether `expr` reads a column of the relations `inner_names`, whose columns are
/// `inner_schema`. An unqualified name binds to them when they have it; a qualified one
/// when its qualifier names one of them or its first part is one of their (struct)
/// columns.
pub(crate) fn reads_inner_column(
    expr: &SQLExpr,
    inner_names: &[RelationName],
    inner_schema: &Schema,
) -> bool {
    visit_expressions(expr, |e| {
        let inner = match e {
            SQLExpr::Identifier(ident) => inner_schema.contains(&ident.value),
            SQLExpr::CompoundIdentifier(parts) => {
                match_qualifier(parts, inner_names.iter()).is_some()
                    || inner_schema.contains(&parts[0].value)
            },
            _ => false,
        };
        if inner {
            ControlFlow::Break(())
        } else {
            ControlFlow::Continue(())
        }
    })
    .is_break()
}
