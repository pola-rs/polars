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
use std::ops::ControlFlow;

use polars_core::prelude::*;
use sqlparser::ast::helpers::attached_token::AttachedToken;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, CaseWhen, Expr as SQLExpr, GroupByExpr, Query, Select,
    SelectItem, SetExpr, UnaryOperator as SQLUnaryOperator, Value as SQLValue, VisitorMut,
};

use crate::SQLContext;
use crate::context::{RelationName, match_qualifier};
use crate::dyn_visit::{self, visit_expressions};
use crate::functions::PolarsSQLFunctions;
use crate::sql_visitors::{
    expr_contains_subquery, expr_has_window_functions, expr_references_any_column,
    has_column_selector,
};
use crate::subquery::strip_nested;
use crate::subquery_pull_up::has_aggregate;

impl SQLContext {
    /// `select` with the subqueries of its clauses whose SELECT item reads only columns of
    /// queries around them rewritten to read those columns here, or `None` if there are none.
    /// The subqueries of its derived tables are left to those tables.
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
}

fn when_then(condition: SQLExpr, result: SQLExpr, else_result: Option<SQLExpr>) -> SQLExpr {
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
fn calls_query_function(ctx: &SQLContext, expr: &SQLExpr) -> bool {
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
