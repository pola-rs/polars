//! A correlated subquery with GROUP BY whose correlation is a set of equalities `i = o`
//! between an inner and an outer expression. Its groups are computed in a derived table
//! that also groups by each `i`, and the subquery reads that table:
//!
//! ```sql
//! (SELECT p FROM r WHERE l AND i = o GROUP BY g HAVING h ORDER BY q LIMIT 1)
//! -- (SELECT p' FROM (SELECT i AS k, g AS g', agg AS a FROM r WHERE l GROUP BY i, g HAVING h) d
//! --  WHERE d.k = o ORDER BY q' LIMIT 1)
//! ```
//!
//! For one outer row the equalities fix `i`, so its groups of `(i, g)` are the groups of `g`
//! of the original subquery. The rewritten subquery has no GROUP BY, so it decorrelates like
//! any other.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_utils::format_pl_smallstr;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Distinct, Expr as SQLExpr, FunctionArg, FunctionArgExpr,
    FunctionArguments, GroupByExpr, Ident, OrderBy, OrderByKind, Query, Select, SelectItem,
    SetExpr, TableAlias, TableFactor, TableWithJoins, Visitor, VisitorMut,
};

use crate::context::{RelationName, combine_conditions, match_qualifier};
use crate::functions::PolarsSQLFunctions;
use crate::sql_expr::parse_sql_expr;
use crate::sql_visitors::{expr_contains_subquery, expr_has_window_functions, is_column_selector};
use crate::subquery::{strip_nested, visit_foreign_qualifiers};
use crate::subquery_outer_select::calls_query_function;
use crate::subquery_pull_up::{ScopeColumns, compares_exactly};
use crate::{SQLContext, dyn_visit, unique_column_name};

/// The relations of the subquery, whose columns are inner.
struct Inner<'a> {
    names: &'a [RelationName],
    schema: &'a Schema,
}

impl Inner<'_> {
    /// Whether the column `expr` names is one of the subquery's own.
    fn is_column(&self, expr: &SQLExpr) -> bool {
        match expr {
            SQLExpr::Identifier(ident) => self.schema.contains(&ident.value),
            SQLExpr::CompoundIdentifier(parts) => {
                match_qualifier(parts, self.names.iter()).is_some()
                    || self.schema.contains(&parts[0].value)
            },
            _ => false,
        }
    }
}

/// The columns an expression reads: of the subquery's own relations, or of the queries around
/// it. A nested subquery reads an outer column if it reads a relation it does not declare and
/// the subquery doesn't either.
#[derive(Default)]
struct Reads {
    inner: bool,
    outer: bool,
}

fn reads(ctx: &SQLContext, expr: &SQLExpr, inner: &Inner) -> Reads {
    struct Finder<'a, 'b> {
        ctx: &'a SQLContext,
        inner: &'a Inner<'b>,
        query_depth: usize,
        reads: Reads,
    }

    impl Visitor for Finder<'_, '_> {
        type Break = ();

        fn pre_visit_query(&mut self, query: &Query) -> ControlFlow<()> {
            if self.query_depth == 0 {
                visit_foreign_qualifiers(query, self.ctx.catalog_spellings(), |parts| {
                    if match_qualifier(parts, self.inner.names.iter()).is_some() {
                        self.reads.inner = true;
                    } else {
                        self.reads.outer = true;
                    }
                    ControlFlow::Continue(())
                });
            }
            self.query_depth += 1;
            ControlFlow::Continue(())
        }

        fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
            self.query_depth -= 1;
            ControlFlow::Continue(())
        }

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            if self.query_depth == 0
                && matches!(
                    expr,
                    SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_)
                )
            {
                if self.inner.is_column(expr) {
                    self.reads.inner = true;
                } else {
                    self.reads.outer = true;
                }
            }
            ControlFlow::Continue(())
        }
    }

    let mut finder = Finder {
        ctx,
        inner,
        query_depth: 0,
        reads: Reads::default(),
    };
    let _ = dyn_visit::visit(expr, &mut finder);
    finder.reads
}

impl Inner<'_> {
    /// Whether `a` and `b` name the same column. Besides equal names, that is `t.x` and `x`
    /// when `t` names the only relation.
    fn same_column(&self, a: &SQLExpr, b: &SQLExpr) -> bool {
        let a = strip_nested(a);
        let b = strip_nested(b);
        if a == b {
            return true;
        }
        let qualified_by_only_relation = |parts: &[Ident], column: &Ident| {
            self.names.len() == 1
                && self.schema.contains(&column.value)
                && matches!(parts, [_, last] if last == column)
                && match_qualifier(parts, self.names.iter()).is_some_and(|(_, len)| len == 1)
        };
        match (a, b) {
            (SQLExpr::Identifier(column), SQLExpr::CompoundIdentifier(parts))
            | (SQLExpr::CompoundIdentifier(parts), SQLExpr::Identifier(column)) => {
                qualified_by_only_relation(parts, column)
            },
            _ => false,
        }
    }
}

fn is_aggregate_call(ctx: &SQLContext, expr: &SQLExpr) -> bool {
    matches!(expr, SQLExpr::Function(function) if function.over.is_none()
        && PolarsSQLFunctions::try_from_sql(function, ctx).is_ok_and(|f| f.is_builtin_aggregate()))
}

fn is_count_star(expr: &SQLExpr) -> bool {
    let SQLExpr::Function(function) = expr else {
        return false;
    };
    let FunctionArguments::List(list) = &function.args else {
        return false;
    };
    function.name.0.len() == 1
        && function.name.0[0]
            .as_ident()
            .is_some_and(|ident| ident.value.eq_ignore_ascii_case("count"))
        && list.duplicate_treatment.is_none()
        && matches!(
            list.args.as_slice(),
            [FunctionArg::Unnamed(FunctionArgExpr::Wildcard)]
        )
}

fn split_conjuncts(expr: &SQLExpr, out: &mut Vec<SQLExpr>) {
    match strip_nested(expr) {
        SQLExpr::BinaryOp {
            left,
            op: SQLBinaryOperator::And,
            right,
        } => {
            split_conjuncts(left, out);
            split_conjuncts(right, out);
        },
        e => out.push(e.clone()),
    }
}

/// Replaces the aggregates and group keys of the SELECT items and ORDER BY of the subquery by
/// the columns of the derived table holding them.
struct GroupOutputs<'a, 'b> {
    ctx: &'a SQLContext,
    inner: &'a Inner<'b>,
    table: &'a Ident,
    /// Expressions the derived table holds, each with its column name.
    keys: Vec<(SQLExpr, Ident)>,
    aggregates: Vec<(SQLExpr, Ident)>,
    /// Names the expressions may read without reading a column: SELECT aliases.
    aliases: Vec<Ident>,
    /// A column outside the keys and aggregates, or an aggregate of an outer column.
    declined: bool,
}

impl GroupOutputs<'_, '_> {
    fn column(&self, name: &Ident) -> SQLExpr {
        SQLExpr::CompoundIdentifier(vec![self.table.clone(), name.clone()])
    }

    fn key(&self, expr: &SQLExpr) -> Option<Ident> {
        self.keys
            .iter()
            .find_map(|(key, name)| self.inner.same_column(key, expr).then(|| name.clone()))
    }

    /// A column for the aggregate `expr`. Equal aggregates are not shared, as they may call
    /// a function such as `RANDOM()` that yields a new value on every call.
    fn aggregate(&mut self, expr: &SQLExpr) -> Ident {
        let name = Ident::new(format_pl_smallstr!(
            "__POLARS_GROUP_AGG_{}",
            self.aggregates.len()
        ));
        self.aggregates.push((expr.clone(), name.clone()));
        name
    }

    fn is_alias(&self, expr: &SQLExpr) -> bool {
        matches!(expr, SQLExpr::Identifier(ident)
            if self.aliases.iter().any(|a| a.value == ident.value))
    }
}

impl VisitorMut for GroupOutputs<'_, '_> {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
        if self.is_alias(expr) {
            if self.inner.is_column(expr) {
                self.declined = true;
                return ControlFlow::Break(());
            }
            return ControlFlow::Continue(());
        }
        let name = if let Some(name) = self.key(expr) {
            name
        } else if is_aggregate_call(self.ctx, expr) {
            if reads(self.ctx, expr, self.inner).outer {
                self.declined = true;
                return ControlFlow::Break(());
            }
            self.aggregate(expr)
        } else {
            let is_column = matches!(
                expr,
                SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_)
            );
            if is_column {
                self.declined = true;
                return ControlFlow::Break(());
            }
            return ControlFlow::Continue(());
        };
        *expr = self.column(&name);
        ControlFlow::Continue(())
    }
}

/// `query` without its GROUP BY, HAVING, ORDER BY and LIMIT.
fn without_grouping(query: &Query, select: &Select) -> Query {
    let mut select = select.clone();
    select.group_by = GroupByExpr::Expressions(vec![], vec![]);
    select.having = None;
    Query {
        body: Box::new(SetExpr::Select(Box::new(select))),
        order_by: None,
        limit_clause: None,
        fetch: None,
        ..query.clone()
    }
}

impl SQLContext {
    /// `subquery` with its GROUP BY computed in a derived table (see the module docs), or
    /// `None` if it has no GROUP BY or another shape. `outer_schema` holds the columns of
    /// the query around it.
    pub(crate) fn group_by_in_derived_table(
        &mut self,
        outer_schema: &Schema,
        subquery: &Query,
    ) -> Option<Query> {
        if subquery.with.is_some()
            || subquery.for_clause.is_some()
            || subquery.settings.is_some()
            || !subquery.pipe_operators.is_empty()
        {
            return None;
        }
        let SetExpr::Select(select) = subquery.body.as_ref() else {
            return None;
        };
        let GroupByExpr::Expressions(group_by, modifiers) = &select.group_by else {
            return None;
        };
        if group_by.is_empty()
            || !modifiers.is_empty()
            || select.from.is_empty()
            || matches!(select.distinct, Some(Distinct::On(_)))
            || select.top.is_some()
            || select.into.is_some()
            || select.prewhere.is_some()
            || select.qualify.is_some()
            || !select.named_window.is_empty()
            || !select.lateral_views.is_empty()
            || !select.connect_by.is_empty()
            || select.value_table_mode.is_some()
            || select.select_modifiers.is_some()
            || !select.optimizer_hints.is_empty()
        {
            return None;
        }
        let order_by_items = match &subquery.order_by {
            None => &[][..],
            Some(OrderBy {
                kind: OrderByKind::Expressions(items),
                ..
            }) => items.as_slice(),
            Some(_) => return None,
        };
        // A column selector would read the derived table, except in `COUNT(*)`, which only
        // counts the rows.
        let has_selector = |e: &SQLExpr| {
            dyn_visit::visit_expressions(e, |e| {
                if is_column_selector(e) && !is_count_star(e) {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                }
            })
            .is_break()
        };
        let reads_query = |e: &SQLExpr| {
            expr_has_window_functions(e) || calls_query_function(self, e) || has_selector(e)
        };
        let mut select_items = Vec::with_capacity(select.projection.len());
        for item in &select.projection {
            let (SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. }) = item
            else {
                return None;
            };
            select_items.push(expr);
        }
        if select_items
            .iter()
            .copied()
            .chain(order_by_items.iter().map(|o| &o.expr))
            .chain(&select.having)
            .any(reads_query)
        {
            return None;
        }

        // A nested subquery reading the outer query moves into the FROM as it does without
        // GROUP BY. Each row matches at most one row of the moved table, so the groups stay.
        let ungrouped = without_grouping(subquery, select);
        let pulled_up = self.pull_up_nested(ScopeColumns::Frame(outer_schema), &ungrouped);
        let mut base = select.as_ref().clone();
        if let Some(Query { body, .. }) = &pulled_up
            && let SetExpr::Select(pulled) = body.as_ref()
        {
            base.from = pulled.from.clone();
            base.selection = pulled.selection.clone();
        }

        let mut ctx = self.isolated();
        let (names, lf, schema) = ctx.resolve_subquery_from(&base).ok()??;
        let inner = Inner {
            names: &names,
            schema: &schema,
        };
        let inner_only = |ctx: &SQLContext, e: &SQLExpr| {
            let r = reads(ctx, e, &inner);
            r.inner && !r.outer
        };

        for key in group_by {
            let is_plain = !matches!(
                strip_nested(key),
                SQLExpr::Rollup(_) | SQLExpr::Cube(_) | SQLExpr::GroupingSets(_)
            );
            if !is_plain || !inner_only(self, key) || expr_contains_subquery(key) {
                return None;
            }
        }
        // A SELECT alias is not a column of the derived table, so HAVING may not read one.
        let aliases: Vec<Ident> = select
            .projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::ExprWithAlias { alias, .. } => Some(alias.clone()),
                _ => None,
            })
            .collect();
        if let Some(having) = &select.having
            && (reads(self, having, &inner).outer
                || dyn_visit::visit_expressions(having, |e| match e {
                    SQLExpr::Identifier(ident)
                        if aliases.iter().any(|a| a.value == ident.value) =>
                    {
                        ControlFlow::Break(())
                    },
                    _ => ControlFlow::Continue(()),
                })
                .is_break())
        {
            return None;
        }

        // The WHERE conjuncts that read the outer query must be `inner = outer` equalities,
        // where distinct inner values never equal one outer value.
        let mut local = Vec::new();
        let mut correlation = Vec::new();
        let mut conjuncts = Vec::new();
        if let Some(selection) = &base.selection {
            split_conjuncts(selection, &mut conjuncts);
        }
        for conjunct in conjuncts {
            if !reads(self, &conjunct, &inner).outer {
                local.push(conjunct);
                continue;
            }
            let SQLExpr::BinaryOp {
                left,
                op: SQLBinaryOperator::Eq,
                right,
            } = &conjunct
            else {
                return None;
            };
            let (inner_key, outer_key) = if inner_only(self, left) {
                (left.as_ref(), right.as_ref())
            } else {
                (right.as_ref(), left.as_ref())
            };
            let outer_reads = reads(self, outer_key, &inner);
            if !inner_only(self, inner_key)
                || outer_reads.inner
                || expr_contains_subquery(inner_key)
            {
                return None;
            }
            let Ok(Some(outer_expr)) = self.try_parse_outer_only_expr(outer_key, outer_schema)
            else {
                return None;
            };
            let outer_dtype = outer_expr.to_field(outer_schema).ok()?.dtype;
            let parsed = parse_sql_expr(inner_key, &mut ctx, Some(&schema)).ok()?;
            let mut probe = lf.clone().select([parsed]);
            let probe_schema = ctx.get_frame_schema(&mut probe).ok()?;
            let (_, inner_dtype) = probe_schema.get_at_index(0)?;
            if !compares_exactly(inner_dtype, &outer_dtype) {
                return None;
            }
            correlation.push((inner_key.clone(), outer_key.clone()));
        }
        if correlation.is_empty() {
            return None;
        }

        let table = Ident::new(format_pl_smallstr!(
            "__POLARS_GROUPS_{}",
            unique_column_name()
        ));
        let correlation_names: Vec<Ident> = (0..correlation.len())
            .map(|i| Ident::new(format!("__POLARS_GROUP_CORR_{i}")))
            .collect();
        let mut outputs = GroupOutputs {
            ctx: self,
            inner: &inner,
            table: &table,
            keys: group_by
                .iter()
                .enumerate()
                .map(|(i, key)| (key.clone(), Ident::new(format!("__POLARS_GROUP_KEY_{i}"))))
                .collect(),
            aggregates: Vec::new(),
            aliases: Vec::new(),
            declined: false,
        };

        // HAVING reads group keys and aggregates only, as it would without the correlation
        // keys that the derived table also groups by.
        if let Some(having) = &select.having {
            let mut check = GroupOutputs {
                keys: outputs.keys.clone(),
                aggregates: Vec::new(),
                aliases: Vec::new(),
                ..outputs
            };
            let _ = dyn_visit::visit_mut(&mut having.clone(), &mut check);
            if check.declined {
                return None;
            }
        }

        // The SELECT items and ORDER BY read the derived table only.
        let mut projection = select.projection.clone();
        for item in &mut projection {
            let (SelectItem::UnnamedExpr(expr) | SelectItem::ExprWithAlias { expr, .. }) = item
            else {
                unreachable!()
            };
            if expr_contains_subquery(expr) || reads(self, expr, &inner).outer {
                return None;
            }
            let _ = dyn_visit::visit_mut(expr, &mut outputs);
        }
        // ORDER BY may also name a SELECT alias.
        outputs.aliases = aliases;
        let mut order_by = subquery.order_by.clone();
        if let Some(OrderBy {
            kind: OrderByKind::Expressions(items),
            ..
        }) = &mut order_by
        {
            for item in items {
                if expr_contains_subquery(&item.expr) {
                    return None;
                }
                // A bare name binds to a SELECT alias first.
                if outputs.is_alias(strip_nested(&item.expr)) {
                    continue;
                }
                let _ = dyn_visit::visit_mut(&mut item.expr, &mut outputs);
            }
        }
        if outputs.declined {
            return None;
        }

        let mut derived = base.clone();
        derived.distinct = None;
        derived.projection = correlation
            .iter()
            .map(|(key, _)| key)
            .zip(&correlation_names)
            .chain(outputs.keys.iter().map(|(key, name)| (key, name)))
            .chain(outputs.aggregates.iter().map(|(agg, name)| (agg, name)))
            .map(|(expr, name)| SelectItem::ExprWithAlias {
                expr: expr.clone(),
                alias: name.clone(),
            })
            .collect();
        derived.selection = combine_conditions(local, SQLBinaryOperator::And);
        derived.group_by = GroupByExpr::Expressions(
            correlation
                .iter()
                .map(|(key, _)| key.clone())
                .chain(group_by.iter().cloned())
                .collect(),
            vec![],
        );
        let derived_query = Query {
            body: Box::new(SetExpr::Select(Box::new(derived))),
            order_by: None,
            limit_clause: None,
            fetch: None,
            ..subquery.clone()
        };

        let mut reading = base;
        reading.projection = projection;
        reading.from = vec![TableWithJoins {
            relation: TableFactor::Derived {
                lateral: false,
                subquery: Box::new(derived_query),
                alias: Some(TableAlias {
                    explicit: true,
                    name: table.clone(),
                    columns: vec![],
                    at: None,
                }),
                sample: None,
            },
            joins: vec![],
        }];
        reading.selection = combine_conditions(
            correlation
                .into_iter()
                .zip(&correlation_names)
                .map(|((_, outer), name)| SQLExpr::BinaryOp {
                    left: Box::new(outputs.column(name)),
                    op: SQLBinaryOperator::Eq,
                    right: Box::new(outer),
                })
                .collect(),
            SQLBinaryOperator::And,
        );
        reading.group_by = GroupByExpr::Expressions(vec![], vec![]);
        reading.having = None;
        Some(Query {
            body: Box::new(SetExpr::Select(Box::new(reading))),
            order_by,
            ..subquery.clone()
        })
    }
}
