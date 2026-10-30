//! A derived table in a join that reads columns of the relations joined before it (a LATERAL
//! join, written or not) is planned once, over the distinct values of those columns:
//!
//! ```sql
//! l JOIN (SELECT p FROM r WHERE r.k = l.k) d ON c
//! -- l' JOIN (SELECT p, dom.id AS rid FROM dom CROSS JOIN r WHERE r.k = dom.v) d
//! --   ON c AND l'.lid = d.rid
//! ```
//!
//! `dom` holds each distinct value `v` of `l.k` with an `id`, and `l'` is `l` with the `id` of
//! its value as `lid`. A GROUP BY of the derived table also groups by `dom.id`.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_defs::join::{JoinCoalesce, MaintainOrderJoin};
use polars_lazy::prelude::*;
use polars_utils::format_pl_smallstr;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Distinct, Expr as SQLExpr, GroupByExpr, Ident, Join,
    JoinConstraint, JoinOperator, ObjectName, OrderBy, OrderByKind, Query, SelectItem,
    SelectItemQualifiedWildcardKind, SetExpr, TableFactor, TableWithJoins, Value as SQLValue,
    ValueWithSpan, Visitor, VisitorMut,
};

use crate::context::{RelationName, match_qualifier, relation_name};
use crate::sql_visitors::{expr_contains_subquery, is_column_selector};
use crate::subquery::{row_indexed_once, strip_nested};
use crate::subquery_group_by::{is_aggregate_call, is_count_star};
use crate::subquery_outer_select::calls_query_function;
use crate::subquery_pull_up::has_aggregate;
use crate::{SQLContext, col, dyn_visit, unique_column_name};

/// The inputs of a LATERAL join: the left input with a key column, the derived table with a
/// key column, and the join on those keys. `keys` names the key columns, which the join drops.
pub(crate) struct LateralJoin {
    pub(crate) left: LazyFrame,
    pub(crate) name: String,
    pub(crate) right: LazyFrame,
    pub(crate) join_operator: JoinOperator,
    pub(crate) keys: [PlSmallStr; 2],
}

/// The columns of the relations of the derived table, whose names bind to them first.
struct Inner<'a> {
    names: &'a [RelationName],
    schema: &'a Schema,
    aliases: &'a [Ident],
}

impl Inner<'_> {
    fn binds(&self, expr: &SQLExpr) -> bool {
        match expr {
            SQLExpr::Identifier(ident) => {
                self.schema.contains(&ident.value)
                    || self.aliases.iter().any(|a| a.value == ident.value)
            },
            SQLExpr::CompoundIdentifier(parts) => {
                match_qualifier(parts, self.names.iter()).is_some()
                    || self.schema.contains(&parts[0].value)
            },
            _ => false,
        }
    }
}

/// The columns the derived table reads outside its own relations, or `Break` if it reads the
/// query in a way the rewrite can't keep: a window or ranking function, a column selector, or
/// an aggregate of outer columns only, which aggregates the outer query.
struct OuterColumns<'a, 'b> {
    ctx: &'a SQLContext,
    inner: &'a Inner<'b>,
    query_depth: usize,
    found: Vec<SQLExpr>,
}

impl Visitor for OuterColumns<'_, '_> {
    type Break = ();

    fn pre_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
        self.query_depth += 1;
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &Query) -> ControlFlow<()> {
        self.query_depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
        if self.query_depth > 1 {
            return ControlFlow::Continue(());
        }
        match expr {
            SQLExpr::Function(function)
                if function.over.is_some() || calls_query_function(self.ctx, expr) =>
            {
                return ControlFlow::Break(());
            },
            // The columns a nested query reads may be its own.
            _ if is_aggregate_call(self.ctx, expr) && expr_contains_subquery(expr) => {
                return ControlFlow::Break(());
            },
            _ if is_aggregate_call(self.ctx, expr) => {
                let (mut reads_inner, mut reads_outer) = (false, false);
                let _ = dyn_visit::visit_expressions(expr, |e| {
                    if matches!(e, SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_)) {
                        if self.inner.binds(e) {
                            reads_inner = true;
                        } else {
                            reads_outer = true;
                        }
                    }
                    ControlFlow::<()>::Continue(())
                });
                if reads_outer && !reads_inner {
                    return ControlFlow::Break(());
                }
            },
            SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_)
                if !self.inner.binds(expr) && !self.found.contains(expr) =>
            {
                self.found.push(expr.clone());
            },
            _ if is_column_selector(expr) && !is_count_star(expr) => return ControlFlow::Break(()),
            _ => {},
        }
        ControlFlow::Continue(())
    }
}

/// Replaces the outer columns of the derived table by the columns of the domain table.
struct ReadDomain<'a> {
    columns: &'a [(SQLExpr, SQLExpr)],
    query_depth: usize,
}

impl VisitorMut for ReadDomain<'_> {
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
        if self.query_depth == 1
            && let Some((_, column)) = self.columns.iter().find(|(outer, _)| outer == expr)
        {
            *expr = column.clone();
        }
        ControlFlow::Continue(())
    }
}

fn table(name: &Ident) -> TableFactor {
    TableFactor::Table {
        name: ObjectName::from(vec![name.clone()]),
        alias: None,
        args: None,
        with_hints: vec![],
        version: None,
        with_ordinality: false,
        partitions: vec![],
        json_path: None,
        sample: None,
        index_hints: vec![],
    }
}

/// The operator of a join on `constraint`, of the same kind as `op`, or `None` for a kind
/// the rewrite doesn't support. A cross join becomes an inner join.
fn with_constraint(op: &JoinOperator, constraint: JoinConstraint) -> Option<JoinOperator> {
    Some(match op {
        JoinOperator::Join(_) => JoinOperator::Join(constraint),
        JoinOperator::Inner(_) | JoinOperator::CrossJoin(JoinConstraint::None) => {
            JoinOperator::Inner(constraint)
        },
        JoinOperator::Left(_) => JoinOperator::Left(constraint),
        JoinOperator::LeftOuter(_) => JoinOperator::LeftOuter(constraint),
        _ => return None,
    })
}

impl SQLContext {
    /// The inputs of the join of `lf` with `relation`, a derived table that reads columns of
    /// `lf` (see the module docs), or `None` if it has another shape.
    pub(crate) fn lateral_join(
        &mut self,
        relation: &TableFactor,
        join_operator: &JoinOperator,
        lf: &LazyFrame,
    ) -> Option<LateralJoin> {
        let TableFactor::Derived {
            subquery,
            alias: Some(alias),
            sample: None,
            ..
        } = relation
        else {
            return None;
        };
        let on = match join_operator {
            JoinOperator::Join(constraint)
            | JoinOperator::Inner(constraint)
            | JoinOperator::Left(constraint)
            | JoinOperator::LeftOuter(constraint)
            | JoinOperator::CrossJoin(constraint) => match constraint {
                JoinConstraint::On(on) => Some(on),
                JoinConstraint::None => None,
                _ => return None,
            },
            _ => return None,
        };
        // A CTE of the derived table may hide a relation whose columns would be read instead.
        if !alias.columns.is_empty()
            || subquery.with.is_some()
            || subquery.limit_clause.is_some()
            || subquery.fetch.is_some()
            || !subquery.locks.is_empty()
            || subquery.for_clause.is_some()
            || subquery.settings.is_some()
            || subquery.format_clause.is_some()
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
        // An empty group yields a row for an outer row without inner rows.
        let empty_group = group_by.is_empty()
            && (select.having.is_some()
                || has_aggregate(self, &select.projection)
                || has_aggregate(self, &subquery.order_by));
        if empty_group
            || !modifiers.is_empty()
            || group_by.iter().any(|key| {
                matches!(
                    strip_nested(key),
                    SQLExpr::Rollup(_) | SQLExpr::Cube(_) | SQLExpr::GroupingSets(_)
                ) || matches!(key, SQLExpr::Tuple(keys) if keys.is_empty())
            })
            || matches!(select.distinct, Some(Distinct::On(_)))
            || select.top.is_some()
            || select.qualify.is_some()
            || !select.named_window.is_empty()
            || !select.lateral_views.is_empty()
            || select.value_table_mode.is_some()
            || select.select_modifiers.is_some()
        {
            return None;
        }
        // The domain table joins before the first FROM item, so a RIGHT or FULL join would
        // add rows without an outer value.
        if select.from.first().is_some_and(|tbl| {
            tbl.joins.iter().any(|join| {
                matches!(
                    join.join_operator,
                    JoinOperator::Right(_)
                        | JoinOperator::RightOuter(_)
                        | JoinOperator::FullOuter(_)
                        | JoinOperator::RightSemi(_)
                        | JoinOperator::RightAnti(_)
                )
            })
        }) {
            return None;
        }
        // An ordinal past the SELECT items would read the added id column.
        let order_by_items = match &subquery.order_by {
            Some(OrderBy {
                kind: OrderByKind::Expressions(items),
                ..
            }) => items.iter().map(|item| &item.expr).collect(),
            Some(_) => return None,
            None => vec![],
        };
        let has_qualified_wildcard = select
            .projection
            .iter()
            .any(|item| matches!(item, SelectItem::QualifiedWildcard(..)));
        let invalid_ordinal = |e: &SQLExpr| match e {
            SQLExpr::Value(ValueWithSpan {
                value: SQLValue::Number(n, _),
                ..
            }) => {
                has_qualified_wildcard
                    || n.parse::<usize>()
                        .map_or(true, |n| n > select.projection.len())
            },
            _ => false,
        };
        if group_by.iter().chain(order_by_items).any(invalid_ordinal) {
            return None;
        }

        // The columns of the derived table's own relations.
        let factors: Vec<&TableFactor> = select
            .from
            .iter()
            .flat_map(|tbl| {
                std::iter::once(&tbl.relation).chain(tbl.joins.iter().map(|j| &j.relation))
            })
            .collect();
        let names: Vec<RelationName> = factors
            .iter()
            .filter_map(|factor| {
                Some(relation_name(factor)?.spelled(factor, self.catalog_spellings()))
            })
            .collect();
        let mut schema = Schema::default();
        let mut ctx = self.isolated();
        for factor in &factors {
            let (_, mut frame) = ctx.get_table(factor).ok()?;
            schema.merge((*ctx.get_frame_schema(&mut frame).ok()?).clone());
        }
        let aliases: Vec<Ident> = select
            .projection
            .iter()
            .filter_map(|item| match item {
                SelectItem::ExprWithAlias { alias, .. } => Some(alias.clone()),
                _ => None,
            })
            .collect();
        let inner = Inner {
            names: &names,
            schema: &schema,
            aliases: &aliases,
        };
        // A wildcard would also select the columns of the domain table, and one that renames
        // or replaces columns names columns that are not inner.
        for item in &select.projection {
            match item {
                SelectItem::Wildcard(_) => return None,
                SelectItem::QualifiedWildcard(_, options)
                    if options.opt_rename.is_some() || options.opt_replace.is_some() =>
                {
                    return None;
                },
                SelectItem::QualifiedWildcard(
                    SelectItemQualifiedWildcardKind::ObjectName(name),
                    _,
                ) => {
                    let parts: Vec<Ident> = name
                        .0
                        .iter()
                        .filter_map(|p| p.as_ident().cloned())
                        .collect();
                    if !names.iter().any(|n| n.named_by(&parts, false)) {
                        return None;
                    }
                },
                SelectItem::QualifiedWildcard(..) => return None,
                _ => {},
            }
        }

        let mut outer = OuterColumns {
            ctx: self,
            inner: &inner,
            query_depth: 0,
            found: Vec::new(),
        };
        if dyn_visit::visit(subquery.as_ref(), &mut outer).is_break() {
            return None;
        }
        let found = outer.found;

        // The domain and the left input must see the same rows.
        let mut lf = lf.clone().cache();
        let left_schema = self.get_frame_schema(&mut lf).ok()?;
        let prefix = format!("__POLARS_LATERAL_{}", unique_column_name());
        let domain_name = Ident::new(&prefix);
        let id = format_pl_smallstr!("{prefix}_ID");
        let left_key = format_pl_smallstr!("{prefix}_L");
        let right_key = format_pl_smallstr!("{prefix}_R");
        let mut outer_exprs: Vec<Expr> = Vec::new();
        // The name of each outer value, as the derived table would name a column of it.
        let mut value_names: Vec<PlSmallStr> = Vec::new();
        let mut columns: Vec<(SQLExpr, SQLExpr)> = Vec::new();
        for sql_expr in found {
            let name = match &sql_expr {
                SQLExpr::Identifier(ident) => {
                    // A name of several relations joined before is ambiguous.
                    let joined_name = format!("{}:", ident.value);
                    if left_schema
                        .iter_names()
                        .any(|name| name.starts_with(&joined_name))
                    {
                        return None;
                    }
                    &ident.value
                },
                SQLExpr::CompoundIdentifier(parts) => &parts.last()?.value,
                _ => unreachable!(),
            };
            let Ok(Some(expr)) = self.try_parse_outer_only_expr(&sql_expr, &left_schema) else {
                continue;
            };
            let index = match outer_exprs.iter().position(|e| *e == expr) {
                Some(index) => index,
                None => {
                    outer_exprs.push(expr);
                    value_names.push(name.into());
                    outer_exprs.len() - 1
                },
            };
            let column = SQLExpr::CompoundIdentifier(vec![
                domain_name.clone(),
                Ident::new(format!("{prefix}_V{index}")),
            ]);
            columns.push((sql_expr, column));
        }
        if outer_exprs.is_empty() {
            return None;
        }

        let values: Vec<PlSmallStr> = (0..outer_exprs.len())
            .map(|i| format_pl_smallstr!("{prefix}_V{i}"))
            .collect();
        let domain = row_indexed_once(
            lf.clone()
                .select(
                    outer_exprs
                        .iter()
                        .zip(&values)
                        .map(|(e, name)| e.clone().alias(name.clone()))
                        .collect::<Vec<_>>(),
                )
                .unique(None, UniqueKeepStrategy::Any),
            id.clone(),
        );
        let left = lf
            .join_builder()
            .with(
                domain.clone().select(
                    values
                        .iter()
                        .map(|name| col(name.clone()))
                        .chain(std::iter::once(col(id.clone()).alias(left_key.clone())))
                        .collect::<Vec<_>>(),
                ),
            )
            .left_on(outer_exprs)
            .right_on(
                values
                    .iter()
                    .map(|name| col(name.clone()))
                    .collect::<Vec<_>>(),
            )
            .how(JoinType::Left)
            .join_nulls(true)
            .coalesce(JoinCoalesce::KeepColumns)
            .maintain_order(MaintainOrderJoin::Left)
            .finish()
            .ok()?
            .drop(Selector::ByName {
                names: values.iter().cloned().collect(),
                strict: true,
            });

        // The derived table reads the domain table, and keeps the id of the outer values.
        let mut query = subquery.as_ref().clone();
        let _ = dyn_visit::visit_mut(
            &mut query,
            &mut ReadDomain {
                columns: &columns,
                query_depth: 0,
            },
        );
        let SetExpr::Select(select) = query.body.as_mut() else {
            unreachable!()
        };
        let domain_column =
            |name: &str| SQLExpr::CompoundIdentifier(vec![domain_name.clone(), Ident::new(name)]);
        match select.from.first_mut() {
            None => select.from.push(TableWithJoins {
                relation: table(&domain_name),
                joins: vec![],
            }),
            Some(first) => {
                let relation = std::mem::replace(&mut first.relation, table(&domain_name));
                first.joins.insert(
                    0,
                    Join {
                        relation,
                        global: false,
                        join_operator: JoinOperator::CrossJoin(JoinConstraint::None),
                    },
                );
            },
        }
        select.projection.push(SelectItem::ExprWithAlias {
            expr: domain_column(&id),
            alias: Ident::new(right_key.as_str()),
        });
        if let GroupByExpr::Expressions(group_by, _) = &mut select.group_by
            && !group_by.is_empty()
        {
            group_by.extend(
                std::iter::once(id.as_str())
                    .chain(values.iter().map(|v| v.as_str()))
                    .map(domain_column),
            );
        }
        let mut right = self
            .execute_isolated(|ctx| {
                ctx.table_map
                    .write()
                    .unwrap()
                    .insert(domain_name.value.clone(), domain);
                ctx.execute_query(&query)
            })
            .ok()?;
        // An unnamed SELECT item named after an outer column takes that column's name.
        let right_schema = self.get_frame_schema(&mut right).ok()?;
        let (renamed, names): (Vec<_>, Vec<_>) = values
            .iter()
            .zip(&value_names)
            .filter(|(value, _)| right_schema.contains(value))
            .map(|(value, name)| (value.clone(), name.clone()))
            .unzip();
        let right = right.rename(renamed, names, true);

        let key_condition = SQLExpr::BinaryOp {
            left: Box::new(SQLExpr::Identifier(Ident::new(left_key.as_str()))),
            op: SQLBinaryOperator::Eq,
            right: Box::new(SQLExpr::Identifier(Ident::new(right_key.as_str()))),
        };
        let condition = match on {
            None
            | Some(SQLExpr::Value(ValueWithSpan {
                value: SQLValue::Boolean(true),
                ..
            })) => key_condition,
            Some(on) => SQLExpr::BinaryOp {
                left: Box::new(on.clone()),
                op: SQLBinaryOperator::And,
                right: Box::new(key_condition),
            },
        };
        let name = alias.name.value.clone();
        self.from_item_frames.insert(
            name.clone(),
            right.clone().drop(Selector::ByName {
                names: [right_key.clone()].into(),
                strict: true,
            }),
        );
        Some(LateralJoin {
            left,
            name,
            right,
            join_operator: with_constraint(join_operator, JoinConstraint::On(condition))?,
            keys: [left_key, right_key],
        })
    }
}
