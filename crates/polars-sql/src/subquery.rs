//! Rewrites of `[NOT] EXISTS` / `[NOT] IN (subquery)` predicates into semi /
//! anti joins, decorrelating equi-correlation predicates into join keys.
//! Subquery shapes the rewrites can't soundly express return `None` so the
//! caller falls back to the generic filter path.
use polars_core::prelude::*;
use polars_lazy::prelude::*;
#[cfg(feature = "semi_anti_join")]
use polars_plan::utils::{expr_to_leaf_column_names_iter, has_expr};
#[cfg(feature = "semi_anti_join")]
use polars_utils::aliases::PlHashSet;
use sqlparser::ast::{BinaryOperator as SQLBinaryOperator, Expr as SQLExpr, Query};
#[cfg(feature = "semi_anti_join")]
use sqlparser::ast::{Distinct, GroupByExpr, Select, SelectItem, SetExpr, TableWithJoins};

use crate::SQLContext;
use crate::context::FilterMode;
#[cfg(feature = "semi_anti_join")]
use crate::context::get_table_name;
#[cfg(feature = "semi_anti_join")]
use crate::sql_expr::parse_sql_expr;

impl SQLContext {
    // Entry point: offer each WHERE conjunct to the rewrite, returning the
    // (possibly join-extended) frame together with the conjuncts left for the
    // ordinary filter path. In `KeepTrue` mode each top-level AND-conjunct is
    // offered independently. In `RemoveTrue` mode conjuncts can't be split
    // (`NOT (a AND b)` is a disjunction), so only a sole (possibly
    // parenthesized) subquery predicate rewrites, and anything else is
    // returned whole as the residual.
    pub(crate) fn rewrite_subquery_conjuncts<'a>(
        &mut self,
        mut lf: LazyFrame,
        expr: &'a SQLExpr,
        filter_mode: FilterMode,
        schema: &Schema,
    ) -> PolarsResult<(LazyFrame, Vec<&'a SQLExpr>)> {
        let residual = match filter_mode {
            FilterMode::RemoveTrue => {
                let mut unwrapped = expr;
                while let SQLExpr::Nested(inner) = unwrapped {
                    unwrapped = inner;
                }
                match self.try_rewrite_subquery_conjunct(&lf, unwrapped, filter_mode, schema)? {
                    Some(new_lf) => {
                        lf = new_lf;
                        Vec::new()
                    },
                    None => vec![expr],
                }
            },
            FilterMode::KeepTrue => {
                let mut residual = Vec::new();
                for conj in MintermIter::new(expr) {
                    if let Some(new_lf) =
                        self.try_rewrite_subquery_conjunct(&lf, conj, filter_mode, schema)?
                    {
                        lf = new_lf;
                    } else {
                        residual.push(conj);
                    }
                }
                residual
            },
        };
        Ok((lf, residual))
    }

    // Dispatch one conjunct to the matching rewrite. `RemoveTrue` mode (DELETE)
    // flips the join polarity. Removing `IN` rows keeps
    // rows whose membership is false or NULL, exactly what an anti-join
    // produces; removing `NOT IN` rows would additionally keep NULL keys, which
    // a semi join can't express, so it stays on the filter path.
    fn try_rewrite_subquery_conjunct(
        &mut self,
        lf: &LazyFrame,
        conj: &SQLExpr,
        filter_mode: FilterMode,
        schema: &Schema,
    ) -> PolarsResult<Option<LazyFrame>> {
        let removing = filter_mode == FilterMode::RemoveTrue;
        match conj {
            SQLExpr::Exists { subquery, negated } => {
                self.try_rewrite_exists_as_join(lf, subquery, *negated != removing, schema)
            },
            SQLExpr::InSubquery {
                expr: lhs,
                subquery,
                negated,
            } if !(*negated && removing) => self.try_rewrite_in_subquery_as_join(
                lf,
                lhs,
                subquery,
                *negated != removing,
                filter_mode,
                schema,
            ),
            _ => Ok(None),
        }
    }

    // Lower `[NOT] EXISTS (SELECT ... FROM rel WHERE rel.k = outer.k ...)` to a
    // semi / anti join by decorrelating the equi-correlation predicate(s) into
    // join keys. DISTINCT is ignored: existence is invariant under
    // deduplication.
    #[cfg(feature = "semi_anti_join")]
    fn try_rewrite_exists_as_join(
        &mut self,
        lf: &LazyFrame,
        subquery: &Query,
        negated: bool,
        outer_schema: &Schema,
    ) -> PolarsResult<Option<LazyFrame>> {
        let Some(select) = eligible_subquery_select(subquery) else {
            return Ok(None);
        };
        let Some(selection) = &select.selection else {
            return Ok(None);
        };
        // Resolve and parse the inner relation in an isolated context so its
        // table/alias registrations don't leak into the outer query's scope.
        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) =
            ctx.resolve_subquery_from(&select.from[0])?
        else {
            return Ok(None);
        };
        let Some(SubqueryConjuncts {
            left_on,
            right_on,
            local_filters,
        }) = ctx.split_subquery_conjuncts(selection, &inner_names, &inner_schema, outer_schema)?
        else {
            return Ok(None);
        };
        // An uncorrelated EXISTS (no correlation key found) has no join key to
        // build from, so leave it to the existing path.
        if left_on.is_empty() {
            return Ok(None);
        }
        Ok(Some(ctx.finish_decorrelated_join(
            lf,
            inner_lf,
            left_on,
            right_on,
            local_filters,
            negated,
        )))
    }

    // Lower `lhs [NOT] IN (SELECT col FROM rel ...)` to a semi / anti join: the
    // projected column is the membership key and any equi-correlations in the
    // subquery WHERE become additional join keys.
    #[cfg(feature = "semi_anti_join")]
    fn try_rewrite_in_subquery_as_join(
        &mut self,
        lf: &LazyFrame,
        lhs: &SQLExpr,
        subquery: &Query,
        anti: bool,
        filter_mode: FilterMode,
        outer_schema: &Schema,
    ) -> PolarsResult<Option<LazyFrame>> {
        let Some(select) = eligible_subquery_select(subquery) else {
            return Ok(None);
        };
        // DISTINCT is membership-invariant, but DISTINCT ON drops rows per key.
        if matches!(&select.distinct, Some(Distinct::On(_))) {
            return Ok(None);
        }
        let [SelectItem::UnnamedExpr(proj) | SelectItem::ExprWithAlias { expr: proj, .. }] =
            select.projection.as_slice()
        else {
            return Ok(None);
        };

        let left_key = parse_sql_expr(lhs, self, Some(outer_schema))?
            .meta()
            .undo_aliases();
        if has_expr(&left_key, |e| matches!(e, Expr::SubPlan(_, _)))
            || !expr_to_leaf_column_names_iter(&left_key)
                .all(|name| outer_schema.contains(name.as_str()))
        {
            return Ok(None);
        }

        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) =
            ctx.resolve_subquery_from(&select.from[0])?
        else {
            return Ok(None);
        };
        // The membership key must be a plain expression over the inner relation;
        // any alias it carries is cosmetic and not allowed in a join key.
        let Some(right_key) = ctx.try_parse_inner_only_expr(proj, &inner_schema, outer_schema)?
        else {
            return Ok(None);
        };
        let right_key = right_key.meta().undo_aliases();

        let SubqueryConjuncts {
            mut left_on,
            mut right_on,
            local_filters,
        } = match &select.selection {
            Some(selection) => {
                let Some(split) = ctx.split_subquery_conjuncts(
                    selection,
                    &inner_names,
                    &inner_schema,
                    outer_schema,
                )?
                else {
                    return Ok(None);
                };
                split
            },
            None => SubqueryConjuncts::default(),
        };
        left_on.insert(0, left_key.clone());
        right_on.insert(0, right_key);
        let joined =
            ctx.finish_decorrelated_join(lf, inner_lf, left_on, right_on, local_filters, anti);
        // `lhs NOT IN (...)` is NULL (and so excludes the row) when `lhs` is
        // NULL, but an anti-join keeps null keys since they match nothing; drop
        // them. In `RemoveTrue` mode a NULL membership test keeps the row, so the
        // anti join is already exact.
        Ok(Some(if anti && filter_mode == FilterMode::KeepTrue {
            joined.filter(left_key.is_not_null())
        } else {
            joined
        }))
    }

    // Apply the local filters to the inner relation, hand this (isolated, now
    // finished) context's arenas to it, and build the semi / anti join against
    // the outer frame. Consumes the context: nothing may be parsed in the
    // subquery's scope after the join is built.
    #[cfg(feature = "semi_anti_join")]
    fn finish_decorrelated_join(
        self,
        lf: &LazyFrame,
        inner_lf: LazyFrame,
        left_on: Vec<Expr>,
        right_on: Vec<Expr>,
        local_filters: Vec<Expr>,
        anti: bool,
    ) -> LazyFrame {
        let inner_lf = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
        inner_lf.set_cached_arena(self.lp_arena, self.expr_arena);

        let join_type = if anti { JoinType::Anti } else { JoinType::Semi };
        lf.clone()
            .join_builder()
            .with(inner_lf)
            .left_on(left_on)
            .right_on(right_on)
            .how(join_type)
            .finish()
    }

    // Resolve the subquery's FROM (a single relation, possibly with joins) into
    // the inner LazyFrame, its schema, and the set of relation names/aliases
    // used to classify qualified correlation columns.
    #[cfg(feature = "semi_anti_join")]
    fn resolve_subquery_from(
        &mut self,
        tbl_expr: &TableWithJoins,
    ) -> PolarsResult<Option<(PlHashSet<String>, LazyFrame, SchemaRef)>> {
        let Some(inner_names) = std::iter::once(&tbl_expr.relation)
            .chain(tbl_expr.joins.iter().map(|j| &j.relation))
            .map(get_table_name)
            .collect::<Option<PlHashSet<_>>>()
        else {
            return Ok(None);
        };
        let mut inner_lf = self.execute_from_statement(tbl_expr)?;
        let inner_schema = self.get_frame_schema(&mut inner_lf)?;
        Ok(Some((inner_names, inner_lf, inner_schema)))
    }

    // Split a subquery's WHERE conjuncts into a `SubqueryConjuncts`, or `None`
    // when a conjunct is neither a correlation key pair nor an inner-only
    // filter (an outer column in a non-equi shape, an unresolvable name).
    #[cfg(feature = "semi_anti_join")]
    fn split_subquery_conjuncts(
        &mut self,
        selection: &SQLExpr,
        inner_names: &PlHashSet<String>,
        inner_schema: &Schema,
        outer_schema: &Schema,
    ) -> PolarsResult<Option<SubqueryConjuncts>> {
        let mut left_on = Vec::new();
        let mut right_on = Vec::new();
        let mut local_filters = Vec::new();
        for conj in MintermIter::new(selection) {
            if let Some((outer_key, inner_key)) =
                correlation_key_pair(conj, inner_names, inner_schema, outer_schema)
            {
                left_on.push(col(outer_key));
                right_on.push(col(inner_key));
                continue;
            }
            let Some(filter) = self.try_parse_inner_only_expr(conj, inner_schema, outer_schema)?
            else {
                return Ok(None);
            };
            local_filters.push(filter);
        }
        Ok(Some(SubqueryConjuncts {
            left_on,
            right_on,
            local_filters,
        }))
    }

    // Parse a subquery expression as one over the inner relation only, or `None`
    // if it references any outer column (a correlation shape we don't handle) or
    // contains a nested subquery.
    #[cfg(feature = "semi_anti_join")]
    fn try_parse_inner_only_expr(
        &mut self,
        sql_expr: &SQLExpr,
        inner_schema: &Schema,
        outer_schema: &Schema,
    ) -> PolarsResult<Option<Expr>> {
        let expr = parse_sql_expr(sql_expr, self, Some(inner_schema))?;
        // A nested subquery parses to `Expr::SubPlan`, which is only valid after
        // `process_subqueries` lowering; it can't be used as a plain expression.
        if has_expr(&expr, |e| matches!(e, Expr::SubPlan(_, _))) {
            return Ok(None);
        }
        let only_inner = expr_to_leaf_column_names_iter(&expr).all(|name| {
            inner_schema.contains(name.as_str()) && !outer_schema.contains(name.as_str())
        });
        Ok(only_inner.then_some(expr))
    }

    #[cfg(not(feature = "semi_anti_join"))]
    fn try_rewrite_exists_as_join(
        &mut self,
        _lf: &LazyFrame,
        _subquery: &Query,
        _negated: bool,
        _outer_schema: &Schema,
    ) -> PolarsResult<Option<LazyFrame>> {
        Ok(None)
    }

    #[cfg(not(feature = "semi_anti_join"))]
    #[expect(clippy::too_many_arguments)]
    fn try_rewrite_in_subquery_as_join(
        &mut self,
        _lf: &LazyFrame,
        _lhs: &SQLExpr,
        _subquery: &Query,
        _anti: bool,
        _filter_mode: FilterMode,
        _outer_schema: &Schema,
    ) -> PolarsResult<Option<LazyFrame>> {
        Ok(None)
    }
}

/// An iterator over all the minterms in an SQL boolean expression: the terms
/// that `AND` together to form it, descending through parenthesized `Nested`
/// expressions. The SQL-AST analogue of the `AExpr`-level
/// `polars_plan::plans::aexpr::MintermIter`.
struct MintermIter<'a> {
    stack: Vec<&'a SQLExpr>,
}

impl<'a> Iterator for MintermIter<'a> {
    type Item = &'a SQLExpr;

    fn next(&mut self) -> Option<Self::Item> {
        let mut top = self.stack.pop()?;
        loop {
            match top {
                SQLExpr::Nested(inner) => top = inner,
                SQLExpr::BinaryOp {
                    left,
                    op: SQLBinaryOperator::And,
                    right,
                } => {
                    self.stack.push(right);
                    top = left;
                },
                _ => return Some(top),
            }
        }
    }
}

impl<'a> MintermIter<'a> {
    fn new(root: &'a SQLExpr) -> Self {
        Self { stack: vec![root] }
    }
}

#[cfg(feature = "semi_anti_join")]
enum CorrelationSide {
    Inner,
    Outer,
}

// A subquery WHERE split into equi-correlation join keys (outer side in
// `left_on`, inner side in `right_on`) and filters over inner columns only.
#[cfg(feature = "semi_anti_join")]
#[derive(Default)]
struct SubqueryConjuncts {
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    local_filters: Vec<Expr>,
}

// An equi-correlation conjunct `inner.col = outer.col` (either way round) as a
// `(outer key, inner key)` column-name pair, or `None` when the conjunct is
// anything else (non-equality, unresolvable names, both columns on the same
// side).
#[cfg(feature = "semi_anti_join")]
fn correlation_key_pair(
    conj: &SQLExpr,
    inner_names: &PlHashSet<String>,
    inner_schema: &Schema,
    outer_schema: &Schema,
) -> Option<(PlSmallStr, PlSmallStr)> {
    let SQLExpr::BinaryOp {
        left,
        op: SQLBinaryOperator::Eq,
        right,
    } = conj
    else {
        return None;
    };
    let (lside, lname) =
        classify_correlation_column(left, inner_names, inner_schema, outer_schema)?;
    let (rside, rname) =
        classify_correlation_column(right, inner_names, inner_schema, outer_schema)?;
    match (lside, rside) {
        (CorrelationSide::Outer, CorrelationSide::Inner) => Some((lname, rname)),
        (CorrelationSide::Inner, CorrelationSide::Outer) => Some((rname, lname)),
        _ => None,
    }
}

// Classify a correlation operand as an inner- or outer-query column and return
// its bare name. A qualified identifier (`tbl.col`) resolves by its qualifier:
// an inner relation's name/alias means inner, anything else means outer (so
// same-named columns like `o.id = c.id` resolve). An unqualified identifier
// resolves by schema membership. `None` for non-identifiers or names that can't
// be placed (in neither schema, or ambiguous).
#[cfg(feature = "semi_anti_join")]
fn classify_correlation_column(
    expr: &SQLExpr,
    inner_names: &PlHashSet<String>,
    inner_schema: &Schema,
    outer_schema: &Schema,
) -> Option<(CorrelationSide, PlSmallStr)> {
    let (qualifier, name): (Option<&str>, PlSmallStr) = match expr {
        SQLExpr::Identifier(ident) => (None, ident.value.as_str().into()),
        SQLExpr::CompoundIdentifier(parts) => {
            let (last, init) = parts.split_last()?;
            // Only the table part: catalog/schema prefixes are dropped, just
            // as `get_table_name` drops them when building `inner_names`.
            (
                init.last().map(|q| q.value.as_str()),
                last.value.as_str().into(),
            )
        },
        _ => return None,
    };
    match qualifier {
        Some(q) if inner_names.contains(q) => inner_schema
            .contains(name.as_str())
            .then_some((CorrelationSide::Inner, name)),
        // Any other qualifier is taken as outer: the outer query may span
        // several relations and their names/aliases aren't visible here, so
        // only the column's schema membership can be checked.
        Some(_) => outer_schema
            .contains(name.as_str())
            .then_some((CorrelationSide::Outer, name)),
        None => match (
            inner_schema.contains(name.as_str()),
            outer_schema.contains(name.as_str()),
        ) {
            (true, false) => Some((CorrelationSide::Inner, name)),
            (false, true) => Some((CorrelationSide::Outer, name)),
            _ => None,
        },
    }
}

// Shared eligibility gate for the rewrites: bail on any clause that changes
// which rows the subquery yields. Exhaustive destructuring (no `..`) is on
// purpose: a new sqlparser clause must not compile until it gets an explicit
// keep-or-bail decision here.
#[cfg(feature = "semi_anti_join")]
fn eligible_subquery_select(subquery: &Query) -> Option<&Select> {
    let Query {
        with, // CTEs aren't resolved inside the rewrite: bail
        body,
        order_by: _,      // row order can't affect existence/membership
        limit_clause,     // LIMIT/OFFSET change the yielded rows: bail
        fetch,            // FETCH FIRST is LIMIT spelled differently: bail
        locks: _,         // row locking doesn't change the rows
        for_clause,       // FOR XML/JSON reshape the result: bail
        settings,         // ClickHouse SETTINGS can change results: bail
        format_clause: _, // output serialization only
        pipe_operators,   // `|>` operators transform the rows: bail
    } = subquery;
    if with.is_some()
        || limit_clause.is_some()
        || fetch.is_some()
        || for_clause.is_some()
        || settings.is_some()
        || !pipe_operators.is_empty()
    {
        return None;
    }
    let SetExpr::Select(select) = body.as_ref() else {
        return None;
    };
    let Select {
        select_token: _,
        // Deduplication is existence/membership-invariant; the IN rewrite
        // separately bails on `DISTINCT ON`, which is not.
        distinct: _,
        top,                    // TOP is LIMIT spelled differently: bail
        top_before_distinct: _, // only meaningful with `top`
        // The projection is validated by the callers: EXISTS ignores it, IN
        // requires a single plain expression (which also rules out wildcards
        // and the `exclude` modifier).
        projection: _,
        exclude: _,
        into,          // SELECT INTO is not a pure subquery: bail
        from,          // must be one (possibly joined) relation
        lateral_views, // row-multiplying: bail
        prewhere,      // an extra filter we don't fold in: bail
        selection: _,  // split into join keys/filters by callers
        group_by,      // aggregation changes the yielded rows: bail
        cluster_by: _, // layout/order hints: row-set preserving
        distribute_by: _,
        sort_by: _,
        having,                   // aggregation filter: bail
        named_window: _,          // definitions only; uses are parsed later
        qualify,                  // post-window filter changes the rows: bail
        window_before_qualify: _, // only meaningful with `qualify`
        value_table_mode,         // changes what a row is: bail
        connect_by,               // hierarchical recursion: bail
        flavor: _,                // surface syntax only
    } = select.as_ref();
    let no_group_by = matches!(
        group_by,
        GroupByExpr::Expressions(e, m) if e.is_empty() && m.is_empty()
    );
    if from.len() != 1
        || !no_group_by
        || top.is_some()
        || into.is_some()
        || having.is_some()
        || qualify.is_some()
        || prewhere.is_some()
        || connect_by.is_some()
        || value_table_mode.is_some()
        || !lateral_views.is_empty()
    {
        return None;
    }
    Some(select)
}
