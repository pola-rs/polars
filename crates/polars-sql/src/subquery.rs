//! Rewrites of `[NOT] EXISTS` / `[NOT] IN (subquery)` predicates into semi /
//! anti joins, decorrelating equi-correlation predicates into join keys; and
//! decorrelation of scalar-aggregate / `EXISTS` subqueries (any comparison
//! operator, not just equality) into a join, for use in general expression
//! position (SELECT list, WHERE, HAVING).
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_ops::frame::JoinCoalesce;
use polars_plan::prelude::AggExpr;
use polars_plan::utils::{expr_to_leaf_column_names_iter, has_expr};
use polars_utils::aliases::PlHashSet;
use polars_utils::{format_pl_smallstr, unique_column_name};
#[cfg(feature = "semi_anti_join")]
use sqlparser::ast::Distinct;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Expr as SQLExpr, GroupByExpr, Ident, Query, Select,
    SelectItem, SetExpr, TableWithJoins, VisitMut, VisitorMut,
};

use crate::SQLContext;
use crate::context::{FilterMode, get_table_name};
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

        // Correlation keys for the "NOT IN" 3VL correction
        let corr_outer = left_on.clone();
        let corr_inner = right_on.clone();
        left_on.insert(0, left_key.clone());
        right_on.insert(0, right_key.clone());

        // Inline, so filtered inner frame can be reused for the correction
        let inner_lf = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
        inner_lf.set_cached_arena(ctx.lp_arena, ctx.expr_arena);
        let joined = build_semi_anti_join(lf, inner_lf.clone(), left_on, right_on, anti);

        // Only `KeepTrue` "NOT IN" needs 3VL correction.
        if !(anti && filter_mode == FilterMode::KeepTrue) {
            return Ok(Some(joined));
        }
        Ok(Some(refine_not_in_anti_join(
            joined,
            inner_lf,
            &left_key,
            &right_key,
            &corr_outer,
            &corr_inner,
        )))
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
        build_semi_anti_join(lf, inner_lf, left_on, right_on, anti)
    }

    // Resolve the subquery's FROM (a single relation, possibly with joins) into
    // the inner LazyFrame, its schema, and the set of relation names/aliases
    // used to classify qualified correlation columns.
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

    // Split a subquery's WHERE conjuncts into correlation predicates
    // (comparisons, any operator, between one outer and one inner column,
    // built into `join_where` exprs against `prefix`'s renamed inner
    // columns) and inner-only filters, or `None` when a conjunct is neither
    // (an outer column in an unsupported shape, an unresolvable name, a
    // nested subquery).
    fn split_correlated_conjuncts(
        &mut self,
        selection: &SQLExpr,
        inner_names: &PlHashSet<String>,
        inner_schema: &Schema,
        outer_schema: &Schema,
        prefix: &str,
    ) -> PolarsResult<Option<(Vec<Expr>, Vec<Expr>)>> {
        let mut join_preds = Vec::new();
        let mut local_filters = Vec::new();
        for conj in MintermIter::new(selection) {
            if let Some(pred) =
                scalar_correlation_predicate(conj, inner_names, inner_schema, outer_schema)
            {
                join_preds.push(pred.to_expr(prefix));
                continue;
            }
            let Some(filter) = self.try_parse_inner_only_expr(conj, inner_schema, outer_schema)?
            else {
                return Ok(None);
            };
            local_filters.push(filter);
        }
        Ok(Some((join_preds, local_filters)))
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

    // Rewrite every correlated subquery / EXISTS reachable from `expr` into a
    // join against `lf`, substituting each one with the column its lowering
    // materialised. Subqueries that are uncorrelated (or aren't a shape we
    // can soundly lower) are left in place for the existing scalar-subquery
    // path to handle. `cache` dedups an identical subquery appearing more
    // than once across the whole SELECT statement (SELECT list / WHERE /
    // HAVING all share one cache), since decorrelation is expensive.
    pub(crate) fn decorrelate_expr(
        &mut self,
        lf: LazyFrame,
        outer_schema: &Schema,
        expr: &SQLExpr,
        cache: &mut CorrelationCache,
    ) -> PolarsResult<(LazyFrame, SQLExpr)> {
        let mut rewritten = expr.clone();
        let mut visitor = CorrelatedRewriter {
            ctx: self,
            lf,
            outer_schema,
            cache,
        };
        if let ControlFlow::Break(e) = rewritten.visit(&mut visitor) {
            return Err(e);
        }
        Ok((visitor.lf, rewritten))
    }

    // Attempt to decorrelate a scalar-aggregate subquery:
    //   (SELECT AGG(...) FROM inner WHERE <corr-preds> AND <inner-filters>)
    // via `join_back_grouped`. `COUNT`-shaped aggregates fill unmatched rows
    // with 0 (SQL semantics); any other aggregate is left NULL. Returns
    // `None` when the subquery is uncorrelated, or isn't a single
    // scalar-aggregate-over-one-relation shape this can soundly lower.
    fn try_decorrelate_scalar_subquery(
        &mut self,
        lf: LazyFrame,
        outer_schema: &Schema,
        subquery: &Query,
    ) -> PolarsResult<Option<(LazyFrame, PlSmallStr)>> {
        let Some(select) = eligible_subquery_select(subquery) else {
            return Ok(None);
        };
        let Some(selection) = &select.selection else {
            return Ok(None);
        };
        let [SelectItem::UnnamedExpr(proj) | SelectItem::ExprWithAlias { expr: proj, .. }] =
            select.projection.as_slice()
        else {
            return Ok(None);
        };

        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) =
            ctx.resolve_subquery_from(&select.from[0])?
        else {
            return Ok(None);
        };

        // The projection must be a scalar aggregate over the inner relation.
        let agg_expr = parse_sql_expr(proj, &mut ctx, Some(&inner_schema))?;
        if has_expr(&agg_expr, |e| matches!(e, Expr::SubPlan(_, _)))
            || !has_expr(&agg_expr, |e| matches!(e, Expr::Agg(_) | Expr::Len))
        {
            return Ok(None);
        }
        let count_like = is_count_like(&agg_expr);

        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let Some((join_preds, local_filters)) = ctx.split_correlated_conjuncts(
            selection,
            &inner_names,
            &inner_schema,
            outer_schema,
            &prefix,
        )?
        else {
            return Ok(None);
        };
        // No correlation: leave it to the uncorrelated scalar-subquery path.
        if join_preds.is_empty() {
            return Ok(None);
        }

        let result_name = format_pl_smallstr!("{prefix}res");

        let inner_renamed = rename_inner(&prefix, inner_lf, local_filters, &inner_schema, ctx);
        let agg_expr = agg_expr.map_expr(|e| match e {
            Expr::Column(name) if inner_schema.contains(name.as_str()) => {
                col(prefixed_inner(&prefix, &name))
            },
            other => other,
        });

        let (mut joined, idx_name) = join_back_grouped(
            lf,
            &prefix,
            inner_renamed,
            join_preds,
            agg_expr.alias(result_name.clone()),
        );
        if count_like {
            joined = joined.with_columns([col(result_name.clone()).fill_null(lit(0))]);
        }
        let joined = joined.drop(Selector::ByName {
            names: Arc::from([idx_name]),
            strict: true,
        });
        Ok(Some((joined, result_name)))
    }

    // Attempt to decorrelate an `[NOT] EXISTS (...)` used in general
    // expression position (not the top-level WHERE conjunct case
    // `try_rewrite_exists_as_join` already handles) into a boolean flag
    // column: `count(*) > 0` over the correlated match set (`== 0` when
    // negated). Returns `None` when uncorrelated or not an eligible shape.
    fn try_decorrelate_exists_subquery(
        &mut self,
        lf: LazyFrame,
        outer_schema: &Schema,
        subquery: &Query,
        negated: bool,
    ) -> PolarsResult<Option<(LazyFrame, PlSmallStr)>> {
        let Some(select) = eligible_subquery_select(subquery) else {
            return Ok(None);
        };
        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) =
            ctx.resolve_subquery_from(&select.from[0])?
        else {
            return Ok(None);
        };

        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let (join_preds, local_filters) = match &select.selection {
            Some(selection) => {
                let Some(split) = ctx.split_correlated_conjuncts(
                    selection,
                    &inner_names,
                    &inner_schema,
                    outer_schema,
                    &prefix,
                )?
                else {
                    return Ok(None);
                };
                split
            },
            None => (Vec::new(), Vec::new()),
        };
        if join_preds.is_empty() {
            return Ok(None);
        }

        let flag_name = format_pl_smallstr!("{prefix}flag");

        let inner_renamed = rename_inner(&prefix, inner_lf, local_filters, &inner_schema, ctx);

        let (joined, idx_name) = join_back_grouped(
            lf,
            &prefix,
            inner_renamed,
            join_preds,
            len().alias(flag_name.clone()),
        );
        let joined = joined.with_columns([col(flag_name.clone()).fill_null(lit(0))]);
        let flag_expr = if negated {
            col(flag_name.clone()).eq(lit(0))
        } else {
            col(flag_name.clone()).gt(lit(0))
        };
        let joined = joined
            .with_columns([flag_expr.alias(flag_name.clone())])
            .drop(Selector::ByName {
                names: Arc::from([idx_name]),
                strict: true,
            });
        Ok(Some((joined, flag_name)))
    }
}

// Semi/anti join the outer frame against the (filtered, arena-cached) inner.
#[cfg(feature = "semi_anti_join")]
fn build_semi_anti_join(
    lf: &LazyFrame,
    inner_lf: LazyFrame,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    anti: bool,
) -> LazyFrame {
    let join_type = if anti { JoinType::Anti } else { JoinType::Semi };
    lf.clone()
        .join_builder()
        .with(inner_lf)
        .left_on(left_on)
        .right_on(right_on)
        .how(join_type)
        .finish()
}

// Account for 3VL interaction with NULL values
#[cfg(feature = "semi_anti_join")]
fn refine_not_in_anti_join(
    joined: LazyFrame,
    inner_lf: LazyFrame,
    left_key: &Expr,
    right_key: &Expr,
    corr_outer: &[Expr],
    corr_inner: &[Expr],
) -> LazyFrame {
    if corr_inner.is_empty() {
        // Uncorrelated
        let flag_name = unique_column_name();
        let flag = when(len().eq(lit(0u32)))
            .then(lit(NULL).cast(DataType::Boolean))
            .otherwise(right_key.clone().is_null().any(true))
            .alias(flag_name.clone());
        let keep = when(col(flag_name.clone()).is_null())
            .then(lit(true)) // empty set
            .when(col(flag_name.clone())) // set has a NULL
            .then(lit(false))
            .otherwise(left_key.clone().is_not_null());

        return joined
            .join_builder()
            .with(inner_lf.select([flag]))
            .how(JoinType::Cross)
            .finish()
            .filter(keep)
            .drop(Selector::ByName {
                names: [flag_name].into(),
                strict: true,
            });
    }

    // Correlated
    let corr_keys = |lf: LazyFrame| lf.select(corr_inner).unique(None, UniqueKeepStrategy::Any);
    let exclude_groups = |rows: LazyFrame, groups: LazyFrame| {
        rows.join_builder()
            .with(groups)
            .left_on(corr_outer)
            .right_on(corr_inner)
            .how(JoinType::Anti)
            .finish()
    };
    let kept_non_null = exclude_groups(
        joined.clone().filter(left_key.clone().is_not_null()),
        corr_keys(inner_lf.clone().filter(right_key.clone().is_null())),
    );
    let kept_null = exclude_groups(
        joined.filter(left_key.clone().is_null()),
        corr_keys(inner_lf),
    );

    concat(
        [kept_non_null, kept_null],
        UnionArgs {
            rechunk: false,
            parallel: true,
            ..Default::default()
        },
    )
    .expect("'NOT IN' 3VL union has identical schemas")
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
    let pred = scalar_correlation_predicate(conj, inner_names, inner_schema, outer_schema)?;
    (pred.op == SQLBinaryOperator::Eq).then_some((pred.outer_col, pred.inner_col))
}

// Classify a correlation operand as an inner- or outer-query column and return
// its bare name. A qualified identifier (`tbl.col`) resolves by its qualifier:
// an inner relation's name/alias means inner, anything else means outer (so
// same-named columns like `o.id = c.id` resolve). An unqualified identifier
// resolves by schema membership. `None` for non-identifiers or names that can't
// be placed (in neither schema, or ambiguous).
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
        optimizer_hints,          // unsupported: bail
        select_modifiers,         // unsupported: bail
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
        || !connect_by.is_empty()
        || value_table_mode.is_some()
        || !lateral_views.is_empty()
        || !optimizer_hints.is_empty()
        || select_modifiers.is_some()
    {
        return None;
    }
    Some(select)
}

// ---------------------------------------------------------------------------
// General-position correlated subquery / EXISTS decorrelation
// ---------------------------------------------------------------------------

/// Prefix for columns materialised by [`SQLContext::try_decorrelate_scalar_subquery`]
/// / [`SQLContext::try_decorrelate_exists_subquery`]. Kept distinctive so it's
/// identifiable in `explain()` plan text (e.g. to confirm decorrelation ran
/// exactly once for a repeated subquery).
pub(crate) const CORRELATED_COL_PREFIX: &str = "__POLARS_CORR_";

fn prefixed_inner(prefix: &str, name: &str) -> PlSmallStr {
    format_pl_smallstr!("{prefix}{name}")
}

// A subquery WHERE conjunct relating one outer column to one inner column
// through a comparison operator (not just equality, unlike the semi/anti
// join rewrite above): `outer_col op inner_col`.
struct ScalarCorrPredicate {
    op: SQLBinaryOperator,
    outer_col: PlSmallStr,
    inner_col: PlSmallStr,
}

impl ScalarCorrPredicate {
    // Build the join predicate, referencing the inner side by its
    // collision-free renamed column (so a self-correlated subquery over the
    // same table as the outer query doesn't clash names).
    fn to_expr(&self, prefix: &str) -> Expr {
        let outer = col(self.outer_col.clone());
        let inner = col(prefixed_inner(prefix, &self.inner_col));
        match self.op {
            SQLBinaryOperator::Eq => outer.eq(inner),
            SQLBinaryOperator::NotEq => outer.neq(inner),
            SQLBinaryOperator::Lt => outer.lt(inner),
            SQLBinaryOperator::LtEq => outer.lt_eq(inner),
            SQLBinaryOperator::Gt => outer.gt(inner),
            SQLBinaryOperator::GtEq => outer.gt_eq(inner),
            _ => unreachable!("scalar_correlation_predicate only yields comparison operators"),
        }
    }
}

// The reverse of a comparison operator: `a OP b` <=> `b reverse(OP) a`.
// Eq/NotEq are symmetric.
fn reverse_cmp_op(op: SQLBinaryOperator) -> SQLBinaryOperator {
    match op {
        SQLBinaryOperator::Lt => SQLBinaryOperator::Gt,
        SQLBinaryOperator::LtEq => SQLBinaryOperator::GtEq,
        SQLBinaryOperator::Gt => SQLBinaryOperator::Lt,
        SQLBinaryOperator::GtEq => SQLBinaryOperator::LtEq,
        same => same,
    }
}

// Classify a WHERE conjunct as a correlation predicate between one outer and
// one inner column under any comparison operator, or `None` if it isn't that
// shape (reuses `classify_correlation_column`'s alias-aware classification).
fn scalar_correlation_predicate(
    conj: &SQLExpr,
    inner_names: &PlHashSet<String>,
    inner_schema: &Schema,
    outer_schema: &Schema,
) -> Option<ScalarCorrPredicate> {
    let SQLExpr::BinaryOp { left, op, right } = conj else {
        return None;
    };
    if !matches!(
        op,
        SQLBinaryOperator::Eq
            | SQLBinaryOperator::NotEq
            | SQLBinaryOperator::Lt
            | SQLBinaryOperator::LtEq
            | SQLBinaryOperator::Gt
            | SQLBinaryOperator::GtEq
    ) {
        return None;
    }
    let (lside, lname) =
        classify_correlation_column(left, inner_names, inner_schema, outer_schema)?;
    let (rside, rname) =
        classify_correlation_column(right, inner_names, inner_schema, outer_schema)?;
    match (lside, rside) {
        (CorrelationSide::Outer, CorrelationSide::Inner) => Some(ScalarCorrPredicate {
            op: op.clone(),
            outer_col: lname,
            inner_col: rname,
        }),
        (CorrelationSide::Inner, CorrelationSide::Outer) => Some(ScalarCorrPredicate {
            op: reverse_cmp_op(op.clone()),
            outer_col: rname,
            inner_col: lname,
        }),
        _ => None,
    }
}

// Apply inner-only filters, then rename inner columns to collision-free names
// so they can't clash with outer columns of the same name (as in a
// self-correlated `t1 AS x` subquery over outer `t1`). Consumes the isolated
// context: its arenas move onto the returned frame.
fn rename_inner(
    prefix: &str,
    inner_lf: LazyFrame,
    local_filters: Vec<Expr>,
    inner_schema: &Schema,
    ctx: SQLContext,
) -> LazyFrame {
    let inner_filtered = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
    let rename_from: Vec<PlSmallStr> = inner_schema.iter_names().cloned().collect();
    let rename_to: Vec<PlSmallStr> = rename_from
        .iter()
        .map(|name| prefixed_inner(prefix, name))
        .collect();
    let inner_renamed = inner_filtered.rename(&rename_from, &rename_to, true);
    inner_renamed.set_cached_arena(ctx.lp_arena, ctx.expr_arena);
    inner_renamed
}

// Row-index `lf`, inner-join it against `inner_renamed` via `join_where`
// (so inequality correlation works too), aggregate per outer row with
// `agg_expr` (already aliased to its output name), then left-join the
// aggregate back onto `lf` by row index. The caller drops the returned
// row-index column once it's done with it.
fn join_back_grouped(
    lf: LazyFrame,
    prefix: &str,
    inner_renamed: LazyFrame,
    join_preds: Vec<Expr>,
    agg_expr: Expr,
) -> (LazyFrame, PlSmallStr) {
    let idx_name = format_pl_smallstr!("{prefix}idx");
    let outer_indexed = lf.with_row_index(idx_name.clone(), None);
    let matched = outer_indexed
        .clone()
        .join_builder()
        .with(inner_renamed)
        .how(JoinType::Inner)
        .join_where(join_preds);
    let grouped = matched.group_by([col(idx_name.clone())]).agg([agg_expr]);
    let joined = outer_indexed
        .join_builder()
        .with(grouped)
        .left_on([col(idx_name.clone())])
        .right_on([col(idx_name.clone())])
        .how(JoinType::Left)
        .coalesce(JoinCoalesce::CoalesceColumns)
        .finish();
    (joined, idx_name)
}

// The root of a scalar-aggregate expression, unwrapping alias wrappers, to
// tell `COUNT`-shaped aggregates (which must yield 0 over no matches) apart
// from every other aggregate (which must yield NULL).
fn is_count_like(expr: &Expr) -> bool {
    fn root(e: &Expr) -> &Expr {
        match e {
            Expr::Alias(inner, _) => root(inner),
            other => other,
        }
    }
    matches!(root(expr), Expr::Len | Expr::Agg(AggExpr::Count { .. }))
}

pub(crate) type CorrelationCache = PlHashMap<(CorrKind, String), PlSmallStr>;

// Distinguishes the two decorrelation shapes so identical subquery text used
// once as `EXISTS` and once as a scalar value doesn't share a cache slot: one
// materialises a boolean flag, the other a value column.
#[derive(PartialEq, Eq, Hash)]
pub(crate) enum CorrKind {
    Scalar,
    Exists { negated: bool },
}

// Mutably walks a SQL expression tree, replacing every correlated
// `Subquery`/`Exists` node with a plain column reference and threading the
// (possibly join-extended) outer frame through the walk. Nodes it can't
// decorrelate are left untouched.
struct CorrelatedRewriter<'a> {
    ctx: &'a mut SQLContext,
    lf: LazyFrame,
    outer_schema: &'a Schema,
    cache: &'a mut CorrelationCache,
}

impl CorrelatedRewriter<'_> {
    // Shared "compute the decorrelated result, cache it, and substitute it
    // into the AST" tail for a cache miss, used by both the `Subquery` and
    // `Exists` arms of `pre_visit_expr` below (each does its own cache
    // lookup first: cheap, just the subquery's `Display` text, and cloning
    // nothing unless it's actually a miss).
    fn substitute_on_miss(
        &mut self,
        expr: &mut SQLExpr,
        key: (CorrKind, String),
        try_decorrelate: impl FnOnce(
            &mut SQLContext,
            LazyFrame,
            &Schema,
        ) -> PolarsResult<Option<(LazyFrame, PlSmallStr)>>,
    ) -> ControlFlow<PolarsError> {
        let outer_lf = self.lf.clone();
        match try_decorrelate(self.ctx, outer_lf, self.outer_schema) {
            Ok(Some((new_lf, name))) => {
                self.lf = new_lf;
                self.cache.insert(key, name.clone());
                *expr = SQLExpr::Identifier(Ident::new(name.as_str()));
                ControlFlow::Continue(())
            },
            Ok(None) => ControlFlow::Continue(()),
            Err(e) => ControlFlow::Break(e),
        }
    }
}

impl VisitorMut for CorrelatedRewriter<'_> {
    type Break = PolarsError;

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<Self::Break> {
        match expr {
            SQLExpr::Subquery(sq) => {
                let key = (CorrKind::Scalar, sq.to_string());
                if let Some(name) = self.cache.get(&key).cloned() {
                    *expr = SQLExpr::Identifier(Ident::new(name.as_str()));
                    return ControlFlow::Continue(());
                }
                let sq = sq.clone();
                self.substitute_on_miss(expr, key, |ctx, lf, schema| {
                    ctx.try_decorrelate_scalar_subquery(lf, schema, &sq)
                })
            },
            SQLExpr::Exists { subquery, negated } => {
                let negated = *negated;
                let key = (CorrKind::Exists { negated }, subquery.to_string());
                if let Some(name) = self.cache.get(&key).cloned() {
                    *expr = SQLExpr::Identifier(Ident::new(name.as_str()));
                    return ControlFlow::Continue(());
                }
                let subquery = subquery.clone();
                self.substitute_on_miss(expr, key, |ctx, lf, schema| {
                    ctx.try_decorrelate_exists_subquery(lf, schema, &subquery, negated)
                })
            },
            _ => ControlFlow::Continue(()),
        }
    }
}

// Replace every `Subquery` node in `expr` with a placeholder identifier,
// parsing each one standalone and aliasing it to that name.
pub(crate) fn lift_uncorrelated_subqueries(
    ctx: &mut SQLContext,
    expr: &mut SQLExpr,
    schema: &Schema,
) -> PolarsResult<Vec<Expr>> {
    struct Lifter<'a> {
        ctx: &'a mut SQLContext,
        schema: &'a Schema,
        out: Vec<Expr>,
    }
    impl VisitorMut for Lifter<'_> {
        type Break = PolarsError;

        fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
            if let SQLExpr::Subquery(sq) = expr {
                let name =
                    format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_res", unique_column_name());
                let parsed = match parse_sql_expr(
                    &SQLExpr::Subquery(sq.clone()),
                    self.ctx,
                    Some(self.schema),
                ) {
                    Ok(parsed) => parsed,
                    Err(e) => return ControlFlow::Break(e),
                };
                self.out.push(parsed.alias(name.clone()));
                *expr = SQLExpr::Identifier(Ident::new(name.as_str()));
            }
            ControlFlow::Continue(())
        }
    }
    let mut lifter = Lifter {
        ctx,
        schema,
        out: Vec::new(),
    };
    if let ControlFlow::Break(e) = expr.visit(&mut lifter) {
        return Err(e);
    }
    Ok(lifter.out)
}
