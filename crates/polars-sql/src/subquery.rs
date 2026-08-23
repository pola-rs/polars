//! Rewrites of `[NOT] EXISTS` / `[NOT] IN (subquery)` predicates into semi /
//! anti joins, decorrelating equi-correlation predicates into join keys.
//! Subquery shapes the rewrites can't soundly express return `None` so the
//! caller falls back to the generic filter path.
use std::borrow::Cow;
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_ops::frame::{JoinCoalesce, MaintainOrderJoin};
use polars_plan::prelude::{AggExpr, Selector};
use polars_plan::utils::{expr_to_leaf_column_names_iter, has_expr};
use polars_utils::aliases::PlHashSet;
use polars_utils::{format_pl_smallstr, unique_column_name};
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Distinct, Expr as SQLExpr, GroupByExpr, Ident, Query,
    Select, SelectItem, SetExpr, Statement, TableFactor, TableWithJoins,
    UnaryOperator as SQLUnaryOperator, Visit, VisitMut, Visitor, VisitorMut, visit_expressions,
};

use crate::SQLContext;
use crate::context::{CORRELATED_COL_PREFIX, FilterMode, get_table_name};
use crate::sql_expr::{parse_sql_expr, sql_in_membership};
use crate::sql_visitors::expr_contains_subquery;

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
    // deduplication. A non-equality correlation predicate falls back to the
    // count-filter rewrite below.
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
        if let Some(SubqueryConjuncts {
            left_on,
            right_on,
            local_filters,
        }) =
            ctx.split_subquery_conjuncts(selection, &inner_names, &inner_schema, outer_schema)?
        {
            // An uncorrelated EXISTS (no correlation key found) has no join key
            // to build from, so leave it to the existing path.
            return Ok(if left_on.is_empty() {
                None
            } else {
                Some(ctx.finish_decorrelated_join(
                    lf,
                    inner_lf,
                    left_on,
                    right_on,
                    local_filters,
                    negated,
                )?)
            });
        }
        ctx.try_rewrite_exists_as_count_filter(
            lf,
            inner_lf,
            selection,
            &inner_names,
            &inner_schema,
            outer_schema,
            negated,
        )
    }

    // Lower `[NOT] EXISTS (SELECT ... FROM rel WHERE <comparison-correlated>)`,
    // whose correlation predicates aren't all plain equalities, to a
    // `count(*) > 0` filter (`== 0` for NOT EXISTS).
    #[cfg(feature = "semi_anti_join")]
    #[expect(clippy::too_many_arguments)]
    fn try_rewrite_exists_as_count_filter(
        mut self,
        lf: &LazyFrame,
        inner_lf: LazyFrame,
        selection: &SQLExpr,
        inner_names: &PlHashSet<String>,
        inner_schema: &Schema,
        outer_schema: &Schema,
        negated: bool,
    ) -> PolarsResult<Option<LazyFrame>> {
        let mut corr_preds = Vec::new();
        let mut local_filters = Vec::new();
        for conj in MintermIter::new(selection) {
            if let Some(pred) =
                scalar_correlation_predicate(conj, inner_names, inner_schema, outer_schema)
            {
                corr_preds.push(pred);
            } else if let Some(filter) =
                self.try_parse_inner_only_expr(conj, inner_names, inner_schema)?
            {
                local_filters.push(filter);
            } else {
                return Ok(None);
            }
        }
        // No correlation: leave it to the equi-only path.
        if corr_preds.is_empty() {
            return Ok(None);
        }

        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let idx_name = format_pl_smallstr!("{prefix}idx");
        let count_name = format_pl_smallstr!("{prefix}cnt");

        // Apply inner-only filters, then rename inner columns to collision-free
        // names so they can't clash with outer columns of the same name (as in
        // a self-correlated `t1 AS x` subquery over outer `t1`).
        let inner_filtered = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
        let rename_from: Vec<PlSmallStr> = inner_schema.iter_names().cloned().collect();
        let rename_to: Vec<PlSmallStr> = rename_from
            .iter()
            .map(|name| prefixed_inner(&prefix, name))
            .collect();
        let inner_renamed = inner_filtered.rename(&rename_from, &rename_to, true);
        inner_renamed.set_cached_arena(self.lp_arena, self.expr_arena);

        let join_preds: Vec<Expr> = corr_preds.iter().map(|p| p.to_expr(&prefix)).collect();

        let outer_indexed = row_indexed_once(lf.clone(), idx_name.clone());
        let matched = outer_indexed
            .clone()
            .join_builder()
            .with(inner_renamed)
            .how(JoinType::Inner)
            .join_where(join_preds);
        let grouped = matched
            .group_by([col(idx_name.clone())])
            .agg([len().alias(count_name.clone())]);

        let joined = outer_indexed
            .join_builder()
            .with(grouped)
            .left_on([col(idx_name.clone())])
            .right_on([col(idx_name.clone())])
            .how(JoinType::Left)
            .coalesce(JoinCoalesce::CoalesceColumns)
            .maintain_order(MaintainOrderJoin::Left)
            .finish()?
            .with_columns([col(count_name.clone()).fill_null(lit(0))]);

        let matches = if negated {
            col(count_name.clone()).eq(lit(0))
        } else {
            col(count_name.clone()).gt(lit(0))
        };
        Ok(Some(joined.filter(matches).drop(Selector::ByName {
            names: Arc::from([idx_name, count_name]),
            strict: true,
        })))
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
        let Some(right_key) = ctx.try_parse_inner_only_expr(proj, &inner_names, &inner_schema)?
        else {
            return Ok(None);
        };
        let right_key = right_key.meta().undo_aliases();
        if !usable_as_join_key(&left_key) || !usable_as_join_key(&right_key) {
            return Ok(None);
        }

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
        let joined = build_semi_anti_join(lf, inner_lf.clone(), left_on, right_on, anti)?;

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
        )?))
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
    ) -> PolarsResult<LazyFrame> {
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
            let Some(filter) = self.try_parse_inner_only_expr(conj, inner_names, inner_schema)?
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
        inner_names: &PlHashSet<String>,
        inner_schema: &Schema,
    ) -> PolarsResult<Option<Expr>> {
        if !binds_to_inner_relation(sql_expr, inner_names, inner_schema) {
            return Ok(None);
        }
        Ok(Some(parse_sql_expr(sql_expr, self, Some(inner_schema))?))
    }

    // Lower every correlated subquery reachable from `expr` into a decorrelated
    // join over `lf`, substituting each one for the column its lowering
    // materialised. Returns the updated frame together with the rewritten
    // expression, which borrows `expr` unchanged when nothing was lowered.
    // Subqueries that can't be lowered are left in place.
    //
    // See [`SubqueryBindings`] for when a set may be shared across calls.
    pub(crate) fn lower_correlated_subqueries<'a>(
        &mut self,
        lf: LazyFrame,
        outer_schema: &Schema,
        expr: &'a SQLExpr,
        scope: LowerScope,
        bindings: &mut SubqueryBindings,
    ) -> PolarsResult<(LazyFrame, Cow<'a, SQLExpr>)> {
        // The mutable visit needs an owned expression; skip the clone when there is
        // nothing to rewrite.
        if !expr_contains_subquery(expr) {
            return Ok((lf, Cow::Borrowed(expr)));
        }
        let mut rewritten = expr.clone();
        let mut lowering = CorrelatedLowering {
            ctx: self,
            lf,
            outer_schema,
            scope,
            bindings,
            query_depth: 0,
            changed: false,
        };
        if let ControlFlow::Break(e) = VisitMut::visit(&mut rewritten, &mut lowering) {
            return Err(e);
        }
        let rewritten = if lowering.changed {
            Cow::Owned(rewritten)
        } else {
            Cow::Borrowed(expr)
        };
        Ok((lowering.lf, rewritten))
    }

    // Attempt the decorrelation of a single scalar aggregate subquery:
    //   (SELECT AGG(...) FROM inner WHERE <corr-preds> AND <inner-filters>)
    // Equality-only correlation: `inner GROUP BY <corr cols>` once, joined onto
    // the outer frame on those columns directly. Otherwise: row-index the outer
    // frame, inner-join outer × inner on the correlation predicates (`join_where`
    // handles inequality), aggregate per outer row, then left-join back on the
    // row index. Either way, `COUNT` over no matches is 0, every other aggregate
    // is NULL. Returns `None` when uncorrelated or not a scalar-aggregate shape.
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
        let count_like = matches!(
            agg_output_root(&agg_expr),
            Expr::Len | Expr::Agg(AggExpr::Count { .. })
        );

        // Split the WHERE into correlation predicates and inner-only filters.
        let mut corr_preds = Vec::new();
        let mut local_filters = Vec::new();
        for conj in MintermIter::new(selection) {
            if let Some(pred) =
                scalar_correlation_predicate(conj, &inner_names, &inner_schema, outer_schema)
            {
                corr_preds.push(pred);
            } else if let Some(filter) =
                ctx.try_parse_inner_only_expr(conj, &inner_names, &inner_schema)?
            {
                local_filters.push(filter);
            } else {
                return Ok(None);
            }
        }
        // No correlation: leave it to the uncorrelated scalar-subquery path.
        if corr_preds.is_empty() {
            return Ok(None);
        }

        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let result_name = format_pl_smallstr!("{prefix}res");

        // Apply inner-only filters, then rename inner columns to collision-free
        // names so they can't clash with outer columns of the same name (as in a
        // self-correlated `t1 AS x` subquery over outer `t1`).
        let inner_filtered = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
        let rename_from: Vec<PlSmallStr> = inner_schema.iter_names().cloned().collect();
        let rename_to: Vec<PlSmallStr> = rename_from
            .iter()
            .map(|name| prefixed_inner(&prefix, name))
            .collect();
        let inner_renamed = inner_filtered.rename(&rename_from, &rename_to, true);
        inner_renamed.set_cached_arena(ctx.lp_arena, ctx.expr_arena);

        let agg_expr = agg_expr.map_expr(|e| match e {
            Expr::Column(name) if inner_schema.contains(name.as_str()) => {
                col(prefixed_inner(&prefix, &name))
            },
            other => other,
        });

        if corr_preds.iter().all(|p| p.op == SQLBinaryOperator::Eq) {
            let outer_on: Vec<Expr> = corr_preds.iter().map(|p| col(p.outer.clone())).collect();
            let inner_on: Vec<Expr> = corr_preds
                .iter()
                .map(|p| col(prefixed_inner(&prefix, &p.inner)))
                .collect();
            let grouped = inner_renamed
                .group_by(inner_on.clone())
                .agg([agg_expr.alias(result_name.clone())]);
            let joined =
                left_join_aggregate(lf, grouped, outer_on, inner_on, &result_name, count_like)?;
            return Ok(Some((joined, result_name)));
        }

        let join_preds: Vec<Expr> = corr_preds.iter().map(|p| p.to_expr(&prefix)).collect();
        let idx_name = format_pl_smallstr!("{prefix}idx");

        let outer_indexed = row_indexed_once(lf, idx_name.clone());
        let matched = outer_indexed
            .clone()
            .join_builder()
            .with(inner_renamed)
            .how(JoinType::Inner)
            .join_where(join_preds);
        let grouped = matched
            .group_by([col(idx_name.clone())])
            .agg([agg_expr.alias(result_name.clone())]);

        let joined = left_join_aggregate(
            outer_indexed,
            grouped,
            vec![col(idx_name.clone())],
            vec![col(idx_name.clone())],
            &result_name,
            count_like,
        )?
        .drop(Selector::ByName {
            names: Arc::from([idx_name]),
            strict: true,
        });
        Ok(Some((joined, result_name)))
    }

    // Attempt the decorrelation of a correlated `IN (subquery)` into a boolean
    // column. The correlated candidate values are collected per outer row into a
    // list, and membership is then evaluated against that list under SQL's
    // three-valued logic. An uncorrelated subquery is left to the generic path.
    fn try_decorrelate_in_subquery(
        &mut self,
        lf: LazyFrame,
        outer_schema: &Schema,
        lhs: &SQLExpr,
        subquery: &Query,
    ) -> PolarsResult<Option<(LazyFrame, PlSmallStr)>> {
        let Some(select) = eligible_subquery_select(subquery) else {
            return Ok(None);
        };
        if matches!(&select.distinct, Some(Distinct::On(_))) {
            return Ok(None);
        }
        let [SelectItem::UnnamedExpr(proj) | SelectItem::ExprWithAlias { expr: proj, .. }] =
            select.projection.as_slice()
        else {
            return Ok(None);
        };
        let Some(selection) = &select.selection else {
            return Ok(None);
        };

        let needle = parse_sql_expr(lhs, self, Some(outer_schema))?
            .meta()
            .undo_aliases();
        if has_expr(&needle, |e| matches!(e, Expr::SubPlan(_, _)))
            || !expr_to_leaf_column_names_iter(&needle)
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
        let Some(value) = ctx.try_parse_inner_only_expr(proj, &inner_names, &inner_schema)? else {
            return Ok(None);
        };
        let value = value.meta().undo_aliases();

        let mut corr_preds = Vec::new();
        let mut local_filters = Vec::new();
        for conj in MintermIter::new(selection) {
            if let Some(pred) =
                scalar_correlation_predicate(conj, &inner_names, &inner_schema, outer_schema)
            {
                corr_preds.push(pred);
            } else if let Some(filter) =
                ctx.try_parse_inner_only_expr(conj, &inner_names, &inner_schema)?
            {
                local_filters.push(filter);
            } else {
                return Ok(None);
            }
        }
        // No correlation: the generic `IN` path already handles this.
        if corr_preds.is_empty() {
            return Ok(None);
        }

        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let set_name = format_pl_smallstr!("{prefix}set");

        let inner_filtered = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
        let rename_from: Vec<PlSmallStr> = inner_schema.iter_names().cloned().collect();
        let rename_to: Vec<PlSmallStr> = rename_from
            .iter()
            .map(|name| prefixed_inner(&prefix, name))
            .collect();
        let inner_renamed = inner_filtered.rename(&rename_from, &rename_to, true);
        inner_renamed.set_cached_arena(ctx.lp_arena, ctx.expr_arena);

        let value = value.map_expr(|e| match e {
            Expr::Column(name) if inner_schema.contains(name.as_str()) => {
                col(prefixed_inner(&prefix, &name))
            },
            other => other,
        });

        // Grouping collects the candidate values of each outer row into a list.
        let joined = if corr_preds.iter().all(|p| p.op == SQLBinaryOperator::Eq) {
            let outer_on: Vec<Expr> = corr_preds.iter().map(|p| col(p.outer.clone())).collect();
            let inner_on: Vec<Expr> = corr_preds
                .iter()
                .map(|p| col(prefixed_inner(&prefix, &p.inner)))
                .collect();
            let grouped = inner_renamed
                .group_by(inner_on.clone())
                .agg([value.alias(set_name.clone())]);
            left_join_aggregate(lf, grouped, outer_on, inner_on, &set_name, false)?
        } else {
            let join_preds: Vec<Expr> = corr_preds.iter().map(|p| p.to_expr(&prefix)).collect();
            let idx_name = format_pl_smallstr!("{prefix}idx");
            let outer_indexed = row_indexed_once(lf, idx_name.clone());
            let matched = outer_indexed
                .clone()
                .join_builder()
                .with(inner_renamed)
                .how(JoinType::Inner)
                .join_where(join_preds);
            let grouped = matched
                .group_by([col(idx_name.clone())])
                .agg([value.alias(set_name.clone())]);
            left_join_aggregate(
                outer_indexed,
                grouped,
                vec![col(idx_name.clone())],
                vec![col(idx_name.clone())],
                &set_name,
                false,
            )?
            .drop(Selector::ByName {
                names: Arc::from([idx_name]),
                strict: true,
            })
        };

        // An outer row with no matching inner rows joins to null, which is the
        // empty candidate set rather than an unknown one.
        let set = col(set_name.clone());
        let is_in = sql_in_membership(
            needle.is_in(set.clone(), false),
            set.clone(),
            set.clone().is_null().or(set.list().len().eq(lit(0u32))),
        );
        // The boolean replaces the candidate list in place.
        let joined = joined.with_columns([is_in.alias(set_name.clone())]);
        Ok(Some((joined, set_name)))
    }

    // Attempt the decorrelation of a single `EXISTS` subquery into a boolean flag
    // column holding `count(*) > 0` over the correlated match set. An uncorrelated
    // subquery yields a single constant boolean broadcast onto every outer row
    // instead, no join being needed to know whether a fixed relation is non-empty.
    // Returns `None` for shapes that can't be soundly classified.
    fn try_decorrelate_exists_subquery(
        &mut self,
        lf: LazyFrame,
        outer_schema: &Schema,
        subquery: &Query,
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

        let mut corr_preds = Vec::new();
        let mut local_filters = Vec::new();
        if let Some(selection) = &select.selection {
            for conj in MintermIter::new(selection) {
                if let Some(pred) =
                    scalar_correlation_predicate(conj, &inner_names, &inner_schema, outer_schema)
                {
                    corr_preds.push(pred);
                } else if let Some(filter) =
                    ctx.try_parse_inner_only_expr(conj, &inner_names, &inner_schema)?
                {
                    local_filters.push(filter);
                } else {
                    return Ok(None);
                }
            }
        }

        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let flag_name = format_pl_smallstr!("{prefix}exists");
        let inner_filtered = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);

        if corr_preds.is_empty() {
            // Uncorrelated: the inner relation doesn't depend on the outer row, so
            // its existence is a single constant boolean broadcast onto every row.
            let inner_flag = inner_filtered.select([len().gt(lit(0)).alias(flag_name.clone())]);
            inner_flag.set_cached_arena(ctx.lp_arena, ctx.expr_arena);
            let joined = concat_lf_horizontal(
                [lf, inner_flag],
                HConcatOptions {
                    broadcast_unit_length: true,
                    ..Default::default()
                },
            )?;
            return Ok(Some((joined, flag_name)));
        }

        let idx_name = format_pl_smallstr!("{prefix}idx");
        let count_name = format_pl_smallstr!("{prefix}cnt");

        // Rename inner columns to collision-free names so they can't clash
        // with outer columns of the same name (as in a self-correlated
        // `t1 AS x` subquery over outer `t1`).
        let rename_from: Vec<PlSmallStr> = inner_schema.iter_names().cloned().collect();
        let rename_to: Vec<PlSmallStr> = rename_from
            .iter()
            .map(|name| prefixed_inner(&prefix, name))
            .collect();
        let inner_renamed = inner_filtered.rename(&rename_from, &rename_to, true);
        inner_renamed.set_cached_arena(ctx.lp_arena, ctx.expr_arena);

        let join_preds: Vec<Expr> = corr_preds.iter().map(|p| p.to_expr(&prefix)).collect();

        let outer_indexed = row_indexed_once(lf, idx_name.clone());
        let matched = outer_indexed
            .clone()
            .join_builder()
            .with(inner_renamed)
            .how(JoinType::Inner)
            .join_where(join_preds);
        let grouped = matched
            .group_by([col(idx_name.clone())])
            .agg([len().alias(count_name.clone())]);

        let joined = outer_indexed
            .join_builder()
            .with(grouped)
            .left_on([col(idx_name.clone())])
            .right_on([col(idx_name.clone())])
            .how(JoinType::Left)
            .coalesce(JoinCoalesce::CoalesceColumns)
            .maintain_order(MaintainOrderJoin::Left)
            .finish()?
            .with_columns([col(count_name.clone())
                .fill_null(lit(0))
                .gt(lit(0))
                .alias(flag_name.clone())])
            .drop(Selector::ByName {
                names: Arc::from([idx_name, count_name]),
                strict: true,
            });
        Ok(Some((joined, flag_name)))
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

// Semi/anti join the outer frame against the (filtered, arena-cached) inner.
#[cfg(feature = "semi_anti_join")]
fn build_semi_anti_join(
    lf: &LazyFrame,
    inner_lf: LazyFrame,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    anti: bool,
) -> PolarsResult<LazyFrame> {
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
) -> PolarsResult<LazyFrame> {
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

        return Ok(joined
            .join_builder()
            .with(inner_lf.select([flag]))
            .how(JoinType::Cross)
            .finish()?
            .filter(keep)
            .drop(Selector::ByName {
                names: [flag_name].into(),
                strict: true,
            }));
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
    )?;
    let kept_null = exclude_groups(
        joined.filter(left_key.clone().is_null()),
        corr_keys(inner_lf),
    )?;

    Ok(concat(
        [kept_non_null, kept_null],
        UnionArgs {
            rechunk: false,
            parallel: true,
            ..Default::default()
        },
    )
    .expect("'NOT IN' 3VL union has identical schemas"))
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

// Whether every column the expression references resolves to the inner relation,
// so it can be evaluated against the inner frame alone. A qualified name resolves
// through its qualifier: only an inner relation's name or alias counts as inner.
// An unqualified name binds to the innermost scope that has it, so the inner
// schema wins over the outer one.
fn binds_to_inner_relation(
    expr: &SQLExpr,
    inner_names: &PlHashSet<String>,
    inner_schema: &Schema,
) -> bool {
    // A nested subquery is its own scope, which this cannot resolve against.
    if expr_contains_subquery(expr) {
        return false;
    }
    visit_expressions(expr, |e| {
        let resolves = match e {
            SQLExpr::Identifier(_) | SQLExpr::CompoundIdentifier(_) => qualifier_and_name(e)
                .is_some_and(|(qualifier, name)| {
                    qualifier.is_none_or(|q| inner_names.contains(q)) && inner_schema.contains(name)
                }),
            _ => true,
        };
        if resolves {
            ControlFlow::Continue(())
        } else {
            ControlFlow::Break(())
        }
    })
    .is_continue()
}

// Split an identifier into its optional table qualifier and its bare column name.
// Catalog and schema prefixes are dropped, matching how `get_table_name` builds
// `inner_names`.
fn qualifier_and_name(expr: &SQLExpr) -> Option<(Option<&str>, &str)> {
    match expr {
        SQLExpr::Identifier(ident) => Some((None, ident.value.as_str())),
        SQLExpr::CompoundIdentifier(parts) => {
            let (last, init) = parts.split_last()?;
            Some((init.last().map(|q| q.value.as_str()), last.value.as_str()))
        },
        _ => None,
    }
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
    let (qualifier, name) = qualifier_and_name(expr)?;
    let name: PlSmallStr = name.into();
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

/// Correlated subqueries already lowered against a frame, each mapped to the column its
/// decorrelation materialised. The `bool` is the `exists` flag: a scalar subquery and an
/// `EXISTS` over the same inner query materialise different columns and must not share a
/// binding.
///
/// A set is only valid while its columns are still present and still mean the same thing
/// per row, so it may be shared across passes over one frame but must be started afresh
/// once the frame has been re-projected or aggregated.
pub(crate) type SubqueryBindings = Vec<(Query, SubqueryKind, PlSmallStr)>;

/// Which predicate a lowered subquery column answers.
#[derive(Clone, PartialEq)]
pub(crate) enum SubqueryKind {
    Scalar,
    Exists,
    /// `IN`, keyed by its left-hand side: one subquery can serve several.
    In(Box<SQLExpr>),
}

/// Whether a subquery reads a qualifier it does not declare, which is what
/// correlates it with the query around it.
pub(crate) fn is_correlated_subquery(query: &Query) -> bool {
    #[derive(Default)]
    struct Declared(PlHashSet<String>);

    impl Visitor for Declared {
        type Break = ();

        fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
            if let Some(name) = get_table_name(factor) {
                self.0.insert(name);
            }
            ControlFlow::Continue(())
        }
    }

    let mut declared = Declared::default();
    let _ = query.visit(&mut declared);

    struct Foreign<'a>(&'a PlHashSet<String>);

    impl Visitor for Foreign<'_> {
        type Break = ();

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            match qualifier_and_name(expr) {
                Some((Some(qualifier), _)) if !self.0.contains(qualifier) => ControlFlow::Break(()),
                _ => ControlFlow::Continue(()),
            }
        }
    }

    query.visit(&mut Foreign(&declared.0)).is_break()
}

/// Rewrite `x = ANY (subquery)` to `x IN (subquery)` and `x <> ALL (subquery)`
/// to `x NOT IN (subquery)`, which they are equivalent to.
struct DesugarQuantified;

impl VisitorMut for DesugarQuantified {
    type Break = ();

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<()> {
        let (left, right, negated) = match &*expr {
            SQLExpr::AnyOp {
                left,
                compare_op: SQLBinaryOperator::Eq,
                right,
                ..
            } => (left, right, false),
            SQLExpr::AllOp {
                left,
                compare_op: SQLBinaryOperator::NotEq,
                right,
            } => (left, right, true),
            _ => return ControlFlow::Continue(()),
        };
        let SQLExpr::Subquery(subquery) = right.as_ref() else {
            return ControlFlow::Continue(());
        };
        *expr = SQLExpr::InSubquery {
            expr: left.clone(),
            subquery: subquery.clone(),
            negated,
        };
        ControlFlow::Continue(())
    }
}

pub(crate) fn desugar_quantified_subqueries(stmt: &mut Statement) {
    let _ = VisitMut::visit(stmt, &mut DesugarQuantified);
}

/// Which correlated-subquery node kinds a lowering pass should claim.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum LowerScope {
    /// Scalar subqueries only, leaving predicate subqueries for the semi/anti join rewrite.
    ScalarOnly,
    /// Scalar subqueries and the predicate subqueries `[NOT] EXISTS` and `[NOT] IN`.
    ScalarAndPredicates,
}

fn lowerable_kind(expr: &SQLExpr, scope: LowerScope) -> bool {
    match expr {
        SQLExpr::Subquery(_) => true,
        SQLExpr::Exists { .. } | SQLExpr::InSubquery { .. } => {
            scope == LowerScope::ScalarAndPredicates
        },
        _ => false,
    }
}

/// Rewrites correlated subqueries in-place into references to the columns their
/// decorrelated lowering materialises, threading the growing `LazyFrame` through the
/// traversal. `query_depth` keeps each subquery a leaf rather than recursing into it.
struct CorrelatedLowering<'a> {
    ctx: &'a mut SQLContext,
    lf: LazyFrame,
    outer_schema: &'a Schema,
    scope: LowerScope,
    bindings: &'a mut SubqueryBindings,
    query_depth: usize,
    changed: bool,
}

impl CorrelatedLowering<'_> {
    /// Lower `subquery` (or reuse an earlier identical lowering), returning the
    /// materialised column name, or `None` if it can't be soundly lowered.
    fn lower(&mut self, subquery: &Query, kind: SubqueryKind) -> PolarsResult<Option<PlSmallStr>> {
        if let Some((_, _, name)) = self
            .bindings
            .iter()
            .find(|(q, bound, _)| *bound == kind && q == subquery)
        {
            return Ok(Some(name.clone()));
        }
        // The frame is cloned rather than moved so it survives a declined lowering.
        let lowered = match &kind {
            SubqueryKind::Exists => self.ctx.try_decorrelate_exists_subquery(
                self.lf.clone(),
                self.outer_schema,
                subquery,
            )?,
            SubqueryKind::Scalar => self.ctx.try_decorrelate_scalar_subquery(
                self.lf.clone(),
                self.outer_schema,
                subquery,
            )?,
            SubqueryKind::In(lhs) => self.ctx.try_decorrelate_in_subquery(
                self.lf.clone(),
                self.outer_schema,
                lhs,
                subquery,
            )?,
        };
        let Some((new_lf, name)) = lowered else {
            return Ok(None);
        };
        self.lf = new_lf;
        self.bindings.push((subquery.clone(), kind, name.clone()));
        Ok(Some(name))
    }
}

impl VisitorMut for CorrelatedLowering<'_> {
    type Break = PolarsError;

    fn pre_visit_query(&mut self, _query: &mut Query) -> ControlFlow<PolarsError> {
        self.query_depth += 1;
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, _query: &mut Query) -> ControlFlow<PolarsError> {
        self.query_depth -= 1;
        ControlFlow::Continue(())
    }

    fn pre_visit_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
        // Inside a nested query: not ours to lower against the outer schema.
        if self.query_depth > 0 || !lowerable_kind(expr, self.scope) {
            return ControlFlow::Continue(());
        }
        let (subquery, kind, negated) = match &*expr {
            SQLExpr::Subquery(subquery) => (subquery.as_ref(), SubqueryKind::Scalar, false),
            SQLExpr::Exists { subquery, negated } => {
                (subquery.as_ref(), SubqueryKind::Exists, *negated)
            },
            SQLExpr::InSubquery {
                expr: lhs,
                subquery,
                negated,
            } => (subquery.as_ref(), SubqueryKind::In(lhs.clone()), *negated),
            _ => unreachable!("guarded by lowerable_kind"),
        };
        let name = match self.lower(subquery, kind) {
            Ok(Some(name)) => name,
            Ok(None) => return ControlFlow::Continue(()),
            Err(e) => return ControlFlow::Break(e),
        };
        // `EXISTS` is never NULL; `IN` may be, and negating a NULL keeps it NULL,
        // which is what SQL asks for.
        let resolved = SQLExpr::Identifier(Ident::new(name.as_str()));
        *expr = if negated {
            SQLExpr::UnaryOp {
                op: SQLUnaryOperator::Not,
                expr: Box::new(resolved),
            }
        } else {
            resolved
        };
        self.changed = true;
        ControlFlow::Continue(())
    }
}

// Whether an expression can serve as a join key. Join keys must be elementwise,
// so that every key is as long as the frame it is built from. Anything this does
// not recognise declines the rewrite, which costs an optimisation rather than
// correctness.
fn usable_as_join_key(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) | Expr::Literal(_) => true,
        Expr::Alias(inner, _) | Expr::Cast { expr: inner, .. } => usable_as_join_key(inner),
        Expr::BinaryExpr { left, op: _, right } => {
            usable_as_join_key(left) && usable_as_join_key(right)
        },
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            usable_as_join_key(predicate) && usable_as_join_key(truthy) && usable_as_join_key(falsy)
        },
        _ => false,
    }
}

// Row-index a frame so the index can serve as a row identity key. The result is
// cached: the index identifies the same row only if every consumer reads one
// materialisation.
fn row_indexed_once(lf: LazyFrame, name: PlSmallStr) -> LazyFrame {
    lf.with_row_index(name, None).cache()
}

fn prefixed_inner(prefix: &str, name: &str) -> PlSmallStr {
    format_pl_smallstr!("{prefix}c_{name}")
}

// Left-join a per-group aggregate onto the outer frame, filling unmatched `COUNT`
// results with 0 (every other aggregate stays NULL for unmatched rows).
fn left_join_aggregate(
    outer: LazyFrame,
    grouped: LazyFrame,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    result_name: &PlSmallStr,
    count_like: bool,
) -> PolarsResult<LazyFrame> {
    let joined = outer
        .join_builder()
        .with(grouped)
        .left_on(left_on)
        .right_on(right_on)
        .how(JoinType::Left)
        .coalesce(JoinCoalesce::CoalesceColumns)
        .maintain_order(MaintainOrderJoin::Left)
        .finish()?;
    Ok(if count_like {
        joined.with_columns([col(result_name.clone()).fill_null(lit(0))])
    } else {
        joined
    })
}

// Peel the alias/cast wrappers SQL puts around an aggregate to reach the
// aggregate that determines the empty-group value.
fn agg_output_root(expr: &Expr) -> &Expr {
    match expr {
        Expr::Alias(inner, _) => agg_output_root(inner.as_ref()),
        Expr::Cast { expr: inner, .. } => agg_output_root(inner.as_ref()),
        other => other,
    }
}

// A correlation conjunct `<inner col> <cmp> <outer col>` (either way round),
// tracking which side is the inner column so the comparison can be rebuilt in
// its original orientation as a join predicate.
struct CorrPredicate {
    outer: PlSmallStr,
    inner: PlSmallStr,
    op: SQLBinaryOperator,
    inner_on_left: bool,
}

impl CorrPredicate {
    fn to_expr(&self, prefix: &str) -> Expr {
        let outer = col(self.outer.clone());
        let inner = col(prefixed_inner(prefix, &self.inner));
        let (l, r) = if self.inner_on_left {
            (inner, outer)
        } else {
            (outer, inner)
        };
        match self.op {
            SQLBinaryOperator::Eq => l.eq(r),
            SQLBinaryOperator::NotEq => l.neq(r),
            SQLBinaryOperator::Lt => l.lt(r),
            SQLBinaryOperator::LtEq => l.lt_eq(r),
            SQLBinaryOperator::Gt => l.gt(r),
            SQLBinaryOperator::GtEq => l.gt_eq(r),
            // Guarded by `scalar_correlation_predicate`.
            _ => unreachable!("non-comparison correlation operator"),
        }
    }
}

// A comparison conjunct with one bare inner column and one bare outer column,
// or `None` for anything else. Both sides must be simple columns; more complex
// correlation shapes leave the whole subquery to the generic path.
fn scalar_correlation_predicate(
    conj: &SQLExpr,
    inner_names: &PlHashSet<String>,
    inner_schema: &Schema,
    outer_schema: &Schema,
) -> Option<CorrPredicate> {
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
        (CorrelationSide::Inner, CorrelationSide::Outer) => Some(CorrPredicate {
            outer: rname,
            inner: lname,
            op: op.clone(),
            inner_on_left: true,
        }),
        (CorrelationSide::Outer, CorrelationSide::Inner) => Some(CorrPredicate {
            outer: lname,
            inner: rname,
            op: op.clone(),
            inner_on_left: false,
        }),
        _ => None,
    }
}
