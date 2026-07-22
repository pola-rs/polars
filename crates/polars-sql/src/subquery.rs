//! Rewrites of `[NOT] EXISTS` / `[NOT] IN (subquery)` predicates into semi /
//! anti joins, decorrelating equi-correlation predicates into join keys.
//! Subquery shapes the rewrites can't soundly express return `None` so the
//! caller falls back to the generic filter path.
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_ops::frame::{JoinCoalesce, MaintainOrderJoin};
use polars_plan::prelude::{AggExpr, Selector};
use polars_plan::utils::{expr_to_leaf_column_names_iter, has_expr};
use polars_utils::aliases::PlHashSet;
use polars_utils::{format_pl_smallstr, unique_column_name};
#[cfg(feature = "semi_anti_join")]
use sqlparser::ast::Distinct;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Expr as SQLExpr, GroupByExpr, Query, Select, SelectItem,
    SetExpr, TableWithJoins,
};

use crate::SQLContext;
use crate::context::{CORRELATED_COL_PREFIX, FilterMode, get_table_name};
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
    // deduplication. When the WHERE mixes in (or only has) a non-equality
    // correlation predicate, falls back to `try_rewrite_exists_as_count_filter`,
    // which handles arbitrary comparisons.
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
                ))
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

    // Lower `[NOT] EXISTS (SELECT ... FROM rel WHERE <comparison-correlated>)`
    // where the correlation predicate(s) aren't all plain equalities (so
    // `split_subquery_conjuncts` bailed) to `count(*) > 0` (`== 0` for NOT
    // EXISTS), reusing the row-index + `join_where` + group_by + left-join-back
    // machinery `try_decorrelate_scalar_subquery` uses for correlated scalar
    // aggregates. Unlike that function, the result is folded straight into a
    // filter rather than left as a joined-on column, since an EXISTS predicate
    // is consumed directly by the WHERE clause.
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
                self.try_parse_inner_only_expr(conj, inner_schema, outer_schema)?
            {
                local_filters.push(filter);
            } else {
                return Ok(None);
            }
        }
        // No correlation: leave it to the existing (equi-only) path, which
        // handles the fully-uncorrelated case identically.
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

        let outer_indexed = lf.clone().with_row_index(idx_name.clone(), None);
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
            .finish()
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

    // Lower every correlated scalar (aggregate) subquery reachable from
    // `sql_exprs` into a decorrelated join over `lf`, registering each result
    // column in `correlated_subqueries` so the expression visitor can resolve
    // the subquery to that column. Uncorrelated subqueries and unsupported
    // shapes are left untouched for the generic scalar-subquery path.
    pub(crate) fn decorrelate_scalar_subqueries(
        &mut self,
        mut lf: LazyFrame,
        outer_schema: &Schema,
        sql_exprs: &[&SQLExpr],
    ) -> PolarsResult<LazyFrame> {
        let mut candidates = Vec::new();
        for e in sql_exprs {
            collect_subqueries(e, &mut candidates);
        }
        for subquery in candidates {
            let key = subquery.to_string();
            if self.correlated_subqueries.contains_key(&key) {
                continue;
            }
            if let Some((new_lf, name)) =
                self.try_decorrelate_scalar_subquery(lf.clone(), outer_schema, subquery)?
            {
                lf = new_lf;
                self.correlated_subqueries.insert(key, name);
            }
        }
        Ok(lf)
    }

    // Attempt the decorrelation of a single scalar aggregate subquery:
    //   (SELECT AGG(...) FROM inner WHERE <corr-preds> AND <inner-filters>)
    // Row-index the outer frame, inner-join outer × inner on the correlation
    // predicates (`join_where` handles inequality correlation), aggregate per
    // outer row, then left-join the aggregate back on the row index. `COUNT`
    // over no matches is 0; every other aggregate is NULL. Returns the updated
    // frame and the materialised result column, or `None` when the subquery is
    // uncorrelated or not an aggregate scalar shape we can soundly lower.
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
                ctx.try_parse_inner_only_expr(conj, &inner_schema, outer_schema)?
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
        let idx_name = format_pl_smallstr!("{prefix}idx");
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
        let join_preds: Vec<Expr> = corr_preds.iter().map(|p| p.to_expr(&prefix)).collect();

        let outer_indexed = lf.with_row_index(idx_name.clone(), None);
        let matched = outer_indexed
            .clone()
            .join_builder()
            .with(inner_renamed)
            .how(JoinType::Inner)
            .join_where(join_preds);
        let grouped = matched
            .group_by([col(idx_name.clone())])
            .agg([agg_expr.alias(result_name.clone())]);

        let mut joined = outer_indexed
            .join_builder()
            .with(grouped)
            .left_on([col(idx_name.clone())])
            .right_on([col(idx_name.clone())])
            .how(JoinType::Left)
            .coalesce(JoinCoalesce::CoalesceColumns)
            .maintain_order(MaintainOrderJoin::Left)
            .finish();
        if count_like {
            joined = joined.with_columns([col(result_name.clone()).fill_null(lit(0))]);
        }
        joined = joined.drop(Selector::ByName {
            names: Arc::from([idx_name]),
            strict: true,
        });
        Ok(Some((joined, result_name)))
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

fn prefixed_inner(prefix: &str, name: &str) -> PlSmallStr {
    format_pl_smallstr!("{prefix}c_{name}")
}

// Peel the alias/cast wrappers SQL puts around an aggregate (`COUNT(*)` lowers
// to `len().cast(Int64)`) to reach the aggregate that determines the
// empty-group value.
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

// Collect the subquery nodes reachable from an SQL expression through the
// common expression-nesting positions. Positions not covered here simply leave
// their subqueries to the generic (uncorrelated) scalar-subquery path.
fn collect_subqueries<'a>(expr: &'a SQLExpr, out: &mut Vec<&'a Query>) {
    use sqlparser::ast::{FunctionArg, FunctionArgExpr, FunctionArguments};
    match expr {
        SQLExpr::Subquery(query) => out.push(query),
        SQLExpr::Nested(e)
        | SQLExpr::UnaryOp { expr: e, .. }
        | SQLExpr::Cast { expr: e, .. }
        | SQLExpr::Ceil { expr: e, .. }
        | SQLExpr::Floor { expr: e, .. }
        | SQLExpr::Extract { expr: e, .. }
        | SQLExpr::IsNull(e)
        | SQLExpr::IsNotNull(e)
        | SQLExpr::IsTrue(e)
        | SQLExpr::IsNotTrue(e)
        | SQLExpr::IsFalse(e)
        | SQLExpr::IsNotFalse(e)
        | SQLExpr::Collate { expr: e, .. } => collect_subqueries(e, out),
        SQLExpr::BinaryOp { left, right, .. } => {
            collect_subqueries(left, out);
            collect_subqueries(right, out);
        },
        SQLExpr::Between {
            expr, low, high, ..
        } => {
            collect_subqueries(expr, out);
            collect_subqueries(low, out);
            collect_subqueries(high, out);
        },
        SQLExpr::InList { expr, list, .. } => {
            collect_subqueries(expr, out);
            list.iter().for_each(|e| collect_subqueries(e, out));
        },
        SQLExpr::Like { expr, pattern, .. } | SQLExpr::ILike { expr, pattern, .. } => {
            collect_subqueries(expr, out);
            collect_subqueries(pattern, out);
        },
        SQLExpr::Case {
            operand,
            conditions,
            else_result,
            ..
        } => {
            if let Some(op) = operand {
                collect_subqueries(op, out);
            }
            for when in conditions {
                collect_subqueries(&when.condition, out);
                collect_subqueries(&when.result, out);
            }
            if let Some(e) = else_result {
                collect_subqueries(e, out);
            }
        },
        SQLExpr::Function(func) => {
            if let FunctionArguments::List(list) = &func.args {
                for arg in &list.args {
                    if let FunctionArg::Unnamed(FunctionArgExpr::Expr(e))
                    | FunctionArg::Named {
                        arg: FunctionArgExpr::Expr(e),
                        ..
                    }
                    | FunctionArg::ExprNamed {
                        arg: FunctionArgExpr::Expr(e),
                        ..
                    } = arg
                    {
                        collect_subqueries(e, out);
                    }
                }
            }
        },
        _ => {},
    }
}
