//! Rewrites of `[NOT] EXISTS` / `[NOT] IN (subquery)` predicates into semi /
//! anti joins, decorrelating equi-correlation predicates into join keys.
//! Subquery shapes the rewrites can't soundly express return `None` so the
//! caller falls back to the generic filter path.
use std::borrow::Cow;
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_core::utils::try_get_supertype;
use polars_defs::join::{JoinCoalesce, JoinType, MaintainOrderJoin};
use polars_lazy::prelude::*;
use polars_plan::plans::visitor::TreeWalker;
use polars_plan::plans::{
    ExprPushdownGroup, ExprToIRContext, NodeStats, attach_dataset_scan_statistics,
    is_elementwise_rec, is_inherently_nondeterministic, node_stats, to_expr_ir,
};
use polars_plan::prelude::{AggExpr, DslPlan, Selector};
use polars_plan::utils::{expr_to_leaf_column_names_iter, has_expr};
use polars_utils::aliases::PlHashSet;
use polars_utils::arena::Arena;
use polars_utils::format_pl_smallstr;
use sqlparser::ast::{
    BinaryOperator as SQLBinaryOperator, Distinct, Expr as SQLExpr, Fetch, GroupByExpr, Ident,
    LimitClause, NamedWindowDefinition, NamedWindowExpr, OrderByKind, Query, Select, SelectItem,
    SetExpr, Statement, TableFactor, TableWithJoins, UnaryOperator as SQLUnaryOperator,
    Value as SQLValue, ValueWithSpan, Visitor, VisitorMut, WindowSpec, WindowType,
};

use crate::catalog::CatalogSpellings;
use crate::context::{
    CORRELATED_COL_PREFIX, FilterMode, RelationName, TableInfo, combine_conditions,
    declared_relations, flatten_and_conditions, get_table_name, match_qualifier, relation_name,
};
use crate::dyn_visit::{self, visit_expressions, visit_expressions_mut};
use crate::functions::PolarsSQLFunctions;
use crate::group_context::{has_marked_aggregate, strip_aggregate_marks};
use crate::sql_expr::{
    order_by_sort_options, parse_sql_expr, reduced_subquery, sql_in_membership, sql_is_in,
};
use crate::sql_visitors::{expr_contains_subquery, expr_has_window_functions, is_subquery_expr};
use crate::subquery_pull_up::ScopeColumns;
use crate::{SQLContext, unique_column_name};

impl SQLContext {
    // Entry point: offer each WHERE conjunct to the rewrites, returning the
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
        stage: RewriteStage,
    ) -> PolarsResult<(LazyFrame, Vec<&'a SQLExpr>)> {
        let residual = match filter_mode {
            FilterMode::RemoveTrue => {
                let unwrapped = unwrap_nested(expr);
                let prepared = self.pull_up_nested_in(ScopeColumns::Frame(schema), unwrapped);
                let unwrapped = prepared.as_ref().unwrap_or(unwrapped);
                match self.try_rewrite_subquery_conjunct(&lf, unwrapped, filter_mode, schema)? {
                    Some(new_lf) => {
                        lf = new_lf;
                        Vec::new()
                    },
                    None => vec![expr],
                }
            },
            FilterMode::KeepTrue => {
                let (plain, with_subquery): (Vec<&SQLExpr>, Vec<&SQLExpr>) =
                    MintermIter::new(expr).partition(|c| !expr_contains_subquery(c));

                if with_subquery.is_empty() {
                    return Ok((lf, plain));
                }

                // A subquery rewrite can row-index the frame, which blocks predicate
                // pushdown. Apply the conjuncts holding no subquery before that.
                let mut residual = Vec::new();
                let mut early = Vec::new();
                for conj in plain {
                    let parsed = parse_sql_expr(conj, self, Some(schema))?;
                    if stage.offers(&parsed, schema) {
                        early.push(parsed);
                    } else {
                        residual.push(conj);
                    }
                }
                if !early.is_empty() {
                    lf = lf.filter(all_horizontal(early)?);
                }

                for conj in with_subquery {
                    if !self.stage_offers_membership(conj, schema, stage)? {
                        residual.push(conj);
                        continue;
                    }
                    let prepared = self.pull_up_nested_in(ScopeColumns::Frame(schema), conj);
                    if let Some(new_lf) = self.try_rewrite_subquery_conjunct(
                        &lf,
                        prepared.as_ref().unwrap_or(conj),
                        filter_mode,
                        schema,
                    )? {
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

    // Whether the stage offers a subquery conjunct to the rewrites: before scalar
    // lowering only when its membership key is harmless on every row. A key
    // holding a subquery of its own can't be parsed before that subquery is
    // lowered.
    fn stage_offers_membership(
        &mut self,
        conj: &SQLExpr,
        schema: &Schema,
        stage: RewriteStage,
    ) -> PolarsResult<bool> {
        if stage == RewriteStage::AfterScalarLowering {
            return Ok(true);
        }
        Ok(match membership_predicate(conj).map(|m| m.key) {
            None => true,
            Some(key) if expr_contains_subquery(key) => false,
            Some(key) => harmless_on_every_row(&parse_sql_expr(key, self, Some(schema))?, schema),
        })
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

    // Lower `lhs [NOT] IN (subquery)` and `lhs = (subquery)` over an uncorrelated
    // subquery of any shape to a semi / anti join against the subquery evaluated
    // whole. A scalar subquery is reduced to its single value first, so more than
    // one row still raises.
    #[cfg(feature = "semi_anti_join")]
    pub(crate) fn try_rewrite_uncorrelated_subquery_conjunct(
        &mut self,
        lf: &LazyFrame,
        conj: &SQLExpr,
        filter_mode: FilterMode,
        outer_schema: &Schema,
    ) -> PolarsResult<Option<LazyFrame>> {
        let removing = filter_mode == FilterMode::RemoveTrue;
        let Some(Membership {
            key: lhs,
            subquery,
            negated,
            scalar,
        }) = membership_predicate(conj)
        else {
            return Ok(None);
        };
        if negated && removing {
            return Ok(None);
        }
        let anti = negated != removing;
        if is_correlated_subquery(subquery, self.catalog_spellings()) || expr_contains_subquery(lhs)
        {
            return Ok(None);
        }
        let Some(left_key) = self.try_parse_outer_only_expr(lhs, outer_schema)? else {
            return Ok(None);
        };

        let mut inner_lf = self.execute_isolated(|ctx| ctx.execute_query(subquery))?;
        let inner_schema = self.get_frame_schema(&mut inner_lf)?;
        polars_ensure!(inner_schema.len() == 1, SQLSyntax: "SQL subquery returns more than one column");
        let right_key = col(inner_schema.get_at_index(0).unwrap().0.clone());
        if scalar {
            inner_lf = inner_lf.select([first().as_expr().item(true)]);
        }

        build_semi_anti_join(
            lf,
            inner_lf,
            outer_schema,
            &inner_schema,
            vec![left_key],
            vec![right_key],
            anti,
            // Only `KeepTrue` "NOT IN" needs 3VL correction.
            anti && !scalar && filter_mode == FilterMode::KeepTrue,
        )
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
        // Both rewrites below split this into conjuncts.
        let selection = factored_selection(selection);
        // Resolve and parse the inner relation in an isolated context so its
        // table/alias registrations don't leak into the outer query's scope.
        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) = ctx.resolve_subquery_from(select)? else {
            return Ok(None);
        };
        // An aggregating subquery is left to `try_decorrelate_exists_subquery`.
        if exists_aggregates(&ctx, subquery, select, &inner_names, &inner_schema) != Some(false) {
            return Ok(None);
        }
        if let Some(conjuncts) = ctx.split_subquery_conjuncts(
            &selection,
            &inner_names,
            &inner_schema,
            outer_schema,
            self,
        )? {
            // An uncorrelated EXISTS (no correlation key found) has no join key
            // to build from, so leave it to the existing path.
            return Ok(if conjuncts.left_on.is_empty() {
                None
            } else {
                ctx.finish_decorrelated_join(
                    lf,
                    inner_lf,
                    outer_schema,
                    &inner_schema,
                    conjuncts,
                    negated,
                )?
            });
        }
        ctx.try_rewrite_exists_as_count_filter(
            lf,
            inner_lf,
            &selection,
            &inner_names,
            &inner_schema,
            outer_schema,
            self,
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
        inner_names: &[RelationName],
        inner_schema: &Schema,
        outer_schema: &Schema,
        outer: &SQLContext,
        negated: bool,
    ) -> PolarsResult<Option<LazyFrame>> {
        let mut corr_preds = Vec::new();
        let mut local_filters = Vec::new();
        for conj in MintermIter::new(selection) {
            if let Some(pred) =
                scalar_correlation_predicate(conj, inner_schema, outer_schema, &self, outer)?
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

        let Some(left_key) = self.try_parse_outer_only_expr(lhs, outer_schema)? else {
            return Ok(None);
        };

        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) = ctx.resolve_subquery_from(select)? else {
            return Ok(None);
        };
        // The membership key must be a plain expression over the inner relation;
        // any alias it carries is cosmetic and not allowed in a join key.
        let Some(right_key) = ctx.try_parse_inner_only_expr(proj, &inner_names, &inner_schema)?
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
                    self,
                )?
                else {
                    return Ok(None);
                };
                split
            },
            None => SubqueryConjuncts::default(),
        };

        left_on.insert(0, left_key);
        right_on.insert(0, right_key);

        let inner_lf = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
        inner_lf.set_cached_arena(ctx.lp_arena, ctx.expr_arena);
        build_semi_anti_join(
            lf,
            inner_lf,
            outer_schema,
            &inner_schema,
            left_on,
            right_on,
            anti,
            // Only `KeepTrue` "NOT IN" needs 3VL correction.
            anti && filter_mode == FilterMode::KeepTrue,
        )
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
        outer_schema: &Schema,
        inner_schema: &Schema,
        conjuncts: SubqueryConjuncts,
        anti: bool,
    ) -> PolarsResult<Option<LazyFrame>> {
        let SubqueryConjuncts {
            left_on,
            right_on,
            local_filters,
        } = conjuncts;
        let inner_lf = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
        inner_lf.set_cached_arena(self.lp_arena, self.expr_arena);
        build_semi_anti_join(
            lf,
            inner_lf,
            outer_schema,
            inner_schema,
            left_on,
            right_on,
            anti,
            false,
        )
    }

    // Resolve the subquery's FROM (relations, possibly with joins) into the inner
    // LazyFrame, its schema, and the set of relation names/aliases used to
    // classify qualified correlation columns. Also checks the SELECT and WHERE
    // for unqualified columns that are ambiguous between the joined relations.
    pub(crate) fn resolve_subquery_from(
        &mut self,
        select: &Select,
    ) -> PolarsResult<Option<(Vec<RelationName>, LazyFrame, SchemaRef)>> {
        let from = select.from.as_slice();
        let Some(inner_names) = from
            .iter()
            .flat_map(|tbl_expr| {
                std::iter::once(&tbl_expr.relation)
                    .chain(tbl_expr.joins.iter().map(|j| &j.relation))
            })
            .map(|factor| Some(relation_name(factor)?.spelled(factor, self.catalog_spellings())))
            .collect::<Option<Vec<_>>>()
        else {
            return Ok(None);
        };
        let Some((first, rest)) = from.split_first() else {
            return Ok(None);
        };
        // Qualifiers in the inner predicates resolve against the inner relations first.
        self.set_active_relations(from);
        let mut inner_lf = self.execute_from_statement(first)?;
        // Comma-separated relations cross join; the equalities linking them are
        // inner-only filters.
        for tbl_expr in rest {
            let Some(r_name) = get_table_name(&tbl_expr.relation) else {
                return Ok(None);
            };
            let mut rf = self.execute_from_statement(tbl_expr)?;
            let left_schema = self.get_frame_schema(&mut inner_lf)?;
            let right_schema = self.get_frame_schema(&mut rf)?;
            inner_lf = inner_lf.cross_join(rf, Some(format_pl_smallstr!(":{r_name}")));
            let joined_schema = self.get_frame_schema(&mut inner_lf)?;
            let right_relations: Vec<String> =
                declared_relations(std::slice::from_ref(tbl_expr), self.catalog_spellings())
                    .iter()
                    .map(RelationName::key)
                    .collect();
            self.record_joined_aliases(
                &r_name,
                &right_relations,
                &left_schema,
                &right_schema,
                &joined_schema,
            )?;
        }
        self.set_join_base(&select.from);
        self.check_select_column_refs(select)?;
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
        inner_names: &[RelationName],
        inner_schema: &Schema,
        outer_schema: &Schema,
        outer: &SQLContext,
    ) -> PolarsResult<Option<SubqueryConjuncts>> {
        let mut left_on = Vec::new();
        let mut right_on = Vec::new();
        let mut local_filters = Vec::new();
        for conj in MintermIter::new(selection) {
            if let Some((outer_key, inner_key)) =
                correlation_key_pair(conj, inner_schema, outer_schema, self, outer)?
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

    // The plan-time row statistics of a frame whose schema this context resolved,
    // or `None` when its plan isn't modelled.
    fn estimated_rows(&mut self, lf: &mut LazyFrame) -> PolarsResult<Option<NodeStats>> {
        self.get_frame_schema(lf)?;
        let DslPlan::IR {
            node: Some(node),
            version,
            ..
        } = &lf.logical_plan
        else {
            return Ok(None);
        };
        if *version != self.lp_arena.version() {
            return Ok(None);
        }
        attach_dataset_scan_statistics(*node, &mut self.lp_arena, &self.expr_arena);
        Ok(node_stats(*node, &self.lp_arena, &self.expr_arena))
    }

    // Split a subquery's WHERE into its correlation comparisons and its
    // inner-only filters, or `None` when a conjunct is neither. A comparison
    // spelled twice is kept once: it would duplicate a group-by key.
    fn split_correlation_conjuncts(
        &mut self,
        selection: &SQLExpr,
        inner_names: &[RelationName],
        inner_schema: &Schema,
        outer_schema: &Schema,
        outer: &SQLContext,
    ) -> PolarsResult<Option<(Vec<CorrPredicate>, Vec<Expr>)>> {
        let mut corr_preds: Vec<CorrPredicate> = Vec::new();
        let mut local_filters = Vec::new();
        let selection = factored_selection(selection);
        for conj in MintermIter::new(&selection) {
            if let Some(pred) =
                scalar_correlation_predicate(conj, inner_schema, outer_schema, self, outer)?
            {
                if !corr_preds.iter().any(|p| p.same_comparison(&pred)) {
                    corr_preds.push(pred);
                }
            } else if let Some(filter) =
                self.try_parse_inner_only_expr(conj, inner_names, inner_schema)?
            {
                local_filters.push(filter);
            } else {
                return Ok(None);
            }
        }
        Ok(Some((corr_preds, local_filters)))
    }

    // Parse an outer-query expression, or `None` if it doesn't stand on the outer
    // relation alone: an unlowered subquery, or a column the outer frame lacks.
    // Any alias is cosmetic here and stripped.
    fn try_parse_outer_only_expr(
        &mut self,
        sql_expr: &SQLExpr,
        outer_schema: &Schema,
    ) -> PolarsResult<Option<Expr>> {
        let expr = parse_sql_expr(sql_expr, self, Some(outer_schema))?
            .meta()
            .undo_aliases();
        if has_expr(&expr, |e| matches!(e, Expr::SubPlan(_, _)))
            || !expr_to_leaf_column_names_iter(&expr)
                .all(|name| outer_schema.contains(name.as_str()))
        {
            return Ok(None);
        }
        Ok(Some(expr))
    }

    // Parse a subquery expression as one over the inner relation only, or `None`
    // if it references any outer column (a correlation shape we don't handle) or
    // contains a nested subquery.
    fn try_parse_inner_only_expr(
        &mut self,
        sql_expr: &SQLExpr,
        inner_names: &[RelationName],
        inner_schema: &Schema,
    ) -> PolarsResult<Option<Expr>> {
        if !binds_to_inner_relation(sql_expr, inner_names, inner_schema) {
            return Ok(None);
        }
        Ok(Some(parse_sql_expr(sql_expr, self, Some(inner_schema))?))
    }

    // `values` sorted by the `ORDER BY` of a `LIMIT 1` subquery, whose first value
    // is then its result. The sort keys are pushed onto `row_columns`, to be
    // computed per inner row. `None` for an order that isn't read from the inner
    // row alone.
    #[allow(clippy::too_many_arguments)]
    fn try_first_row_order(
        &mut self,
        subquery: &Query,
        alias: Option<&Ident>,
        values: Expr,
        prefix: &str,
        inner_names: &[RelationName],
        inner_schema: &Schema,
        row_columns: &mut Vec<Expr>,
    ) -> PolarsResult<Option<Expr>> {
        let Some(order_by) = &subquery.order_by else {
            return Ok(Some(values));
        };
        let OrderByKind::Expressions(exprs) = &order_by.kind else {
            return Ok(None);
        };
        if order_by.interpolate.is_some() || exprs.is_empty() {
            return Ok(None);
        }
        let mut by = Vec::with_capacity(exprs.len());
        let mut descending = Vec::with_capacity(exprs.len());
        let mut nulls_last = Vec::with_capacity(exprs.len());
        for ob in exprs {
            if ob.with_fill.is_some() || expr_has_window_functions(&ob.expr) {
                return Ok(None);
            }
            let mut expr = &ob.expr;
            while let SQLExpr::Nested(inner) = expr {
                expr = inner;
            }
            // The output column, by name or by position.
            let is_output = match expr {
                SQLExpr::Identifier(ident) => alias.is_some_and(|a| a.value == ident.value),
                SQLExpr::Value(ValueWithSpan {
                    value: SQLValue::Number(n, _),
                    ..
                }) => {
                    if n != "1" {
                        return Ok(None);
                    }
                    true
                },
                _ => false,
            };
            let key = if is_output {
                values.clone()
            } else {
                let Some(key) = self.marking_aggregates(|ctx| {
                    ctx.try_parse_inner_only_expr(&ob.expr, inner_names, inner_schema)
                })?
                else {
                    return Ok(None);
                };
                if has_marked_aggregate(&key)
                    || has_expr(&key, |e| matches!(e, Expr::SubPlan(_, _)))
                {
                    return Ok(None);
                }
                let name = format_pl_smallstr!("{prefix}ord{}", row_columns.len());
                row_columns.push(key.alias(name.clone()));
                col(name)
            };
            let options = order_by_sort_options(&ob.options);
            by.push(key);
            descending.push(options.descending);
            nulls_last.push(options.nulls_last);
        }
        Ok(Some(
            values.sort_by(
                by,
                SortMultipleOptions::default()
                    .with_order_descending_multi(descending)
                    .with_nulls_last_multi(nulls_last),
            ),
        ))
    }

    // The join tree of a block whose frame is its FROM relations joined and
    // filtered by the WHERE conjuncts that hold no subquery, and nothing more.
    // `residual` is what the early subquery rewrites left of `where_expr`: the
    // frame is still the tree when that is every subquery conjunct and nothing
    // else. `None` as well when the FROM isn't a list of distinct plain tables
    // whose columns all have distinct names.
    pub(crate) fn outer_join_tree(
        &mut self,
        select: &Select,
        where_expr: &SQLExpr,
        residual: &[&SQLExpr],
    ) -> PolarsResult<Option<OuterJoinTree>> {
        let Some(selection) = &select.selection else {
            return Ok(None);
        };
        let n_with_subquery = MintermIter::new(where_expr)
            .filter(|c| expr_contains_subquery(c))
            .count();
        if select.from.len() < 2
            || residual.len() != n_with_subquery
            || !residual.iter().all(|c| expr_contains_subquery(c))
        {
            return Ok(None);
        }

        let mut relations: Vec<TreeRelation> = Vec::with_capacity(select.from.len());
        for tbl_expr in &select.from {
            let Some((table, name)) = plain_table(tbl_expr) else {
                return Ok(None);
            };
            if relations.iter().any(|r| r.table == table || r.name == name) {
                return Ok(None);
            }
            let mut rel_lf = self.execute_from_statement(tbl_expr)?;
            let schema = self.get_frame_schema(&mut rel_lf)?;
            relations.push(TreeRelation {
                table: table.to_string(),
                name: name.to_string(),
                schema,
            });
        }
        let n_columns: usize = relations.iter().map(|r| r.schema.len()).sum();
        let n_distinct = relations
            .iter()
            .flat_map(|r| r.schema.iter_names())
            .collect::<PlHashSet<_>>()
            .len();
        if n_distinct != n_columns {
            return Ok(None);
        }

        let own: Vec<(&str, usize)> = relations
            .iter()
            .enumerate()
            .map(|(idx, r)| (r.name.as_str(), idx))
            .collect();
        let selection = factored_selection(selection);
        let mut conjuncts = Vec::new();
        for conj in MintermIter::new(&selection).filter(|c| !expr_contains_subquery(c)) {
            let Some(spelled) = spell_conjunct(conj, &own, &relations, false) else {
                return Ok(None);
            };
            conjuncts.push(spelled);
        }
        Ok(Some(OuterJoinTree {
            relations,
            conjuncts,
        }))
    }

    // Lower every correlated subquery reachable from `expr` into a decorrelated
    // join over `lf`, substituting each one for the column its lowering
    // materialised. Returns the updated frame together with the rewritten
    // expression, which borrows `expr` unchanged when nothing was lowered.
    // Subqueries that can't be lowered are left in place.
    //
    // See [`SubqueryBindings`] for when a set may be shared across calls. Pass
    // `join_tree` only while `lf` is still the frame it describes.
    pub(crate) fn lower_correlated_subqueries<'a>(
        &mut self,
        lf: LazyFrame,
        outer_schema: &Schema,
        expr: &'a SQLExpr,
        scope: LowerScope,
        bindings: &mut SubqueryBindings,
        join_tree: Option<&OuterJoinTree>,
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
            join_tree,
            query_depth: 0,
            skips: Vec::new(),
            changed: false,
        };
        if let ControlFlow::Break(e) = dyn_visit::visit_mut(&mut rewritten, &mut lowering) {
            return Err(e);
        }
        let rewritten = if lowering.changed {
            Cow::Owned(rewritten)
        } else {
            Cow::Borrowed(expr)
        };
        Ok((lowering.lf, rewritten))
    }

    // Lower the correlated subqueries of a `JOIN ... ON` condition against the
    // input they read, one conjunct at a time, so that the condition reads
    // plain columns of the inputs. A conjunct that `correlated_join_side`
    // declines is left as it is. Returns the inputs with the lowered columns
    // added and the rewritten condition, or `None` when nothing was lowered.
    pub(crate) fn lower_join_on_subqueries(
        &mut self,
        tbl_left: &TableInfo,
        tbl_right: &TableInfo,
        on: &SQLExpr,
        join_type: &JoinType,
    ) -> PolarsResult<Option<(TableInfo, TableInfo, SQLExpr)>> {
        let mut left = tbl_left.frame.clone();
        let mut right = tbl_right.frame.clone();
        let mut left_bindings = SubqueryBindings::new();
        let mut right_bindings = SubqueryBindings::new();
        let mut changed = false;
        let mut conjuncts = Vec::new();
        for original in flatten_and_conditions(on) {
            let prepared =
                self.pull_up_nested_in(ScopeColumns::Join(tbl_left, tbl_right), original);
            let conj = prepared.as_ref().unwrap_or(original);
            let lowered;
            match correlated_join_side(self, conj, tbl_left, tbl_right) {
                Some(true) => {
                    (left, lowered) = self.lower_correlated_subqueries(
                        left,
                        &tbl_left.schema,
                        conj,
                        LowerScope::ScalarAndPredicates,
                        &mut left_bindings,
                        None,
                    )?;
                },
                Some(false) => {
                    (right, lowered) = self.lower_correlated_subqueries(
                        right,
                        &tbl_right.schema,
                        conj,
                        LowerScope::ScalarAndPredicates,
                        &mut right_bindings,
                        None,
                    )?;
                },
                None => {
                    lowered = match self.try_attach_join_membership(
                        conj, &mut left, &mut right, tbl_left, tbl_right, join_type,
                    )? {
                        Some(rewritten) => Cow::Owned(rewritten),
                        None => Cow::Borrowed(conj),
                    }
                },
            }
            match lowered {
                Cow::Owned(lowered) => {
                    changed = true;
                    conjuncts.push(lowered);
                },
                Cow::Borrowed(_) => conjuncts.push(original.clone()),
            }
        }
        if !changed {
            return Ok(None);
        }
        let left_schema = self.get_frame_schema(&mut left)?;
        let right_schema = self.get_frame_schema(&mut right)?;
        let on = combine_conditions(conjuncts, SQLBinaryOperator::And)
            .expect("a condition has at least one conjunct");
        Ok(Some((
            TableInfo {
                frame: left,
                name: tbl_left.name.clone(),
                schema: left_schema,
                relations: tbl_left.relations.clone(),
            },
            TableInfo {
                frame: right,
                name: tbl_right.name.clone(),
                schema: right_schema,
                relations: tbl_right.relations.clone(),
            },
            on,
        )))
    }

    // Join the distinct inner rows that make `lhs IN (subquery)` or `EXISTS (subquery)`
    // true onto one input of a join, when the subquery correlates with an input that the
    // rest of the conjunct doesn't read. The conjunct then becomes equalities between the
    // other input and the joined columns. For a pair of input rows at most one joined row
    // matches, so only an input whose unmatched rows the join drops can take them.
    fn try_attach_join_membership(
        &mut self,
        conj: &SQLExpr,
        left: &mut LazyFrame,
        right: &mut LazyFrame,
        tbl_left: &TableInfo,
        tbl_right: &TableInfo,
        join_type: &JoinType,
    ) -> PolarsResult<Option<SQLExpr>> {
        let (attach_left, attach_right) = match join_type {
            JoinType::Inner => (true, true),
            JoinType::Left => (false, true),
            JoinType::Right => (true, false),
            _ => return Ok(None),
        };
        let (lhs, subquery) = match strip_nested(conj) {
            SQLExpr::InSubquery {
                expr,
                subquery,
                negated: false,
            } => (Some(expr.as_ref()), subquery.as_ref()),
            SQLExpr::Exists {
                subquery,
                negated: false,
            } => (None, subquery.as_ref()),
            _ => return Ok(None),
        };
        let Some(select) = eligible_subquery_select(subquery) else {
            return Ok(None);
        };
        let Some(selection) = &select.selection else {
            return Ok(None);
        };
        if matches!(&select.distinct, Some(Distinct::On(_))) {
            return Ok(None);
        }
        // Planning the subquery is only worth it if its qualifiers read the other input.
        let lhs_is_left = match lhs.map(|lhs| self.join_inputs_read(lhs, tbl_left, tbl_right)) {
            Some((true, false)) => Some(true),
            Some((false, true)) => Some(false),
            Some(_) => return Ok(None),
            None => None,
        };
        let (corr_left, corr_right) = self.foreign_join_inputs(subquery, tbl_left, tbl_right);
        let reads_both = match lhs_is_left {
            Some(true) => corr_right,
            Some(false) => corr_left,
            None => corr_left && corr_right,
        };
        if !reads_both {
            return Ok(None);
        }
        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) = ctx.resolve_subquery_from(select)? else {
            return Ok(None);
        };
        if exists_aggregates(&ctx, subquery, select, &inner_names, &inner_schema) != Some(false) {
            return Ok(None);
        }

        // Pairs of an input expression and the inner expression it must equal.
        let mut keys: Vec<JoinMembershipKey> = Vec::new();
        if let Some(lhs) = lhs {
            let [SelectItem::UnnamedExpr(proj) | SelectItem::ExprWithAlias { expr: proj, .. }] =
                select.projection.as_slice()
            else {
                return Ok(None);
            };
            let Some(value) = ctx.try_parse_inner_only_expr(proj, &inner_names, &inner_schema)?
            else {
                return Ok(None);
            };
            // The value is taken per inner row, before the rows are grouped by outer row.
            if !is_elementwise_over(&value, &inner_schema) {
                return Ok(None);
            }
            let is_left = lhs_is_left.unwrap();
            let tbl = if is_left { tbl_left } else { tbl_right };
            let Some(outer) = self.try_parse_outer_only_expr(lhs, &tbl.schema)? else {
                return Ok(None);
            };
            keys.push(JoinMembershipKey {
                is_left,
                outer,
                inner: value.meta().undo_aliases(),
            });
        }
        let mut local_filters = Vec::new();
        let selection = factored_selection(selection);
        for c in MintermIter::new(&selection) {
            if let Some(key) =
                self.join_correlation_key(c, &ctx, &inner_schema, tbl_left, tbl_right)?
            {
                keys.push(key);
            } else if let Some(filter) =
                ctx.try_parse_inner_only_expr(c, &inner_names, &inner_schema)?
                && !has_expr(&filter, |e| matches!(e, Expr::SubPlan(_, _)))
            {
                local_filters.push(filter);
            } else {
                return Ok(None);
            }
        }
        let reads_left = keys.iter().any(|k| k.is_left);
        let reads_right = keys.iter().any(|k| !k.is_left);
        if !(reads_left && reads_right) {
            return Ok(None);
        }
        // With a choice, the rows join onto the input the `IN` operand doesn't read.
        let to_left = match (attach_left, attach_right) {
            (true, true) => lhs.is_none() || !keys[0].is_left,
            (to_left, _) => to_left,
        };

        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let names: Vec<PlSmallStr> = (0..keys.len())
            .map(|i| format_pl_smallstr!("{prefix}{i}"))
            .collect();
        let mut rows = local_filters
            .into_iter()
            .fold(inner_lf, LazyFrame::filter)
            .select(
                keys.iter()
                    .zip(&names)
                    .map(|(k, name)| k.inner.clone().alias(name.clone()))
                    .collect::<Vec<_>>(),
            );
        let rows_schema = ctx.get_frame_schema(&mut rows)?;
        rows.set_cached_arena(ctx.lp_arena, ctx.expr_arena);

        // Each key is compared as a column of its input, both sides cast to a shared type.
        let (mut left_frame, mut right_frame) = (left.clone(), right.clone());
        let mut row_casts = Vec::with_capacity(keys.len());
        let (mut target_on, mut rows_on, mut equalities) = (Vec::new(), Vec::new(), Vec::new());
        for (i, (key, name)) in keys.into_iter().zip(names).enumerate() {
            let frame = if key.is_left {
                &mut left_frame
            } else {
                &mut right_frame
            };
            let key_name = format_pl_smallstr!("{prefix}key{i}");
            let mut keyed = frame.clone().with_column(key.outer.alias(key_name.clone()));
            let keyed_schema = self.get_frame_schema(&mut keyed)?;
            let Some((outer, inner)) = try_prepare_join_keys(
                &keyed_schema,
                &rows_schema,
                vec![col(key_name.clone())],
                vec![col(name.clone())],
            ) else {
                return Ok(None);
            };
            let [outer] = <[Expr; 1]>::try_from(outer).unwrap();
            let [inner] = <[Expr; 1]>::try_from(inner).unwrap();
            *frame = keyed.with_column(outer.alias(key_name.clone()));
            row_casts.push(inner.alias(name.clone()));
            if key.is_left == to_left {
                target_on.push(col(key_name));
                rows_on.push(col(name));
            } else {
                equalities.push(SQLExpr::BinaryOp {
                    left: Box::new(SQLExpr::Identifier(Ident::new(key_name.as_str()))),
                    op: SQLBinaryOperator::Eq,
                    right: Box::new(SQLExpr::Identifier(Ident::new(name.as_str()))),
                });
            }
        }
        let rows = rows
            .with_columns(row_casts)
            .unique(None, UniqueKeepStrategy::Any);
        let (target, other, mut target_frame, other_frame) = if to_left {
            (left, right, left_frame, right_frame)
        } else {
            (right, left, right_frame, left_frame)
        };
        target_frame = target_frame
            .join_builder()
            .with(rows)
            .left_on(target_on)
            .right_on(rows_on)
            .how(JoinType::Inner)
            .coalesce(JoinCoalesce::KeepColumns)
            .finish()?;
        *target = target_frame;
        *other = other_frame;
        Ok(combine_conditions(equalities, SQLBinaryOperator::And))
    }

    // Which inputs of a join the qualifiers that `subquery` doesn't declare name, as
    // `(reads_left, reads_right)`.
    fn foreign_join_inputs(
        &self,
        subquery: &Query,
        tbl_left: &TableInfo,
        tbl_right: &TableInfo,
    ) -> (bool, bool) {
        let (mut reads_left, mut reads_right) = (false, false);
        visit_foreign_qualifiers(subquery, self.catalog_spellings(), |parts| {
            if let Ok(Some((name, _))) = self.resolve_qualifier(parts) {
                if self.is_right_relation(&name, tbl_right) {
                    reads_right = true;
                } else if self.is_joined_relation(&name, tbl_left) {
                    reads_left = true;
                }
            }
            ControlFlow::Continue(())
        });
        (reads_left, reads_right)
    }

    // An equality between a column of the subquery and a column of one input of the
    // join around it, either way round.
    fn join_correlation_key(
        &self,
        conj: &SQLExpr,
        inner: &SQLContext,
        inner_schema: &Schema,
        tbl_left: &TableInfo,
        tbl_right: &TableInfo,
    ) -> PolarsResult<Option<JoinMembershipKey>> {
        let SQLExpr::BinaryOp {
            left,
            op: SQLBinaryOperator::Eq,
            right,
        } = strip_nested(conj)
        else {
            return Ok(None);
        };
        for (inner_sql, outer_sql) in [(left, right), (right, left)] {
            let Some((CorrelationSide::Inner, inner_name)) = classify_correlation_column(
                inner_sql,
                inner_schema,
                &Schema::default(),
                inner,
                self,
            )?
            else {
                continue;
            };
            let Some(is_left) = self.column_join_input(outer_sql, tbl_left, tbl_right) else {
                continue;
            };
            let tbl = if is_left { tbl_left } else { tbl_right };
            let Some((CorrelationSide::Outer, outer_name)) =
                classify_correlation_column(outer_sql, inner_schema, &tbl.schema, inner, self)?
            else {
                continue;
            };
            return Ok(Some(JoinMembershipKey {
                is_left,
                outer: col(outer_name),
                inner: col(inner_name),
            }));
        }
        Ok(None)
    }

    // Attempt the decorrelation of a single scalar subquery:
    //   (SELECT AGG(...) FROM inner WHERE <corr-preds> AND <inner-filters>)
    // A projection without an aggregate is taken as the `item` of its values,
    // which raises when an outer row matches more than one inner row.
    // Equality-only correlation: `inner GROUP BY <corr cols>` once, joined onto
    // the outer frame on those columns directly. Otherwise: row-index the outer
    // frame, inner-join outer × inner on the correlation predicates (`join_where`
    // handles inequality), aggregate per outer row, then left-join back on the
    // row index. Either way, `COUNT` over no matches is 0, every other aggregate
    // is NULL. Returns `None` when uncorrelated or not a supported shape.
    // When `join_tree` describes the outer frame and the subquery repeats a part
    // of it, the aggregate is taken over the outer frame itself instead.
    fn try_decorrelate_scalar_subquery(
        &mut self,
        mut lf: LazyFrame,
        outer_schema: &Schema,
        subquery: &Query,
        join_tree: Option<&OuterJoinTree>,
    ) -> PolarsResult<Option<(LazyFrame, PlSmallStr)>> {
        let Some(first_row) = first_row_limit(subquery) else {
            return Ok(None);
        };
        let Some(select) = eligible_subquery_body(subquery) else {
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
        let Some((inner_names, inner_lf, inner_schema)) = ctx.resolve_subquery_from(select)? else {
            return Ok(None);
        };

        // A window gives one value per inner row, not one per group.
        if expr_has_window_functions(proj) {
            return Ok(None);
        }
        let Some(proj_expr) = ctx.marking_aggregates(|ctx| {
            ctx.try_parse_inner_only_expr(proj, &inner_names, &inner_schema)
        })?
        else {
            return Ok(None);
        };
        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let has_aggregate = has_marked_aggregate(&proj_expr);
        let proj_expr = strip_aggregate_marks(proj_expr);
        // The projection is either a scalar aggregate over the inner relation, or a
        // value per inner row. Without `LIMIT 1`, each outer row may match at most
        // one inner row. With it, the first row in the subquery's order is taken.
        let mut row_columns: Vec<Expr> = Vec::new();
        let (agg_expr, single_row_check) =
            if has_expr(&proj_expr, |e| matches!(e, Expr::Agg(_) | Expr::Len)) {
                // An aggregate within a row-expanding function gives several rows.
                if !is_scalar_over(&proj_expr, &inner_schema) {
                    return Ok(None);
                }
                (proj_expr, false)
            } else if !has_aggregate
                && !matches!(select.distinct, Some(Distinct::On(_)))
                && !has_expr(&proj_expr, |e| matches!(e, Expr::SubPlan(_, _)))
            {
                let value_name = format_pl_smallstr!("{prefix}val");
                let values = col(value_name.clone());
                row_columns.push(proj_expr.alias(value_name));
                if first_row {
                    let alias = match &select.projection[0] {
                        SelectItem::ExprWithAlias { alias, .. } => Some(alias),
                        _ => None,
                    };
                    let Some(ordered) = ctx.try_first_row_order(
                        subquery,
                        alias,
                        values,
                        &prefix,
                        &inner_names,
                        &inner_schema,
                        &mut row_columns,
                    )?
                    else {
                        return Ok(None);
                    };
                    (ordered.first(), false)
                } else {
                    let values = match select.distinct {
                        Some(Distinct::Distinct) => values.unique(),
                        _ => values,
                    };
                    (values.item(true), true)
                }
            } else {
                return Ok(None);
            };

        let Some((corr_preds, local_filters)) = ctx.split_correlation_conjuncts(
            selection,
            &inner_names,
            &inner_schema,
            outer_schema,
            self,
        )?
        else {
            return Ok(None);
        };
        // No correlation: leave it to the uncorrelated scalar-subquery path.
        if corr_preds.is_empty() {
            return Ok(None);
        }

        let result_name = format_pl_smallstr!("{prefix}res");
        // Without a semi join to restrict the inner groups, a value per row is taken
        // per matched outer row instead.
        let equality_only = corr_preds.iter().all(|p| p.op == SQLBinaryOperator::Eq)
            && (cfg!(feature = "semi_anti_join") || !single_row_check);

        // When the subquery repeats a part of the outer join tree, the outer frame
        // holds each outer row's subquery rows, each one repeated once per match
        // of the row's other relations on the keys. An aggregate that ignores such
        // repeats is then read from the outer frame, grouped by the outer keys.
        if equality_only
            && ignores_uniform_repeats(&agg_expr)
            && let Some(tree) = join_tree
            && tree.covers(&select.from, &factored_selection(selection), &inner_schema)
        {
            let mut keys: Vec<Expr> = Vec::with_capacity(corr_preds.len());
            for p in &corr_preds {
                let key = col(p.outer.clone());
                if !keys.contains(&key) {
                    keys.push(key);
                }
            }
            if expr_to_leaf_column_names_iter(&agg_expr)
                .chain(corr_preds.iter().map(|p| p.outer.clone()))
                .all(|name| outer_schema.contains(&name))
            {
                let outer = lf.cache();
                let aggregate = ScalarAggregate::new(agg_expr, &result_name, &prefix, outer_schema);
                let joined = equality_correlated_aggregate(
                    outer.clone(),
                    outer,
                    keys.clone(),
                    keys,
                    &aggregate,
                    false,
                )?;
                return Ok(Some((joined, result_name)));
            }
        }

        // Apply inner-only filters, then rename inner columns to collision-free
        // names so they can't clash with outer columns of the same name (as in a
        // self-correlated `t1 AS x` subquery over outer `t1`).
        let mut inner_filtered = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);
        if !row_columns.is_empty() {
            inner_filtered = inner_filtered.with_columns(row_columns);
        }
        let rename_from: Vec<PlSmallStr> = inner_schema.iter_names().cloned().collect();
        let rename_to: Vec<PlSmallStr> = rename_from
            .iter()
            .map(|name| prefixed_inner(&prefix, name))
            .collect();
        let mut inner_renamed = inner_filtered.rename(&rename_from, &rename_to, true);
        // The single-row check must only see the groups that outer rows ask for.
        let restrict = equality_only
            && group_local_aggregate(&agg_expr)
            && (single_row_check || {
                let inner_rows = ctx.estimated_rows(&mut inner_renamed)?;
                let outer_rows = self.estimated_rows(&mut lf)?;
                restriction_pays_off(outer_rows.as_ref(), inner_rows.as_ref())
            });
        inner_renamed.set_cached_arena(ctx.lp_arena, ctx.expr_arena);

        let aggregate = ScalarAggregate::new(agg_expr, &result_name, &prefix, &inner_schema)
            .map_inputs(|e| {
                e.map_expr(|e| match e {
                    Expr::Column(name) if inner_schema.contains(name.as_str()) => {
                        col(prefixed_inner(&prefix, &name))
                    },
                    other => other,
                })
            });

        if equality_only {
            let outer_on: Vec<Expr> = corr_preds.iter().map(|p| col(p.outer.clone())).collect();
            let inner_on: Vec<Expr> = corr_preds
                .iter()
                .map(|p| col(prefixed_inner(&prefix, &p.inner)))
                .collect();
            let joined = equality_correlated_aggregate(
                lf,
                inner_renamed,
                outer_on,
                inner_on,
                &aggregate,
                restrict,
            )?;
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
            .agg(aggregate.aggs.clone());

        let joined = aggregate
            .finish(left_join_aggregate(
                outer_indexed,
                grouped,
                vec![col(idx_name.clone())],
                vec![col(idx_name.clone())],
            )?)
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

        let Some(needle) = self.try_parse_outer_only_expr(lhs, outer_schema)? else {
            return Ok(None);
        };

        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) = ctx.resolve_subquery_from(select)? else {
            return Ok(None);
        };
        let Some(value) = ctx.try_parse_inner_only_expr(proj, &inner_names, &inner_schema)? else {
            return Ok(None);
        };
        let value = value.meta().undo_aliases();

        let Some((corr_preds, local_filters)) = ctx.split_correlation_conjuncts(
            selection,
            &inner_names,
            &inner_schema,
            outer_schema,
            self,
        )?
        else {
            return Ok(None);
        };
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
            left_join_aggregate(lf, grouped, outer_on, inner_on)?
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
            sql_is_in(needle, set.clone()),
            set.clone(),
            set.clone().is_null().or(set.list().len().eq(lit(0u32))),
        );
        // The boolean replaces the candidate list in place.
        let joined = joined.with_columns([is_in.alias(set_name.clone())]);
        Ok(Some((joined, set_name)))
    }

    /// Whether the uncorrelated `subquery` aggregates without `GROUP BY`, and so returns
    /// exactly one row.
    pub(crate) fn exists_over_aggregate(&mut self, subquery: &Query) -> PolarsResult<bool> {
        let Some(select) = eligible_subquery_select(subquery) else {
            return Ok(false);
        };
        let mut ctx = self.isolated();
        let Some((inner_names, inner_lf, inner_schema)) = ctx.resolve_subquery_from(select)? else {
            return Ok(false);
        };
        match exists_aggregates(&ctx, subquery, select, &inner_names, &inner_schema) {
            Some(true) => {
                // The rows are not read, but WHERE must still be valid.
                if let Some(selection) = &select.selection {
                    let mut filtered = ctx.process_where(
                        inner_lf,
                        Some(selection),
                        FilterMode::KeepTrue,
                        Some(inner_schema),
                    )?;
                    ctx.get_frame_schema(&mut filtered)?;
                }
                Ok(true)
            },
            Some(false) => Ok(false),
            None => polars_bail!(
                SQLInterface:
                "EXISTS subquery is not currently supported in this position: {:?}", subquery
            ),
        }
    }

    // Attempt the decorrelation of a single `EXISTS` subquery into a boolean flag
    // column holding `count(*) > 0` over the correlated match set. An uncorrelated
    // subquery is one value instead, kept in `exists_values` for `visit_exists`, and
    // `None` is returned. Also `None` for shapes that can't be soundly classified.
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
        let Some((inner_names, inner_lf, inner_schema)) = ctx.resolve_subquery_from(select)? else {
            return Ok(None);
        };

        let (corr_preds, local_filters) = match &select.selection {
            Some(selection) => {
                let Some(split) = ctx.split_correlation_conjuncts(
                    selection,
                    &inner_names,
                    &inner_schema,
                    outer_schema,
                    self,
                )?
                else {
                    return Ok(None);
                };
                split
            },
            None => (Vec::new(), Vec::new()),
        };

        let prefix = format_pl_smallstr!("{CORRELATED_COL_PREFIX}{}_", unique_column_name());
        let flag_name = format_pl_smallstr!("{prefix}exists");
        // An uncorrelated subquery is one value, which `visit_exists` takes.
        if corr_preds.is_empty() && !is_correlated_subquery(subquery, self.catalog_spellings()) {
            let exists =
                match exists_aggregates(&ctx, subquery, select, &inner_names, &inner_schema) {
                    None => return Ok(None),
                    Some(true) => lit(true),
                    Some(false) => {
                        let inner = local_filters
                            .into_iter()
                            .fold(inner_lf, LazyFrame::filter)
                            .select([len().gt(lit(0))]);
                        inner.set_cached_arena(ctx.lp_arena, ctx.expr_arena);
                        reduced_subquery(inner, first().as_expr().item(true))
                    },
                };
            self.exists_values.push((subquery.clone(), exists));
            return Ok(None);
        }
        match exists_aggregates(&ctx, subquery, select, &inner_names, &inner_schema) {
            None => return Ok(None),
            // An aggregate without GROUP BY returns one row, even over no rows.
            Some(true) => {
                return Ok(Some((
                    lf.with_column(lit(true).alias(flag_name.clone())),
                    flag_name,
                )));
            },
            Some(false) => {},
        }
        let inner_filtered = local_filters.into_iter().fold(inner_lf, LazyFrame::filter);

        if corr_preds.is_empty() {
            // Outer columns are only in the projection, which EXISTS ignores, so the
            // existence is a single constant boolean broadcast onto every row.
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

/// When WHERE conjuncts are offered to the join rewrites.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum RewriteStage {
    /// Before the correlated scalar subqueries are lowered: a conjunct moves
    /// ahead of them only if it is harmless on every row.
    BeforeScalarLowering,
    /// After: every conjunct is offered.
    AfterScalarLowering,
}

impl RewriteStage {
    fn offers(self, expr: &Expr, schema: &Schema) -> bool {
        self == Self::AfterScalarLowering || harmless_on_every_row(expr, schema)
    }
}

// A `[NOT] IN (subquery)` or `= (subquery)` conjunct taken apart.
#[cfg_attr(not(feature = "semi_anti_join"), allow(dead_code))]
struct Membership<'a> {
    /// The outer-side expression compared against the subquery.
    key: &'a SQLExpr,
    subquery: &'a Query,
    negated: bool,
    /// `= (subquery)`: the subquery stands for its single value.
    scalar: bool,
}

fn membership_predicate(conj: &SQLExpr) -> Option<Membership<'_>> {
    match unwrap_nested(conj) {
        SQLExpr::InSubquery {
            expr,
            subquery,
            negated,
        } => Some(Membership {
            key: expr,
            subquery,
            negated: *negated,
            scalar: false,
        }),
        SQLExpr::BinaryOp {
            left,
            op: SQLBinaryOperator::Eq,
            right,
        } => match (left.as_ref(), right.as_ref()) {
            (key, SQLExpr::Subquery(subquery)) | (SQLExpr::Subquery(subquery), key) => {
                Some(Membership {
                    key,
                    subquery,
                    negated: false,
                    scalar: true,
                })
            },
            _ => None,
        },
        _ => None,
    }
}

fn unwrap_nested(mut expr: &SQLExpr) -> &SQLExpr {
    while let SQLExpr::Nested(inner) = expr {
        expr = inner;
    }
    expr
}

// Whether the expression can be evaluated on rows a later filter removes: it
// is elementwise, infallible and deterministic by the planner's classification.
fn harmless_on_every_row(expr: &Expr, schema: &Schema) -> bool {
    let mut arena = Arena::new();
    let mut ctx = ExprToIRContext::new(&mut arena, schema);
    let Ok(expr) = to_expr_ir(expr.clone(), &mut ctx) else {
        return false;
    };
    let mut group = ExprPushdownGroup::Pushable;
    group.update_with_expr_rec(arena.get(expr.node()), &arena, None);
    matches!(group, ExprPushdownGroup::Pushable)
        && !is_inherently_nondeterministic(expr.node(), &arena)
}

// Semi/anti join the outer frame against the (filtered, arena-cached) inner, or
// `None` for keys the join cannot stand in for the comparison on. The first key
// pair is the membership key, the rest correlation keys. With `correct_not_in`
// the anti join is refined to `NOT IN`'s three-valued logic.
#[cfg(feature = "semi_anti_join")]
#[allow(clippy::too_many_arguments)]
fn build_semi_anti_join(
    lf: &LazyFrame,
    inner_lf: LazyFrame,
    outer_schema: &Schema,
    inner_schema: &Schema,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
    anti: bool,
    correct_not_in: bool,
) -> PolarsResult<Option<LazyFrame>> {
    let join_type = if anti { JoinType::Anti } else { JoinType::Semi };
    let Some((left_on, right_on)) =
        try_prepare_join_keys(outer_schema, inner_schema, left_on, right_on)
    else {
        return Ok(None);
    };
    let joined = lf
        .clone()
        .join_builder()
        .with(inner_lf.clone())
        .left_on(left_on.clone())
        .right_on(right_on.clone())
        .how(join_type)
        .finish()?;
    if !correct_not_in {
        return Ok(Some(joined));
    }
    let (left_key, corr_outer) = left_on.split_first().unwrap();
    let (right_key, corr_inner) = right_on.split_first().unwrap();
    Ok(Some(refine_not_in_anti_join(
        joined, inner_lf, left_key, right_key, corr_outer, corr_inner,
    )?))
}

// The keys of a semi / anti join standing in for a comparison, numeric pairs cast
// to their supertype as the comparison would. `None` for a pair the join cannot
// stand in for: not elementwise over columns, reading no column, holding an
// untyped literal (whose type the comparison would take from the other operand),
// of unresolvable types, or of different non-numeric types.
fn try_prepare_join_keys(
    outer_schema: &Schema,
    inner_schema: &Schema,
    mut left_on: Vec<Expr>,
    mut right_on: Vec<Expr>,
) -> Option<(Vec<Expr>, Vec<Expr>)> {
    let untyped_literal = |e: &Expr| matches!(e, Expr::Literal(LiteralValue::Dyn(_)));
    for (left, right) in left_on.iter_mut().zip(right_on.iter_mut()) {
        if !is_elementwise(left)
            || !is_elementwise(right)
            || expr_to_leaf_column_names_iter(left).next().is_none()
            || has_expr(left, untyped_literal)
            || has_expr(right, untyped_literal)
        {
            return None;
        }
        let l = left.to_field(outer_schema).ok()?.dtype;
        let r = right.to_field(inner_schema).ok()?.dtype;
        if l == r {
            continue;
        }
        if !(l.is_primitive_numeric() && r.is_primitive_numeric()) {
            return None;
        }
        let st = try_get_supertype(&l, &r).ok()?;
        *left = std::mem::take(left).cast(st.clone());
        *right = std::mem::take(right).cast(st);
    }
    Some((left_on, right_on))
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

/// An iterator over the terms that `OR` together to form an SQL boolean
/// expression, descending through parenthesized `Nested` expressions.
struct DisjunctIter<'a> {
    stack: Vec<&'a SQLExpr>,
}

impl<'a> Iterator for DisjunctIter<'a> {
    type Item = &'a SQLExpr;

    fn next(&mut self) -> Option<Self::Item> {
        let mut top = self.stack.pop()?;
        loop {
            match top {
                SQLExpr::Nested(inner) => top = inner,
                SQLExpr::BinaryOp {
                    left,
                    op: SQLBinaryOperator::Or,
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

impl<'a> DisjunctIter<'a> {
    fn new(root: &'a SQLExpr) -> Self {
        Self { stack: vec![root] }
    }
}

/// Rewrite `(A AND X) OR (A AND Y)` as `A AND (X OR Y)`.
///
/// The SQL-AST analogue of `polars_plan::plans::aexpr::or_factoring`, which runs
/// after the decorrelation decision made here.
fn factor_common_conjuncts(expr: &SQLExpr) -> Option<SQLExpr> {
    let branches: Vec<Vec<&SQLExpr>> = DisjunctIter::new(expr)
        .map(|branch| MintermIter::new(branch).collect())
        .collect();
    let (first, rest) = branches.split_first()?;
    if rest.is_empty() {
        return None;
    }
    let common: Vec<&SQLExpr> = first
        .iter()
        .copied()
        .filter(|term| rest.iter().all(|branch| branch.contains(term)))
        .collect();
    let factored = combine_conditions(
        common.iter().copied().cloned().collect(),
        SQLBinaryOperator::And,
    )?;

    // A branch holding nothing but the common terms absorbs the others:
    // `A OR (A AND Y)` is just `A`.
    let remainders: Option<Vec<SQLExpr>> = branches
        .iter()
        .map(|terms| {
            combine_conditions(
                terms
                    .iter()
                    .copied()
                    .filter(|term| !common.contains(term))
                    .cloned()
                    .collect(),
                SQLBinaryOperator::And,
            )
        })
        .collect();
    let Some(remainders) = remainders else {
        return Some(factored);
    };
    let disjunction = combine_conditions(remainders, SQLBinaryOperator::Or)?;
    Some(SQLExpr::BinaryOp {
        left: Box::new(factored),
        op: SQLBinaryOperator::And,
        right: Box::new(SQLExpr::Nested(Box::new(disjunction))),
    })
}

/// Hoist predicates shared by every branch of an `OR` to the top level.
fn factored_selection(selection: &SQLExpr) -> Cow<'_, SQLExpr> {
    let factored: Vec<Option<SQLExpr>> = MintermIter::new(selection)
        .map(factor_common_conjuncts)
        .collect();
    if factored.iter().all(Option::is_none) {
        return Cow::Borrowed(selection);
    }
    let conjuncts = MintermIter::new(selection)
        .zip(factored)
        .map(|(conj, factored)| factored.unwrap_or_else(|| conj.clone()))
        .collect();
    combine_conditions(conjuncts, SQLBinaryOperator::And)
        .map_or(Cow::Borrowed(selection), Cow::Owned)
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
    inner_schema: &Schema,
    outer_schema: &Schema,
    inner: &SQLContext,
    outer: &SQLContext,
) -> PolarsResult<Option<(PlSmallStr, PlSmallStr)>> {
    let SQLExpr::BinaryOp {
        left,
        op: SQLBinaryOperator::Eq,
        right,
    } = conj
    else {
        return Ok(None);
    };
    let Some((lside, lname)) =
        classify_correlation_column(left, inner_schema, outer_schema, inner, outer)?
    else {
        return Ok(None);
    };
    let Some((rside, rname)) =
        classify_correlation_column(right, inner_schema, outer_schema, inner, outer)?
    else {
        return Ok(None);
    };
    Ok(match (lside, rside) {
        (CorrelationSide::Outer, CorrelationSide::Inner) => Some((lname, rname)),
        (CorrelationSide::Inner, CorrelationSide::Outer) => Some((rname, lname)),
        _ => None,
    })
}

// Whether every column the expression references resolves to the inner relation,
// so it can be evaluated against the inner frame alone. A qualified name resolves
// through its qualifier: only an inner relation's name or alias counts as inner.
// An unqualified name binds to the innermost scope that has it, so the inner
// schema wins over the outer one.
fn binds_to_inner_relation(
    expr: &SQLExpr,
    inner_names: &[RelationName],
    inner_schema: &Schema,
) -> bool {
    visit_expressions(expr, |e| {
        let resolves = match e {
            // A nested subquery is its own scope, which this cannot resolve against.
            _ if is_subquery_expr(e) => false,
            SQLExpr::Identifier(ident) => inner_schema.contains(&ident.value),
            SQLExpr::CompoundIdentifier(parts) => match_qualifier(parts, inner_names.iter())
                .is_some_and(|(_, len)| inner_schema.contains(&parts[len].value)),
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

// Classify a correlation operand as an inner- or outer-query column and return
// its column name in that frame. `None` for non-identifiers or names in neither
// schema.
fn classify_correlation_column(
    expr: &SQLExpr,
    inner_schema: &Schema,
    outer_schema: &Schema,
    inner: &SQLContext,
    outer: &SQLContext,
) -> PolarsResult<Option<(CorrelationSide, PlSmallStr)>> {
    let (qualifier, name) = match expr {
        SQLExpr::Identifier(ident) => (None, ident.value.as_str()),
        SQLExpr::CompoundIdentifier(parts) => {
            // A relation of the outer query is outer (so same-named columns like
            // `o.id = c.id` resolve). Any other qualifier, or a struct column of the
            // subquery, is not an outer column.
            let (side, q, len) = match inner.resolve_qualifier(parts)? {
                Some((q, len)) => (CorrelationSide::Inner, q, len),
                None => match outer.resolve_qualifier(parts)? {
                    Some((q, len))
                        if !matches!(
                            inner_schema.get(&parts[0].value),
                            Some(DataType::Struct(_))
                        ) =>
                    {
                        (CorrelationSide::Outer, q, len)
                    },
                    _ => return Ok(None),
                },
            };
            // A struct field is not a column of either frame.
            if len + 1 != parts.len() {
                return Ok(None);
            }
            (Some((side, q)), parts[len].value.as_str())
        },
        _ => return Ok(None),
    };
    // A joined relation's column may be stored under another name.
    Ok(match qualifier {
        Some((CorrelationSide::Inner, q)) => {
            let name = inner.resolve_name(&q, name);
            inner_schema
                .contains(name.as_str())
                .then(|| (CorrelationSide::Inner, name.into()))
        },
        Some((CorrelationSide::Outer, q)) => {
            let name = outer.resolve_name(&q, name);
            outer_schema
                .contains(name.as_str())
                .then(|| (CorrelationSide::Outer, name.into()))
        },
        // An unqualified name binds to the innermost scope that holds it. An
        // outer one raises if it is ambiguous between the outer joins.
        None => match (inner_schema.contains(name), outer_schema.contains(name)) {
            (true, _) => Some((CorrelationSide::Inner, name.into())),
            (false, true) => {
                outer.check_unqualified_column(name)?;
                Some((CorrelationSide::Outer, name.into()))
            },
            (false, false) => None,
        },
    })
}

/// How many rows an `EXISTS` subquery that passed `eligible_subquery_select` returns because
/// of its aggregates: `Some(true)` if it aggregates its own rows and so returns exactly one,
/// `Some(false)` if it does not aggregate, and `None` for shapes left unsupported (an
/// aggregate over outer columns only, which belongs to the outer query, or `UNNEST`).
pub(crate) fn exists_aggregates(
    ctx: &SQLContext,
    query: &Query,
    select: &Select,
    inner_names: &[RelationName],
    inner_schema: &Schema,
) -> Option<bool> {
    /// Whether the columns an aggregate call reads, outside of its subqueries, are inner
    /// or outer ones.
    struct Columns<'a> {
        inner_names: &'a [RelationName],
        inner_schema: &'a Schema,
        query_depth: usize,
        inner: bool,
        outer: bool,
    }

    impl Visitor for Columns<'_> {
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
            let is_inner = match expr {
                _ if self.query_depth > 0 => return ControlFlow::Continue(()),
                SQLExpr::Identifier(ident) => self.inner_schema.contains(&ident.value),
                SQLExpr::CompoundIdentifier(parts) => {
                    match_qualifier(parts, self.inner_names.iter()).is_some()
                        || self.inner_schema.contains(&parts[0].value)
                },
                _ => return ControlFlow::Continue(()),
            };
            if is_inner {
                self.inner = true;
            } else {
                self.outer = true;
            }
            ControlFlow::Continue(())
        }
    }

    struct Aggregates<'a> {
        ctx: &'a SQLContext,
        inner_names: &'a [RelationName],
        inner_schema: &'a Schema,
        query_depth: usize,
        local: bool,
        outer: bool,
        unsupported: bool,
        /// Named windows used so far, whose definitions may hold aggregates.
        windows: Vec<String>,
    }

    impl Visitor for Aggregates<'_> {
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
            let SQLExpr::Function(function) = expr else {
                return ControlFlow::Continue(());
            };
            if self.query_depth > 0 {
                return ControlFlow::Continue(());
            }
            match &function.over {
                Some(WindowType::NamedWindow(name))
                | Some(WindowType::WindowSpec(WindowSpec {
                    window_name: Some(name),
                    ..
                })) => self.windows.push(name.value.clone()),
                _ => {},
            }
            match PolarsSQLFunctions::try_from_sql(function, self.ctx) {
                Ok(PolarsSQLFunctions::Explode) => self.unsupported = true,
                Ok(f) if f.is_builtin_aggregate() && function.over.is_none() => {
                    // An aggregate belongs to the outer query if it reads only outer columns,
                    // in its arguments or its FILTER clause.
                    let mut columns = Columns {
                        inner_names: self.inner_names,
                        inner_schema: self.inner_schema,
                        query_depth: 0,
                        inner: false,
                        outer: false,
                    };
                    let _ = dyn_visit::visit(&function.args, &mut columns);
                    let _ = dyn_visit::visit(&function.filter, &mut columns);
                    if columns.inner || !columns.outer {
                        self.local = true;
                    } else {
                        self.outer = true;
                    }
                },
                _ => {},
            }
            ControlFlow::Continue(())
        }
    }

    let mut aggregates = Aggregates {
        ctx,
        inner_names,
        inner_schema,
        query_depth: 0,
        local: false,
        outer: false,
        unsupported: false,
        windows: vec![],
    };
    let _ = dyn_visit::visit(&select.projection, &mut aggregates);
    let _ = dyn_visit::visit(&query.order_by, &mut aggregates);
    let mut visited = PlHashSet::new();
    while let Some(name) = aggregates.windows.pop() {
        if !visited.insert(name.clone()) {
            continue;
        }
        let definition = select
            .named_window
            .iter()
            .find(|NamedWindowDefinition(ident, _)| ident.value == name);
        match definition {
            Some(NamedWindowDefinition(_, NamedWindowExpr::NamedWindow(base))) => {
                aggregates.windows.push(base.value.clone())
            },
            Some(NamedWindowDefinition(_, NamedWindowExpr::WindowSpec(spec))) => {
                aggregates
                    .windows
                    .extend(spec.window_name.iter().map(|i| i.value.clone()));
                let _ = dyn_visit::visit(spec, &mut aggregates);
            },
            None => {},
        }
    }
    (!aggregates.unsupported && !aggregates.outer).then_some(aggregates.local)
}

// Shared eligibility gate for the rewrites: bail on any clause that changes
// which rows the subquery yields.
fn eligible_subquery_select(subquery: &Query) -> Option<&Select> {
    if subquery.limit_clause.is_some() || subquery.fetch.is_some() {
        return None;
    }
    eligible_subquery_body(subquery)
}

// Whether `subquery` keeps only its first row: `Some(true)` for `LIMIT 1` or
// `FETCH FIRST 1 ROW ONLY`, `Some(false)` without a limit, `None` for any other
// limit.
pub(crate) fn first_row_limit(subquery: &Query) -> Option<bool> {
    let is_one = |e: &SQLExpr| matches!(e, SQLExpr::Value(ValueWithSpan { value: SQLValue::Number(n, _), .. }) if n == "1");
    match (&subquery.limit_clause, &subquery.fetch) {
        (None, None) => Some(false),
        (
            Some(LimitClause::LimitOffset {
                limit: Some(limit),
                offset: None,
                limit_by,
            }),
            None,
        ) if limit_by.is_empty() && is_one(limit) => Some(true),
        (
            None,
            Some(Fetch {
                with_ties: false,
                percent: false,
                quantity,
            }),
        ) if quantity.as_ref().is_none_or(is_one) => Some(true),
        _ => None,
    }
}

// `eligible_subquery_select` without the limit check.
pub(crate) fn eligible_subquery_body(subquery: &Query) -> Option<&Select> {
    eligible_query_block(subquery).filter(|select| !select.from.is_empty())
}

// `eligible_subquery_body` that also accepts a SELECT without FROM. Exhaustive
// destructuring (no `..`) is on purpose: a new sqlparser clause must not compile
// until it gets an explicit keep-or-bail decision here.
pub(crate) fn eligible_query_block(subquery: &Query) -> Option<&Select> {
    let Query {
        with, // CTEs aren't resolved inside the rewrite: bail
        body,
        order_by: _,      // row order can't affect existence/membership
        limit_clause: _,  // checked by the callers
        fetch: _,         // checked by the callers
        locks: _,         // row locking doesn't change the rows
        for_clause,       // FOR XML/JSON reshape the result: bail
        settings,         // ClickHouse SETTINGS can change results: bail
        format_clause: _, // output serialization only
        pipe_operators,   // `|>` operators transform the rows: bail
    } = subquery;
    if with.is_some() || for_clause.is_some() || settings.is_some() || !pipe_operators.is_empty() {
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
        from: _,       // comma-separated relations cross join
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
    if !no_group_by
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
pub(crate) fn is_correlated_subquery(query: &Query, spellings: &CatalogSpellings) -> bool {
    let mut is_correlated = false;
    visit_foreign_qualifiers(query, spellings, |_| {
        is_correlated = true;
        ControlFlow::Break(())
    });
    is_correlated
}

/// Which input of a join the correlated subqueries in `conj` read, together with the
/// left operand of an `IN`: `Some(true)` for the left input, `Some(false)` for the right
/// one. `None` if they read neither or both, or a relation whose columns can't be found
/// by name.
fn correlated_join_side(
    ctx: &SQLContext,
    conj: &SQLExpr,
    tbl_left: &TableInfo,
    tbl_right: &TableInfo,
) -> Option<bool> {
    struct JoinSides<'a> {
        ctx: &'a SQLContext,
        tbl_left: &'a TableInfo,
        tbl_right: &'a TableInfo,
        query_depth: usize,
        is_correlated: bool,
        unresolved: bool,
        reads_left: bool,
        reads_right: bool,
    }

    impl Visitor for JoinSides<'_> {
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
            // A nested subquery's qualifiers may name relations of the subquery
            // around it.
            if self.query_depth > 0 {
                return ControlFlow::Continue(());
            }
            let (subquery, lhs) = match expr {
                SQLExpr::Subquery(subquery) | SQLExpr::Exists { subquery, .. } => (subquery, None),
                SQLExpr::InSubquery {
                    expr: lhs,
                    subquery,
                    ..
                } => (subquery, Some(lhs)),
                _ => return ControlFlow::Continue(()),
            };
            visit_foreign_qualifiers(subquery, self.ctx.catalog_spellings(), |parts| {
                self.is_correlated = true;
                match self.ctx.resolve_qualifier(parts) {
                    // A struct field path is not resolved.
                    Ok(Some((name, len))) if len + 1 == parts.len() => {
                        if self.ctx.is_right_relation(&name, self.tbl_right) {
                            self.reads_right = true;
                        } else if self.ctx.is_joined_relation(&name, self.tbl_left) {
                            self.reads_left = true;
                        } else {
                            self.unresolved = true;
                        }
                    },
                    _ => self.unresolved = true,
                }
                ControlFlow::Continue(())
            });
            // The left operand of a lowered `IN` is read on the same input.
            if let Some(lhs) = lhs {
                let (reads_left, reads_right) =
                    self.ctx
                        .join_inputs_read(lhs, self.tbl_left, self.tbl_right);
                self.reads_left |= reads_left;
                self.reads_right |= reads_right;
            }
            ControlFlow::Continue(())
        }
    }

    let mut sides = JoinSides {
        ctx,
        tbl_left,
        tbl_right,
        query_depth: 0,
        is_correlated: false,
        unresolved: false,
        reads_left: false,
        reads_right: false,
    };
    let _ = dyn_visit::visit(conj, &mut sides);
    if !sides.is_correlated || sides.unresolved {
        return None;
    }
    match (sides.reads_left, sides.reads_right) {
        (true, false) => Some(true),
        (false, true) => Some(false),
        _ => None,
    }
}

pub(crate) fn strip_nested(mut expr: &SQLExpr) -> &SQLExpr {
    while let SQLExpr::Nested(inner) = expr {
        expr = inner;
    }
    expr
}

/// Call `f` on each compound identifier that `query` reads without naming one of
/// its own relations.
fn visit_foreign_qualifiers(
    query: &Query,
    spellings: &CatalogSpellings,
    f: impl FnMut(&[Ident]) -> ControlFlow<()>,
) {
    let declared = declared_anywhere(query, spellings);

    struct Foreign<'a, F>(&'a [RelationName], F);

    impl<F: FnMut(&[Ident]) -> ControlFlow<()>> Visitor for Foreign<'_, F> {
        type Break = ();

        fn pre_visit_expr(&mut self, expr: &SQLExpr) -> ControlFlow<()> {
            match expr {
                SQLExpr::CompoundIdentifier(parts)
                    if match_qualifier(parts, self.0.iter()).is_none() =>
                {
                    (self.1)(parts)
                },
                _ => ControlFlow::Continue(()),
            }
        }
    }

    let _ = dyn_visit::visit(query, &mut Foreign(&declared, f));
}

/// The relations declared anywhere in `query`, its subqueries included.
fn declared_anywhere(query: &Query, spellings: &CatalogSpellings) -> Vec<RelationName> {
    struct Declared<'a>(Vec<RelationName>, &'a CatalogSpellings);

    impl Visitor for Declared<'_> {
        type Break = ();

        fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
            let name = relation_name(factor).map(|name| name.spelled(factor, self.1));
            self.0.extend(name);
            ControlFlow::Continue(())
        }
    }

    let mut declared = Declared(Vec::new(), spellings);
    let _ = dyn_visit::visit(query, &mut declared);
    declared.0
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
    let _ = dyn_visit::visit_mut(stmt, &mut DesugarQuantified);
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
/// `skips` holds, per expression being visited, whether it was skipped as a whole.
struct CorrelatedLowering<'a> {
    ctx: &'a mut SQLContext,
    lf: LazyFrame,
    outer_schema: &'a Schema,
    scope: LowerScope,
    bindings: &'a mut SubqueryBindings,
    join_tree: Option<&'a OuterJoinTree>,
    query_depth: usize,
    skips: Vec<bool>,
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
        let prepared = self
            .ctx
            .pull_up_nested(ScopeColumns::Frame(self.outer_schema), subquery);
        let original = subquery;
        let subquery = prepared.as_ref().unwrap_or(subquery);
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
                self.join_tree,
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
        self.bindings.push((original.clone(), kind, name.clone()));
        Ok(Some(name))
    }

    /// Replace `expr` by the column of its lowering when it is a lowerable subquery.
    fn lower_expr(&mut self, expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
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
        // The subquery of `x > ANY (subquery)` is a set of values rather than a
        // scalar: only the left operand is lowered, and the rest is skipped like a
        // nested query.
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
        self.lower_expr(expr)
    }

    fn post_visit_expr(&mut self, _expr: &mut SQLExpr) -> ControlFlow<PolarsError> {
        if self.skips.pop() == Some(true) {
            self.query_depth -= 1;
        }
        ControlFlow::Continue(())
    }
}

// Whether an expression is elementwise: one value per row, from that row alone.
// Join keys must be, so that every key is as long as the frame it is built from.
// Anything this does not recognise declines the rewrite, which costs an
// optimisation rather than correctness.
fn is_elementwise(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) | Expr::Literal(_) => true,
        Expr::Alias(inner, _) | Expr::Cast { expr: inner, .. } => is_elementwise(inner),
        Expr::BinaryExpr { left, op: _, right } => is_elementwise(left) && is_elementwise(right),
        // SQL operators that are lowered once the dtypes are known are elementwise.
        Expr::Function {
            input,
            function: FunctionExpr::Sql(_),
        } => input.iter().all(is_elementwise),
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => is_elementwise(predicate) && is_elementwise(truthy) && is_elementwise(falsy),
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

// The inner frame has to be this many times larger than the outer frame for the
// restriction of its groups to the outer keys to pay for the semi join.
const RESTRICTION_LOPSIDED_FACTOR: f64 = 2.0;

// Whether the inner frame is large enough relative to the outer frame, by the
// sizes of their sources or by their estimates after filtering, for the key
// restriction to be worth its semi join. Unknown statistics decline.
fn restriction_pays_off(outer: Option<&NodeStats>, inner: Option<&NodeStats>) -> bool {
    let (Some(outer), Some(inner)) = (outer, inner) else {
        return false;
    };
    outer.unfiltered * RESTRICTION_LOPSIDED_FACTOR <= inner.unfiltered
        || outer.filtered * RESTRICTION_LOPSIDED_FACTOR <= inner.filtered
}

/// The relations and filters of a query block whose frame is its FROM relations
/// joined and filtered by its WHERE conjuncts that hold no subquery, and nothing
/// more. A correlated scalar subquery over some of these relations can then be
/// answered from the frame itself.
pub(crate) struct OuterJoinTree {
    relations: Vec<TreeRelation>,
    conjuncts: Vec<SpelledConjunct>,
}

struct TreeRelation {
    // The name of the table it reads.
    table: String,
    // The name that qualifies its columns: its alias, else its table name.
    name: String,
    schema: SchemaRef,
}

// A conjunct with each column spelled `<relation index>.<column>`, so that one
// condition written in two scopes, under different aliases, compares equal.
struct SpelledConjunct {
    expr: SQLExpr,
    // The relations it reads through its own scope's FROM.
    own: Vec<usize>,
    // The relations it reads as correlated outer columns.
    outer: Vec<usize>,
}

impl OuterJoinTree {
    // Whether a subquery over `from` filtered by `selection` reads, for each row
    // of the frame, exactly the rows of its relations that the row was joined
    // with: its relations are tree relations, its filters are exactly the
    // tree's conditions on those relations alone, and its correlation
    // predicates are exactly the tree's conditions linking them to the other
    // relations. `inner_schema` is the schema the subquery's FROM resolved to.
    fn covers(&self, from: &[TableWithJoins], selection: &SQLExpr, inner_schema: &Schema) -> bool {
        let mut own: Vec<(&str, usize)> = Vec::with_capacity(from.len());
        for tbl_expr in from {
            let Some((table, name)) = plain_table(tbl_expr) else {
                return false;
            };
            let Some(idx) = self.relations.iter().position(|r| r.table == table) else {
                return false;
            };
            if own.iter().any(|&(n, i)| n == name || i == idx) {
                return false;
            }
            own.push((name, idx));
        }
        // An alias spelled like a table name can make the subquery read another table.
        let n_columns: usize = own
            .iter()
            .map(|&(_, idx)| self.relations[idx].schema.len())
            .sum();
        let same_tables = own.iter().all(|&(_, idx)| {
            self.relations[idx]
                .schema
                .iter()
                .all(|(name, dtype)| inner_schema.get(name) == Some(dtype))
        });
        if n_columns != inner_schema.len() || !same_tables {
            return false;
        }

        let is_own = |idx: &usize| own.iter().any(|&(_, i)| i == *idx);
        let mut filters = Vec::new();
        let mut links = Vec::new();
        for conj in MintermIter::new(selection) {
            let Some(spelled) = spell_conjunct(conj, &own, &self.relations, true) else {
                return false;
            };
            if spelled.outer.is_empty() {
                filters.push(spelled.expr);
            } else if spelled.outer.iter().any(is_own) {
                // The outer row's column of a relation the subquery reads itself.
                return false;
            } else {
                links.push(spelled.expr);
            }
        }
        let mut tree_filters = Vec::new();
        let mut tree_links = Vec::new();
        for conj in &self.conjuncts {
            match conj.own.iter().filter(|idx| is_own(idx)).count() {
                n if n == conj.own.len() => tree_filters.push(&conj.expr),
                0 => {},
                _ => tree_links.push(&conj.expr),
            }
        }
        let same = |ours: &[SQLExpr], tree: &[&SQLExpr]| {
            ours.iter().all(|e| tree.contains(&e)) && tree.iter().all(|e| ours.contains(e))
        };
        same(&filters, &tree_filters) && same(&links, &tree_links)
    }
}

// A FROM item that is a plain table without joins: the table's name and the
// name that qualifies its columns.
fn plain_table(tbl_expr: &TableWithJoins) -> Option<(&str, &str)> {
    let TableFactor::Table {
        name,
        alias,
        args: None,
        ..
    } = &tbl_expr.relation
    else {
        return None;
    };
    let [part] = name.0.as_slice() else {
        return None;
    };
    let table = part.as_ident()?.value.as_str();
    if !tbl_expr.joins.is_empty() || alias.as_ref().is_some_and(|a| !a.columns.is_empty()) {
        return None;
    }
    let qualifier = alias.as_ref().map_or(table, |a| a.name.value.as_str());
    Some((table, qualifier))
}

// Spell a conjunct's columns by relation index. A column resolves against the
// relations its scope declares, `own` as (name, relation index), and then, when
// `correlated`, against all relations as an outer column. `None` when a column
// doesn't resolve to exactly one relation, or the conjunct holds a subquery.
fn spell_conjunct(
    expr: &SQLExpr,
    own: &[(&str, usize)],
    relations: &[TreeRelation],
    correlated: bool,
) -> Option<SpelledConjunct> {
    let has = |idx: usize, column: &str| relations[idx].schema.contains(column);
    let resolve = |qualifier: Option<&str>, column: &str| -> Option<(usize, bool)> {
        let own_matches: Vec<usize> = own
            .iter()
            .filter(|&&(name, idx)| qualifier.map_or(has(idx, column), |q| q == name))
            .map(|&(_, idx)| idx)
            .collect();
        match own_matches.as_slice() {
            [idx] => return has(*idx, column).then_some((*idx, false)),
            [] if correlated => {},
            _ => return None,
        }
        let outer_matches: Vec<usize> = (0..relations.len())
            .filter(|&idx| qualifier.is_none_or(|q| relations[idx].name == q) && has(idx, column))
            .collect();
        match outer_matches.as_slice() {
            [idx] => Some((*idx, true)),
            _ => None,
        }
    };

    let mut spelled = expr.clone();
    let mut own_read = Vec::new();
    let mut outer_read = Vec::new();
    let flow = visit_expressions_mut(&mut spelled, |e| {
        if is_subquery_expr(e) {
            return ControlFlow::Break(());
        }
        if let SQLExpr::Nested(inner) = e {
            let inner = inner.as_ref().clone();
            *e = inner;
            return ControlFlow::Continue(());
        }
        // An (in)equality reads the same either way round.
        if let SQLExpr::BinaryOp {
            left,
            op: SQLBinaryOperator::Eq | SQLBinaryOperator::NotEq,
            right,
        } = e
        {
            if left.to_string() > right.to_string() {
                std::mem::swap(left, right);
            }
            return ControlFlow::Continue(());
        }
        let resolved = match &*e {
            SQLExpr::Identifier(ident) => resolve(None, &ident.value).zip(Some(ident.clone())),
            SQLExpr::CompoundIdentifier(parts) => match parts.as_slice() {
                [qualifier, ident] => {
                    resolve(Some(&qualifier.value), &ident.value).zip(Some(ident.clone()))
                },
                _ => None,
            },
            _ => return ControlFlow::Continue(()),
        };
        let Some(((idx, outer), ident)) = resolved else {
            return ControlFlow::Break(());
        };
        if outer {
            outer_read.push(idx);
        } else {
            own_read.push(idx);
        }
        *e = SQLExpr::CompoundIdentifier(vec![Ident::new(idx.to_string()), ident]);
        ControlFlow::Continue(())
    });
    flow.is_continue().then_some(SpelledConjunct {
        expr: spelled,
        own: own_read,
        outer: outer_read,
    })
}

// Whether an aggregate keeps its exact value when every input row is repeated
// the same number of times. A mean does not: summing the repeats rounds
// differently, and can overflow.
fn ignores_uniform_repeats(agg: &Expr) -> bool {
    match agg {
        Expr::Agg(
            AggExpr::Min { input, .. } | AggExpr::Max { input, .. } | AggExpr::NUnique(input),
        ) => is_elementwise(input),
        Expr::Alias(inner, _) | Expr::Cast { expr: inner, .. } => ignores_uniform_repeats(inner),
        Expr::BinaryExpr { left, op: _, right } => {
            ignores_uniform_repeats(left) && ignores_uniform_repeats(right)
        },
        Expr::Function {
            input,
            function: FunctionExpr::Sql(_),
        } => input.iter().all(ignores_uniform_repeats),
        Expr::Literal(_) => true,
        _ => false,
    }
}

// Lower an equality-correlated scalar aggregate: group the inner frame by the
// correlation keys and left-join the per-group result onto the outer frame,
// with the inner frame first restricted to the groups the outer keys ask for
// when `restrict`.
fn equality_correlated_aggregate(
    outer: LazyFrame,
    inner: LazyFrame,
    outer_on: Vec<Expr>,
    inner_on: Vec<Expr>,
    aggregate: &ScalarAggregate,
    restrict: bool,
) -> PolarsResult<LazyFrame> {
    let (outer, inner) = if restrict {
        restrict_to_outer_keys(outer, inner, &outer_on, &inner_on)?
    } else {
        (outer, inner)
    };
    let grouped = inner.group_by(inner_on.clone()).agg(aggregate.aggs.clone());
    Ok(aggregate.finish(left_join_aggregate(outer, grouped, outer_on, inner_on)?))
}

// Restrict the inner frame to the rows whose keys occur in the outer frame by a
// semi join against the outer keys. The outer frame is cached so that both the
// restriction and the join back read one evaluation of it. A semi join never
// duplicates inner rows.
#[cfg(feature = "semi_anti_join")]
fn restrict_to_outer_keys(
    outer: LazyFrame,
    inner: LazyFrame,
    outer_on: &[Expr],
    inner_on: &[Expr],
) -> PolarsResult<(LazyFrame, LazyFrame)> {
    let outer = outer.cache();
    // One outer column may serve several key pairs; it is projected once.
    let mut requested: Vec<Expr> = Vec::with_capacity(outer_on.len());
    for key in outer_on {
        if !requested.contains(key) {
            requested.push(key.clone());
        }
    }
    let requested_keys = outer.clone().select(requested);
    let restricted = inner
        .join_builder()
        .with(requested_keys)
        .left_on(inner_on)
        .right_on(outer_on)
        .how(JoinType::Semi)
        .finish()?;
    Ok((outer, restricted))
}

#[cfg(not(feature = "semi_anti_join"))]
fn restrict_to_outer_keys(
    outer: LazyFrame,
    inner: LazyFrame,
    _outer_on: &[Expr],
    _inner_on: &[Expr],
) -> PolarsResult<(LazyFrame, LazyFrame)> {
    Ok((outer, inner))
}

// Whether a per-group aggregate reads nothing but its own group's rows, so
// dropping other groups cannot change its value.
fn group_local_aggregate(agg: &Expr) -> bool {
    !has_expr(agg, |e| {
        matches!(
            e,
            Expr::AnonymousFunction { .. } | Expr::Over { .. } | Expr::SubPlan(_, _)
        )
    })
}

// Left-join a per-group aggregate onto the outer frame. Outer rows without a
// group get NULL.
fn left_join_aggregate(
    outer: LazyFrame,
    grouped: LazyFrame,
    left_on: Vec<Expr>,
    right_on: Vec<Expr>,
) -> PolarsResult<LazyFrame> {
    outer
        .join_builder()
        .with(grouped)
        .left_on(left_on)
        .right_on(right_on)
        .how(JoinType::Left)
        .coalesce(JoinCoalesce::CoalesceColumns)
        .finish()
}

// The projection of a correlated scalar aggregate subquery, as the aggregates
// computed per group plus what turns them into the result after the left join.
//
// An outer row without inner rows must get the projection's value over no
// rows. The left join gives NULL there. That is right for most projections,
// but not for `COUNT(*)`, `COUNT(*) + 1` or `COALESCE(MAX(x), 0)`. For those,
// each aggregate gets its value over no rows, and a projection that combines
// several aggregates is evaluated per outer row after the join. It then only
// runs on outer rows that exist, so it can't fail on a value no row needs.
struct ScalarAggregate {
    // Computed per group.
    aggs: Vec<Expr>,
    // Columns of `aggs` and the value they take for an outer row without inner
    // rows, where that is not NULL.
    fill: Vec<(PlSmallStr, Expr)>,
    // The result over the `aggs` columns, and those columns to drop after.
    combine: Option<(Expr, Vec<PlSmallStr>)>,
}

impl ScalarAggregate {
    // `schema` is the frame `agg` reads.
    fn new(agg: Expr, result_name: &PlSmallStr, prefix: &str, schema: &Schema) -> Self {
        let whole = |fill: Vec<(PlSmallStr, Expr)>| Self {
            aggs: vec![agg.clone().alias(result_name.clone())],
            fill,
            combine: None,
        };
        match agg_output_root(&agg) {
            e if zero_over_no_rows(e) && widening_integer_casts(&agg, schema) => {
                return whole(vec![(result_name.clone(), lit(0))]);
            },
            // A cast of NULL is NULL.
            e @ Expr::Agg(_) if !zero_over_no_rows(e) => return whole(vec![]),
            _ => {},
        }

        let mut aggs = Vec::new();
        let combine = split_aggregates(agg.clone(), prefix, &mut aggs);
        let Ok(fields) = aggs
            .iter()
            .map(|e| e.to_field(schema))
            .collect::<PolarsResult<Vec<_>>>()
        else {
            return whole(vec![]);
        };
        let agg_schema = Schema::from_iter(fields);
        let fill: Vec<(PlSmallStr, Expr)> = aggs
            .iter()
            .zip(agg_schema.iter())
            .filter(|(e, _)| zero_over_no_rows(agg_output_root(e)))
            .map(|(_, (name, dtype))| (name.clone(), lit(0).cast(dtype.clone())))
            .collect();
        if !is_elementwise_over(&combine, &agg_schema) {
            return whole(vec![]);
        }
        // Only a cheap expression is evaluated here. Any other one is split,
        // which is right whatever its value over no rows is.
        if is_cheap(&combine) && null_over_no_rows(&combine, &agg_schema, &fill) {
            return whole(vec![]);
        }
        let names = agg_schema.iter_names().cloned().collect();
        Self {
            aggs,
            fill,
            combine: Some((combine.alias(result_name.clone()), names)),
        }
    }

    // Apply `f` to every per-group aggregate.
    fn map_inputs(mut self, f: impl Fn(Expr) -> Expr) -> Self {
        self.aggs = self.aggs.into_iter().map(f).collect();
        self
    }

    // Turn the left-joined aggregate columns into the result column.
    fn finish(&self, joined: LazyFrame) -> LazyFrame {
        let joined = if self.fill.is_empty() {
            joined
        } else {
            joined.with_columns(
                self.fill
                    .iter()
                    .map(|(name, value)| col(name.clone()).fill_null(value.clone()))
                    .collect::<Vec<_>>(),
            )
        };
        match &self.combine {
            Some((combine, names)) => joined.with_column(combine.clone()).drop(Selector::ByName {
                names: Arc::from(names.as_slice()),
                strict: true,
            }),
            None => joined,
        }
    }
}

// Whether an aggregate is 0 over no rows rather than NULL. A bare `sum` is: SQL
// `SUM` puts its NULL over no rows around it.
fn zero_over_no_rows(agg: &Expr) -> bool {
    matches!(
        agg,
        Expr::Len | Expr::Agg(AggExpr::Count { .. } | AggExpr::NUnique(_) | AggExpr::Sum(_))
    )
}

// Replace every aggregate in `expr` with a column named after it, pushing the
// aliased aggregate onto `aggs`.
fn split_aggregates(expr: Expr, prefix: &str, aggs: &mut Vec<Expr>) -> Expr {
    match expr {
        Expr::Len | Expr::Agg(_) => {
            let name = format_pl_smallstr!("{prefix}agg{}", aggs.len());
            aggs.push(expr.alias(name.clone()));
            col(name)
        },
        e => e
            .map_children(&mut |c, _| Ok(split_aggregates(c, prefix, aggs)), &mut ())
            .unwrap(),
    }
}

// Whether `combine` is NULL when its aggregates hold their values over no rows:
// `fill`, else NULL. An error counts as not NULL.
fn null_over_no_rows(combine: &Expr, agg_schema: &Schema, fill: &[(PlSmallStr, Expr)]) -> bool {
    let columns = agg_schema
        .iter()
        .map(|(name, dtype)| {
            let value = fill
                .iter()
                .find(|(n, _)| n == name)
                .map_or(lit(LiteralValue::untyped_null()), |(_, v)| v.clone());
            value.cast(dtype.clone()).alias(name.clone())
        })
        .collect::<Vec<_>>();
    DataFrame::empty()
        .lazy()
        .select(columns)
        .select([combine.clone()])
        .collect()
        .is_ok_and(|df| df.height() == 1 && df.columns()[0].null_count() == 1)
}

// Whether `expr`, over the columns of `schema`, gives a single value.
fn is_scalar_over(expr: &Expr, schema: &Schema) -> bool {
    let mut arena = Arena::new();
    let mut ctx = ExprToIRContext::new(&mut arena, schema);
    let Ok(expr) = to_expr_ir(expr.clone(), &mut ctx) else {
        return false;
    };
    expr.is_scalar(&arena)
}

// Whether `combine`, over the columns of `agg_schema`, gives one value per row.
fn is_elementwise_over(combine: &Expr, agg_schema: &Schema) -> bool {
    let mut arena = Arena::new();
    let mut ctx = ExprToIRContext::new(&mut arena, agg_schema);
    let Ok(combine) = to_expr_ir(combine.clone(), &mut ctx) else {
        return false;
    };
    is_elementwise_rec(combine.node(), &arena)
}

// Whether `expr` is built from operations that are cheap on one row and run no
// user code: arithmetic, comparisons, casts, `CASE` and `COALESCE`.
fn is_cheap(expr: &Expr) -> bool {
    match expr {
        Expr::Column(_) | Expr::Literal(_) => true,
        Expr::Alias(inner, _) | Expr::Cast { expr: inner, .. } => is_cheap(inner),
        Expr::BinaryExpr { left, op: _, right } => is_cheap(left) && is_cheap(right),
        Expr::Ternary {
            predicate,
            truthy,
            falsy,
        } => is_cheap(predicate) && is_cheap(truthy) && is_cheap(falsy),
        Expr::Function {
            input,
            function:
                FunctionExpr::Sql(_)
                | FunctionExpr::Coalesce
                | FunctionExpr::FillNull
                | FunctionExpr::Boolean(BooleanFunction::IsNull | BooleanFunction::IsNotNull),
        } => input.iter().all(is_cheap),
        _ => false,
    }
}

// Whether every cast around the aggregate in `expr` turns an integer into an
// integer type that holds all its values. Filling in 0 after such casts is the
// same as before them.
fn widening_integer_casts(expr: &Expr, schema: &Schema) -> bool {
    match expr {
        Expr::Alias(inner, _) => widening_integer_casts(inner, schema),
        Expr::Cast {
            expr: inner, dtype, ..
        } => {
            let Some(to) = dtype.as_literal() else {
                return false;
            };
            inner.to_field(schema).is_ok_and(|from| {
                from.dtype.is_integer()
                    && to.is_integer()
                    && try_get_supertype(&from.dtype, to).is_ok_and(|st| &st == to)
            }) && widening_integer_casts(inner, schema)
        },
        _ => true,
    }
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

// An expression of one input of a join and the inner expression of a subquery it must
// equal.
struct JoinMembershipKey {
    is_left: bool,
    outer: Expr,
    inner: Expr,
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
    // Whether both conjuncts compare the same columns the same way; an equality
    // reads the same from either side.
    fn same_comparison(&self, other: &Self) -> bool {
        let symmetric = self.op == SQLBinaryOperator::Eq && other.op == SQLBinaryOperator::Eq;
        self.outer == other.outer
            && self.inner == other.inner
            && (symmetric || self.op == other.op && self.inner_on_left == other.inner_on_left)
    }

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
    inner_schema: &Schema,
    outer_schema: &Schema,
    inner: &SQLContext,
    outer: &SQLContext,
) -> PolarsResult<Option<CorrPredicate>> {
    let SQLExpr::BinaryOp { left, op, right } = conj else {
        return Ok(None);
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
        return Ok(None);
    }
    let Some((lside, lname)) =
        classify_correlation_column(left, inner_schema, outer_schema, inner, outer)?
    else {
        return Ok(None);
    };
    let Some((rside, rname)) =
        classify_correlation_column(right, inner_schema, outer_schema, inner, outer)?
    else {
        return Ok(None);
    };
    Ok(match (lside, rside) {
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
    })
}
