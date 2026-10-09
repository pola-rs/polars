//! The group context of a SELECT block: which parts of its expressions are computed per
//! group, and which run on the aggregated rows.

use std::sync::LazyLock;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::plans::visitor::{TreeWalker, VisitRecursion, Visitor};
use polars_plan::prelude::*;
use polars_utils::aliases::{PlHashSet, PlIndexSet};
use polars_utils::format_pl_smallstr;

use crate::context::is_correlated_result_col;
use crate::unique_column_name;

/// The expressions computed in the group context, without aggregate marks.
pub(crate) struct AggregateOutputs {
    exprs: Vec<Expr>,
}

impl AggregateOutputs {
    pub(crate) fn with_capacity(capacity: usize) -> Self {
        Self {
            exprs: Vec::with_capacity(capacity),
        }
    }

    pub(crate) fn push(&mut self, expr: Expr) {
        self.exprs.push(strip_aggregate_marks(expr));
    }

    pub(crate) fn iter(&self) -> impl Iterator<Item = &Expr> {
        self.exprs.iter()
    }

    pub(crate) fn into_exprs(self) -> Vec<Expr> {
        self.exprs
    }

    /// Reference to the output computing `expr`, reusing an existing aggregate with
    /// the same expression (from SELECT, or the same aggregate repeated in a window).
    fn get_or_insert_hoisted(&mut self, expr: Expr) -> Expr {
        let expr = strip_aggregate_marks(expr);
        let existing = self.exprs.iter().find_map(|agg| match agg {
            Expr::Alias(inner, name) if **inner == expr => Some(name.clone()),
            _ => None,
        });
        let name = existing.unwrap_or_else(|| {
            let name = format_pl_smallstr!("__POLARS_HOISTED_AGG_{}", unique_column_name());
            self.exprs.push(expr.alias(name.clone()));
            name
        });
        col(name)
    }
}

/// Splits post-aggregation expressions from the aggregates they contain.
pub(crate) struct GroupContextSplitter<'a> {
    pub(crate) keys: &'a Schema,
    /// Each group key without its alias, and the column holding it after aggregation.
    pub(crate) key_exprs: Vec<(Expr, PlSmallStr)>,
    pub(crate) whole_frame_partition: Option<&'a PlSmallStr>,
    /// Columns holding the result of a scalar subquery.
    pub(crate) subquery_names: &'a PlHashSet<PlSmallStr>,
    /// Whether a scalar subquery outside the aggregates is read from the aggregated rows.
    /// Without GROUP BY there is one aggregated row even when there is no input row, so
    /// only that row can hold its value.
    pub(crate) subqueries_after_aggregation: bool,
    /// The subqueries that are read from the aggregated rows.
    pub(crate) subqueries_read_after: PlIndexSet<PlSmallStr>,
    pub(crate) aggregates: AggregateOutputs,
}

impl GroupContextSplitter<'_> {
    /// Whether `expr` has one value per group without being an aggregate call: a scalar
    /// subquery, or a correlated column.
    fn is_group_value(&self, expr: &Expr) -> bool {
        matches!(expr, Expr::Agg(AggExpr::First(inner)) if matches!(inner.as_ref(), Expr::Column(name)
            if self.subquery_names.contains(name) || is_correlated_result_col(name)))
    }

    /// The scalar subquery that `expr` reads from the aggregated rows.
    fn subquery_read_after<'e>(&self, expr: &'e Expr) -> Option<&'e PlSmallStr> {
        match expr {
            Expr::Agg(AggExpr::First(inner)) if self.subqueries_after_aggregation => {
                match inner.as_ref() {
                    Expr::Column(name) if self.subquery_names.contains(name) => Some(name),
                    _ => None,
                }
            },
            _ => None,
        }
    }

    fn read_after_aggregation(&mut self, name: PlSmallStr) -> Expr {
        self.subqueries_read_after.insert(name.clone());
        Expr::Column(name)
    }

    /// Whether a SELECT projection must be processed in the group context rather
    /// than passed through as a group key: it contains an aggregate, a window, or a
    /// function over a non-key column.
    pub(crate) fn requires_group_processing(&self, expr: &Expr) -> bool {
        has_expr(expr, |e| match e {
            _ if is_marked_aggregate(e) => true,
            Expr::Agg(_) | Expr::Len | Expr::Over { .. } => true,
            #[cfg(feature = "dynamic_group_by")]
            Expr::Rolling { .. } => true,
            Expr::AnonymousFunction { options, .. } => options.returns_scalar(),
            Expr::Function { function: func, .. }
                if !matches!(func, FunctionExpr::StructExpr(_)) =>
            {
                has_expr(
                    e,
                    |e| matches!(e, Expr::Column(name) if !self.keys.contains(name)),
                )
            },
            _ => false,
        })
    }

    /// Whether `expr` must run after aggregation: it holds a window or a subquery read from
    /// the aggregated rows, or combines an aggregate with a grouped key.
    pub(crate) fn needs_post_aggregation(&self, expr: &Expr) -> bool {
        struct Finder<'a, 'b> {
            splitter: &'a GroupContextSplitter<'b>,
            has_window: bool,
            has_subquery_read_after: bool,
            has_group_key: bool,
            has_aggregate: bool,
        }
        impl Visitor for Finder<'_, '_> {
            type Node = Expr;
            type Arena = ();

            fn pre_visit(&mut self, node: &Expr, _: &()) -> PolarsResult<VisitRecursion> {
                // Siblings are still visited so that no aggregate goes unnoticed.
                Ok(match node {
                    Expr::Over { .. } => {
                        self.has_window = true;
                        VisitRecursion::Skip
                    },
                    Expr::Column(name) => {
                        self.has_group_key |= self.splitter.keys.contains(name);
                        VisitRecursion::Skip
                    },
                    _ if self.splitter.subquery_read_after(node).is_some() => {
                        self.has_subquery_read_after = true;
                        VisitRecursion::Skip
                    },
                    _ if is_marked_aggregate(node) || self.splitter.is_group_value(node) => {
                        self.has_aggregate = true;
                        VisitRecursion::Skip
                    },
                    _ => VisitRecursion::Continue,
                })
            }
        }
        let mut finder = Finder {
            splitter: self,
            has_window: false,
            has_subquery_read_after: false,
            has_group_key: false,
            has_aggregate: false,
        };
        let _ = expr.visit(&mut finder, &());
        finder.has_window
            || finder.has_subquery_read_after
            || (finder.has_group_key && finder.has_aggregate)
    }

    /// Replace every aggregate in `expr` with a reference to a hoisted aggregation output,
    /// recorded in `self.aggregates`. A window runs on the aggregated rows; its inputs are
    /// bound by `bind_window_input`.
    pub(crate) fn hoist(&mut self, expr: Expr) -> PolarsResult<Expr> {
        if let Some(name) = self.subquery_read_after(&expr) {
            return Ok(self.read_after_aggregation(name.clone()));
        }
        match expr {
            Expr::Over {
                function,
                partition_by,
                order_by,
                mapping,
            } => Ok(Expr::Over {
                function: Arc::new(self.bind_window_input(Arc::unwrap_or_clone(function))?),
                partition_by: partition_by
                    .into_iter()
                    .map(|e| match e {
                        Expr::Column(name) if Some(&name) == self.whole_frame_partition => {
                            Ok(lit(1))
                        },
                        e => self.bind_window_input(e),
                    })
                    .collect::<PolarsResult<_>>()?,
                order_by: order_by
                    .map(|(e, options)| {
                        let e = self.bind_window_input(Arc::unwrap_or_clone(e))?;
                        PolarsResult::Ok((Arc::new(e), options))
                    })
                    .transpose()?,
                mapping,
            }),
            e if is_marked_aggregate(&e) || self.is_group_value(&e) => {
                Ok(self.aggregates.get_or_insert_hoisted(e))
            },
            e => e.map_children(&mut |c, _| self.hoist(c), &mut ()),
        }
    }

    /// Check that `expr`, which runs in the group context, reads the input columns only in
    /// group keys and aggregates: `x` has no single value per group in `HAVING x > 1`.
    pub(crate) fn check_columns_per_group(&self, expr: &Expr) -> PolarsResult<()> {
        struct Finder<'a, 'b> {
            splitter: &'a GroupContextSplitter<'b>,
            column: Option<PlSmallStr>,
        }
        impl Visitor for Finder<'_, '_> {
            type Node = Expr;
            type Arena = ();

            fn pre_visit(&mut self, node: &Expr, _: &()) -> PolarsResult<VisitRecursion> {
                let splitter = self.splitter;
                Ok(match node {
                    // The inputs of a window are checked when they are bound.
                    Expr::Over { .. } => VisitRecursion::Skip,
                    _ if is_marked_aggregate(node)
                        || splitter.is_group_value(node)
                        || splitter.key_exprs.iter().any(|(key, _)| key == node) =>
                    {
                        VisitRecursion::Skip
                    },
                    Expr::Column(name)
                        if !splitter.keys.contains(name)
                            && !splitter.subquery_names.contains(name)
                            && !is_correlated_result_col(name) =>
                    {
                        self.column = Some(name.clone());
                        VisitRecursion::Stop
                    },
                    _ => VisitRecursion::Continue,
                })
            }
        }
        let mut finder = Finder {
            splitter: self,
            column: None,
        };
        expr.visit(&mut finder, &())?;
        if let Some(name) = finder.column {
            polars_bail!(SQLSyntax: "'{}' should participate in the GROUP BY clause or an aggregate function", name);
        }
        Ok(())
    }

    /// Read the group keys in `expr`, which runs in the group context, as one value per
    /// group, as HAVING reads them.
    pub(crate) fn read_keys_per_group(&self, expr: Expr) -> Expr {
        if self.key_exprs.iter().any(|(key, _)| *key == expr) {
            return expr.first();
        }
        match expr {
            e if is_marked_aggregate(&e) || self.is_group_value(&e) => e,
            e => e
                .map_children(&mut |c, _| Ok(self.read_keys_per_group(c)), &mut ())
                .unwrap(),
        }
    }

    /// Bind an input of a window function to the aggregated rows: an aggregate reads its
    /// hoisted output and a group key reads its key column.
    fn bind_window_input(&mut self, expr: Expr) -> PolarsResult<Expr> {
        if let Some((_, name)) = self.key_exprs.iter().find(|(key, _)| *key == expr) {
            return Ok(col(name.clone()));
        }
        if let Some(name) = self.subquery_read_after(&expr) {
            return Ok(self.read_after_aggregation(name.clone()));
        }
        match expr {
            e if is_marked_aggregate(&e) || self.is_group_value(&e) => {
                Ok(self.aggregates.get_or_insert_hoisted(e))
            },
            Expr::Column(name) if self.subquery_names.contains(&name) => {
                if self.subqueries_after_aggregation {
                    Ok(self.read_after_aggregation(name))
                } else {
                    Ok(self.aggregates.get_or_insert_hoisted(col(name).first()))
                }
            },
            Expr::Column(name) => {
                polars_ensure!(
                    self.keys.contains(&name),
                    SQLSyntax: "'{}' should participate in the GROUP BY clause or an aggregate function", name
                );
                Ok(Expr::Column(name))
            },
            e => e.map_children(&mut |c, _| self.bind_window_input(c), &mut ()),
        }
    }
}

/// The names that QUALIFY and ORDER BY can read besides the input columns: columns renamed
/// by `SELECT * RENAME`, then SELECT aliases. Each stands for an expression over the input
/// columns, of a type that is known if it could be inferred.
pub(crate) struct OutputNames {
    names: PlHashMap<PlSmallStr, (Expr, Option<DataType>)>,
}

impl OutputNames {
    /// The output names of `projections` and `renames` over the input `schema`. An input
    /// column comes before an output name of the same name. `typed_schema` is `schema` with
    /// the columns that the projections read before the block resolves them.
    pub(crate) fn new(
        projections: &[Expr],
        renames: &PlHashMap<PlSmallStr, PlSmallStr>,
        schema: &Schema,
        typed_schema: &Schema,
    ) -> Self {
        let mut names = PlHashMap::new();
        for (before, after) in renames {
            if !schema.contains(after) {
                names
                    .entry(after.clone())
                    .or_insert_with(|| (col(before.clone()), schema.get(before).cloned()));
            }
        }
        for projection in projections {
            if let Expr::Alias(inner, name) = projection
                && !schema.contains(name)
                && !names.contains_key(name)
            {
                let dtype = projection.to_field(typed_schema).ok().map(|f| f.dtype);
                names.insert(name.clone(), (inner.as_ref().clone(), dtype));
            }
        }
        Self { names }
    }

    /// `schema` with the output names of a known type, to parse an expression that can
    /// read them.
    pub(crate) fn extend_schema(&self, schema: &Schema) -> Schema {
        let mut schema = schema.clone();
        for (name, (_, dtype)) in &self.names {
            if let Some(dtype) = dtype {
                schema.with_column(name.clone(), dtype.clone());
            }
        }
        schema
    }

    /// Replace the output names in `expr` by their expressions.
    pub(crate) fn resolve(&self, expr: Expr) -> Expr {
        expr.map_expr(|e| match e {
            Expr::Column(name) => match self.names.get(&name) {
                Some((expr, _)) => expr.clone(),
                None => Expr::Column(name),
            },
            e => e,
        })
    }
}

/// Whether a block without GROUP BY has both windows and aggregates, as in
/// `SELECT SUM(x), COUNT(*) OVER () FROM t`.
pub(crate) fn has_windows_over_aggregates(projections: &[Expr]) -> bool {
    projections
        .iter()
        .any(|e| has_expr(e, |e| matches!(e, Expr::Over { .. })))
        && projections.iter().any(has_marked_aggregate)
}

/// Whether a block without GROUP BY reads a scalar subquery outside its aggregates, as in
/// `SELECT COUNT(*), (SELECT 1) FROM t`. Without input rows, the subquery only has a value
/// in the aggregated row.
pub(crate) fn has_subquery_outside_aggregates(
    projections: &[Expr],
    subquery_names: &PlHashSet<PlSmallStr>,
) -> bool {
    struct Finder<'a> {
        subquery_names: &'a PlHashSet<PlSmallStr>,
        found: bool,
    }
    impl Visitor for Finder<'_> {
        type Node = Expr;
        type Arena = ();

        fn pre_visit(&mut self, node: &Expr, _: &()) -> PolarsResult<VisitRecursion> {
            Ok(match node {
                _ if is_marked_aggregate(node) => VisitRecursion::Skip,
                Expr::Column(name) if self.subquery_names.contains(name) => {
                    self.found = true;
                    VisitRecursion::Stop
                },
                _ => VisitRecursion::Continue,
            })
        }
    }
    if subquery_names.is_empty() || !projections.iter().any(has_marked_aggregate) {
        return false;
    }
    projections.iter().any(|expr| {
        let mut finder = Finder {
            subquery_names,
            found: false,
        };
        let _ = expr.visit(&mut finder, &());
        finder.found
    })
}

/// Check that a block that aggregates without GROUP BY reads the input columns only in its
/// aggregates: `x` has no single value in `SELECT SUM(x), x FROM t`.
pub(crate) fn check_columns_in_aggregates(
    projections: &[Expr],
    subquery_names: &PlHashSet<PlSmallStr>,
) -> PolarsResult<()> {
    struct Finder<'a> {
        subquery_names: &'a PlHashSet<PlSmallStr>,
        column: Option<PlSmallStr>,
    }
    impl Visitor for Finder<'_> {
        type Node = Expr;
        type Arena = ();

        fn pre_visit(&mut self, node: &Expr, _: &()) -> PolarsResult<VisitRecursion> {
            Ok(match node {
                _ if is_marked_aggregate(node) => VisitRecursion::Skip,
                Expr::Column(name)
                    if !self.subquery_names.contains(name) && !is_correlated_result_col(name) =>
                {
                    self.column = Some(name.clone());
                    VisitRecursion::Stop
                },
                _ => VisitRecursion::Continue,
            })
        }
    }
    if !projections.iter().any(has_marked_aggregate) {
        return Ok(());
    }
    for expr in projections {
        let mut finder = Finder {
            subquery_names,
            column: None,
        };
        expr.visit(&mut finder, &())?;
        if let Some(name) = finder.column {
            polars_bail!(SQLSyntax: "'{}' should participate in the GROUP BY clause or an aggregate function", name);
        }
    }
    Ok(())
}

/// Marks the call of an aggregate function: a SQL aggregate, or a user-defined function that
/// returns one value. The parser sets it in the clauses that run in the group context of a
/// block (the SELECT list, QUALIFY, HAVING and the aggregates of ORDER BY), and in the inputs
/// of window functions, as on `COUNT(*)` in `SUM(COUNT(*)) OVER ()`. A window function is not
/// marked, nor are the reductions that its lowering adds.
///
/// The mark is a rename that keeps the name, by a callback that only this module holds, so no
/// alias written in the query is taken for it. `GroupContextSplitter` computes each marked
/// aggregate per group, and a block without GROUP BY or windows removes the marks, so none
/// reach the plan.
static AGGREGATE_MARK: LazyLock<PlanCallback<PlSmallStr, PlSmallStr>> =
    LazyLock::new(|| PlanCallback::new(Ok));

pub(crate) fn mark_aggregate(expr: Expr) -> Expr {
    Expr::RenameAlias {
        function: RenameAliasFn::Map(AGGREGATE_MARK.clone()),
        expr: Arc::new(expr),
    }
}

pub(crate) fn is_marked_aggregate(expr: &Expr) -> bool {
    matches!(expr, Expr::RenameAlias { function: RenameAliasFn::Map(f), .. } if *f == *AGGREGATE_MARK)
}

/// Whether `expr` holds a marked aggregate.
pub(crate) fn has_marked_aggregate(expr: &Expr) -> bool {
    has_expr(expr, is_marked_aggregate)
}

/// `expr` without aggregate marks.
pub(crate) fn strip_aggregate_marks(expr: Expr) -> Expr {
    expr.map_expr(|e| match e {
        Expr::RenameAlias { expr, .. } if is_marked_aggregate(&e) => Arc::unwrap_or_clone(expr),
        e => e,
    })
}

/// `expr`, evaluated in the groups of a GROUP BY on a column, which have at least one row.
/// There, the check `len() > 0` that the aggregate of a constant makes (see
/// `SQLFunctionVisitor::rows_read`) is true, and the optimizer drops it.
pub(crate) fn assume_groups_have_rows(expr: Expr) -> Expr {
    expr.map_expr(|e| match e {
        Expr::BinaryExpr {
            ref left,
            op: Operator::Gt,
            ref right,
        } if matches!(**left, Expr::Len)
            && matches!(
                **right,
                Expr::Literal(LiteralValue::Dyn(DynLiteralValue::Int(0)))
            ) =>
        {
            lit(true)
        },
        e => e,
    })
}
