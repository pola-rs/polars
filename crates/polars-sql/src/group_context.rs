//! The group context of a SELECT block: which parts of its expressions are computed per
//! group, and which run on the aggregated rows.

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::plans::visitor::{TreeWalker, VisitRecursion, Visitor};
use polars_plan::plans::{ArenaExprIter, ExprToIRContext, is_scalar_ae, to_expr_ir};
use polars_plan::prelude::*;
use polars_utils::aliases::PlHashSet;
use polars_utils::format_pl_smallstr;

use crate::context::{is_correlated_result_col, strip_outer_alias, without_resolved_subqueries};
use crate::{SQLContext, unique_column_name};

/// The expressions computed in the group context.
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
        self.exprs.push(expr);
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

/// Splits post-aggregation expressions from the group-context reductions they
/// contain, over the pre-aggregation schema.
pub(crate) struct GroupContextSplitter<'a> {
    pub(crate) schema: &'a Schema,
    pub(crate) keys: &'a Schema,
    /// Each group key without its alias, and the column holding it after aggregation.
    pub(crate) key_exprs: Vec<(Expr, PlSmallStr)>,
    pub(crate) whole_frame_partition: Option<&'a PlSmallStr>,
    /// Columns holding the result of a scalar subquery.
    pub(crate) subquery_names: &'a PlHashSet<PlSmallStr>,
    pub(crate) aggregates: AggregateOutputs,
}

impl GroupContextSplitter<'_> {
    fn is_reduction(&self, expr: &Expr) -> bool {
        is_reduction(expr, self.schema)
    }

    /// Whether a SELECT projection must be processed in the group context rather
    /// than passed through as a group key: it contains an aggregate, a window, a
    /// function over a non-key column, or reduces to a scalar (which also covers
    /// aggregates lowered to plain functions, such as `COVAR_POP`).
    pub(crate) fn requires_group_processing(&self, expr: &Expr) -> bool {
        has_expr(expr, |e| match e {
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
        }) || self.is_reduction(&strip_outer_alias(expr))
    }

    /// Whether `expr` must run after aggregation: it holds a window, or combines
    /// a reduction with a grouped key.
    pub(crate) fn needs_post_aggregation(&self, expr: &Expr) -> bool {
        struct Finder<'a, 'b> {
            splitter: &'a GroupContextSplitter<'b>,
            has_window: bool,
            has_group_key: bool,
            has_reduction: bool,
        }
        impl Visitor for Finder<'_, '_> {
            type Node = Expr;
            type Arena = ();

            fn pre_visit(&mut self, node: &Expr, _: &()) -> PolarsResult<VisitRecursion> {
                // Siblings are still visited so that no reduction goes unnoticed.
                Ok(match node {
                    Expr::Over { .. } => {
                        self.has_window = true;
                        VisitRecursion::Skip
                    },
                    Expr::Column(name) => {
                        self.has_group_key |= self.splitter.keys.contains(name);
                        VisitRecursion::Skip
                    },
                    _ if self.splitter.is_reduction(node) => {
                        self.has_reduction = true;
                        VisitRecursion::Skip
                    },
                    _ => VisitRecursion::Continue,
                })
            }
        }
        let mut finder = Finder {
            splitter: self,
            has_window: false,
            has_group_key: false,
            has_reduction: false,
        };
        let _ = expr.visit(&mut finder, &());
        finder.has_window || (finder.has_group_key && finder.has_reduction)
    }

    /// Replace every reduction in `expr` with a reference to a hoisted aggregation
    /// output, recorded in `self.aggregates`. A window runs on the aggregated rows, so in
    /// its inputs only the marked aggregates are hoisted (see `WINDOW_AGGREGATE`).
    pub(crate) fn hoist(&mut self, expr: Expr) -> PolarsResult<Expr> {
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
            e if is_window_aggregate(&e) => {
                Ok(self.aggregates.get_or_insert_hoisted(strip_outer_alias(&e)))
            },
            e if self.is_reduction(&e) => Ok(self.aggregates.get_or_insert_hoisted(e)),
            e => e.map_children(&mut |c, _| self.hoist(c), &mut ()),
        }
    }

    /// Bind an input of a window function to the aggregated rows: a marked aggregate
    /// reads its hoisted output and a group key reads its key column.
    fn bind_window_input(&mut self, expr: Expr) -> PolarsResult<Expr> {
        if let Some((_, name)) = self.key_exprs.iter().find(|(key, _)| *key == expr) {
            return Ok(col(name.clone()));
        }
        match expr {
            e if is_window_aggregate(&e) => {
                Ok(self.aggregates.get_or_insert_hoisted(strip_outer_alias(&e)))
            },
            // A scalar subquery, or a correlated column, which has one value per group.
            Expr::Agg(AggExpr::First(inner))
                if matches!(inner.as_ref(), Expr::Column(name)
                    if self.subquery_names.contains(name) || is_correlated_result_col(name)) =>
            {
                Ok(self
                    .aggregates
                    .get_or_insert_hoisted(Expr::Agg(AggExpr::First(inner))))
            },
            Expr::Column(name) if self.subquery_names.contains(&name) => {
                Ok(self.aggregates.get_or_insert_hoisted(col(name).first()))
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

/// Whether `expr` reduces its input columns to one value, judged by the plan it lowers
/// to; this covers SQL aggregates lowered to non-`Agg` reductions.
///
/// An expression that fails to lower is not treated as a reduction; it is left in place
/// and reports its error when the plan is built.
pub(crate) fn is_reduction(expr: &Expr, schema: &Schema) -> bool {
    let mut arena = Arena::new();
    let mut ctx = ExprToIRContext::new(&mut arena, schema);
    ctx.allow_unknown = true;
    ctx.check_column_names = false;
    let Ok(ir) = to_expr_ir(expr.clone(), &mut ctx) else {
        return false;
    };
    // A literal expression is scalar too, but reduces nothing.
    let reduces = arena.iter(ir.node()).any(|(_, ae)| match ae {
        AExpr::Agg(_) | AExpr::AnonymousAgg { .. } | AExpr::Len => true,
        AExpr::Function { options, .. } | AExpr::AnonymousFunction { options, .. } => {
            options.returns_scalar()
        },
        _ => false,
    });
    reduces && is_scalar_ae(ir.node(), &arena)
}

/// The type of `expr`, or `Unknown` if it can't be inferred.
pub(crate) fn infer_dtype(expr: &Expr, schema: &Schema) -> DataType {
    let mut arena = Arena::new();
    let mut ctx = ExprToIRContext::new(&mut arena, schema);
    ctx.allow_unknown = true;
    ctx.check_column_names = false;
    to_expr_ir(expr.clone(), &mut ctx)
        .and_then(|ir| ir.dtype(schema, &arena).cloned())
        .unwrap_or(DataType::Unknown(Default::default()))
}

/// The names that QUALIFY and ORDER BY can read besides the input columns: columns renamed
/// by `SELECT * RENAME`, then SELECT aliases. Each stands for an expression over the input
/// columns, of a type that is known if it could be inferred.
pub(crate) struct OutputNames<'a> {
    names: PlHashMap<PlSmallStr, (Expr, Option<DataType>)>,
    /// Columns holding the result of a scalar subquery.
    subquery_names: &'a PlHashSet<PlSmallStr>,
}

impl<'a> OutputNames<'a> {
    /// The output names of `projections` and `renames` over the input `schema`. An input
    /// column comes before an output name of the same name. `typed_schema` is `schema` with
    /// the columns that the projections read before the block resolves them.
    pub(crate) fn new(
        projections: &[Expr],
        renames: &PlHashMap<PlSmallStr, PlSmallStr>,
        schema: &Schema,
        typed_schema: &Schema,
        subquery_names: &'a PlHashSet<PlSmallStr>,
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
        Self {
            names,
            subquery_names,
        }
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

    /// Replace the output names in `expr` by their expressions. An aggregate that an alias
    /// brings into the inputs of a window is marked, as the parser marks the aggregates it
    /// finds there.
    pub(crate) fn resolve(&self, expr: Expr, schema: &Schema) -> Expr {
        self.resolve_rec(expr, false, schema)
    }

    fn resolve_rec(&self, expr: Expr, in_window: bool, schema: &Schema) -> Expr {
        match expr {
            Expr::Column(name) => match self.names.get(&name) {
                Some((expr, _)) if in_window => {
                    mark_aggregates(expr.clone(), schema, self.subquery_names)
                },
                Some((expr, _)) => expr.clone(),
                None => Expr::Column(name),
            },
            e => {
                let in_window = in_window || matches!(e, Expr::Over { .. });
                e.map_children(
                    &mut |c, _| Ok(self.resolve_rec(c, in_window, schema)),
                    &mut (),
                )
                .unwrap()
            },
        }
    }
}

/// Whether a block without GROUP BY has both windows and aggregates, as in
/// `SELECT SUM(x), COUNT(*) OVER () FROM t`.
pub(crate) fn has_windows_over_aggregates(
    projections: &[Expr],
    subquery_names: &PlHashSet<PlSmallStr>,
    schema: &Schema,
) -> bool {
    struct Finder<'a> {
        schema: &'a Schema,
        has_window: bool,
        has_reduction: bool,
    }
    impl Visitor for Finder<'_> {
        type Node = Expr;
        type Arena = ();

        fn pre_visit(&mut self, node: &Expr, _: &()) -> PolarsResult<VisitRecursion> {
            Ok(match node {
                Expr::Over { .. } => {
                    self.has_window = true;
                    self.has_reduction |= has_window_aggregate(node);
                    VisitRecursion::Skip
                },
                _ if is_reduction(node, self.schema) => {
                    self.has_reduction = true;
                    VisitRecursion::Skip
                },
                _ => VisitRecursion::Continue,
            })
        }
    }
    let mut finder = Finder {
        schema,
        has_window: false,
        has_reduction: false,
    };
    for e in projections {
        let _ = without_resolved_subqueries(e, subquery_names, schema).visit(&mut finder, &());
    }
    finder.has_window && finder.has_reduction
}

/// Alias that marks a grouped aggregate in the inputs (arguments, FILTER, PARTITION BY,
/// ORDER BY) of a window function, as `COUNT(*)` in `SUM(COUNT(*)) OVER ()`. The aggregate
/// is computed per group, and the window reads its result on the aggregated rows. The
/// reductions that the lowering of a window adds itself, as the row count of
/// `COUNT(*) OVER ()`, are not marked.
///
/// Markers are set when a function call is parsed (`mark_aggregate_call`), and when a SELECT
/// alias brings an aggregate into a window (`OutputNames::resolve`). `GroupContextSplitter::hoist`
/// replaces each one with a column of the aggregation, so none reach the plan.
const WINDOW_AGGREGATE: &str = "__POLARS_WINDOW_AGG";

pub(crate) fn is_window_aggregate(expr: &Expr) -> bool {
    matches!(expr, Expr::Alias(_, name) if name == WINDOW_AGGREGATE)
}

/// Whether `expr` holds a marked aggregate.
pub(crate) fn has_window_aggregate(expr: &Expr) -> bool {
    has_expr(expr, is_window_aggregate)
}

/// Replace each marked aggregate in `expr` by `f` of the aggregate.
pub(crate) fn map_window_aggregates(expr: Expr, f: impl Fn(Expr) -> Expr) -> Expr {
    expr.map_expr(|e| match e {
        Expr::Alias(inner, name) if name == WINDOW_AGGREGATE => f(Arc::unwrap_or_clone(inner)),
        e => e,
    })
}

impl SQLContext {
    /// Mark `expr`, a function call parsed in the inputs of a window, if it is an aggregate.
    pub(crate) fn mark_aggregate_call(
        &mut self,
        expr: Expr,
        schema: &Schema,
    ) -> PolarsResult<Expr> {
        // A subquery, which is not resolved yet, and an aggregate already marked inside the
        // call are values here, not reductions. Each is read as a NULL of its type, as
        // lowering checks types.
        let mut subquery_dtypes = PlHashMap::new();
        for e in &expr {
            if let Expr::SubPlan(lp, names) = e {
                let mut lf = LazyFrame::from((***lp).clone()).select([names[0].1.clone()]);
                let dtype = self
                    .get_frame_schema(&mut lf)
                    .ok()
                    .and_then(|schema| schema.iter_values().next().cloned())
                    .unwrap_or(DataType::Unknown(Default::default()));
                subquery_dtypes.insert(names[0].0.clone(), dtype);
            }
        }
        let unmarked = expr.clone().map_expr(|e| match e {
            Expr::SubPlan(_, names) => Scalar::null(subquery_dtypes[&names[0].0].clone()).lit(),
            // `IN (subquery)` also reads the subquery's column.
            Expr::Agg(AggExpr::First(inner)) => match inner.as_ref() {
                Expr::Column(name) if subquery_dtypes.contains_key(name) => {
                    Scalar::null(subquery_dtypes[name].clone()).lit()
                },
                _ => Expr::Agg(AggExpr::First(inner)),
            },
            Expr::Alias(inner, name) if name == WINDOW_AGGREGATE => {
                Scalar::null(infer_dtype(&inner, schema)).lit()
            },
            e => e,
        });
        if !is_reduction(&unmarked, schema) {
            return Ok(expr);
        }
        polars_ensure!(
            !has_window_aggregate(&expr),
            SQLSyntax: "aggregate function calls cannot be nested"
        );
        Ok(expr.alias(WINDOW_AGGREGATE))
    }
}

/// Mark the aggregates of an expression that a SELECT alias brings into the inputs of a
/// window.
fn mark_aggregates(expr: Expr, schema: &Schema, subquery_names: &PlHashSet<PlSmallStr>) -> Expr {
    match expr {
        Expr::Over { .. } => expr,
        // The result of a scalar subquery is one value, not an aggregate of the rows.
        e if is_reduction(
            &without_resolved_subqueries(&e, subquery_names, schema),
            schema,
        ) =>
        {
            e.alias(WINDOW_AGGREGATE)
        },
        e => e
            .map_children(
                &mut |c, _| Ok(mark_aggregates(c, schema, subquery_names)),
                &mut (),
            )
            .unwrap(),
    }
}
