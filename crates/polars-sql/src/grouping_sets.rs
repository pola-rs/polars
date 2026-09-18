//! `GROUP BY GROUPING SETS / ROLLUP / CUBE` lowered to a `UNION ALL` of ordinary
//! aggregations over one shared input, plus the `GROUPING()` metadata that goes with it.
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::plans::typed_lit;
use polars_plan::plans::visitor::{RewriteRecursion, RewritingVisitor, TreeWalker};
use polars_plan::utils::has_expr;
use polars_utils::aliases::PlIndexSet;
use polars_utils::{format_pl_smallstr, unique_column_name};
use sqlparser::ast::Expr as SQLExpr;

use crate::context::strip_outer_alias;

/// Cap on the number of grouping sets a single `GROUP BY` may expand to.
const MAX_GROUPING_SETS: usize = 4096;

/// Widest `GROUPING()` call whose bits fit an `Int64`.
pub(crate) const MAX_GROUPING_ARGS: usize = 63;

const PLACEHOLDER_PREFIX: &str = "__POLARS_GROUPING_";

const GLOBAL_COUNT_COL: &str = "__POLARS_GSET_COUNT";

/// A `GROUPING(...)` call parsed in the current query block.
///
/// Its arguments are bound to the grouping keys once the `GROUP BY` clause is
/// known; until then the call is a reference to `placeholder`.
#[derive(Clone)]
pub(crate) struct GroupingCall {
    pub args: Vec<SQLExpr>,
    pub placeholder: PlSmallStr,
}

pub(crate) fn new_placeholder() -> PlSmallStr {
    format_pl_smallstr!("{}{}", PLACEHOLDER_PREFIX, unique_column_name())
}

/// Whether `expr` refers to any of the given `GROUPING()` placeholders.
pub(crate) fn contains_grouping_placeholder<'a>(
    expr: &Expr,
    placeholders: impl Iterator<Item = &'a PlSmallStr>,
) -> bool {
    let placeholders: Vec<&PlSmallStr> = placeholders.collect();
    has_expr(
        expr,
        |e| matches!(e, Expr::Column(name) if placeholders.contains(&name)),
    )
}

/// The `GROUP BY` items of a query with grouping sets, expanded into the distinct
/// key expressions they reference and the sets, given as indices into those keys.
pub(crate) struct ExpandedGroupBy {
    pub keys: Vec<SQLExpr>,
    pub sets: Vec<Vec<usize>>,
}

/// The alternative key lists a single `GROUP BY` item contributes, or `None` for a
/// plain expression (which every grouping set then includes).
fn grouping_item_alternatives(item: &SQLExpr) -> PolarsResult<Option<Vec<Vec<SQLExpr>>>> {
    let flatten =
        |groups: &[Vec<SQLExpr>]| -> Vec<SQLExpr> { groups.iter().flatten().cloned().collect() };
    Ok(Some(match item {
        // ROLLUP(a, b, c) => (a,b,c), (a,b), (a), ()
        SQLExpr::Rollup(groups) => (0..=groups.len())
            .rev()
            .map(|n| flatten(&groups[..n]))
            .collect(),
        // CUBE(a, b, c) => every subset of (a, b, c)
        SQLExpr::Cube(groups) => {
            let n = groups.len();
            polars_ensure!(
                n < usize::BITS as usize && (1usize << n) <= MAX_GROUPING_SETS,
                SQLInterface: "CUBE with {} grouping expressions exceeds the {} grouping set limit", n, MAX_GROUPING_SETS
            );
            (0..(1usize << n))
                .rev()
                .map(|mask| {
                    (0..n)
                        .filter(|i| (mask >> i) & 1 == 1)
                        .flat_map(|i| groups[i].iter().cloned())
                        .collect()
                })
                .collect()
        },
        SQLExpr::GroupingSets(sets) => sets.clone(),
        // `GROUP BY ()`
        SQLExpr::Tuple(items) if items.is_empty() => vec![vec![]],
        _ => return Ok(None),
    }))
}

/// Expand a `GROUP BY` item list; `None` when it holds no grouping-set construct
/// and can be grouped directly. Multiple items combine as a Cartesian product.
pub(crate) fn expand_grouping_sets(items: &[SQLExpr]) -> PolarsResult<Option<ExpandedGroupBy>> {
    let alternatives = items
        .iter()
        .map(grouping_item_alternatives)
        .collect::<PolarsResult<Vec<_>>>()?;
    if alternatives.iter().all(Option::is_none) {
        return Ok(None);
    }

    let mut keys: PlIndexSet<SQLExpr> = PlIndexSet::default();
    let mut sets: Vec<Vec<usize>> = vec![Vec::new()];
    for (item, alts) in items.iter().zip(alternatives) {
        let alts = alts.unwrap_or_else(|| vec![vec![item.clone()]]);
        let n_expanded = sets
            .len()
            .checked_mul(alts.len())
            .filter(|n| *n <= MAX_GROUPING_SETS);
        let Some(n_expanded) = n_expanded else {
            polars_bail!(SQLInterface: "GROUP BY expands to more than {} grouping sets", MAX_GROUPING_SETS);
        };
        let mut expanded = Vec::with_capacity(n_expanded);
        for base in &sets {
            for alt in &alts {
                let mut set = base.clone();
                for e in alt {
                    let (idx, _) = keys.insert_full(e.clone());
                    if !set.contains(&idx) {
                        set.push(idx);
                    }
                }
                expanded.push(set);
            }
        }
        sets = expanded;
    }
    Ok(Some(ExpandedGroupBy {
        keys: keys.into_iter().collect(),
        sets,
    }))
}

/// Merge keys that resolved to the same expression, remapping the sets onto the
/// canonical key list.
pub(crate) fn canonicalize_keys(
    resolved: Vec<Expr>,
    sets: Vec<Vec<usize>>,
) -> (Vec<Expr>, Vec<Vec<usize>>) {
    let mut keys: Vec<Expr> = Vec::with_capacity(resolved.len());
    let mut stripped: Vec<Expr> = Vec::with_capacity(resolved.len());
    let mut remap = Vec::with_capacity(resolved.len());
    for key in resolved {
        let s = strip_outer_alias(&key);
        let idx = stripped.iter().position(|k| *k == s).unwrap_or_else(|| {
            keys.push(key);
            stripped.push(s);
            keys.len() - 1
        });
        remap.push(idx);
    }
    let sets = sets
        .into_iter()
        .map(|set| {
            let mut out: Vec<usize> = Vec::with_capacity(set.len());
            for i in set {
                if !out.contains(&remap[i]) {
                    out.push(remap[i]);
                }
            }
            out
        })
        .collect();
    (keys, sets)
}

/// Everything the aggregation stage needs to know about a grouping-sets query.
pub(crate) struct GroupingSets {
    /// Canonical key expressions, as handed to `group_by`.
    keys: Vec<Expr>,
    /// The keys without their output alias, for matching by expression.
    stripped_keys: Vec<Expr>,
    /// One entry per grouping-set occurrence, holding the active canonical keys.
    sets: Vec<Vec<usize>>,
    /// Each `GROUPING()` placeholder with the canonical keys of its arguments.
    calls: Vec<(PlSmallStr, Vec<usize>)>,
}

impl GroupingSets {
    /// Bind every parsed `GROUPING()` call to the canonical keys.
    pub(crate) fn new(
        keys: Vec<Expr>,
        sets: Vec<Vec<usize>>,
        calls: &[GroupingCall],
        resolved_args: Vec<Vec<Expr>>,
    ) -> PolarsResult<Self> {
        let stripped_keys: Vec<Expr> = keys.iter().map(strip_outer_alias).collect();
        let calls = calls
            .iter()
            .zip(resolved_args)
            .map(|(call, args)| {
                let indices = args
                    .iter()
                    .zip(&call.args)
                    .map(|(arg, sql_arg)| {
                        let arg = strip_outer_alias(arg);
                        stripped_keys.iter().position(|k| *k == arg).ok_or_else(|| {
                            polars_err!(SQLSyntax: "GROUPING() argument '{}' does not appear in the GROUP BY clause", sql_arg)
                        })
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;
                Ok((call.placeholder.clone(), indices))
            })
            .collect::<PolarsResult<Vec<_>>>()?;
        Ok(Self {
            keys,
            stripped_keys,
            sets,
            calls,
        })
    }

    pub(crate) fn placeholders(&self) -> impl Iterator<Item = &PlSmallStr> {
        self.calls.iter().map(|(name, _)| name)
    }

    pub(crate) fn contains_placeholder(&self, expr: &Expr) -> bool {
        contains_grouping_placeholder(expr, self.placeholders())
    }

    /// Point computed grouping expressions in a post-aggregation expression at the
    /// columns holding their grouped values. A key is matched as a whole before its
    /// children are visited, and aggregate arguments keep reading the original input.
    pub(crate) fn bind_stored_keys(&self, expr: Expr, key_schema: &Schema) -> Expr {
        struct Binder<'a> {
            keys: Vec<(Expr, &'a PlSmallStr)>,
        }
        impl RewritingVisitor for Binder<'_> {
            type Node = Expr;
            type Arena = ();

            fn pre_visit(&mut self, node: &Expr, _: &mut ()) -> PolarsResult<RewriteRecursion> {
                Ok(match node {
                    Expr::Agg(_) | Expr::Len => RewriteRecursion::Stop,
                    _ if self.keys.iter().any(|(k, _)| k == node) => {
                        RewriteRecursion::MutateAndStop
                    },
                    _ => RewriteRecursion::NoMutateAndContinue,
                })
            }

            fn mutate(&mut self, node: Expr, _: &mut ()) -> PolarsResult<Expr> {
                let (_, name) = self.keys.iter().find(|(k, _)| *k == node).unwrap();
                Ok(col((*name).clone()))
            }
        }
        let keys = self
            .stripped_keys
            .iter()
            .zip(key_schema.iter_names())
            .map(|(key, name)| (key.clone(), name))
            .filter(|(key, _)| !matches!(key, Expr::Column(_)))
            .collect::<Vec<_>>();
        if keys.is_empty() {
            return expr;
        }
        expr.rewrite(&mut Binder { keys }, &mut ()).unwrap()
    }

    /// The `GROUPING()` bits of a call within one set: an argument the set omits
    /// contributes a set bit, the last argument being the least significant.
    fn call_value(set: &[usize], args: &[usize]) -> i64 {
        args.iter()
            .fold(0i64, |acc, i| (acc << 1) | i64::from(!set.contains(i)))
    }

    /// Run the ordinary aggregation once per grouping set over one shared input and
    /// concatenate the branches onto a common schema of canonical keys (NULL where a
    /// set omits them), aggregate outputs and `GROUPING()` values.
    ///
    /// `group_aggs` run in `group_by().agg()`; `global_aggs` are the same aggregates
    /// as they run in `select()` for the empty set.
    pub(crate) fn aggregate(
        &self,
        lf: LazyFrame,
        key_schema: &Schema,
        group_aggs: &[Expr],
        global_aggs: &[Expr],
        agg_names: &[PlSmallStr],
    ) -> PolarsResult<LazyFrame> {
        // Compute non-column keys once, before the input is shared.
        let mut prepared = Vec::new();
        let group_keys: Vec<Expr> = self
            .keys
            .iter()
            .zip(&self.stripped_keys)
            .zip(key_schema.iter_names())
            .map(|((key, inner), name)| {
                if matches!(inner, Expr::Column(_)) {
                    key.clone()
                } else {
                    let hidden = unique_column_name();
                    prepared.push(inner.clone().alias(hidden.clone()));
                    col(hidden).alias(name.clone())
                }
            })
            .collect();
        let lf = if prepared.is_empty() {
            lf
        } else {
            lf.with_columns(prepared)
        };
        let shared = if self.sets.len() > 1 { lf.cache() } else { lf };

        // With neither keys nor aggregates only a row count can carry the height.
        let need_count = key_schema.is_empty() && agg_names.is_empty();

        let branches = self
            .sets
            .iter()
            .map(|set| {
                let branch = if set.is_empty() {
                    let mut aggs = global_aggs.to_vec();
                    if aggs.is_empty() {
                        aggs.push(len().alias(GLOBAL_COUNT_COL));
                    }
                    shared.clone().select(aggs)
                } else {
                    let keys: Vec<Expr> = set.iter().map(|&i| group_keys[i].clone()).collect();
                    shared.clone().group_by(keys).agg(group_aggs)
                };

                let mut out: Vec<Expr> =
                    Vec::with_capacity(key_schema.len() + agg_names.len() + self.calls.len() + 1);
                for (i, (name, dtype)) in key_schema.iter().enumerate() {
                    out.push(if set.contains(&i) {
                        col(name.clone())
                    } else {
                        Scalar::null(dtype.clone()).lit().alias(name.clone())
                    });
                }
                out.extend(agg_names.iter().map(|n| col(n.clone())));
                out.extend(self.calls.iter().map(|(placeholder, args)| {
                    typed_lit(Self::call_value(set, args)).alias(placeholder.clone())
                }));
                if need_count {
                    out.push(col(GLOBAL_COUNT_COL));
                }
                branch.select(out)
            })
            .collect::<Vec<_>>();

        if branches.len() == 1 {
            Ok(branches.into_iter().next().unwrap())
        } else {
            concat(
                branches,
                UnionArgs {
                    maintain_order: false,
                    ..Default::default()
                },
            )
        }
    }
}
