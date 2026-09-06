//! Two-phase aggregation over a surrogate key.
//!
//! A group-by whose keys all come from one input of a join can group on that
//! input's row number instead. Row numbers are finer than the attributes they
//! stand for, so a second aggregation over the partial results restores the
//! original groups without needing to know that the attributes are unique.
//!
//! ```text
//!   GroupBy[a, b, y]                 GroupBy[a, b, y]           (merge)
//!     Join(fact, dim)         =>       Join(.., dim[sk, a, b])
//!                                        GroupBy[sk, y]         (partial)
//!                                          Join(fact, dim[sk])
//! ```

use std::sync::Arc;

use polars_core::prelude::DataType;
use polars_core::schema::{Schema, SchemaRef};
use polars_utils::aliases::InitHashMaps;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;
use polars_utils::{IdxSize, format_pl_smallstr};

use crate::plans::lit::LiteralValue;
use crate::plans::stats::{StatsCache, node_stats_with_cache};
use crate::plans::{
    AExpr, AExprBuilder, ArenaLpIter, ExprIR, IR, IRAggExpr, IRBuilder, JoinOptionsIR,
    JoinTypeOptionsIR, OutputName, ToFieldContext, has_aexpr,
};
use crate::prelude::{JoinArgs, JoinCoalesce, JoinType, Operator};

/// Base name of the injected row-number column.
const SURROGATE_KEY: &str = "__POLARS_SURROGATE_KEY";

/// How many rows must flow into the group-by per surrogate row before the extra
/// aggregation and join pay for themselves.
const MIN_REDUCTION: f64 = 4.0;

/// Number of keys that must come from one join input for the rewrite to be worth it.
const MIN_SURROGATE_KEYS: usize = 3;

/// Fraction of the key width the surrogate must replace, as a percentage. The
/// second aggregation costs a group-by over the partials plus a join, so the first
/// one has to become much cheaper to pay for it.
const MIN_WIDTH_REPLACED_PCT: usize = 75;

/// Encoded width of the surrogate key itself, in bytes.
const SURROGATE_WIDTH: usize = 1 + size_of::<IdxSize>();

/// Assumed encoded width of a key whose width is not fixed.
const VARIABLE_KEY_WIDTH: usize = 16;

/// Rough width of a key in the row encoding: a null byte plus the value.
fn encoded_width(dtype: &DataType) -> usize {
    1 + dtype
        .byte_width()
        .map_or(VARIABLE_KEY_WIDTH, |w| w.ceil() as usize)
}

/// A name for the row-number column that no column in `schemas` shares a prefix
/// with, so neither it nor the `_PARTIAL_` names derived from it can collide.
fn unique_surrogate_name(schemas: &[SchemaRef]) -> PlSmallStr {
    let mut n = 0;
    loop {
        let candidate = match n {
            0 => PlSmallStr::from_static(SURROGATE_KEY),
            n => format_pl_smallstr!("{SURROGATE_KEY}_{n}"),
        };
        let taken = schemas
            .iter()
            .flat_map(|schema| schema.iter_names())
            .any(|name| name.starts_with(candidate.as_str()));
        if !taken {
            return candidate;
        }
        n += 1;
    }
}

pub fn surrogate_group_by(root: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) {
    let group_bys: Vec<Node> = ir_arena
        .iter(root)
        .filter_map(|(node, ir)| matches!(ir, IR::GroupBy { .. }).then_some(node))
        .collect();

    for node in group_bys {
        if let Some(rewritten) = try_rewrite(node, ir_arena, expr_arena) {
            let ir = ir_arena.take(rewritten);
            ir_arena.replace(node, ir);
        }
    }
}

/// The join input that supplies most of the group keys, and the joins above it.
struct Surrogate {
    /// The subplan whose row number becomes the surrogate key.
    node: Node,
    /// The nodes from the group-by input down to `node`, top first.
    path: Vec<PathStep>,
    /// Key columns living in `node`.
    keys: Vec<PlSmallStr>,
}

/// A node between the group-by and the surrogate, which must be rebuilt so it
/// carries the row index up.
enum PathStep {
    /// A join: the input the descent did not follow, and which side it took.
    Join {
        other: Node,
        went_left: bool,
        options: Arc<JoinOptionsIR>,
    },
    /// The columns a projection selects.
    Projection(SchemaRef),
}

/// Rebuilds `path` on top of `indexed`, so every level passes the row index up.
fn rebuild_path(
    path: &[PathStep],
    indexed: Node,
    sk: &PlSmallStr,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    let mut child = indexed;
    for step in path.iter().rev() {
        child = match step {
            PathStep::Join {
                other,
                went_left,
                options,
            } => {
                let (left, right) = if *went_left {
                    (child, *other)
                } else {
                    (*other, child)
                };
                IRBuilder::new(left, expr_arena, ir_arena)
                    .join(right, options.clone())
                    .node()
            },
            PathStep::Projection(columns) => {
                let mut names = Vec::with_capacity(columns.len() + 1);
                names.push(sk.clone());
                names.extend(columns.iter_names().cloned());
                IRBuilder::new(child, expr_arena, ir_arena)
                    .project_simple(names)
                    .ok()?
                    .node()
            },
        };
    }
    Some(child)
}

fn find_surrogate(
    input: Node,
    key_names: &[PlSmallStr],
    ir_arena: &Arena<IR>,
) -> Option<Surrogate> {
    let mut cur = input;
    let mut path = Vec::new();
    let mut keys: Vec<PlSmallStr> = Vec::new();

    loop {
        // A simple projection only narrows, so the names below are a superset.
        if let IR::SimpleProjection { input, columns } = ir_arena.get(cur) {
            let (input, columns) = (*input, columns.clone());
            path.push(PathStep::Projection(columns));
            cur = input;
            continue;
        }

        let IR::Join {
            input_left,
            input_right,
            options,
            ..
        } = ir_arena.get(cur)
        else {
            break;
        };
        if !matches!(options.args.how, JoinType::Inner) || !options.is_pure_equi() {
            break;
        }

        let (left, right, options) = (*input_left, *input_right, options.clone());
        let left_schema = ir_arena.get(left).schema(ir_arena).into_owned();
        let right_schema = ir_arena.get(right).schema(ir_arena).into_owned();

        let mut in_left = Vec::new();
        let mut in_right = Vec::new();
        let mut ambiguous = false;
        for name in key_names {
            let (l, r) = (left_schema.contains(name), right_schema.contains(name));
            // A name on both sides is renamed by the join, so we cannot follow it.
            ambiguous |= l && r;
            if l {
                in_left.push(name.clone());
            }
            if r {
                in_right.push(name.clone());
            }
        }
        if ambiguous {
            break;
        }

        let (next, other, next_keys) = if in_left.len() >= MIN_SURROGATE_KEYS {
            (left, right, in_left)
        } else if in_right.len() >= MIN_SURROGATE_KEYS {
            (right, left, in_right)
        } else {
            break;
        };

        path.push(PathStep::Join {
            other,
            went_left: next == left,
            options,
        });
        keys = next_keys;
        cur = next;
    }

    (cur != input && keys.len() >= MIN_SURROGATE_KEYS).then_some(Surrogate {
        node: cur,
        path,
        keys,
    })
}

/// Rewrites one aggregation expression into partial aggregations and a merge
/// expression over their results.
///
/// The expression may wrap its aggregations in elementwise operations, as SQL's
/// `sum` does with `when(count() > 0).then(sum())`. Those wrappers are rebuilt
/// verbatim around the merged partials.
fn split_expr(
    node: Node,
    sk: &PlSmallStr,
    schema: &Schema,
    partials: &mut Vec<ExprIR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    let ae = expr_arena.get(node).clone();

    if matches!(ae, AExpr::Agg(_) | AExpr::Len) {
        return split_aggregation(node, &ae, sk, schema, partials, expr_arena);
    }

    // A bare column in a group-by is the whole group's values, which the second
    // phase no longer has.
    if matches!(ae, AExpr::Column(_)) || !ae.is_elementwise_top_level() {
        return None;
    }

    let mut inputs = Vec::new();
    ae.inputs_rev(&mut inputs);
    for input in &mut inputs {
        *input = split_expr(*input, sk, schema, partials, expr_arena)?;
    }
    inputs.reverse();
    Some(expr_arena.add(ae.replace_inputs(&inputs)))
}

/// Emits the partials for one aggregation and returns the expression merging them.
fn split_aggregation(
    node: Node,
    ae: &AExpr,
    sk: &PlSmallStr,
    schema: &Schema,
    partials: &mut Vec<ExprIR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    // An aggregation of an aggregation is not a group-wise decomposition.
    let mut nested = Vec::new();
    ae.inputs_rev(&mut nested);
    for input in nested {
        if has_aexpr(input, expr_arena, |ae| {
            matches!(ae, AExpr::Agg(_) | AExpr::Len)
        }) {
            return None;
        }
    }

    // Aggregates `expr` in the first phase and returns a column reading it back.
    fn partial(
        expr: Node,
        sk: &PlSmallStr,
        partials: &mut Vec<ExprIR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Node {
        let name = format_pl_smallstr!("{sk}_PARTIAL_{}", partials.len());
        partials.push(ExprIR::new(expr, OutputName::Alias(name.clone())));
        AExprBuilder::col(name, expr_arena).node()
    }

    /// [`partial`], merged by summing what the first phase produced.
    fn partial_sum(
        expr: Node,
        sk: &PlSmallStr,
        partials: &mut Vec<ExprIR>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Node {
        let col = partial(expr, sk, partials, expr_arena);
        AExprBuilder::new_from_node(col).sum(expr_arena).node()
    }

    let merge = match ae {
        AExpr::Len | AExpr::Agg(IRAggExpr::Sum(_)) | AExpr::Agg(IRAggExpr::Count { .. }) => {
            partial_sum(node, sk, partials, expr_arena)
        },
        AExpr::Agg(IRAggExpr::Min { propagate_nans, .. }) => {
            let input = partial(node, sk, partials, expr_arena);
            AExprBuilder::agg(
                IRAggExpr::Min {
                    input,
                    propagate_nans: *propagate_nans,
                },
                expr_arena,
            )
            .node()
        },
        AExpr::Agg(IRAggExpr::Max { propagate_nans, .. }) => {
            let input = partial(node, sk, partials, expr_arena);
            AExprBuilder::agg(
                IRAggExpr::Max {
                    input,
                    propagate_nans: *propagate_nans,
                },
                expr_arena,
            )
            .node()
        },
        // A mean does not merge, but the sum and the count it is made of do.
        AExpr::Agg(IRAggExpr::Mean(input)) => {
            let dtype = expr_arena
                .get(*input)
                .to_dtype(&ToFieldContext::new(expr_arena, schema))
                .ok()?;
            // A mean accumulates in `f64`, so the partial sums have to as well.
            // Summing integers as integers rounds differently and can overflow.
            let input = match dtype {
                DataType::Float64 => *input,
                dt if dt.is_integer() || dt.is_bool() => AExprBuilder::new_from_node(*input)
                    .cast(DataType::Float64, expr_arena)
                    .node(),
                // Any other mean has a dtype of its own that a quotient loses.
                _ => return None,
            };

            let sum = AExprBuilder::agg(IRAggExpr::Sum(input), expr_arena).node();
            let count = AExprBuilder::new_from_node(input)
                .count_opt_nulls(false, expr_arena)
                .node();
            let total = partial_sum(sum, sk, partials, expr_arena);
            let n = partial_sum(count, sk, partials, expr_arena);

            let mean =
                AExprBuilder::new_from_node(total).binary_op(n, Operator::TrueDivide, expr_arena);
            // Averaging nothing is null rather than a division by zero.
            let zero = AExprBuilder::lit(LiteralValue::new_idxsize(0), expr_arena);
            let null = AExprBuilder::lit(LiteralValue::untyped_null(), expr_arena);
            AExprBuilder::new_from_node(n)
                .binary_op(zero, Operator::Gt, expr_arena)
                .ternary(mean, null, expr_arena)
                .node()
        },
        _ => return None,
    };

    Some(merge)
}

fn try_rewrite(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    let IR::GroupBy {
        input,
        keys,
        aggs,
        schema,
        maintain_order,
        options,
        apply,
    } = ir_arena.get(node)
    else {
        return None;
    };

    if *maintain_order
        || apply.is_some()
        || options.slice.is_some()
        || aggs.is_empty()
        || keys.len() < MIN_SURROGATE_KEYS + 1
    {
        return None;
    }
    #[cfg(feature = "dynamic_group_by")]
    if options.rolling.is_some() || options.dynamic.is_some() {
        return None;
    }

    let (input, keys, aggs, schema, options) = (
        *input,
        keys.clone(),
        aggs.clone(),
        schema.clone(),
        options.clone(),
    );

    // Only plain column keys; a computed key may not be reproducible after the split.
    let mut key_names = Vec::with_capacity(keys.len());
    for key in &keys {
        let AExpr::Column(name) = expr_arena.get(key.node()) else {
            return None;
        };
        if name != key.output_name() {
            return None;
        }
        key_names.push(name.clone());
    }

    let surrogate = find_surrogate(input, &key_names, ir_arena)?;
    if surrogate.keys.len() == keys.len() {
        // Nothing is grouped alongside the surrogate; the second phase would repeat
        // the first for no gain.
        return None;
    }

    // Only worth it when the surrogate replaces most of the key. Otherwise the
    // second aggregation hashes nearly as much as the first.
    let input_schema = ir_arena.get(input).schema(ir_arena).into_owned();
    let width = |name: &PlSmallStr| input_schema.get(name).map(encoded_width);
    let surrogate_width: usize = surrogate.keys.iter().map(width).sum::<Option<usize>>()?;
    let total_width: usize = key_names.iter().map(width).sum::<Option<usize>>()?;
    if surrogate_width <= SURROGATE_WIDTH
        || (surrogate_width - SURROGATE_WIDTH) * 100 < total_width * MIN_WIDTH_REPLACED_PCT
    {
        return None;
    }

    // Only worth it when the first phase collapses many rows into few. It groups on
    // the surrogate alongside the keys that stay behind, so both bound how far it
    // can reduce.
    // The two nodes share descendants, so they share one cache.
    let cache = &mut StatsCache::new();
    let in_stats = node_stats_with_cache(input, ir_arena, expr_arena, cache)?;
    let in_rows = in_stats.filtered;
    let surrogate_rows =
        node_stats_with_cache(surrogate.node, ir_arena, expr_arena, cache)?.filtered;

    let remaining: Vec<&PlSmallStr> = key_names
        .iter()
        .filter(|name| !surrogate.keys.contains(name))
        .collect();
    // The first phase emits one row per surrogate at least, and a key that stays
    // behind can only split that further. Its value domain is all we know of how
    // far, so a key that could be near-unique disqualifies the rewrite.
    let remaining_domain = in_stats.key_domain_product(&remaining).unwrap_or_default();
    let phase1_rows = surrogate_rows.max(remaining_domain);

    if polars_core::config::verbose() {
        eprintln!(
            "surrogate group-by candidate: {in_rows:.0} rows over {phase1_rows:.0} \
             partial rows ({surrogate_rows:.0} surrogate), key \
             {surrogate_width}/{total_width} bytes",
        );
    }
    if in_rows < MIN_REDUCTION * phase1_rows {
        return None;
    }

    // The row number and the partials named after it must not shadow a column the
    // query already has.
    let sk = unique_surrogate_name(&{
        let mut schemas = vec![input_schema.clone()];
        schemas.push(ir_arena.get(surrogate.node).schema(ir_arena).into_owned());
        for step in &surrogate.path {
            if let PathStep::Join { other, .. } = step {
                schemas.push(ir_arena.get(*other).schema(ir_arena).into_owned());
            }
        }
        schemas
    });

    let mut partial_aggs = Vec::with_capacity(aggs.len());
    let mut merge_aggs = Vec::with_capacity(aggs.len());
    for agg in &aggs {
        let before = partial_aggs.len();
        let merge = split_expr(
            agg.node(),
            &sk,
            &input_schema,
            &mut partial_aggs,
            expr_arena,
        )?;
        if partial_aggs.len() == before {
            // Nothing to aggregate; the expression is constant per group.
            return None;
        }
        merge_aggs.push(ExprIR::new(
            merge,
            OutputName::Alias(agg.output_name().clone()),
        ));
    }

    // Both the narrowed join input and the attribute side read the numbered
    // surrogate, and they only agree if the numbering happens once. A cache is
    // evaluated a single time and shared, so the two cannot diverge.
    let indexed = IRBuilder::new(surrogate.node, expr_arena, ir_arena)
        .row_index(sk.clone(), None)
        .node();
    let indexed = ir_arena.add(IR::Cache {
        input: indexed,
        id: UniqueId::new(),
    });

    // Rebuild the joins above the surrogate so they carry the row index up.
    let child = rebuild_path(&surrogate.path, indexed, &sk, ir_arena, expr_arena)?;

    let mut phase1_keys = Vec::with_capacity(keys.len());
    phase1_keys.push(ExprIR::from_column_name(sk.clone(), expr_arena));
    phase1_keys.extend(
        keys.iter()
            .zip(&key_names)
            .filter(|(_, name)| !surrogate.keys.contains(name))
            .map(|(key, _)| key.clone()),
    );

    let phase1 = IRBuilder::new(child, expr_arena, ir_arena)
        .group_by(phase1_keys, partial_aggs, None, false, options.clone())
        .ok()?
        .node();

    let attrs = IRBuilder::new(indexed, expr_arena, ir_arena)
        .project_simple({
            let mut names = Vec::with_capacity(surrogate.keys.len() + 1);
            names.push(sk.clone());
            names.extend(surrogate.keys.iter().cloned());
            names
        })
        .ok()?
        .node();

    let sk_left = ExprIR::from_column_name(sk.clone(), expr_arena);
    let sk_right = ExprIR::from_column_name(sk.clone(), expr_arena);
    let join_options = JoinOptionsIR {
        allow_parallel: true,
        force_parallel: false,
        args: JoinArgs {
            how: JoinType::Inner,
            coalesce: JoinCoalesce::CoalesceColumns,
            ..Default::default()
        },
        options: JoinTypeOptionsIR::Equi {
            on: vec![(sk_left, sk_right)],
        },
    };

    let joined = IRBuilder::new(phase1, expr_arena, ir_arena)
        .join(attrs, Arc::new(join_options))
        .node();

    let phase2 = IRBuilder::new(joined, expr_arena, ir_arena)
        .group_by(keys, merge_aggs, None, false, options)
        .ok()?
        .node();

    // The split must be invisible to everything above.
    let IR::GroupBy {
        schema: new_schema, ..
    } = ir_arena.get(phase2)
    else {
        unreachable!()
    };
    (new_schema == &schema).then_some(phase2)
}
