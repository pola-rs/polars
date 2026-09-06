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

use polars_core::prelude::DataType;
use polars_utils::arena::{Arena, Node};
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;
use polars_utils::{IdxSize, format_pl_smallstr};

use crate::plans::stats::node_stats;
use crate::plans::{
    AExpr, ExprIR, IR, IRAggExpr, IRBuilder, JoinOptionsIR, JoinTypeOptionsIR, OutputName,
    has_aexpr,
};
use crate::prelude::{JoinArgs, JoinCoalesce, JoinType};

/// Name of the injected row-number column.
const SURROGATE_KEY: &str = "__POLARS_SURROGATE_KEY";

/// How much smaller the group-by output must be than its input before the extra
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

/// Rough width of a key in the row encoding, in bytes.
fn encoded_width(dtype: &DataType) -> usize {
    use DataType::*;
    1 + match dtype {
        Boolean | Int8 | UInt8 => 1,
        Int16 | UInt16 | Float16 => 2,
        Int32 | UInt32 | Float32 | Date | Time => 4,
        Int64 | UInt64 | Float64 | Datetime(_, _) | Duration(_) => 8,
        Int128 | UInt128 => 16,
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => 16,
        #[cfg(feature = "dtype-categorical")]
        Categorical(_, _) | Enum(_, _) => 4,
        _ => VARIABLE_KEY_WIDTH,
    }
}

pub fn surrogate_group_by(root: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) {
    let mut stack = vec![root];
    let mut group_bys = Vec::new();
    while let Some(node) = stack.pop() {
        let ir = ir_arena.get(node);
        if matches!(ir, IR::GroupBy { .. }) {
            group_bys.push(node);
        }
        ir.copy_inputs(&mut stack);
    }

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
    /// A join, and whether the descent went into its left input.
    Join(Node, bool),
    /// A column selection.
    Projection(Node),
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
        if let IR::SimpleProjection { input, .. } = ir_arena.get(cur) {
            let input = *input;
            path.push(PathStep::Projection(cur));
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

        let (left, right) = (*input_left, *input_right);
        let left_schema = ir_arena.get(left).schema(ir_arena).into_owned();
        let right_schema = ir_arena.get(right).schema(ir_arena).into_owned();

        let in_left: Vec<PlSmallStr> = key_names
            .iter()
            .filter(|n| left_schema.contains(n))
            .cloned()
            .collect();
        let in_right: Vec<PlSmallStr> = key_names
            .iter()
            .filter(|n| right_schema.contains(n))
            .cloned()
            .collect();

        // A name on both sides is renamed by the join, so we cannot follow it.
        if in_left.iter().any(|n| right_schema.contains(n)) {
            break;
        }

        let (next, next_keys) = if in_left.len() >= MIN_SURROGATE_KEYS {
            (left, in_left)
        } else if in_right.len() >= MIN_SURROGATE_KEYS {
            (right, in_right)
        } else {
            break;
        };

        path.push(PathStep::Join(cur, next == left));
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
    partials: &mut Vec<ExprIR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Node> {
    let ae = expr_arena.get(node).clone();

    if matches!(ae, AExpr::Agg(_) | AExpr::Len) {
        return split_aggregation(node, &ae, partials, expr_arena);
    }

    // A bare column in a group-by is the whole group's values, which the second
    // phase no longer has.
    if matches!(ae, AExpr::Column(_)) || !ae.is_elementwise_top_level() {
        return None;
    }

    let mut inputs = Vec::new();
    ae.inputs_rev(&mut inputs);
    for input in &mut inputs {
        *input = split_expr(*input, partials, expr_arena)?;
    }
    inputs.reverse();
    Some(expr_arena.add(ae.replace_inputs(&inputs)))
}

/// Emits the partial for one aggregation and returns the expression merging it.
fn split_aggregation(
    node: Node,
    ae: &AExpr,
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

    let partial_name = format_pl_smallstr!("__POLARS_PARTIAL_{}", partials.len());
    let col = |expr_arena: &mut Arena<AExpr>| expr_arena.add(AExpr::Column(partial_name.clone()));

    let merge = match ae {
        AExpr::Len | AExpr::Agg(IRAggExpr::Sum(_)) | AExpr::Agg(IRAggExpr::Count { .. }) => {
            let input = col(expr_arena);
            expr_arena.add(AExpr::Agg(IRAggExpr::Sum(input)))
        },
        AExpr::Agg(IRAggExpr::Min { propagate_nans, .. }) => {
            let input = col(expr_arena);
            expr_arena.add(AExpr::Agg(IRAggExpr::Min {
                input,
                propagate_nans: *propagate_nans,
            }))
        },
        AExpr::Agg(IRAggExpr::Max { propagate_nans, .. }) => {
            let input = col(expr_arena);
            expr_arena.add(AExpr::Agg(IRAggExpr::Max {
                input,
                propagate_nans: *propagate_nans,
            }))
        },
        _ => return None,
    };

    partials.push(ExprIR::new(node, OutputName::Alias(partial_name)));
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

    // Only worth it when the group-by discards most of its input, and when the
    // surrogate is much smaller than what flows into the group-by.
    let in_rows = node_stats(input, ir_arena, expr_arena)?.filtered;
    let out_rows = node_stats(node, ir_arena, expr_arena)?.filtered;
    let surrogate_rows = node_stats(surrogate.node, ir_arena, expr_arena)?.filtered;
    if polars_core::config::verbose() {
        eprintln!(
            "surrogate group-by candidate: {in_rows:.0} rows -> {out_rows:.0} groups \
             (reduction {:.1}x), surrogate {surrogate_rows:.0} rows, \
             key {surrogate_width}/{total_width} bytes",
            in_rows / out_rows,
        );
    }
    if in_rows < MIN_REDUCTION * out_rows || in_rows < MIN_REDUCTION * surrogate_rows {
        return None;
    }

    let mut partial_aggs = Vec::with_capacity(aggs.len());
    let mut merge_aggs = Vec::with_capacity(aggs.len());
    for agg in &aggs {
        let before = partial_aggs.len();
        let merge = split_expr(agg.node(), &mut partial_aggs, expr_arena)?;
        if partial_aggs.len() == before {
            // Nothing to aggregate; the expression is constant per group.
            return None;
        }
        merge_aggs.push(ExprIR::new(
            merge,
            OutputName::Alias(agg.output_name().clone()),
        ));
    }

    let sk = PlSmallStr::from_static(SURROGATE_KEY);

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
    let mut child = indexed;
    for step in surrogate.path.iter().rev() {
        child = match step {
            PathStep::Join(join, went_left) => {
                let IR::Join {
                    input_left,
                    input_right,
                    options,
                    ..
                } = ir_arena.get(*join)
                else {
                    unreachable!()
                };
                let (left, right, options) = (*input_left, *input_right, options.clone());
                let (left, right) = if *went_left {
                    (child, right)
                } else {
                    (left, child)
                };
                IRBuilder::new(left, expr_arena, ir_arena)
                    .join(right, options)
                    .node()
            },
            PathStep::Projection(node) => {
                let IR::SimpleProjection { columns, .. } = ir_arena.get(*node) else {
                    unreachable!()
                };
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

    let mut phase1_keys = Vec::with_capacity(keys.len());
    phase1_keys.push(ExprIR::new(
        expr_arena.add(AExpr::Column(sk.clone())),
        OutputName::ColumnLhs(sk.clone()),
    ));
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

    let sk_left = ExprIR::new(
        expr_arena.add(AExpr::Column(sk.clone())),
        OutputName::ColumnLhs(sk.clone()),
    );
    let sk_right = ExprIR::new(
        expr_arena.add(AExpr::Column(sk.clone())),
        OutputName::ColumnLhs(sk.clone()),
    );
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
        .join(attrs, std::sync::Arc::new(join_options))
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
