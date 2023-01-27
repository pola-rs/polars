//! Common Subplan Elimination

use std::collections::{BTreeMap, BTreeSet};
use std::hash::{BuildHasher, Hash, Hasher};

use polars_core::prelude::*;

use crate::prelude::*;

// nodes into an alogicalplan.
type Trail = Vec<Node>;

// we use mutation of `id` to get a unique trail
// we traverse left first, so the `id` remains the same for an all left traversal.
// every right node may increment `id` and because it's shared mutable there will
// be no collisions as the increment is communicated upward with mutation.
pub(super) fn collect_trails(
    root: Node,
    lp_arena: &Arena<ALogicalPlan>,
    // every branch gets its own trail
    // note to self:
    // don't use a vec, as different branches can have collisions
    trails: &mut BTreeMap<u32, Trail>,
    id: &mut u32,
    // if trails should be collected
    collect: bool,
) -> Option<()> {
    // TODO! remove recursion and use a stack
    if collect {
        trails.get_mut(id).unwrap().push(root);
    }

    use ALogicalPlan::*;
    match lp_arena.get(root) {
        // if we find a cache anywhere, that means the users has set caches and we don't interfere
        Cache { .. } => return None,
        // we start collecting from first encountered join
        // later we must unions as well
        Join {
            input_left,
            input_right,
            ..
        } => {
            // make sure that the new branch has the same trail history
            let new_trail = trails.get(id).unwrap().clone();
            collect_trails(*input_left, lp_arena, trails, id, true)?;

            *id += 1;
            trails.insert(*id, new_trail);
            collect_trails(*input_right, lp_arena, trails, id, true)?;
        }
        Union { inputs, .. } => {
            if inputs.len() > 200 {
                // don't even bother with cse on this many inputs
                return None;
            }
            let new_trail = trails.get(id).unwrap().clone();

            let last_i = inputs.len() - 1;

            for (i, input) in inputs.iter().enumerate() {
                collect_trails(*input, lp_arena, trails, id, true)?;

                // don't add a trail on the last iteration as that would only add a Union
                // without any inputs
                if i != last_i {
                    *id += 1;
                    trails.insert(*id, new_trail.clone());
                }
            }
        }
        ExtContext { .. } => {
            // block for now.
        }
        lp => {
            // other nodes have only a single input
            let nodes = &mut [None];
            lp.copy_inputs(nodes);
            if let Some(input) = nodes[0] {
                collect_trails(input, lp_arena, trails, id, collect)?
            }
        }
    }
    Some(())
}

fn expr_nodes_equal(a: &[Node], b: &[Node], expr_arena: &Arena<AExpr>) -> bool {
    a.len() == b.len()
        && a.iter()
            .zip(b)
            .all(|(a, b)| node_to_expr(*a, expr_arena) == node_to_expr(*b, expr_arena))
}

fn predicate_equal(a: Option<Node>, b: Option<Node>, expr_arena: &Arena<AExpr>) -> bool {
    match (a, b) {
        (Some(l), Some(r)) => node_to_expr(l, expr_arena) == node_to_expr(r, expr_arena),
        (None, None) => true,
        _ => false,
    }
}

fn lp_node_equal(a: &ALogicalPlan, b: &ALogicalPlan, expr_arena: &Arena<AExpr>) -> bool {
    use ALogicalPlan::*;
    match (a, b) {
        (
            DataFrameScan {
                df: left_df,
                projection: None,
                selection: None,
                ..
            },
            DataFrameScan {
                df: right_df,
                projection: None,
                selection: None,
                ..
            },
        ) => Arc::ptr_eq(left_df, right_df),
        #[cfg(feature = "parquet")]
        (
            ParquetScan {
                path: path_left,
                predicate: predicate_l,
                options: options_l,
                ..
            },
            ParquetScan {
                path: path_right,
                predicate: predicate_r,
                options: options_r,
                ..
            },
        ) => {
            path_left == path_right
                && options_l == options_r
                && predicate_equal(*predicate_l, *predicate_r, expr_arena)
        }
        #[cfg(feature = "ipc")]
        (
            IpcScan {
                path: path_left,
                predicate: predicate_l,
                options: options_l,
                ..
            },
            IpcScan {
                path: path_right,
                predicate: predicate_r,
                options: options_r,
                ..
            },
        ) => {
            path_left == path_right
                && options_l == options_r
                && predicate_equal(*predicate_l, *predicate_r, expr_arena)
        }
        #[cfg(feature = "csv-file")]
        (
            CsvScan {
                path: path_left,
                predicate: predicate_l,
                options: options_l,
                ..
            },
            CsvScan {
                path: path_right,
                predicate: predicate_r,
                options: options_r,
                ..
            },
        ) => {
            path_left == path_right
                && options_l == options_r
                && predicate_equal(*predicate_l, *predicate_r, expr_arena)
        }
        (Selection { predicate: l, .. }, Selection { predicate: r, .. }) => {
            node_to_expr(*l, expr_arena) == node_to_expr(*r, expr_arena)
        }
        (Projection { expr: l, .. }, Projection { expr: r, .. })
        | (HStack { exprs: l, .. }, HStack { exprs: r, .. }) => expr_nodes_equal(l, r, expr_arena),
        (Melt { args: l, .. }, Melt { args: r, .. }) => Arc::ptr_eq(l, r),
        (
            Slice {
                offset: offset_l,
                len: len_l,
                ..
            },
            Slice {
                offset: offset_r,
                len: len_r,
                ..
            },
        ) => offset_l == offset_r && len_l == len_r,
        (
            Sort {
                by_column: by_l,
                args: args_l,
                ..
            },
            Sort {
                by_column: by_r,
                args: args_r,
                ..
            },
        ) => expr_nodes_equal(by_l, by_r, expr_arena) && args_l == args_r,
        (Explode { columns: l, .. }, Explode { columns: r, .. }) => l == r,
        (Distinct { options: l, .. }, Distinct { options: r, .. }) => l == r,
        (MapFunction { function: l, .. }, MapFunction { function: r, .. }) => l == r,
        (
            Aggregate {
                keys: keys_l,
                aggs: agg_l,
                apply: None,
                maintain_order: maintain_order_l,
                options: options_l,
                ..
            },
            Aggregate {
                keys: keys_r,
                aggs: agg_r,
                apply: None,
                maintain_order: maintain_order_r,
                options: options_r,
                ..
            },
        ) => {
            maintain_order_l == maintain_order_r
                && options_l == options_r
                && expr_nodes_equal(keys_l, keys_r, expr_arena)
                && expr_nodes_equal(agg_l, agg_r, expr_arena)
        }
        #[cfg(feature = "python")]
        (PythonScan { options: l, .. }, PythonScan { options: r, .. }) => l == r,
        _ => {
            // joins and unions are also false
            // they do not originate from a single trail
            // so we would need to follow every leaf that
            // is below the joining/union root
            // that gets complicated quick
            false
        }
    }
}

/// Iterate from two leaf location upwards and find the latest matching node.
///
/// Returns the matching nodes
fn longest_subgraph(
    trail_a: &Trail,
    trail_b: &Trail,
    lp_arena: &Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) -> Option<(Node, Node, bool)> {
    if trail_a.is_empty() || trail_b.is_empty() {
        return None;
    }
    let mut prev_node_a = Node(0);
    let mut prev_node_b = Node(0);
    let mut is_equal;
    let mut i = 0;
    let mut entirely_equal = trail_a.len() == trail_b.len();

    // iterates from the leaves upwards
    for (node_a, node_b) in trail_a.iter().rev().zip(trail_b.iter().rev()) {
        // we never include the root that splits a trail
        // e.g. don't want to cache the join/union, but
        // we want to cache the similar inputs
        if *node_a == *node_b {
            break;
        }
        let a = lp_arena.get(*node_a);
        let b = lp_arena.get(*node_b);

        is_equal = lp_node_equal(a, b, expr_arena);

        if !is_equal {
            entirely_equal = false;
            break;
        }

        prev_node_a = *node_a;
        prev_node_b = *node_b;
        i += 1;
    }
    // previous node was equal
    if i > 0 {
        Some((prev_node_a, prev_node_b, entirely_equal))
    } else {
        None
    }
}

pub(crate) fn elim_cmn_subplans(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    expr_arena: &Arena<AExpr>,
) -> (Node, bool) {
    let mut trails = BTreeMap::new();
    let mut id = 0;
    trails.insert(id, Vec::new());
    if collect_trails(root, lp_arena, &mut trails, &mut id, false).is_none() {
        // early return because we encountered a cache set by the caller
        // we will not interfere with those
        return (root, false);
    }
    let trails = trails.into_values().collect::<Vec<_>>();

    // search from the leafs upwards and find the longest shared subplans
    let mut trail_ends = vec![];
    // if i matches j
    // we don't need to search with j as they are equal
    // this is very important as otherwise we get quadratic behavior
    let mut to_skip = BTreeSet::new();

    for i in 0..trails.len() {
        if to_skip.contains(&i) {
            continue;
        }
        let trail_i = &trails[i];

        // we only look forwards, then we traverse all combinations
        for (j, trail_j) in trails.iter().enumerate().skip(i + 1) {
            if let Some((a, b, all_equal)) =
                longest_subgraph(trail_i, trail_j, lp_arena, expr_arena)
            {
                // then we can skip `j` as we already searched with trail `i` which is equal
                if all_equal {
                    to_skip.insert(j);
                }
                trail_ends.push((a, b))
            }
        }
    }

    let lp_cache = lp_arena as *const Arena<ALogicalPlan> as usize;

    let hb = ahash::RandomState::new();
    let mut changed = false;

    let mut cache_mapping = BTreeMap::new();
    let mut cache_counts = PlHashMap::with_capacity(trail_ends.len());

    for combination in trail_ends.iter() {
        // both are the same, but only point to a different location
        // in our arena so we hash one and store the hash for both locations
        // this will ensure all matches have the same hash.
        let node1 = combination.0 .0;
        let node2 = combination.1 .0;

        let cache_id = match (cache_mapping.get(&node1), cache_mapping.get(&node2)) {
            (Some(h), _) => *h,
            (_, Some(h)) => *h,
            _ => {
                let mut h = hb.build_hasher();
                node1.hash(&mut h);
                let hash = h.finish();
                let mut cache_id = lp_cache.wrapping_add(hash as usize);
                // this ensures we can still add branch ids without overflowing
                // during the dot representation
                if (usize::MAX - cache_id) < 2048 {
                    cache_id -= 2048
                }

                cache_mapping.insert(node1, cache_id);
                cache_mapping.insert(node2, cache_id);
                cache_id
            }
        };
        *cache_counts.entry(cache_id).or_insert(0usize) += 1;
    }

    // insert cache nodes
    for combination in trail_ends.iter() {
        // both are the same, but only point to a different location
        // in our arena so we hash one and store the hash for both locations
        // this will ensure all matches have the same hash.
        let node1 = combination.0 .0;
        let node2 = combination.1 .0;

        let cache_id = match (cache_mapping.get(&node1), cache_mapping.get(&node2)) {
            // (Some(_), Some(_)) => {
            //     continue
            // }
            (Some(h), _) => *h,
            (_, Some(h)) => *h,
            _ => {
                unreachable!()
            }
        };
        let cache_count = *cache_counts.get(&cache_id).unwrap();

        // reassign old nodes to another location as we are going to replace
        // them with a cache node
        for inp_node in [combination.0, combination.1] {
            if let ALogicalPlan::Cache { count, .. } = lp_arena.get_mut(inp_node) {
                *count = cache_count;
            } else {
                let lp = lp_arena.get(inp_node).clone();

                let node = lp_arena.add(lp);

                let cache_lp = ALogicalPlan::Cache {
                    input: node,
                    id: cache_id,
                    // remove after one cache hit.
                    count: cache_count,
                };
                lp_arena.replace(inp_node, cache_lp.clone());
            };
        }

        changed = true;
    }

    (root, changed)
}

// ensure the file count counters are decremented with the cache counts
pub(crate) fn decrement_file_counters_by_cache_hits(
    root: Node,
    lp_arena: &mut Arena<ALogicalPlan>,
    _expr_arena: &Arena<AExpr>,
    acc_count: FileCount,
    scratch: &mut Vec<Node>,
) {
    use ALogicalPlan::*;
    match lp_arena.get_mut(root) {
        #[cfg(feature = "parquet")]
        ParquetScan { options, .. } => {
            if acc_count >= options.file_counter {
                options.file_counter = 1;
            } else {
                options.file_counter -= acc_count as FileCount
            }
        }
        #[cfg(feature = "ipc")]
        IpcScan { options, .. } => {
            if acc_count >= options.file_counter {
                options.file_counter = 1;
            } else {
                options.file_counter -= acc_count as FileCount
            }
        }
        #[cfg(feature = "csv-file")]
        CsvScan { options, .. } => {
            if acc_count >= options.file_counter {
                options.file_counter = 1;
            } else {
                options.file_counter -= acc_count as FileCount
            }
        }
        Cache { count, input, .. } => {
            // we use usize::MAX for an infinite cache.
            let new_count = if *count != usize::MAX {
                acc_count + *count as FileCount
            } else {
                acc_count
            };
            decrement_file_counters_by_cache_hits(*input, lp_arena, _expr_arena, new_count, scratch)
        }
        lp => {
            lp.copy_inputs(scratch);
            while let Some(input) = scratch.pop() {
                decrement_file_counters_by_cache_hits(
                    input,
                    lp_arena,
                    _expr_arena,
                    acc_count,
                    scratch,
                )
            }
        }
    }
}
