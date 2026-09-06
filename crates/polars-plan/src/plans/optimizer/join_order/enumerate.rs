//! Choosing a join order.
//!
//! The search is greedy over left-deep plans. [`order`] is the only place the order
//! is decided, so a different search can replace it on its own.

use polars_utils::pl_str::PlSmallStr;

use super::cluster::Cluster;
use crate::plans::{composite_key_domain, join_cardinality, key_domain};

/// Greedy left-deep ordering of a cluster's leaves.
///
/// Returns leaf indices in join order: the first is the base relation, each later
/// leaf is joined onto the accumulated result.
///
/// Every leaf is tried as the base and the chain with the smallest total
/// intermediate size wins. A cluster wider than [`MAX_STARTS`] only tries the
/// largest leaf, as each start is a full greedy pass.
pub(super) fn order(cluster: &Cluster) -> Vec<usize> {
    let n = cluster.leaves.len();
    let largest = || {
        (0..n).fold(0, |best, i| {
            if cluster.leaves[i].stats.filtered > cluster.leaves[best].stats.filtered {
                i
            } else {
                best
            }
        })
    };

    // Ties keep the earlier start so the order is deterministic.
    let starts: Box<dyn Iterator<Item = usize>> = if n > MAX_STARTS {
        Box::new(std::iter::once(largest()))
    } else {
        Box::new(0..n)
    };
    starts
        .map(|anchor| chain_from(cluster, anchor))
        .min_by(|(a, _), (b, _)| a.total_cmp(b))
        .map_or_else(Vec::new, |(_, order)| order)
}

/// Leaves above which every-anchor search is skipped.
const MAX_STARTS: usize = 24;

/// One greedy left-deep chain starting at `anchor`, with the total of the
/// intermediate results it materialises.
///
/// Only leaves connected to an already placed leaf are considered, so this never
/// adds a cross product that the plan did not have. The base relation's own rows
/// are not counted: it is read whichever leaf it is, and charging for it would
/// always favour the smallest leaf as the base.
fn chain_from(cluster: &Cluster, anchor: usize) -> (f64, Vec<usize>) {
    let n = cluster.leaves.len();
    let mut is_placed = vec![false; n];
    let mut placed = Vec::with_capacity(n);

    let mut card = cluster.leaves[anchor].stats.filtered;
    let mut cost = 0.0;
    placed.push(anchor);
    is_placed[anchor] = true;

    while placed.len() < n {
        let best = (0..n)
            .filter(|&i| !is_placed[i])
            .filter_map(|candidate| {
                let denominator = key_domain_product(cluster, &is_placed, candidate)?;
                let out =
                    join_cardinality(card, cluster.leaves[candidate].stats.filtered, denominator);
                Some((out, candidate))
            })
            .min_by(|(a, _), (b, _)| a.total_cmp(b));

        match best {
            Some((out, candidate)) => {
                card = out;
                cost += out;
                placed.push(candidate);
                is_placed[candidate] = true;
            },
            // The rest are disconnected. Emit them in source order; they become the
            // same cross joins the plan already had.
            None => {
                placed.extend((0..n).filter(|&i| !is_placed[i]));
                break;
            },
        }
    }

    (cost, placed)
}

/// Product of the key domains bridging `candidate` to the placed leaves, or `None`
/// if it is not connected to them at all.
///
/// Several edges can carry the same key: the implied edges of a coalesced name reach
/// every leaf holding it. They are all estimates of that one key's domain, so they
/// contribute one factor between them rather than one each, and the smallest wins.
/// Only keys that are actually independent multiply.
///
/// A key with no output name cannot be matched up, so it counts on its own.
fn key_domain_product(cluster: &Cluster, is_placed: &[bool], candidate: usize) -> Option<f64> {
    let mut per_key: Vec<(Option<&PlSmallStr>, f64)> = Vec::new();

    for bridge in cluster.bridging(is_placed, candidate) {
        let domain = key_domain(
            &cluster.leaves[bridge.placed_leaf].stats,
            bridge.placed_name,
            &cluster.leaves[candidate].stats,
            bridge.candidate_name,
        );
        let name = bridge.candidate_name;

        match per_key
            .iter_mut()
            .find(|(seen, _)| name.is_some() && *seen == name)
        {
            Some((_, smallest)) => *smallest = smallest.min(domain),
            None => per_key.push((name, domain)),
        }
    }

    // The largest relation the composite key is read from, which bounds the product.
    let max_rows = cluster
        .bridging(is_placed, candidate)
        .map(|bridge| cluster.leaves[bridge.placed_leaf].stats.unfiltered)
        .fold(cluster.leaves[candidate].stats.unfiltered, f64::max);

    (!per_key.is_empty())
        .then(|| composite_key_domain(per_key.iter().map(|(_, domain)| *domain), max_rows))
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::{DataType, Schema};
    use polars_utils::arena::{Arena, Node};

    use super::super::cluster::{Edge, Leaf};
    use super::*;
    use crate::plans::{ExprIR, JoinOptionsIR, JoinTypeOptionsIR, NodeStats};
    use crate::prelude::JoinArgs;

    fn dummy_options() -> std::sync::Arc<JoinOptionsIR> {
        std::sync::Arc::new(JoinOptionsIR {
            allow_parallel: true,
            force_parallel: false,
            args: JoinArgs::default(),
            options: JoinTypeOptionsIR::Equi { on: Vec::new() },
        })
    }

    fn leaf(node: usize, rows: f64) -> Leaf {
        let mut schema = Schema::default();
        schema.insert("k".into(), DataType::Int64);
        Leaf {
            node: Node(node),
            schema: schema.into(),
            stats: NodeStats::of_rows(rows),
        }
    }

    /// The implied edges of a coalesced name reach every leaf holding it, so a
    /// candidate can bridge the same key to several placed leaves. Multiplying one
    /// factor per edge would divide by that key's domain more than once.
    #[test]
    fn one_key_reached_twice_divides_once() {
        let mut expr_arena = Arena::new();
        let key = ExprIR::from_column_name("k".into(), &mut expr_arena);

        let leaves = vec![leaf(0, 100.0), leaf(1, 200.0), leaf(2, 400.0)];
        // The clique over the three holders of `k`.
        let edges = vec![
            Edge {
                left_leaf: 0,
                right_leaf: 1,
                left_key: key.clone(),
                right_key: key.clone(),
                left_name: Some("k".into()),
                right_name: Some("k".into()),
            },
            Edge {
                left_leaf: 0,
                right_leaf: 2,
                left_key: key.clone(),
                right_key: key.clone(),
                left_name: Some("k".into()),
                right_name: Some("k".into()),
            },
            Edge {
                left_leaf: 1,
                right_leaf: 2,
                left_key: key.clone(),
                right_key: key.clone(),
                left_name: Some("k".into()),
                right_name: Some("k".into()),
            },
        ];

        let cluster = Cluster {
            leaves,
            edges,
            output_schema: Schema::default().into(),
            restore: Vec::new(),
            options: dummy_options(),
            residuals: Vec::new(),
        };

        // Leaves 0 and 1 placed, 2 the candidate: it bridges on `k` to both.
        let two_edges = key_domain_product(&cluster, &[true, true, false], 2).unwrap();
        // Only leaf 0 placed: the same key reached once.
        let one_edge = key_domain_product(&cluster, &[true, false, false], 2).unwrap();

        // The smallest of the estimates of `k`'s domain, counted a single time.
        assert_eq!(two_edges, 100.0);
        assert_eq!(one_edge, 100.0);
    }
}
