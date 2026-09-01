//! Choosing a join order.
//!
//! The search is greedy over left-deep plans. [`order`] is the only place the order
//! is decided, so a different search can replace it on its own.

use polars_utils::pl_str::PlSmallStr;

use super::cluster::Cluster;
use crate::plans::{join_cardinality, key_domain};

/// Greedy left-deep ordering of a cluster's leaves.
///
/// Returns leaf indices in join order: the first is the base relation, each later
/// leaf is joined onto the accumulated result.
///
/// The largest relation anchors the chain and the rest are folded in
/// most-selective-first. Only leaves connected to an already placed leaf are
/// considered, so this never adds a cross product that the plan did not have.
pub(super) fn order(cluster: &Cluster) -> Vec<usize> {
    let n = cluster.leaves.len();
    let mut is_placed = vec![false; n];
    let mut placed = Vec::with_capacity(n);

    // Ties keep the earlier leaf so the order is deterministic.
    let anchor = (0..n).fold(0, |best, i| {
        if cluster.leaves[i].stats.filtered > cluster.leaves[best].stats.filtered {
            i
        } else {
            best
        }
    });

    let mut card = cluster.leaves[anchor].stats.filtered;
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

    placed
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
            bridge.placed_key,
            &cluster.leaves[candidate].stats,
            bridge.candidate_key,
        );
        let name = bridge.candidate_key.output_name_inner().get();

        match per_key
            .iter_mut()
            .find(|(seen, _)| name.is_some() && *seen == name)
        {
            Some((_, smallest)) => *smallest = smallest.min(domain),
            None => per_key.push((name, domain)),
        }
    }

    (!per_key.is_empty()).then(|| per_key.iter().map(|(_, domain)| domain).product())
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
            },
            Edge {
                left_leaf: 0,
                right_leaf: 2,
                left_key: key.clone(),
                right_key: key.clone(),
            },
            Edge {
                left_leaf: 1,
                right_leaf: 2,
                left_key: key.clone(),
                right_key: key.clone(),
            },
        ];

        let cluster = Cluster {
            leaves,
            edges,
            output_schema: Schema::default().into(),
            options: dummy_options(),
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
