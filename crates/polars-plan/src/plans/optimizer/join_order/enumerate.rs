//! Choosing a join order.
//!
//! The search is greedy over left-deep plans. [`order`] is the only place the order
//! is decided, so a different search can replace it on its own.

use super::cluster::Cluster;
use super::stats::{join_cardinality, key_domain};

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
fn key_domain_product(cluster: &Cluster, is_placed: &[bool], candidate: usize) -> Option<f64> {
    let mut product = None;
    for bridge in cluster.bridging(is_placed, candidate) {
        let domain = key_domain(
            &cluster.leaves[bridge.placed_leaf].stats,
            &cluster.leaves[candidate].stats,
        );
        *product.get_or_insert(1.0) *= domain;
    }
    product
}
