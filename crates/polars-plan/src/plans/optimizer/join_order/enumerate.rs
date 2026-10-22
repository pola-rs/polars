//! Choosing a join order.
//!
//! The search is greedy over left-deep plans. [`order`] is the only place the order
//! is decided, so a different search can replace it on its own.

use polars_utils::pl_str::PlSmallStr;

use super::cluster::{Bridge, Cluster};
use crate::plans::{composite_key_domain, join_cardinality, key_domain, unique_side_rows};

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
                let denominator =
                    key_domain_product(cluster, &is_placed, placed.len() == 1, candidate)?;
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

#[derive(Clone, Copy, PartialEq, Eq)]
enum DomainIdentity<'a> {
    Class(usize),
    Column(&'a PlSmallStr),
}

/// The key of `bridge`, as seen from the candidate.
fn identity_of<'a>(bridge: &Bridge<'a>) -> Option<DomainIdentity<'a>> {
    bridge
        .class
        .map(DomainIdentity::Class)
        .or_else(|| bridge.candidate_name.map(DomainIdentity::Column))
}

struct DomainEstimate<'a> {
    identity: Option<DomainIdentity<'a>>,
    domain: f64,
    /// The rows of a leaf the key is unique on, see [`unique_side_rows`].
    unique_rows: Option<f64>,
}

/// Product of the key domains bridging `candidate` to the placed leaves, or `None`
/// if it is not connected to them at all.
///
/// Keys in one equality class estimate the same domain, so they contribute one
/// factor between them, with the smallest estimate winning. Direct edges on the
/// same candidate column also contribute only one factor.
/// Only keys that are actually independent multiply.
///
/// A key with no output name cannot be matched up, so it counts on its own.
///
/// A key unique on one of several placed leaves can repeat in their join, so a
/// placed leaf only counts as unique when `single_placed`.
fn key_domain_product(
    cluster: &Cluster,
    is_placed: &[bool],
    single_placed: bool,
    candidate: usize,
) -> Option<f64> {
    let mut per_key: Vec<DomainEstimate<'_>> = Vec::new();
    let candidate_stats = &cluster.leaves[candidate].stats;
    let mut max_rows = candidate_stats.unfiltered;
    let domain_of = |bridge: &Bridge<'_>| {
        key_domain(
            &cluster.leaves[bridge.placed_leaf].stats,
            bridge.placed_name,
            candidate_stats,
            bridge.candidate_name,
        )
    };

    for bridge in cluster.bridging(is_placed, candidate) {
        let domain = domain_of(&bridge);
        let identity = identity_of(&bridge);
        max_rows = max_rows.max(cluster.leaves[bridge.placed_leaf].stats.unfiltered);

        match per_key
            .iter_mut()
            .find(|estimate| identity.is_some() && estimate.identity == identity)
        {
            Some(estimate) => estimate.domain = estimate.domain.min(domain),
            None => per_key.push(DomainEstimate {
                identity,
                domain,
                unique_rows: None,
            }),
        }
    }

    // Uniqueness only bounds a key of several parts, so it is only looked up then,
    // from every bridge that gave a part its domain.
    if per_key.len() > 1 {
        let unnamed: Vec<usize> = (0..per_key.len())
            .filter(|&i| per_key[i].identity.is_none())
            .collect();
        let mut unnamed = unnamed.into_iter();
        for bridge in cluster.bridging(is_placed, candidate) {
            let identity = identity_of(&bridge);
            let index = match identity {
                Some(_) => per_key
                    .iter()
                    .position(|estimate| estimate.identity == identity),
                None => unnamed.next(),
            };
            let Some(estimate) = index.map(|i| &mut per_key[i]) else {
                continue;
            };
            let domain = domain_of(&bridge);
            if domain != estimate.domain {
                continue;
            }
            let placed_stats = &cluster.leaves[bridge.placed_leaf].stats;
            let unique_rows =
                unique_side_rows(candidate_stats, bridge.candidate_name, placed_stats, domain)
                    .or_else(|| {
                        unique_side_rows(placed_stats, bridge.placed_name, candidate_stats, domain)
                            .filter(|_| single_placed)
                    });
            estimate.unique_rows = match (estimate.unique_rows, unique_rows) {
                (Some(a), Some(b)) => Some(a.max(b)),
                (a, b) => a.or(b),
            };
        }
    }

    (!per_key.is_empty()).then(|| {
        composite_key_domain(
            per_key
                .iter()
                .map(|estimate| (estimate.domain, estimate.unique_rows)),
            max_rows,
        )
    })
}

#[cfg(test)]
mod tests {
    use polars_core::prelude::{DataType, Schema};
    use polars_utils::arena::{Arena, Node};

    use super::super::cluster::{ColumnKey, Leaf};
    use super::*;
    use crate::plans::{ExprIR, JoinOptionsIR, JoinTypeOptionsIR, NodeStats};
    use crate::prelude::JoinArgs;

    fn dummy_options() -> std::sync::Arc<JoinOptionsIR> {
        std::sync::Arc::new(JoinOptionsIR {
            allow_parallel: true,
            force_parallel: false,
            args: JoinArgs::default(),
            options: JoinTypeOptionsIR::Equi {
                on: Vec::new(),
                fused_predicate: None,
            },
            runtime_filters: Vec::new(),
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
        let key_classes = vec![
            (0..3)
                .map(|leaf| ColumnKey {
                    leaf,
                    key: key.clone(),
                })
                .collect(),
        ];

        let cluster = Cluster {
            leaves,
            edges: Vec::new(),
            key_classes,
            classes_by_leaf: vec![vec![0]; 3],
            output_schema: Schema::default().into(),
            restore: Vec::new(),
            options: dummy_options(),
            residuals: Vec::new(),
        };

        // Leaves 0 and 1 placed, 2 the candidate: it bridges on `k` to both.
        let two_edges = key_domain_product(&cluster, &[true, true, false], false, 2).unwrap();
        // Only leaf 0 placed: the same key reached once.
        let one_edge = key_domain_product(&cluster, &[true, false, false], true, 2).unwrap();

        // The smallest of the estimates of `k`'s domain, counted a single time.
        assert_eq!(two_edges, 100.0);
        assert_eq!(one_edge, 100.0);
    }
}
