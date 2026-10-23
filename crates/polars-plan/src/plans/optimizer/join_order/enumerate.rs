//! Choosing a join order.
//!
//! The search is greedy over left-deep plans. [`order`] is the only place the order
//! is decided, so a different search can replace it on its own.

use polars_utils::pl_str::PlSmallStr;

use super::cluster::{Bridge, Cluster};
use crate::plans::{
    NodeStats, composite_key_domain, join_cardinality, key_domain, unique_side_rows,
};

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
    let guessed = GuessedKeys::new(cluster);
    starts
        .map(|anchor| chain_from(cluster, &guessed, anchor))
        .min_by(|(a, _), (b, _)| a.total_cmp(b))
        .map_or_else(Vec::new, |(_, order)| order)
}

/// Leaves above which every-anchor search is skipped.
const MAX_STARTS: usize = 24;

/// Domain of a key that does not narrow the join.
const MIN_DOMAIN: f64 = 1.0;

/// One greedy left-deep chain starting at `anchor`, with the total of the
/// intermediate results it materialises.
///
/// Only leaves connected to an already placed leaf are considered, so this never
/// adds a cross product that the plan did not have. The base relation's own rows
/// are not counted: it is read whichever leaf it is, and charging for it would
/// always favour the smallest leaf as the base.
fn chain_from(cluster: &Cluster, guessed: &GuessedKeys<'_>, anchor: usize) -> (f64, Vec<usize>) {
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
                    key_domain_product(cluster, guessed, &is_placed, placed.len() == 1, candidate)?;
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

/// Whether the key joining `leaf` to `other` is only guessed to be unique on `leaf`.
///
/// With no distinct count or range on either side, [`key_domain`] takes the key to
/// be unique on the smaller side.
fn guessed_unique(
    leaf: &NodeStats,
    leaf_key: Option<&PlSmallStr>,
    other: &NodeStats,
    other_key: Option<&PlSmallStr>,
) -> bool {
    let has_stats = |stats: &NodeStats, key: Option<&PlSmallStr>| {
        key.is_some_and(|name| stats.key_distinct_estimate(name).is_some())
    };
    leaf.unfiltered <= other.unfiltered
        && !has_stats(leaf, leaf_key)
        && !has_stats(other, other_key)
}

/// The keys of `bridge` as seen from the candidate and from the placed leaf.
fn identities<'a>(bridge: &Bridge<'a>) -> (Option<DomainIdentity<'a>>, Option<DomainIdentity<'a>>) {
    let class = bridge.class.map(DomainIdentity::Class);
    (
        class.or_else(|| bridge.candidate_name.map(DomainIdentity::Column)),
        class.or_else(|| bridge.placed_name.map(DomainIdentity::Column)),
    )
}

/// A key pair seen from one leaf.
struct KeyPair<'a> {
    partner: usize,
    identity: Option<DomainIdentity<'a>>,
    /// Column names of the key on the leaf and on the partner, when plain columns.
    key: Option<&'a PlSmallStr>,
    partner_key: Option<&'a PlSmallStr>,
}

/// Every key pair joining `leaf` to another leaf.
///
/// Read from the classes and edges directly, so all leaves together cost the size
/// of the key graph rather than a [`Cluster::bridging`] scan per pair of leaves.
fn key_pairs(cluster: &Cluster, leaf: usize) -> impl Iterator<Item = KeyPair<'_>> {
    let classes = cluster.classes_by_leaf[leaf]
        .iter()
        .flat_map(move |&class| {
            let keys = &cluster.key_classes[class];
            keys.iter()
                .filter(move |own| own.leaf == leaf)
                .flat_map(move |own| {
                    keys.iter()
                        .filter(move |other| other.leaf != leaf)
                        .map(move |other| KeyPair {
                            partner: other.leaf,
                            identity: Some(DomainIdentity::Class(class)),
                            key: Some(own.key.output_name()),
                            partner_key: Some(other.key.output_name()),
                        })
                })
        });
    let edges = cluster.edges.iter().filter_map(move |edge| {
        let (key, partner, partner_key) = if edge.left_leaf == leaf {
            (
                edge.left_name.as_ref(),
                edge.right_leaf,
                edge.right_name.as_ref(),
            )
        } else if edge.right_leaf == leaf {
            (
                edge.right_name.as_ref(),
                edge.left_leaf,
                edge.left_name.as_ref(),
            )
        } else {
            return None;
        };
        (partner != leaf).then(|| KeyPair {
            partner,
            identity: key.map(DomainIdentity::Column),
            key,
            partner_key,
        })
    });
    classes.chain(edges)
}

/// Keys guessed unique on a leaf that are taken not to narrow a join.
///
/// A leaf can be the smaller side of several edges, on different keys, and is then
/// guessed unique on each of them. That is rarely true: a dimension is unique on the
/// key the fact refers to, not on its attributes. So each leaf keeps the guess on the
/// keys reaching its most fact-like partner: the one joined to the most smaller
/// leaves, then the largest.
///
/// A guess on another key is dropped only when its partner also joins the kept one.
/// That edge closes a triangle, and mostly restates what the kept partner already
/// ties together, like a region shared by a fact's unit and its price.
struct GuessedKeys<'a> {
    n: usize,
    /// `n * n` flags, set where two leaves share a key.
    adjacent: Vec<bool>,
    /// Per leaf, its most fact-like partner and the keys reaching it.
    kept: Vec<Option<(usize, Vec<DomainIdentity<'a>>)>>,
    /// Whether any guess in the cluster is dropped at all.
    any_dropped: bool,
}

impl<'a> GuessedKeys<'a> {
    fn new(cluster: &'a Cluster) -> Self {
        let n = cluster.leaves.len();
        let rows = |leaf: usize| cluster.leaves[leaf].stats.unfiltered;
        let mut adjacent = vec![false; n * n];
        for leaf in 0..n {
            for pair in key_pairs(cluster, leaf) {
                adjacent[leaf * n + pair.partner] = true;
            }
        }
        let smaller_neighbours: Vec<usize> = (0..n)
            .map(|leaf| {
                (0..n)
                    .filter(|&other| adjacent[leaf * n + other] && rows(other) <= rows(leaf))
                    .count()
            })
            .collect();
        let score = |leaf: usize| (smaller_neighbours[leaf], rows(leaf));

        let kept = (0..n)
            .map(|leaf| {
                let mut kept: Option<(usize, Vec<DomainIdentity<'a>>)> = None;
                for pair in key_pairs(cluster, leaf) {
                    let Some(identity) = pair.identity else {
                        continue;
                    };
                    if !guessed_unique(
                        &cluster.leaves[leaf].stats,
                        pair.key,
                        &cluster.leaves[pair.partner].stats,
                        pair.partner_key,
                    ) {
                        continue;
                    }
                    match &mut kept {
                        Some((best, keys)) if *best == pair.partner => {
                            if !keys.contains(&identity) {
                                keys.push(identity);
                            }
                        },
                        Some((best, _)) if score(pair.partner) <= score(*best) => {},
                        _ => kept = Some((pair.partner, vec![identity])),
                    }
                }
                kept
            })
            .collect();

        let mut guessed = Self {
            n,
            adjacent,
            kept,
            any_dropped: false,
        };
        guessed.any_dropped = (0..n).any(|leaf| {
            key_pairs(cluster, leaf).any(|pair| {
                guessed.dropped(leaf, pair.identity, pair.partner)
                    && guessed_unique(
                        &cluster.leaves[leaf].stats,
                        pair.key,
                        &cluster.leaves[pair.partner].stats,
                        pair.partner_key,
                    )
            })
        });
        guessed
    }

    /// Whether `bridge`, joining `candidate` to a placed leaf, rests on a dropped
    /// guess.
    fn drops(&self, cluster: &Cluster, bridge: &Bridge<'_>, candidate: usize) -> bool {
        if !self.any_dropped {
            return false;
        }
        let (candidate_identity, placed_identity) = identities(bridge);
        let candidate_stats = &cluster.leaves[candidate].stats;
        let placed_stats = &cluster.leaves[bridge.placed_leaf].stats;
        (self.dropped(candidate, candidate_identity, bridge.placed_leaf)
            && guessed_unique(
                candidate_stats,
                bridge.candidate_name,
                placed_stats,
                bridge.placed_name,
            ))
            || (self.dropped(bridge.placed_leaf, placed_identity, candidate)
                && guessed_unique(
                    placed_stats,
                    bridge.placed_name,
                    candidate_stats,
                    bridge.candidate_name,
                ))
    }

    /// Whether the guess that `identity` is unique on `leaf`, joined to `partner`, is
    /// dropped.
    fn dropped(&self, leaf: usize, identity: Option<DomainIdentity<'_>>, partner: usize) -> bool {
        let (Some(identity), Some((kept_partner, keys))) = (identity, &self.kept[leaf]) else {
            return false;
        };
        !keys.contains(&identity) && self.adjacent[partner * self.n + kept_partner]
    }
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
///
/// A key in [`GuessedKeys`] is taken not to narrow the join.
fn key_domain_product(
    cluster: &Cluster,
    guessed: &GuessedKeys<'_>,
    is_placed: &[bool],
    single_placed: bool,
    candidate: usize,
) -> Option<f64> {
    let mut per_key: Vec<DomainEstimate<'_>> = Vec::new();
    let candidate_stats = &cluster.leaves[candidate].stats;
    let mut max_rows = candidate_stats.unfiltered;
    let domain_of = |bridge: &Bridge<'_>| {
        if guessed.drops(cluster, bridge, candidate) {
            MIN_DOMAIN
        } else {
            key_domain(
                &cluster.leaves[bridge.placed_leaf].stats,
                bridge.placed_name,
                candidate_stats,
                bridge.candidate_name,
            )
        }
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
        let guessed = GuessedKeys::new(&cluster);
        let two_edges =
            key_domain_product(&cluster, &guessed, &[true, true, false], false, 2).unwrap();
        // Only leaf 0 placed: the same key reached once.
        let one_edge =
            key_domain_product(&cluster, &guessed, &[true, false, false], true, 2).unwrap();

        // The smallest of the estimates of `k`'s domain, counted a single time.
        assert_eq!(two_edges, 100.0);
        assert_eq!(one_edge, 100.0);
    }
}
