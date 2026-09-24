//! Finding the parts of a plan whose joins may be reordered.
//!
//! A *cluster* is a maximal contiguous run of inner equi-joins. Its *leaves* are the
//! subtrees hanging off that run, which are opaque here: reordering permutes the
//! leaves and never looks inside one.
//!
//! Anything not known to be safe to reorder across ends the cluster instead.

use std::sync::Arc;

use polars_core::prelude::{PlIndexMap, PlIndexSet};
use polars_core::schema::{Schema, SchemaRef};
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use recursive::recursive;

use crate::plans::aexpr::{ExprPushdownGroup, is_inherently_nondeterministic};
use crate::plans::optimizer::join_utils::plain_inner_equi_join;
use crate::plans::schema::join_right_output_names;
use crate::plans::{
    AExpr, ExprIR, IR, JoinOptionsIR, MintermIter, NodeStats, OutputName, ProjectionOptions,
    aexpr_to_leaf_names_iter, node_stats,
};
use crate::prelude::JoinArgs;
use crate::utils::rename_columns;

/// With two leaves there is only one order, so a cluster needs at least three.
const MIN_LEAVES: usize = 3;

/// Renames carrying names from somewhere inside a cluster up into the namespace the
/// cluster root uses. A name absent from the map is unchanged.
type Renames = PlIndexMap<PlSmallStr, PlSmallStr>;

pub(super) struct Leaf {
    pub(super) node: Node,
    pub(super) schema: SchemaRef,
    pub(super) stats: NodeStats,
}

/// One equi-key pair, resolved to the leaves it connects.
///
/// One join can contribute several, and they need not touch the same pair of leaves:
/// in `(a ⋈ b) ⋈ c ON a.x = c.x AND b.y = c.y` the outer join bridges `a`–`c` and
/// `b`–`c`.
pub(super) struct Edge {
    pub(super) left_leaf: usize,
    pub(super) right_leaf: usize,
    pub(super) left_key: ExprIR,
    pub(super) right_key: ExprIR,
    /// Column each key reads, when it is a plain column reference. A computed key
    /// has none: it is named after its left-most column, whose statistics and
    /// identity are not the expression's.
    pub(super) left_name: Option<PlSmallStr>,
    pub(super) right_name: Option<PlSmallStr>,
}

/// A plain column in a same-dtype equality class.
pub(super) struct ColumnKey {
    pub(super) leaf: usize,
    pub(super) key: ExprIR,
}

/// An edge oriented against the leaves joined so far.
pub(super) struct Bridge<'a> {
    pub(super) class: Option<usize>,
    /// The already-joined leaf this edge reaches back to.
    pub(super) placed_leaf: usize,
    /// Key belonging to the accumulated (left) side.
    pub(super) placed_key: &'a ExprIR,
    /// Key belonging to the candidate (right) side.
    pub(super) candidate_key: &'a ExprIR,
    /// Column names of those keys, when they are plain column references.
    pub(super) placed_name: Option<&'a PlSmallStr>,
    pub(super) candidate_name: Option<&'a PlSmallStr>,
}

pub(super) struct Cluster {
    pub(super) leaves: Vec<Leaf>,
    /// Keys that cannot participate in transitive equality, such as casts.
    pub(super) edges: Vec<Edge>,
    pub(super) key_classes: Vec<Vec<ColumnKey>>,
    pub(super) classes_by_leaf: Vec<Vec<usize>>,
    /// Schema of the cluster root before reordering. The rebuilt plan is projected
    /// back to it.
    pub(super) output_schema: SchemaRef,
    /// Reads of [`output_schema`](Self::output_schema)'s columns under the names the
    /// leaves carry, aliased back. Empty unless a leaf was renamed.
    pub(super) restore: Vec<ExprIR>,
    /// Options used for every rebuilt join. [`same_settings`] guarantees all joins
    /// in the cluster agree on everything but their keys.
    pub(super) options: Arc<JoinOptionsIR>,
    /// Conjuncts that sat between the cluster's joins, in the root namespace. The
    /// joins are all inner, so these commute with them and are re-applied as soon as
    /// the chain has the columns they read.
    pub(super) residuals: Vec<ExprIR>,
}

impl Cluster {
    /// Edges bridging `candidate` to anything already placed, oriented so the
    /// accumulated side is `placed_key` and `candidate` is `candidate_key`.
    ///
    /// An empty iterator means joining the candidate would be a cross product.
    /// Costing considers every implied equality; rebuilding emits a spanning tree
    /// from the same classes.
    pub(super) fn bridging<'a>(
        &'a self,
        is_placed: &'a [bool],
        candidate: usize,
    ) -> impl Iterator<Item = Bridge<'a>> + 'a {
        let classes = self.classes_by_leaf[candidate]
            .iter()
            .flat_map(move |&class| {
                let keys = &self.key_classes[class];
                keys.iter()
                    .filter(move |key| key.leaf == candidate)
                    .flat_map(move |candidate_key| {
                        keys.iter()
                            .filter(move |key| is_placed[key.leaf])
                            .map(move |placed| Bridge {
                                class: Some(class),
                                placed_leaf: placed.leaf,
                                placed_key: &placed.key,
                                candidate_key: &candidate_key.key,
                                placed_name: Some(placed.key.output_name()),
                                candidate_name: Some(candidate_key.key.output_name()),
                            })
                    })
            });
        classes.chain(self.direct_bridging(is_placed, candidate))
    }

    pub(super) fn direct_bridging<'a>(
        &'a self,
        is_placed: &'a [bool],
        candidate: usize,
    ) -> impl Iterator<Item = Bridge<'a>> + 'a {
        self.edges.iter().filter_map(move |edge| {
            if edge.right_leaf == candidate && is_placed[edge.left_leaf] {
                Some(Bridge {
                    class: None,
                    placed_leaf: edge.left_leaf,
                    placed_key: &edge.left_key,
                    candidate_key: &edge.right_key,
                    placed_name: edge.left_name.as_ref(),
                    candidate_name: edge.right_name.as_ref(),
                })
            } else if edge.left_leaf == candidate && is_placed[edge.right_leaf] {
                Some(Bridge {
                    class: None,
                    placed_leaf: edge.right_leaf,
                    placed_key: &edge.right_key,
                    candidate_key: &edge.left_key,
                    placed_name: edge.right_name.as_ref(),
                    candidate_name: edge.left_name.as_ref(),
                })
            } else {
                None
            }
        })
    }
}

/// Whether a join node may be reordered against its neighbours.
///
/// Coalescing joins pass here but are constrained further in [`coalesce_keys`]:
/// coalescing folds a key pair into one column under the left name, so only pairs of
/// identically named columns survive the inputs being swapped.
/// If this evaluates false we don't rewrite a cluster and leave it as is.
fn reorderable(options: &JoinOptionsIR) -> bool {
    plain_inner_equi_join(options)
}

/// A leaf as found, with the renames that carry its columns into the root namespace.
struct RawLeaf {
    node: Node,
    renames: Arc<Renames>,
}

/// One conjunct found between two joins, with the origin of every column it reads.
struct RawResidual {
    predicate: ExprIR,
    origins: PlIndexMap<PlSmallStr, Origin>,
}

/// A leaf and the name of one of its columns, as the leaf itself emits it.
type Origin = (usize, PlSmallStr);

/// A key pair as written, with the origin of every column either key reads.
///
/// A key expression is resolved against its own input, so the sides are looked up
/// separately; the same name can occur on both.
struct RawKey {
    left_key: ExprIR,
    right_key: ExprIR,
    left_origins: PlIndexMap<PlSmallStr, Origin>,
    right_origins: PlIndexMap<PlSmallStr, Origin>,
}

/// Extract the cluster rooted at `root`, or `None` if it cannot be reordered.
pub(super) fn extract(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<Cluster> {
    let IR::Join { options, .. } = ir_arena.get(root_join(root, ir_arena, expr_arena)?) else {
        return None;
    };
    if !reorderable(options) {
        return None;
    }
    let options = options.clone();

    let (raw_leaves, raw_keys, raw_residuals, root_origins) =
        Collector::run(root, ir_arena, expr_arena, &options)?;

    if raw_leaves.len() < MIN_LEAVES {
        return None;
    }

    let mut nodes = Vec::with_capacity(raw_leaves.len());
    let mut schemas = Vec::with_capacity(raw_leaves.len());
    let mut leaf_renames = Vec::with_capacity(raw_leaves.len());
    for raw in raw_leaves {
        let schema = ir_arena.get(raw.node).schema(ir_arena).into_owned();
        let (node, schema) = rename_leaf(raw.node, schema, &raw.renames, ir_arena, expr_arena)?;
        nodes.push(node);
        schemas.push(schema);
        leaf_renames.push(raw.renames);
    }

    let mut edges = Vec::with_capacity(raw_keys.len());
    for raw in raw_keys {
        let (left_key, left_leaf) =
            resolve_key(&raw.left_key, &raw.left_origins, &leaf_renames, expr_arena)?;
        let (right_key, right_leaf) = resolve_key(
            &raw.right_key,
            &raw.right_origins,
            &leaf_renames,
            expr_arena,
        )?;
        let left_name = left_key.plain_column(expr_arena).cloned();
        let right_name = right_key.plain_column(expr_arena).cloned();
        edges.push(Edge {
            left_leaf,
            right_leaf,
            left_key,
            right_key,
            left_name,
            right_name,
        });
    }

    let coalesced = if options.args.should_coalesce() {
        coalesce_keys(&schemas, &edges, expr_arena)?
    } else {
        PlIndexSet::default()
    };

    let output_schema = ir_arena.get(root).schema(ir_arena).into_owned();

    // A coalescing cluster folds its key columns away, which the restoring projection
    // does not model, so only a non-coalescing one can be renamed apart.
    let mut collisions = vec![Renames::default(); schemas.len()];
    if !column_names_are_unambiguous(&schemas, &coalesced) {
        if !coalesced.is_empty() {
            return None;
        }
        collisions = collision_renames(&schemas);
        let renames = &collisions;
        for (leaf, renames) in renames.iter().enumerate() {
            let (node, schema) = rename_leaf(
                nodes[leaf],
                schemas[leaf].clone(),
                renames,
                ir_arena,
                expr_arena,
            )?;
            nodes[leaf] = node;
            schemas[leaf] = schema;
        }
        for edge in &mut edges {
            edge.left_key = rename_expr(&edge.left_key, &renames[edge.left_leaf], expr_arena);
            edge.right_key = rename_expr(&edge.right_key, &renames[edge.right_leaf], expr_arena);
            edge.left_name = edge.left_key.plain_column(expr_arena).cloned();
            edge.right_name = edge.right_key.plain_column(expr_arena).cloned();
        }
        // A leaf column already named like a renamed one is still shared afterwards.
        if !column_names_are_unambiguous(&schemas, &coalesced) {
            return None;
        }
    }

    // The name a leaf column carries in the rebuilt joins.
    let rebuilt_name = |(leaf, name): &Origin| {
        let name = leaf_renames[*leaf].get(name).unwrap_or(name);
        collisions[*leaf].get(name).unwrap_or(name).clone()
    };

    // Reads of the cluster's output columns under the names the rebuilt joins use,
    // aliased back to the names the plan above expects. Only needed once leaves have
    // been renamed apart; otherwise the output is selected by name.
    let mut restore = Vec::new();
    if collisions.iter().any(|renames| !renames.is_empty()) {
        for (origin, output_name) in root_origins.iter().zip(output_schema.iter_names()) {
            let read = rebuilt_name(origin);
            let mut e = ExprIR::from_column_name(read.clone(), expr_arena);
            if read != *output_name {
                e.set_alias(output_name.clone());
            }
            restore.push(e);
        }
    }

    let key_classes = key_equivalence_classes(&mut edges, &schemas, expr_arena);
    let mut classes_by_leaf = vec![Vec::new(); schemas.len()];
    for (class, keys) in key_classes.iter().enumerate() {
        for key in keys {
            let classes = &mut classes_by_leaf[key.leaf];
            if classes.last() != Some(&class) {
                classes.push(class);
            }
        }
    }

    let residuals = raw_residuals
        .iter()
        .map(|raw| {
            let renames: Renames = raw
                .origins
                .iter()
                .map(|(name, origin)| (name.clone(), rebuilt_name(origin)))
                .collect();
            rename_expr(&raw.predicate, &renames, expr_arena)
        })
        .collect();

    // Every leaf needs an estimate. Ordering on partial information would order by
    // which leaves happened to be measurable. Taken after renaming so that the
    // per-column statistics are keyed on the names the rebuilt joins use.
    let mut leaves = Vec::with_capacity(nodes.len());
    for (node, schema) in nodes.into_iter().zip(schemas) {
        let stats = node_stats(node, ir_arena, expr_arena)?;
        leaves.push(Leaf {
            node,
            schema,
            stats,
        });
    }

    Some(Cluster {
        leaves,
        edges,
        key_classes,
        classes_by_leaf,
        output_schema,
        restore,
        options,
        residuals,
    })
}

/// The join whose settings the cluster inherits: `root` itself, or the join under a
/// filter on top of the run. The collector carries such a filter's conjuncts as
/// residuals, so they are re-applied as soon as their columns are joined in rather
/// than staying above the whole rebuilt run.
fn root_join(root: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<Node> {
    let node = match ir_arena.get(root) {
        IR::Join { .. } => root,
        IR::Filter { input, .. } => {
            peel_projections(*input, ir_arena, expr_arena, &Arc::new(Renames::default())).0
        },
        _ => return None,
    };
    matches!(ir_arena.get(node), IR::Join { .. }).then_some(node)
}

/// Whether a filter conjunct may be re-applied elsewhere in the join run: the same
/// test predicate-pushdown applies before moving a predicate past a join, plus
/// determinism, since a moved predicate is evaluated on a different set of rows.
fn may_travel(node: Node, expr_arena: &Arena<AExpr>) -> bool {
    let mut group = ExprPushdownGroup::Pushable;
    group.update_with_expr_rec(expr_arena.get(node), expr_arena, None);
    matches!(group, ExprPushdownGroup::Pushable)
        && !is_inherently_nondeterministic(node, expr_arena)
}

/// A name for a column held by more than one leaf, unique across the cluster.
fn disambiguated(leaf: usize, name: &PlSmallStr) -> PlSmallStr {
    format_pl_smallstr!("__POLARS_JOIN_ORDER_{leaf}_{name}")
}

/// Per leaf, the renames pulling its share of a name held by several leaves apart.
///
/// A leaf holding no shared name gets an empty map, which is a no-op to apply.
fn collision_renames(schemas: &[SchemaRef]) -> Vec<Renames> {
    let mut holders: PlIndexMap<&PlSmallStr, usize> = PlIndexMap::default();
    for schema in schemas {
        for name in schema.iter_names() {
            *holders.entry(name).or_default() += 1;
        }
    }

    schemas
        .iter()
        .enumerate()
        .map(|(leaf, schema)| {
            schema
                .iter_names()
                .filter(|name| holders[name] > 1)
                .map(|name| (name.clone(), disambiguated(leaf, name)))
                .collect()
        })
        .collect()
}

/// Walk the contiguous run of reorderable joins, collecting leaves and raw key pairs.
///
/// A join configured differently from the root becomes a leaf instead of being folded
/// in. Rebuilt joins inherit the root's settings, so folding in a join that disagreed
/// on, say, `nulls_equal` would change its meaning.
struct Collector<'a> {
    ir_arena: &'a Arena<IR>,
    expr_arena: &'a Arena<AExpr>,
    root_options: &'a JoinOptionsIR,
    leaves: Vec<RawLeaf>,
    key_pairs: Vec<RawKey>,
    residuals: Vec<RawResidual>,
}

impl<'a> Collector<'a> {
    /// Walk the cluster rooted at `root`, returning its leaves, keys and residuals,
    /// and the origin of each of the root's output columns.
    #[allow(clippy::type_complexity)]
    fn run(
        root: Node,
        ir_arena: &'a Arena<IR>,
        expr_arena: &'a Arena<AExpr>,
        root_options: &'a JoinOptionsIR,
    ) -> Option<(Vec<RawLeaf>, Vec<RawKey>, Vec<RawResidual>, Vec<Origin>)> {
        let mut collector = Self {
            ir_arena,
            expr_arena,
            root_options,
            leaves: Vec::new(),
            key_pairs: Vec::new(),
            residuals: Vec::new(),
        };
        let origins = collector.collect(root, &Arc::new(Renames::default()))?;
        Some((
            collector.leaves,
            collector.key_pairs,
            collector.residuals,
            origins,
        ))
    }

    /// Push `node` as a leaf; its columns are their own origin.
    fn push_leaf(&mut self, node: Node, renames: &Arc<Renames>) -> Vec<Origin> {
        let leaf = self.leaves.len();
        self.leaves.push(RawLeaf {
            node,
            renames: renames.clone(),
        });
        self.leaf_origins(leaf, node)
    }

    fn leaf_origins(&self, leaf: usize, node: Node) -> Vec<Origin> {
        self.ir_arena
            .get(node)
            .schema(self.ir_arena)
            .iter_names()
            .map(|name| (leaf, name.clone()))
            .collect()
    }

    /// Collect the run under `node`, returning the origin of each of its output
    /// columns, in schema order. `None` if a join's schema cannot be determined.
    #[recursive]
    fn collect(&mut self, node: Node, renames: &Arc<Renames>) -> Option<Vec<Origin>> {
        // Column projections commonly sit between joins. They preserve rows, so look
        // past them for the join underneath; otherwise almost every join is its own
        // cluster.
        let (peeled, peeled_renames) =
            peel_projections(node, self.ir_arena, self.expr_arena, renames);

        // A predicate over two of the relations cannot be pushed below their join, so
        // it sits between the joins. Peel it off and carry it, otherwise the cluster
        // ends here and everything below is one leaf, keys and all. Each conjunct
        // travels on its own so it can be re-applied as soon as its own columns are
        // available.
        //
        // Only a predicate that predicate-pushdown would move past a join may travel:
        // elementwise, infallible and deterministic. Anything else depends on which
        // rows reach it, and reordering the joins below changes which rows those are.
        let origins = if let IR::Filter { input, predicate } = self.ir_arena.get(peeled) {
            let expr_arena = self.expr_arena;
            let conjuncts = || MintermIter::new(predicate.node(), expr_arena);
            if !conjuncts().all(|node| may_travel(node, expr_arena)) {
                return Some(self.push_leaf(node, renames));
            }

            let leaf_start = self.leaves.len();
            let origins = self.collect(*input, &peeled_renames)?;

            if self.leaves.len() == leaf_start + 1 {
                // Everything below narrows one leaf, so fold the whole filter into
                // it: the predicate keeps the namespace it was written in, and the
                // leaf is measured through it when the order is picked.
                self.leaves[leaf_start] = RawLeaf {
                    node,
                    renames: renames.clone(),
                };
                return Some(self.leaf_origins(leaf_start, node));
            }

            let input_schema = self.ir_arena.get(*input).schema(self.ir_arena);
            for conjunct in conjuncts() {
                self.residuals.push(RawResidual {
                    predicate: ExprIR::from_node(conjunct, expr_arena),
                    origins: self.origins_read(conjunct, &input_schema, &origins)?,
                });
            }
            origins
        } else {
            match self.ir_arena.get(peeled) {
                IR::Join {
                    input_left,
                    input_right,
                    options,
                    ..
                } if reorderable(options) && same_settings(options, self.root_options) => {
                    let mut origins = self.collect(*input_left, &peeled_renames)?;
                    let right_origins = self.collect(*input_right, &peeled_renames)?;
                    let schema_left = self.ir_arena.get(*input_left).schema(self.ir_arena);
                    let schema_right = self.ir_arena.get(*input_right).schema(self.ir_arena);

                    if let Some(on) = options.options.key_pairs() {
                        for (left_key, right_key) in on {
                            self.key_pairs.push(RawKey {
                                left_key: left_key.clone(),
                                right_key: right_key.clone(),
                                left_origins: self.origins_read(
                                    left_key.node(),
                                    &schema_left,
                                    &origins,
                                )?,
                                right_origins: self.origins_read(
                                    right_key.node(),
                                    &schema_right,
                                    &right_origins,
                                )?,
                            });
                        }
                    }

                    // The join emits its left columns, then its right ones minus any
                    // coalesced away.
                    let right_names =
                        join_right_output_names(&schema_left, &schema_right, options).ok()?;
                    origins.extend(
                        right_origins
                            .into_iter()
                            .zip(right_names)
                            .filter_map(|(origin, name)| name.map(|_| origin)),
                    );
                    origins
                },
                // Keep the unpeeled node, and with it the renames as they stood above
                // it: a projection on a leaf still narrows it, and its own renames are
                // already part of its schema.
                _ => return Some(self.push_leaf(node, renames)),
            }
        };

        Some(self.project_origins(node, peeled, origins))
    }

    /// The origin of every column `expr` reads, resolved against the schema it is
    /// evaluated on and the origins of that schema's columns.
    fn origins_read(
        &self,
        expr: Node,
        schema: &Schema,
        origins: &[Origin],
    ) -> Option<PlIndexMap<PlSmallStr, Origin>> {
        aexpr_to_leaf_names_iter(expr, self.expr_arena)
            .map(|name| Some((name.clone(), origins[schema.index_of(name)?].clone())))
            .collect()
    }

    /// Carry `origins`, which describe `peeled`'s columns, up through the projections
    /// between `peeled` and `node`, so that they describe `node`'s columns instead.
    fn project_origins(&self, node: Node, peeled: Node, mut origins: Vec<Origin>) -> Vec<Origin> {
        let mut chain = Vec::new();
        let mut current = node;
        while current != peeled {
            chain.push(current);
            current = match self.ir_arena.get(current) {
                IR::SimpleProjection { input, .. } | IR::Select { input, .. } => *input,
                _ => unreachable!("only projections are peeled"),
            };
        }

        for projection in chain.into_iter().rev() {
            let (input, reads): (Node, Vec<&PlSmallStr>) = match self.ir_arena.get(projection) {
                IR::SimpleProjection { input, columns } => (*input, columns.iter_names().collect()),
                // `peel_projections` only looks through a select of plain columns.
                IR::Select { input, expr, .. } => (
                    *input,
                    expr.iter()
                        .map(|e| match self.expr_arena.get(e.node()) {
                            AExpr::Column(name) => name,
                            _ => unreachable!("only column reads are peeled"),
                        })
                        .collect(),
                ),
                _ => unreachable!("only projections are peeled"),
            };
            let input_schema = self.ir_arena.get(input).schema(self.ir_arena);
            origins = reads
                .into_iter()
                .map(|name| origins[input_schema.index_of(name).unwrap()].clone())
                .collect();
        }
        origins
    }
}

/// Strip any chain of column projections to reach the node beneath, composing the
/// renames they apply along the way.
///
/// Dropping an interior projection widens the rows flowing through the rebuilt joins.
/// Projection pushdown runs after this pass and narrows them again against the new
/// order, and the cluster is projected back to its original schema, so the extra
/// columns are not observable.
///
/// That argument only holds for a projection which reads existing columns and nothing
/// more: one that computes a column cannot be dropped, because the restoring
/// projection can only pick columns out of what the joins produce. A projection that
/// renames can, as long as the rename is carried down to the leaf that holds the
/// column, which is what the returned map records.
fn peel_projections(
    mut node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    renames: &Arc<Renames>,
) -> (Node, Arc<Renames>) {
    let mut renames = renames.clone();
    // Only simple projections and select renames are peeled.
    // If we re-order, we will not keep extra columns around as
    // projection pushdown runs after this.
    loop {
        node = match ir_arena.get(node) {
            IR::SimpleProjection { input, .. } => *input,
            // A `select` of plain columns is the same thing before `fast_projection`
            // (which runs after this pass) rewrites it into one. A `rename` reaches
            // the IR as such a `select` too, with the new name as the output name.
            IR::Select { input, expr, .. } => {
                let Some(composed) = compose_renames(expr, &renames, expr_arena) else {
                    // We bail if we compute new values here.
                    return (node, renames);
                };
                renames = Arc::new(composed);
                *input
            },
            _ => return (node, renames),
        };
    }
}

/// `renames` pulled through a projection, so that it maps the names of the
/// projection's *input* to the root namespace.
///
/// `None` if the projection does something no rename of the leaves can reproduce:
/// computing a column, or reading one column out under two names.
/// In that case we would bail.
fn compose_renames(
    expr: &[ExprIR],
    renames: &Renames,
    expr_arena: &Arena<AExpr>,
) -> Option<Renames> {
    let read_name = |e: &ExprIR| match expr_arena.get(e.node()) {
        AExpr::Column(read) => Some(read),
        _ => None,
    };

    // A projection that only narrows renames nothing, which is the common shape.
    if renames.is_empty() && expr.iter().all(|e| read_name(e) == Some(e.output_name())) {
        return Some(Renames::default());
    }

    let mut out = Renames::with_capacity_and_hasher(expr.len(), Default::default());
    for e in expr {
        let read = read_name(e)?;
        let output_name = e.output_name();
        let target = renames.get(output_name).unwrap_or(output_name);
        // Reading one column out under two names is not a rename, and pushing it down
        // would leave the leaf holding only one of them.
        if out.insert(read.clone(), target.clone()).is_some() {
            return None;
        }
    }
    out.retain(|read, target| read != target);
    Some(out)
}

/// A leaf rewritten so that its columns carry the names the cluster root uses.
///
/// Reordering rebuilds the joins over the leaves directly, so a rename that sat
/// between two of those joins has to travel down to the leaf holding the column.
/// `None` if it cannot, which leaves the cluster alone.
fn rename_leaf(
    node: Node,
    schema: SchemaRef,
    renames: &Renames,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Option<(Node, SchemaRef)> {
    if !schema.iter_names().any(|name| renames.contains_key(name)) {
        return Some((node, schema));
    }

    let mut expr = Vec::with_capacity(schema.len());
    let mut renamed = Schema::with_capacity(schema.len());
    for (name, dtype) in schema.iter() {
        let new = renames.get(name).unwrap_or(name);
        let mut e = ExprIR::from_column_name(name.clone(), expr_arena);
        if new != name {
            e.set_alias(new.clone());
        }
        expr.push(e);
        // The projection this rename came from may have dropped a column of the new
        // name; the leaf still holds it, and two columns cannot share a name.
        renamed.try_insert(new.clone(), dtype.clone()).ok()?;
    }

    let schema = Arc::new(renamed);
    let node = ir_arena.add(IR::Select {
        input: node,
        expr,
        schema: schema.clone(),
        options: ProjectionOptions {
            run_parallel: false,
            duplicate_check: false,
            should_broadcast: false,
            maintain_dataframe_height: false,
        },
    });
    Some((node, schema))
}

/// A key rewritten into the names its leaf carries after [`rename_leaf`], and that
/// leaf; `None` if the key does not read from exactly one leaf.
fn resolve_key(
    key: &ExprIR,
    origins: &PlIndexMap<PlSmallStr, Origin>,
    leaf_renames: &[Arc<Renames>],
    expr_arena: &mut Arena<AExpr>,
) -> Option<(ExprIR, usize)> {
    let mut owner = None;
    let mut renames = Renames::default();
    for (name, (leaf, column)) in origins {
        // A key spanning two leaves cannot be attributed to one side.
        if owner
            .replace(*leaf)
            .is_some_and(|existing| existing != *leaf)
        {
            return None;
        }
        let read = leaf_renames[*leaf].get(column).unwrap_or(column);
        if read != name {
            renames.insert(name.clone(), read.clone());
        }
    }
    Some((rename_expr(key, &renames, expr_arena), owner?))
}

/// A key or residual rewritten into the names the rebuilt joins use.
fn rename_expr(expr: &ExprIR, renames: &Renames, expr_arena: &mut Arena<AExpr>) -> ExprIR {
    // `rename_columns` re-interns the whole expression, so only pay for it when this
    // expression is one of the things being renamed.
    let touches = |name: &PlSmallStr| renames.contains_key(name.as_str());
    if !aexpr_to_leaf_names_iter(expr.node(), expr_arena).any(&touches)
        && !expr.output_name_inner().get().is_some_and(touches)
    {
        return expr.clone();
    }
    let node = rename_columns(expr.node(), expr_arena, renames);
    let renamed = |name: &PlSmallStr| renames.get(name).unwrap_or(name).clone();
    let output_name = match expr.output_name_inner() {
        OutputName::ColumnLhs(name) => OutputName::ColumnLhs(renamed(name)),
        OutputName::Alias(name) => OutputName::Alias(renamed(name)),
        other => other.clone(),
    };
    ExprIR::new(node, output_name)
}

/// Whether two joins agree on everything that survives being rebuilt.
///
/// The suffix is excluded because the SQL frontend names it after the right-hand
/// table, so no two joins agree on it. This is only sound while
/// [`column_names_are_unambiguous`] holds, since a suffix only applies on a collision.
///
/// Destructured so that a new `JoinArgs` field is a compile error here.
fn same_settings(a: &JoinOptionsIR, b: &JoinOptionsIR) -> bool {
    let JoinArgs {
        how,
        validation,
        suffix: _,
        slice,
        nulls_equal,
        coalesce,
        maintain_order,
        build_side,
    } = &a.args;

    *how == b.args.how
        && *validation == b.args.validation
        && *slice == b.args.slice
        && *nulls_equal == b.args.nulls_equal
        && *coalesce == b.args.coalesce
        && *maintain_order == b.args.maintain_order
        && *build_side == b.args.build_side
        && a.allow_parallel == b.allow_parallel
        && a.force_parallel == b.force_parallel
}

/// Whether every column name in the cluster identifies exactly one output column.
///
/// A name held by two leaves is suffixed on collision, and which side gets the suffix
/// depends on which ends up left, so reordering could rename columns. Coalesced key
/// names are the exception: they are folded into one column rather than suffixed.
///
/// [`collision_renames`] pulls the holders apart so that this holds.
fn column_names_are_unambiguous(schemas: &[SchemaRef], coalesced: &PlIndexSet<PlSmallStr>) -> bool {
    let total: usize = schemas.iter().map(|s| s.len()).sum();
    let mut seen = PlIndexSet::with_capacity_and_hasher(total, Default::default());
    schemas
        .iter()
        .flat_map(|s| s.iter_names())
        .all(|name| coalesced.contains(name.as_str()) || seen.insert(name.as_str()))
}

/// The names a coalescing cluster folds away, or `None` if it cannot be reordered.
///
/// Coalescing keeps the left key's column and drops the right one, so a pair naming
/// different columns would rename the output when the inputs swap. Only pairs of
/// identically named plain columns are accepted.
///
/// Every holder of a name must belong to the same equality class; otherwise
/// reordering could fold away a column that the original joins kept.
fn coalesce_keys(
    schemas: &[SchemaRef],
    edges: &[Edge],
    expr_arena: &Arena<AExpr>,
) -> Option<PlIndexSet<PlSmallStr>> {
    let mut by_name: PlIndexMap<PlSmallStr, Vec<&Edge>> = PlIndexMap::default();
    for edge in edges {
        let name = edge.left_key.plain_column(expr_arena)?;
        if edge.right_key.plain_column(expr_arena) != Some(name) {
            return None;
        }
        by_name.entry(name.clone()).or_default().push(edge);
    }

    for (name, on_name) in &by_name {
        let holders: Vec<usize> = (0..schemas.len())
            .filter(|&i| schemas[i].contains(name.as_str()))
            .collect();
        if !folds_into_one_column(name, &holders, on_name, schemas) {
            return None;
        }
    }

    Some(by_name.into_keys().collect())
}

/// Keep implied equalities as classes, rather than materializing a quadratic clique.
/// Casted and computed keys remain direct edges: a lossy cast cannot prove equality
/// between its input columns.
fn key_equivalence_classes(
    edges: &mut Vec<Edge>,
    schemas: &[SchemaRef],
    expr_arena: &mut Arena<AExpr>,
) -> Vec<Vec<ColumnKey>> {
    let mut columns: PlIndexMap<Origin, usize> = PlIndexMap::default();
    let mut parent = Vec::new();
    edges.retain(|edge| {
        let (Some(left_name), Some(right_name)) = (
            edge.left_key.plain_column(expr_arena),
            edge.right_key.plain_column(expr_arena),
        ) else {
            return true;
        };
        if schemas[edge.left_leaf].get(left_name) != schemas[edge.right_leaf].get(right_name) {
            return true;
        }
        let mut indices = [0; 2];
        for (slot, origin) in indices.iter_mut().zip([
            (edge.left_leaf, left_name.clone()),
            (edge.right_leaf, right_name.clone()),
        ]) {
            let next = columns.len();
            *slot = *columns.entry(origin).or_insert_with(|| {
                parent.push(next);
                next
            });
        }
        let left_root = find(&mut parent, indices[0]);
        let right_root = find(&mut parent, indices[1]);
        parent[left_root] = right_root;
        false
    });
    let mut groups: PlIndexMap<usize, Vec<ColumnKey>> = PlIndexMap::default();
    for ((leaf, name), index) in columns {
        groups
            .entry(find(&mut parent, index))
            .or_default()
            .push(ColumnKey {
                leaf,
                key: ExprIR::from_column_name(name, expr_arena),
            });
    }
    groups.into_values().collect()
}

/// Whether every leaf holding `name` collapses into a single column of that name.
///
/// The dtypes have to agree because coalescing keeps the left column, so the output
/// dtype would otherwise depend on the order. Reachability over the edges on `name`
/// is what makes transitive equality sound: joining two leaves that were not already connected
/// by this name would drop rows the original query kept.
fn folds_into_one_column(
    name: &PlSmallStr,
    holders: &[usize],
    edges: &[&Edge],
    schemas: &[SchemaRef],
) -> bool {
    let mut parent: Vec<usize> = (0..schemas.len()).collect();
    for edge in edges {
        let (left, right) = (
            find(&mut parent, edge.left_leaf),
            find(&mut parent, edge.right_leaf),
        );
        parent[left] = right;
    }

    let Some((&first, rest)) = holders.split_first() else {
        return false;
    };
    let root = find(&mut parent, first);
    let dtype = schemas[first].get(name.as_str());

    rest.iter()
        .all(|&i| find(&mut parent, i) == root && schemas[i].get(name.as_str()) == dtype)
}

fn find(parent: &mut [usize], mut i: usize) -> usize {
    while parent[i] != i {
        parent[i] = parent[parent[i]];
        i = parent[i];
    }
    i
}
