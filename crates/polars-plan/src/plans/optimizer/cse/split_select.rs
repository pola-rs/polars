//! Splitting a select into a pre-select and an elementwise post-select.
//!
//! Consumers that materialize (or spill) the result of a select benefit from that
//! intermediate being as narrow as possible, while still being allowed to finish the
//! computation afterwards. [`split_pre_post_select_minsize_elementwise`] finds the
//! cheapest such split.

use polars_core::prelude::{DataType, InitHashMaps, PlIndexMap, PlIndexSet};
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_column_name;
use recursive::recursive;

use crate::plans::{
    AExpr, CanonicalExprId, CanonicalExprMap, ExprIR, OutputName, ToFieldContext, is_elementwise,
};

/// Assumed payload of one variable-length value, in bits.
const ESTIMATED_VARLEN_PAYLOAD_BITS: u64 = 16 * 8;
/// Assumed number of elements in one list value.
const ESTIMATED_LIST_LEN: u64 = 4;
/// Weight of a value whose size we cannot estimate, and the cap on any estimate. Large
/// enough that we never prefer materializing one over anything we can size, small enough
/// not to overflow when every node in the expression carries it.
const MAX_ESTIMATED_BITS: u64 = 1 << 32;
/// Capacity of a hard constraint; never part of a finite cut.
const INF: u64 = u64::MAX / 4;

/// Expected number of bits stored per row by a column of `dtype`.
fn expected_row_bits(dtype: &DataType) -> u64 {
    // Every column carries a validity bitmap. Counting it also breaks ties towards fewer
    // intermediate columns.
    const VALIDITY: u64 = 1;

    use DataType as D;
    let payload = match dtype.byte_width() {
        Some(bytes) => (bytes * 8.0) as u64,
        None => match dtype {
            // A 16 byte view, plus the heap it points into.
            D::String | D::Binary => 128 + ESTIMATED_VARLEN_PAYLOAD_BITS,
            D::BinaryOffset => 64 + ESTIMATED_VARLEN_PAYLOAD_BITS,
            D::List(inner) => 64 + ESTIMATED_LIST_LEN * expected_row_bits(inner),
            #[cfg(feature = "dtype-array")]
            D::Array(inner, width) => (*width as u64).saturating_mul(expected_row_bits(inner)),
            #[cfg(feature = "dtype-struct")]
            D::Struct(fields) => fields
                .iter()
                .map(|f| expected_row_bits(f.dtype()))
                .fold(0, u64::saturating_add),
            #[cfg(feature = "dtype-map")]
            D::Map(key, value) => {
                64 + ESTIMATED_LIST_LEN * (expected_row_bits(key) + expected_row_bits(value))
            },
            #[cfg(feature = "dtype-extension")]
            D::Extension(_, storage) => expected_row_bits(storage),
            _ => MAX_ESTIMATED_BITS,
        },
    };
    VALIDITY + payload.min(MAX_ESTIMATED_BITS)
}

/// Dinic max-flow, used only to read off a minimum cut.
struct FlowNetwork {
    /// Edge `e` and edge `e ^ 1` are each other's reverse.
    to: Vec<u32>,
    cap: Vec<u64>,
    adj: Vec<Vec<u32>>,
}

impl FlowNetwork {
    fn new(num_nodes: usize) -> Self {
        Self {
            to: Vec::new(),
            cap: Vec::new(),
            adj: vec![Vec::new(); num_nodes],
        }
    }

    fn add_edge(&mut self, from: usize, to: usize, cap: u64) {
        let e = self.to.len() as u32;
        self.to.push(to as u32);
        self.cap.push(cap);
        self.to.push(from as u32);
        self.cap.push(0);
        self.adj[from].push(e);
        self.adj[to].push(e + 1);
    }

    /// Returns, per node, whether it is on the source side of a minimum `src`-`dst` cut.
    fn min_cut_source_side(&mut self, src: usize, dst: usize) -> Vec<bool> {
        let mut level = vec![u32::MAX; self.adj.len()];
        let mut next_edge = vec![0usize; self.adj.len()];
        let mut queue = Vec::new();

        loop {
            // Build the level graph over the residual edges.
            level.fill(u32::MAX);
            level[src] = 0;
            queue.clear();
            queue.push(src);
            let mut head = 0;
            while head < queue.len() {
                let v = queue[head];
                head += 1;
                for i in 0..self.adj[v].len() {
                    let e = self.adj[v][i] as usize;
                    let w = self.to[e] as usize;
                    if self.cap[e] > 0 && level[w] == u32::MAX {
                        level[w] = level[v] + 1;
                        queue.push(w);
                    }
                }
            }

            // The residual graph no longer reaches `dst`, so the flow is maximal and the
            // nodes reached above are exactly the source side of a minimum cut.
            if level[dst] == u32::MAX {
                return level.iter().map(|l| *l != u32::MAX).collect();
            }

            next_edge.fill(0);
            while self.augment(src, dst, INF, &level, &mut next_edge) > 0 {}
        }
    }

    /// Pushes flow along a single `v`-`dst` path in the level graph, if there is one.
    #[recursive]
    fn augment(
        &mut self,
        v: usize,
        dst: usize,
        flow: u64,
        level: &[u32],
        next_edge: &mut [usize],
    ) -> u64 {
        if v == dst {
            return flow;
        }

        while next_edge[v] < self.adj[v].len() {
            let e = self.adj[v][next_edge[v]] as usize;
            let w = self.to[e] as usize;
            if self.cap[e] > 0 && level[w] == level[v] + 1 {
                let pushed = self.augment(w, dst, flow.min(self.cap[e]), level, next_edge);
                if pushed > 0 {
                    self.cap[e] -= pushed;
                    self.cap[e ^ 1] += pushed;
                    return pushed;
                }
            }
            next_edge[v] += 1;
        }

        0
    }
}

/// One distinct sub-expression the split may cut at.
struct DagNode {
    /// An arena node representing this sub-expression.
    node: Node,
    /// Input indices, in the order [`AExpr::replace_inputs`] expects them. Only populated
    /// for splittable nodes; a node that is materialized is computed inside the
    /// pre-select as a whole, so its inputs never reach the intermediate schema.
    inputs: Vec<usize>,
    /// Whether this node's operation may be applied in the post-select.
    splittable: bool,
    /// Expected bits per row when this node is materialized.
    weight: u64,
}

/// Whether `ae`'s own operation may be moved into the post-select, leaving its inputs to
/// be produced by the pre-select. `inputs_rev` must be `ae.inputs_rev()`.
fn is_splittable(ae: &AExpr, inputs_rev: &[Node], expr_arena: &Arena<AExpr>) -> bool {
    match ae {
        AExpr::Column(_) | AExpr::Element => return false,
        #[cfg(feature = "dtype-struct")]
        AExpr::StructEval { .. } => return false,
        _ => {},
    }

    // `is_elementwise` reports which sub-expressions may be split off. Where that differs
    // from `inputs_rev` an input has to stay attached to its parent (the literal
    // right-hand side of `is_in`, say) and `replace_inputs` could no longer put rebuilt
    // inputs back in the right places.
    let mut detachable = UnitVec::new();
    is_elementwise(&mut detachable, ae, expr_arena) && *detachable == *inputs_rev
}

#[recursive]
fn collect_dag(
    id: CanonicalExprId,
    canonical: &mut CanonicalExprMap,
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
    idx_of: &mut PlIndexMap<CanonicalExprId, usize>,
    dag: &mut Vec<DagNode>,
) -> PolarsResult<usize> {
    if let Some(idx) = idx_of.get(&id) {
        return Ok(*idx);
    }

    let node = canonical.representative(id);
    let ae = expr_arena.get(node);
    let weight = expected_row_bits(&ae.to_dtype(&ToFieldContext::new(expr_arena, input_schema))?);
    let mut inputs_rev = UnitVec::new();
    ae.inputs_rev(&mut inputs_rev);
    let splittable = is_splittable(ae, &inputs_rev, expr_arena);

    let mut inputs = Vec::new();
    if splittable {
        for input in inputs_rev.iter().rev() {
            let input_id = canonical.resolve(*input, expr_arena);
            inputs.push(collect_dag(
                input_id,
                canonical,
                expr_arena,
                input_schema,
                idx_of,
                dag,
            )?);
        }
    }

    let idx = dag.len();
    dag.push(DagNode {
        node,
        inputs,
        splittable,
        weight,
    });
    idx_of.insert(id, idx);
    Ok(idx)
}

/// Rebuilds the post-select expressions, materializing into the pre-select wherever the
/// post-select stops.
struct Rebuilder<'a> {
    dag: &'a [DagNode],
    /// Whether a node's operation is applied in the post-select.
    in_post: Vec<bool>,
    /// The post-select node per already-rebuilt DAG node.
    memo: Vec<Option<Node>>,
    used_names: PlIndexSet<PlSmallStr>,
    pre_select: Vec<ExprIR>,
}

impl Rebuilder<'_> {
    #[recursive]
    fn rebuild(&mut self, idx: usize, expr_arena: &mut Arena<AExpr>) -> Node {
        if let Some(node) = self.memo[idx] {
            return node;
        }

        let dag = self.dag;
        let node = if self.in_post[idx] {
            let ae = expr_arena.get(dag[idx].node).clone();
            let inputs = dag[idx]
                .inputs
                .iter()
                .map(|input| self.rebuild(*input, expr_arena))
                .collect::<Vec<_>>();
            expr_arena.add(ae.replace_inputs(&inputs))
        } else {
            let node = dag[idx].node;
            // Pass a column straight through under its own name where we can, rather than
            // renaming it for no reason.
            let name = match expr_arena.get(node) {
                AExpr::Column(name) if !self.used_names.contains(name) => name.clone(),
                _ => unique_column_name(),
            };
            self.used_names.insert(name.clone());
            self.pre_select
                .push(ExprIR::new(node, OutputName::Alias(name.clone())));
            expr_arena.add(AExpr::Column(name))
        };

        self.memo[idx] = Some(node);
        node
    }
}

/// Splits `exprs` into a pre-select and a post-select.
///
/// Selecting the returned pre-select on `input_schema` and then the returned post-select
/// on that result produces exactly what selecting `exprs` on `input_schema` would, and:
///
/// * every post-select expression is elementwise,
/// * every expression in `must_preselect` is an output of the pre-select, under its own
///   output name,
/// * the expected bytes per row of the intermediate schema is minimal.
///
/// The post-select reproduces `exprs` only; `must_preselect` is there for callers that
/// need those values available in the intermediate (a group-by needs its keys there, for
/// instance), and they are also offered to `exprs` as free building blocks, since they
/// are materialized either way.
///
/// Note that the pre-select is empty when `exprs` and `must_preselect` need nothing from
/// the input, which leaves the intermediate height undefined; a caller that cares must
/// add a column of its own.
pub fn split_pre_post_select_minsize_elementwise(
    exprs: &[ExprIR],
    must_preselect: &[ExprIR],
    input_schema: &Schema,
    expr_arena: &mut Arena<AExpr>,
) -> PolarsResult<(Vec<ExprIR>, Vec<ExprIR>)> {
    let mut canonical = CanonicalExprMap::new();
    let mut idx_of = PlIndexMap::new();
    let mut dag = Vec::new();

    let mut roots = Vec::with_capacity(exprs.len());
    for expr in exprs {
        let id = canonical.resolve(expr.node(), expr_arena);
        roots.push(collect_dag(
            id,
            &mut canonical,
            expr_arena,
            input_schema,
            &mut idx_of,
            &mut dag,
        )?);
    }

    // A `must_preselect` expression is materialized whether or not `exprs` uses it, so it
    // is free to build on and must never be split. Seeding the memo with its intermediate
    // column makes the post-select refer to it instead of recomputing it.
    let mut memo: Vec<Option<Node>> = vec![None; dag.len()];
    let mut used_names = PlIndexSet::new();
    let mut pre_select = Vec::with_capacity(must_preselect.len());
    for expr in must_preselect {
        let name = expr.output_name().clone();
        let id = canonical.resolve(expr.node(), expr_arena);
        if let Some(idx) = idx_of.get(&id).copied()
            && memo[idx].is_none()
        {
            dag[idx].splittable = false;
            dag[idx].weight = 0;
            memo[idx] = Some(expr_arena.add(AExpr::Column(name.clone())));
        }
        used_names.insert(name);
        pre_select.push(expr.clone());
    }

    // Minimize the intermediate width as a minimum cut. Per node we have two booleans:
    //
    //   post(n)  the node's operation is applied in the post-select
    //   avail(n) the node's value is available to the post-select
    //
    // with `avail(n) && !post(n)` meaning `n` is materialized, at cost `weight(n)`. The
    // only rule tying nodes together is that a node computed in the post-select needs its
    // inputs available there. Nodes are otherwise free to disagree, because an input of a
    // materialized node is recomputed inside the pre-select rather than passed through
    // the intermediate.
    //
    // Assigning "source side" to `true`, each cost is of the form
    // `w * [x == true && y == false]`, which is exactly an edge `x -> y` of capacity `w`,
    // so the minimum cut is the minimum intermediate width.
    let post = |idx: usize| 2 * idx;
    let avail = |idx: usize| 2 * idx + 1;
    let src = 2 * dag.len();
    let dst = 2 * dag.len() + 1;

    let mut network = FlowNetwork::new(2 * dag.len() + 2);
    for (idx, dag_node) in dag.iter().enumerate() {
        network.add_edge(avail(idx), post(idx), dag_node.weight);
        if dag_node.splittable {
            for input in &dag_node.inputs {
                network.add_edge(post(idx), avail(*input), INF);
            }
        } else {
            network.add_edge(post(idx), dst, INF);
        }
    }
    for root in &roots {
        network.add_edge(src, avail(*root), INF);
    }

    let source_side = network.min_cut_source_side(src, dst);
    let mut rebuilder = Rebuilder {
        in_post: (0..dag.len()).map(|idx| source_side[post(idx)]).collect(),
        dag: &dag,
        memo,
        used_names,
        pre_select,
    };

    let mut post_select = Vec::with_capacity(exprs.len());
    for (expr, root) in exprs.iter().zip(&roots) {
        let node = rebuilder.rebuild(*root, expr_arena);
        post_select.push(ExprIR::new(node, expr.output_name_inner().clone()));
    }

    Ok((rebuilder.pre_select, post_select))
}
