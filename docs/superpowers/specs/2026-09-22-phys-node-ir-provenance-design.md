# Physical node IR provenance in the streaming engine

Date: 2026-09-22

## Goal

Let a consumer of the query observer map streaming-engine `NodeMetrics` back
onto the nodes of the optimized `IRPlan`. Each physical node records the IR
node whose lowering created it. The relationship is one IR node to many
physical nodes; every physical node has exactly one source IR node.

## Current state

The observer's `PlannedQuery` already carries three views of a query:

- `ir: Vec<IrNodeDescription>`, keyed by the IR arena index (`Node.0`).
- `physical: Vec<PhysicalNodeDescription>`, keyed by the `PhysNodeKey`
  slotmap key in FFI form.
- A metrics snapshotter whose rows (`NodeMetricsDescription`) carry that same
  physical key.

Metrics therefore join to physical nodes today. Nothing records which IR node
a physical node came from, so physical nodes cannot join to IR nodes.

Lowering (`crates/polars-stream/src/physical_plan/lower_ir.rs`) is one
recursive function. It and its helpers in `lower_expr.rs`,
`lower_group_by.rs`, and `mod.rs` insert physical nodes directly into a
`SlotMap<PhysNodeKey, PhysNode>` at about 51 sites. After lowering,
`build_physical_plan` runs four passes over the slotmap: inserting
multiplexers, splitting multiplexers over in-memory sources, fusing drops into
filters, and flagging group-by inputs for rechunking.

Three things complicate a naive "stamp the node being lowered" rule:

- Lowering adds temporary IR nodes to the arena and lowers them recursively:
  a memory `Sink` around each bare input of a `SinkMultiple`, an `HStack`
  that materializes a non-trivial range-join key, and a `Sort` inserted
  before a range join. These are never linked into the IR tree, so the IR
  description the observer receives does not contain them.
- The post-lowering passes create physical nodes outside any IR node's
  lowering (multiplexers, cloned in-memory sources) and move node contents
  between slots (drop fusing swaps a filter into a projection's slot).
- `IR::Cache` returns an already-lowered stream and inserts nothing.

## Design

### Data model

`PhysNode` gains a private `ir_node: Option<Node>` field and a public
`ir_node()` getter. Both constructors (`new`, `new_multi_output`) leave it
unset. Because the field lives on `PhysNode` next to `output_schemas` and
`kind`, the clone in the multiplexer-splitting pass and the `mem::swap` in
the drop-fusing pass carry it along with the node kind.

A new `PhysPlanBuilder` in `crates/polars-stream/src/physical_plan/builder.rs`
owns the slotmap while the plan is being built. Owning rather than borrowing
keeps a lifetime parameter out of `LowerExprContext` and every lowering
signature:

```rust
pub struct PhysPlanBuilder {
    phys_sm: SlotMap<PhysNodeKey, PhysNode>,
    /// IR node whose lowering is currently inserting physical nodes.
    pub current_ir_node: Option<Node>,
    /// Length of the IR arena before lowering started. Nodes at or beyond
    /// this index were added by lowering itself and are not in the plan the
    /// observer sees.
    original_ir_len: usize,
}
```

It implements `Deref` and `DerefMut` to the slotmap, so indexing, `get`,
`remove`, and the schema accessors on `PhysStream` keep working unchanged.
Its one inserting method is `insert(&mut self, node: PhysNode) -> PhysNodeKey`,
which sets `node.ir_node = self.current_ir_node` and inserts. It also exposes
`is_original_ir_node(Node) -> bool` and `into_inner() -> SlotMap<..>`. Inherent methods win
over dereferenced ones during method resolution, so the existing
`phys_sm.insert(PhysNode::new(..))` call sites compile unchanged and now
stamp. Bypassing the stamp requires an explicit `(**phys_sm).insert(..)`,
which nothing does.

The parameter type changes from `&mut SlotMap<PhysNodeKey, PhysNode>` to
`&mut PhysPlanBuilder` on `lower_ir`, the `build_*_stream` helpers,
`LowerExprContext::phys_sm`, `build_group_by_stream` and its helpers, and the
two post-lowering passes that insert nodes (`insert_multiplexers`,
`split_multiplexers`). The drop-fusing and rechunk passes and the node
visitors keep taking the plain slotmap and receive it through deref coercion.
`build_physical_plan` takes the slotmap by value and returns it alongside the
root key, so its two callers in `skeleton.rs` destructure the pair. Read-only
consumers (`to_graph`, `to_description`, `fmt`) keep taking the plain
slotmap.

`PhysicalNodeDescription` in `polars-descriptions` gains
`ir_node_id: Option<usize>`, typed to match `IrNodeDescription::id`. The
struct already has `#[serde(default)]`, so payloads from older producers
decode with the field as `None`, and the msgpack the Python observer binding
serializes picks the field up with no binding changes. New producers always
fill it.

### Lowering rules

`build_physical_plan` records `ir_arena.len()` before lowering and constructs
the builder from it and the slotmap it was given. The current body of `lower_ir` becomes
an inner function. The public `lower_ir` wrapper:

1. saves `builder.current_ir_node`;
2. sets it to `Some(node)` if `node.0 < builder.original_ir_len`, and leaves
   it unchanged otherwise;
3. calls the inner function;
4. restores the saved value and returns the result.

The recursive `lower_ir!` macro already calls `lower_ir`, so every child sets
and restores its own node, including original children under a temporary
parent. Early returns inside the body need no handling because the wrapper
owns the restore. The root is always original, so the current node is `Some`
for the whole of lowering, and a temporary IR node inherits the original IR
node whose lowering created it.

Outcomes:

- **Temporary IR nodes attribute upward.** The memory sink wrapped around a
  bare `SinkMultiple` input goes to the `SinkMultiple` IR node. The key
  column and sort inserted for a range join go to the `Join` IR node.
- **Cache IR nodes map to zero physical nodes.** They insert nothing and
  return the shared stream. The cached subtree's nodes belong to their own IR
  nodes.
- **In-memory fallbacks** that build a throwaway IR arena (for example
  order-preserving distinct with keep-last) insert their physical node while
  the original IR node is current, so they need no special handling.
- **Post-lowering passes that insert run through the builder.** Every insert
  in the crate goes through one method. Before inserting a multiplexer, the
  pass sets `builder.current_ir_node` to `builder[stream.node].ir_node()` for
  the stream being fanned out. Fan-out is not only a cache artefact: one IR node's lowering
  can consume a stream twice, so inheriting from the input is the general
  rule. The multiplexer-splitting pass clones an in-memory source per
  consumer; it sets `builder.current_ir_node` to the source's id before
  inserting each clone, so the clones keep the source's id. Drop fusing swaps the filter
  into the projection's slot, so the surviving node keeps the `Filter` IR's
  id; the dropped `SimpleProjection` IR node ends up with no physical nodes,
  and the orphaned slot is unreachable from the root and never described.

At the end of `build_physical_plan`, a `debug_assert!` checks that every node
in the slotmap has `ir_node == Some(n)` with `n.0 < original_ir_len`. This
runs in every debug build that lowers a plan, which includes the whole Python
test suite.

`PhysNode::ir_node()` returns `Option<Node>` because the field is unset
between construction and insert. After `build_physical_plan` returns it is
always `Some`.

### Description plumbing

`physical_plan_to_description` sets
`ir_node_id: node.ir_node().map(|n| n.0)`. The metrics snapshotter,
`PlannedQuery`, and the Python observer binding are untouched.

### Out of scope

- Showing the IR id in the physical plan dot visualizer or in the
  `POLARS_LOG_METRICS` printout.
- Aggregating metrics per IR node on the Rust side. The consumer joins
  metrics rows to physical nodes by `phys_node_key`, then physical nodes to IR
  nodes by `ir_node_id`.
- Exposing temporary IR nodes in the IR description.
- The existing quirk that the `SinkMultiple` physical node maps to its first
  sink's graph node and so duplicates that sink's metrics row.

## Testing

Tests live in `py-polars/tests/unit/lazyframe/test_query_monitoring.py`,
which already fakes the `polars_cloud` observer and decodes the IR and
physical msgpack payloads.

1. **Coverage.** For the sample query, every physical node has a non-null
   `ir_node_id`, and each one is the `id` of some node in the IR payload.
2. **One-to-many.** A filter whose predicate is not elementwise (for example
   `pl.col("a") > pl.col("a").mean()`) lowers to several physical nodes.
   Assert that at least one IR id appears on more than one physical node.
3. **Temporary node inherits upward.** `pl.collect_all([lf])` on a bare plan
   already runs through the observer in this file. Its memory sink is lowered
   from a temporary `Sink` IR node. Assert that the physical node whose
   properties type is `InMemorySink` carries the id of the `SinkMultiple` IR
   node.
4. **Multiplexer inherits from input.** A plan that fans one stream out to
   two consumers produces a `Multiplexer`. The non-elementwise filter from
   test 2 does this (the input feeds both the reduction and the zip that
   broadcasts its result), but the fan-out must not sit directly on an
   in-memory source, because `split_multiplexers` then replaces the
   multiplexer with per-consumer copies of the source. Put a `with_columns`
   between the source and the filter. Assert the multiplexer's `ir_node_id`
   equals the `ir_node_id` of its single input physical node.

No Rust unit tests: `polars-stream` only has them for IO internals, and
plan-level behaviour is tested from Python throughout the repository.

## Verification

- `cargo fmt` and
  `cargo clippy -p polars-stream -p polars-descriptions --all-targets --all-features -- -W clippy::dbg_macro`
  (the `make clippy` flags scoped to the two crates) are clean.
- The four tests above pass, along with the rest of
  `test_query_monitoring.py`, against a debug build of `py-polars` so the
  `debug_assert!` is active.
