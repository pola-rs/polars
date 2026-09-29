# Physical node IR provenance in the streaming engine

Date: 2026-09-29 (revises the 2026-09-22 design)

## Goal

Let a consumer of the query observer map streaming-engine `NodeMetrics` back
onto the nodes of the optimized `IRPlan`. Each physical node records the IR
node whose lowering created it. The relationship is one IR node to many
physical nodes; every physical node has exactly one source IR node.

## Revision

The first version of this design stored the IR node on `PhysNode` and routed
every insert through a `PhysSmBuilder` wrapper that was threaded through the
whole of lowering. Review asked for neither: the attribution should be an
external map, and lowering should not have to know about it. This revision
keeps the goal, the outcomes, and the tests, and replaces the mechanism with
a length diff over the slotmap around each IR node's lowering. The
description of the current state below is relative to `main`.

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

One site removes from the slotmap during lowering: `simplify_input_streams`
in `lower_expr.rs` folds the `Reduce` nodes that `lower_reduce_node` just
inserted for the same expression into one combined `Reduce`, and removes the
originals.

## Design

### Data model

`PhysNode` is unchanged. Attribution lives beside the plan, not in it:

- `phys_sm` changes type from `SlotMap<PhysNodeKey, PhysNode>` to
  `DenseSlotMap<PhysNodeKey, PhysNode>` at every signature that names it
  (about 33 sites in `mod.rs`, `lower_ir.rs`, `lower_expr.rs`,
  `lower_group_by.rs`, `to_graph.rs`, `fmt.rs`, `to_description.rs`, and
  `skeleton.rs`). `DenseSlotMap` keeps its keys in a dense `Vec` in insertion
  order, which is what the lowering rule below relies on. Every operation the
  crate uses (`Index`, `get`, `get_mut`, `get_disjoint_mut`, `insert`,
  `remove`, `iter`, `keys`, `values`, `with_capacity_and_key`) exists on it
  with the same signature, so the change is a rename. No alias or wrapper is
  introduced.
- The attribution is a `SecondaryMap<PhysNodeKey, Node>`, called
  `phys_to_ir`. It is a plain map; the only logic that fills it is in
  `lower_ir` and the post-lowering passes.

`PhysicalNodeDescription` in `polars-descriptions` gains
`ir_node_id: Option<usize>`, typed to match `IrNodeDescription::id`. The
struct already has `#[serde(default)]`, so payloads from older producers
decode with the field as `None`, and the msgpack the Python observer binding
serializes picks the field up with no binding changes. New producers always
fill it.

### Lowering rule

`build_physical_plan` records `original_ir_len = ir_arena.len()` before
lowering. An IR node is *original* when its index is below that length;
anything at or beyond it was added by lowering and is not in the plan the
observer sees.

The current body of `lower_ir` becomes `lower_ir_inner`. The public
`lower_ir` wrapper takes two extra parameters, `phys_to_ir` and
`original_ir_len`, which the `lower_ir!` macro forwards on every recursive
call. The wrapper is the only place that writes `phys_to_ir` during lowering:

```rust
let len_before = phys_sm.len();
let out = lower_ir_inner(node, ..)?;
// Temporary IR nodes are not in the IR the observer sees; the physical
// nodes lowered from them are claimed by the enclosing original node.
if node.0 < original_ir_len {
    // `keys_as_slice` is the dense key vector in insertion order; slicing it
    // is O(1), whereas `keys().skip(len_before)` walks every earlier key.
    for &key in &phys_sm.keys_as_slice()[len_before..] {
        // A nested `lower_ir` call already claimed its own nodes, so the
        // innermost original IR node wins.
        if !phys_to_ir.contains_key(key) {
            phys_to_ir.insert(key, node);
        }
    }
}
Ok(out)
```

Children are lowered inside the parent's window, and each child's wrapper
runs before the parent's, so a node is claimed by the innermost original IR
node whose lowering inserted it. Early returns inside `lower_ir_inner` need
no handling because the wrapper owns the diff. An error aborts the whole
build, so a window that never closes does not matter.

No other lowering signature changes. `build_*_stream`, `LowerExprContext`,
`build_group_by_stream` and its helpers keep taking
`&mut DenseSlotMap<PhysNodeKey, PhysNode>` and know nothing about
attribution.

Every attribution is an id the observer can look up: lowering only descends
through nodes reachable from the root via `IR::inputs()` (including a
`Resolver`'s resolved node, which `inputs()` yields), and
`ir_plan_to_description` walks the same edges, so every `ir_node_id` in the
physical payload names a node in the IR payload.

Outcomes:

- **Temporary IR nodes attribute upward.** The memory sink wrapped around a
  bare `SinkMultiple` input goes to the `SinkMultiple` IR node. The key
  column and sort inserted for a range join go to the `Join` IR node.
- **Cache IR nodes map to zero physical nodes.** They insert nothing and
  return the shared stream, so their window is empty. The cached subtree's
  nodes belong to their own IR nodes.
- **In-memory fallbacks** that build a throwaway IR arena (for example
  order-preserving distinct with keep-last) insert their physical node inside
  the original IR node's window, so they need no special handling.

### Post-lowering passes

A window cannot express "each new node gets a different source", so the
three passes that create or move nodes write `phys_to_ir` directly and take
`&mut SecondaryMap<PhysNodeKey, Node>` as an extra parameter.

- `insert_multiplexers`: a multiplexer only fans out the stream it wraps, so
  it takes that stream's producer: `phys_to_ir.insert(mux,
  phys_to_ir[stream.node])`. Fan-out is not only a cache artefact: one IR
  node's lowering can consume a stream twice, so inheriting from the input is
  the general rule.
- `split_multiplexers`: when it records the in-memory source to clone for a
  multiplexer, it also records the source's IR node
  (`phys_to_ir[input.node]`), and inserts each clone into `phys_to_ir` with
  it.
- `fuse_drops`: in the branch that `mem::swap`s the filter into the
  projection's slot, it swaps the two `phys_to_ir` entries as well, so the
  surviving slot keeps the `Filter` IR node. The dropped `SimpleProjection`
  IR node ends up with no reachable physical node; the orphaned slot holds
  the old projection and is never described.
- `rechunk_group_by_inputs` only flips a flag on existing nodes and is
  unchanged.

### Plumbing

`build_physical_plan` keeps taking `phys_sm: &mut DenseSlotMap<..>` from its
caller, as on `main`, and returns
`PolarsResult<(PhysNodeKey, SecondaryMap<PhysNodeKey, Node>)>`. Its two
callers in `skeleton.rs` construct a `DenseSlotMap` instead of a `SlotMap`;
`visualize_physical_plan` discards the map, and `StreamingQuery::build`
stores it in a new `pub phys_to_ir` field next to `phys_sm`.

`physical_plan_to_description` takes `&SecondaryMap<PhysNodeKey, Node>` as a
new parameter and sets `ir_node_id: phys_to_ir.get(key).map(|n| n.0)`.
`StreamingQuery::to_planned_query` passes `&self.phys_to_ir`. The metrics
snapshotter, `PlannedQuery`, and the Python observer binding are untouched.

### Invariants

**Removal inside a window.** `DenseSlotMap::remove` is a `swap_remove` on
the dense key vector: the last key moves into the removed key's position.
`keys().skip(len_before)` is therefore exactly the keys inserted since
`len_before` only if nothing at a position below `len_before` is removed
while the window is open. The one removal in the crate,
`simplify_input_streams`, only removes `Reduce` nodes inserted while lowering
the same expression, hence inside the innermost open `lower_ir` call, hence
at positions at or beyond every open window's `len_before`. A comment at the
`remove` states this rule (a removal may only touch nodes inserted during the
current `lower_ir` call) and why; a comment where `len_before` is taken
points back to it. Any new removal in lowering must obey the same rule.

**Insertion order.** slotmap documents `DenseSlotMap` iteration order as
arbitrary. The order is insertion order because the values and keys live in
dense `Vec`s that only ever push, or swap-remove; that storage layout is the
reason the type exists, and this design relies on it.

**Safety net.** `build_physical_plan` ends with

```rust
debug_assert!(
    phys_sm.keys().all(|k| phys_to_ir.get(k).is_some_and(|ir| ir.0 < original_ir_len)),
    "every physical node must be attributed to an IR node of the original plan"
);
```

This runs in every debug build that lowers a plan, which includes the whole
Python test suite. It guards the post-lowering passes: a pass that inserts a
node without attributing it fails here. It does not guard the removal rule.
The root's window starts at length 0, so a node that an inner window missed
is claimed by an outer window (at worst the root) rather than left
unclaimed; breaking the rule shows up as a wrong `ir_node_id`, not as a
panic. The rule is enforced by the comments and by review.

### Out of scope

- Showing the IR id in the physical plan dot visualizer or in the
  `POLARS_LOG_METRICS` printout.
- Aggregating metrics per IR node on the Rust side. The consumer joins
  metrics rows to physical nodes by `phys_node_key`, then physical nodes to IR
  nodes by `ir_node_id`.
- Exposing temporary IR nodes in the IR description.
- The existing quirk that the `SinkMultiple` physical node maps to its first
  sink's graph node and so duplicates that sink's metrics row.
- A type alias for the slotmap type. The rename is mechanical and an alias
  would be a second change to review.

## Testing

Tests live in `py-polars/tests/unit/lazyframe/test_query_monitoring.py`,
which already fakes the `polars_cloud` observer and decodes the IR and
physical msgpack payloads. The six tests are already on the branch and are
the acceptance criteria for this revision; they must pass unchanged.

1. **Coverage.** For the sample query, every physical node has a non-null
   `ir_node_id`, and each one is the `id` of some node in the IR payload.
2. **One-to-many.** A filter whose predicate is not elementwise (for example
   `pl.col("a") > pl.col("a").mean()`) lowers to several physical nodes, and
   at least one IR id appears on more than one physical node.
3. **Temporary node inherits upward.** `pl.collect_all([lf])` on a bare plan
   lowers its memory sink from a temporary `Sink` IR node; the `InMemorySink`
   physical node carries the id of the `SinkMultiple` IR node.
4. **Multiplexer inherits from input.** A `with_columns` followed by a
   non-elementwise filter fans one stream out to two consumers through a
   `Multiplexer`; its `ir_node_id` equals that of its single input node.
5. **Split source clones keep the source.** Fanning an in-memory source out
   directly clones it per consumer; every `InMemorySource` carries the
   `DataFrameScan` IR node's id.
6. **Fused drop keeps the filter.** A `select` fused into the filter below it
   leaves one `Filter` physical node attributed to the `Filter` IR node and
   no `SimpleProjection` physical node.

The non-elementwise filter in tests 2, 4, and 5 also exercises the `Reduce`
removal in `simplify_input_streams`, so the removal rule is covered by the
same tests.

No Rust unit tests: `polars-stream` only has them for IO internals, and
plan-level behaviour is tested from Python throughout the repository.

## Verification

- `cargo fmt` and
  `cargo clippy -p polars-stream -p polars-descriptions --all-targets --all-features -- -W clippy::dbg_macro`
  (the `make clippy` flags scoped to the two crates) are clean.
- The six tests above pass, along with the rest of
  `test_query_monitoring.py`, against a debug build of `py-polars` so the
  `debug_assert!` is active.
- `builder.rs` is deleted and `PhysNode` no longer has an `ir_node` field;
  `grep -rn "PhysSmBuilder\|ir_node()" crates/polars-stream` finds nothing.
