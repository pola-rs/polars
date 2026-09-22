# Physical Node IR Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record, for every physical node the streaming engine builds, the IR node whose lowering created it, and expose that id on the physical plan description the query observer receives.

**Architecture:** `PhysNode` gains an `ir_node: Option<Node>` field. A new `PhysPlanBuilder` owns the physical slotmap during plan construction and stamps the current IR node on every insert; `lower_ir` becomes a thin wrapper that sets and restores that current node around the existing body. The two post-lowering passes that insert nodes set the current node explicitly. `physical_plan_to_description` copies the id into a new `ir_node_id` field on `PhysicalNodeDescription`.

**Tech Stack:** Rust (crates `polars-stream`, `polars-descriptions`), `slotmap`, Python tests in `py-polars` via pytest and `msgpack`, jujutsu for version control.

**Spec:** `docs/superpowers/specs/2026-09-22-phys-node-ir-provenance-design.md`

## Global Constraints

- Every physical node in the slotmap must end up with `ir_node == Some(n)` where `n.0 < ir_arena.len()` as measured before lowering started. A `debug_assert!` in `build_physical_plan` enforces this.
- Temporary IR nodes (index at or beyond that recorded length) never become an attribution; nodes lowered from them inherit the original IR node being lowered.
- Wire type is `ir_node_id: Option<usize>` on `PhysicalNodeDescription`, matching `IrNodeDescription::id: usize`. Do not make it non-optional.
- The metrics snapshotter, `PlannedQuery`, and the Python observer binding (`crates/polars-python/src/polars_cloud_observer.rs`) are not modified.
- Read-only consumers of the plan (`to_graph.rs`, `to_description.rs`, `fmt.rs`) keep taking `&SlotMap<PhysNodeKey, PhysNode>`.
- Keep the lowering variable name `phys_sm` everywhere; only its type changes. This keeps the diff to signatures and imports.
- Version control is jujutsu, in the workspace at `/private/tmp/claude-501/-Volumes-sourcecode-polars/ea8956ca-4634-48a4-8bcf-99ec596b3af9/scratchpad/ws-ir-provenance`. Run all commands from that directory. Do not use `git`.
- Every commit message ends with these two trailer lines, verbatim:
  ```
  Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV
  ```
- Rust verification per task: `cargo fmt --all` then `cargo check -p polars-stream --all-features`, expecting zero warnings from `polars-stream` and `polars-descriptions`. Full clippy runs once in Task 5 because it is slow.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `crates/polars-descriptions/src/physical.rs` | Wire types for the physical plan description | Add `ir_node_id` field |
| `crates/polars-stream/src/physical_plan/mod.rs` | `PhysNode` data model, post-lowering passes, `build_physical_plan` | Add `ir_node` field + getter; declare `builder` module; take slotmap by value; passes that insert use the builder; debug assertion |
| `crates/polars-stream/src/physical_plan/builder.rs` (new) | `PhysPlanBuilder`: owns the slotmap during construction, stamps IR node on insert | Create |
| `crates/polars-stream/src/physical_plan/lower_ir.rs` | IR to physical lowering | `lower_ir` wrapper/inner split; parameter types |
| `crates/polars-stream/src/physical_plan/lower_expr.rs` | Expression lowering | `LowerExprContext::phys_sm` type; three entry-point parameter types |
| `crates/polars-stream/src/physical_plan/lower_group_by.rs` | Group-by lowering | Five parameter types |
| `crates/polars-stream/src/physical_plan/to_description.rs` | Physical plan to observer description | Fill `ir_node_id` |
| `crates/polars-stream/src/skeleton.rs` | Builds and runs a streaming query | Two `build_physical_plan` call sites destructure `(root, slotmap)` |
| `py-polars/tests/unit/lazyframe/test_query_monitoring.py` | Observer end-to-end tests | Four new tests plus two helpers |

Line numbers below are as of the spec commit and shift as tasks land. Anchor on the function names.

---

### Task 1: End-to-end tests for IR attribution (written first, failing)

**Files:**
- Modify: `py-polars/tests/unit/lazyframe/test_query_monitoring.py` (imports at lines 10-25; append tests after `test_metrics_handle_snapshot`, which ends at line 380)

**Interfaces:**
- Consumes: the fake observer helpers already in this file: `fake_cloud_observer()`, `mock_module_import`, `_sample_lf()`.
- Produces: the observable contract later tasks satisfy. Physical payload nodes gain key `ir_node_id` (int). IR payload nodes have `id` (int) and `properties.type` (str). Physical payload nodes have `id`, `input_ids`, and `properties.type`.

- [ ] **Step 1: Prepare the Python environment**

Run, from the workspace root:

```bash
cd py-polars && make requirements
.venv/bin/python -c "import msgpack" || uv pip install --python ../.venv/bin/python msgpack
```

If `uv` is not on the path, use `.venv/bin/python -m pip install msgpack`. Without `msgpack` every test in this task reports SKIPPED instead of FAILED, which hides the signal.

- [ ] **Step 2: Build the unchanged extension as a baseline**

Run: `cd py-polars && make build`

This is the dev profile (`maturin develop` without `--release`), so `debug_assert!` is active. A from-scratch build takes tens of minutes. Expected: exits 0 and `.venv/bin/python -c "import polars; print(polars.__file__)"` prints a path inside the workspace-root `.venv`. The venv lives one level above `py-polars` because `py-polars/Makefile` sets `VENV := ../.venv`; every `../.venv/bin/...` command in this plan assumes you are in `py-polars/`.

- [ ] **Step 3: Add the `Any` import**

In `py-polars/tests/unit/lazyframe/test_query_monitoring.py`, change line 10:

```python
from typing import TYPE_CHECKING
```

to:

```python
from typing import TYPE_CHECKING, Any
```

`Callable` is already imported under `TYPE_CHECKING` on line 26 and the file has `from __future__ import annotations`, so annotations below need nothing else.

- [ ] **Step 4: Add the helpers and four tests**

Insert immediately after `test_metrics_handle_snapshot` (after the line `assert sum(r["rows_sent"] for r in rows) > 0`, before `def test_on_query_failed_called`):

```python


def _observe_streaming(run: Callable[[], object]) -> MagicMock:
    """Run `run` with the fake cloud observer installed and return the observer."""
    module, observer = fake_cloud_observer()
    with mock_module_import("polars_cloud", module, replace_if_exists=True):
        pl.Config.enable_monitoring()
        run()
    return observer


def _planned_payloads(
    observer: MagicMock,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Decode the IR and physical payloads handed to `on_query_planned`."""
    msgpack = pytest.importorskip("msgpack")
    _, _, ir_bytes, phys_bytes = observer.on_query_planned.call_args.args
    ir = msgpack.unpackb(ir_bytes, raw=False)
    phys = msgpack.unpackb(phys_bytes, raw=False)
    return ir, phys


def test_physical_nodes_attributed_to_ir_nodes() -> None:
    """Every physical node carries the id of an IR node from the IR payload."""
    observer = _observe_streaming(lambda: _sample_lf().collect(engine="streaming"))
    ir, phys = _planned_payloads(observer)

    ir_ids = {node["id"] for node in ir}
    assert len(phys) > 0
    assert all(node["ir_node_id"] is not None for node in phys)
    assert all(node["ir_node_id"] in ir_ids for node in phys)


def test_ir_node_lowers_to_many_physical_nodes() -> None:
    """A filter with a non-elementwise predicate is one IR node, many physical."""
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]}).filter(
        pl.col("a") > pl.col("a").mean()
    )
    observer = _observe_streaming(lambda: lf.collect(engine="streaming"))
    ir, phys = _planned_payloads(observer)

    (filter_id,) = [n["id"] for n in ir if n["properties"]["type"] == "Filter"]
    assert sum(n["ir_node_id"] == filter_id for n in phys) > 1


def test_temporary_ir_node_inherits_attribution() -> None:
    """`collect_all` wraps bare inputs in memory sinks that lowering invents.

    Those sink IR nodes are never in the IR payload, so their physical sink is
    attributed to the `SinkMultiple` node whose lowering created them.
    """
    observer = _observe_streaming(
        lambda: pl.collect_all([_sample_lf()], engine="streaming")
    )
    ir, phys = _planned_payloads(observer)

    (sink_multiple_id,) = [
        n["id"] for n in ir if n["properties"]["type"] == "SinkMultiple"
    ]
    (memory_sink,) = [n for n in phys if n["properties"]["type"] == "InMemorySink"]
    assert memory_sink["ir_node_id"] == sink_multiple_id


def test_multiplexer_inherits_input_attribution() -> None:
    """A multiplexer inserted after lowering takes its input's IR node."""
    lf = (
        pl.LazyFrame({"a": [1, 2, 3, 4, 5]})
        .with_columns(b=pl.col("a") * 2)
        .filter(pl.col("b") > pl.col("b").mean())
    )
    observer = _observe_streaming(lambda: lf.collect(engine="streaming"))
    _, phys = _planned_payloads(observer)

    by_id = {n["id"]: n for n in phys}
    multiplexers = [n for n in phys if n["properties"]["type"] == "Multiplexer"]
    assert len(multiplexers) > 0
    for multiplexer in multiplexers:
        (input_id,) = multiplexer["input_ids"]
        assert multiplexer["ir_node_id"] == by_id[input_id]["ir_node_id"]
```

Why these plans: the mean predicate is not elementwise, so `build_filter_stream` lowers it through `build_hstack_stream_with_ctx` and produces a reduction, a zip, and the filter, all from one `Filter` IR node. That same lowering consumes its input twice (once for the reduction, once for the zip), which is what makes `insert_multiplexers` add a `Multiplexer`. The `with_columns` in the fourth test matters: without it the fan-out sits directly on an in-memory source, and `split_multiplexers` replaces that multiplexer with per-consumer copies of the source, so no `Multiplexer` would reach the description.

- [ ] **Step 5: Format and lint the test file**

Run:

```bash
cd py-polars && ../.venv/bin/ruff format tests/unit/lazyframe/test_query_monitoring.py && ../.venv/bin/ruff check tests/unit/lazyframe/test_query_monitoring.py
```

Expected: no diagnostics. If `ruff format` rewrapped lines, that is fine.

- [ ] **Step 6: Run the new tests and confirm they fail for the right reason**

Run:

```bash
cd py-polars && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -k "attributed or lowers_to_many or inherits" -v
```

Expected: 4 FAILED, each with `KeyError: 'ir_node_id'`. If `test_multiplexer_inherits_input_attribution` instead fails on `assert len(multiplexers) > 0`, the plan produced no multiplexer. Diagnose with:

```bash
cd py-polars && POLARS_VISUALIZE_PHYSICAL_PLAN=$PWD/phys.dot ../.venv/bin/python -c "import polars as pl; pl.LazyFrame({'a':[1,2,3,4,5]}).with_columns(b=pl.col('a')*2).filter(pl.col('b')>pl.col('b').mean()).collect(engine='streaming')" && grep -ic multiplexer phys.dot; rm -f phys.dot
```

A count of 0 means pick a plan whose fan-out sits on a non-source node (for example add a `.sort("a")` before the filter) and re-run until the failure is the `KeyError`. If any test reports SKIPPED, go back to Step 1.

- [ ] **Step 7: Commit**

```bash
jj describe -m "$(printf 'test(python): Cover physical node IR attribution in query monitoring\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV')" && jj new
```

---

### Task 2: Add the `ir_node` field, the wire field, and the description plumbing

**Files:**
- Modify: `crates/polars-descriptions/src/physical.rs:6-11`
- Modify: `crates/polars-stream/src/physical_plan/mod.rs:76-96` (`PhysNode`)
- Modify: `crates/polars-stream/src/physical_plan/to_description.rs:47-56`

**Interfaces:**
- Produces: `PhysNode::ir_node(&self) -> Option<Node>`; private field `PhysNode::ir_node: Option<Node>` (readable and writable from child modules of `physical_plan`, which Task 3 relies on); `PhysicalNodeDescription::ir_node_id: Option<usize>`.

- [ ] **Step 1: Add the wire field**

In `crates/polars-descriptions/src/physical.rs`, replace:

```rust
pub struct PhysicalNodeDescription {
    pub id: u64,
    pub input_ids: Vec<u64>,
    pub properties: PhysicalPropsDescription,
}
```

with:

```rust
pub struct PhysicalNodeDescription {
    pub id: u64,
    pub input_ids: Vec<u64>,
    /// `id` of the [`IrNodeDescription`](crate::IrNodeDescription) whose lowering created this
    /// node. `None` only in payloads from producers that predate this field.
    pub ir_node_id: Option<usize>,
    pub properties: PhysicalPropsDescription,
}
```

The struct already carries `#[derive(Debug, Clone, Default, Serialize, Deserialize)]` and `#[serde(default)]`, so no attribute changes.

- [ ] **Step 2: Add the field and getter to `PhysNode`**

In `crates/polars-stream/src/physical_plan/mod.rs`, replace the struct and its two constructors:

```rust
#[derive(Clone, Debug)]
pub struct PhysNode {
    output_schemas: UnitVec<Arc<Schema>>,
    kind: PhysNodeKind,
}

impl PhysNode {
    pub fn new(output_schema: Arc<Schema>, kind: PhysNodeKind) -> Self {
        Self {
            output_schemas: unitvec![output_schema],
            kind,
        }
    }

    pub fn new_multi_output(output_schemas: UnitVec<Arc<Schema>>, kind: PhysNodeKind) -> Self {
        Self {
            output_schemas,
            kind,
        }
    }
```

with:

```rust
#[derive(Clone, Debug)]
pub struct PhysNode {
    output_schemas: UnitVec<Arc<Schema>>,
    kind: PhysNodeKind,
    /// The IR node whose lowering created this node. Set by `PhysPlanBuilder::insert`; always
    /// `Some` once `build_physical_plan` has returned.
    ir_node: Option<Node>,
}

impl PhysNode {
    pub fn new(output_schema: Arc<Schema>, kind: PhysNodeKind) -> Self {
        Self {
            output_schemas: unitvec![output_schema],
            kind,
            ir_node: None,
        }
    }

    pub fn new_multi_output(output_schemas: UnitVec<Arc<Schema>>, kind: PhysNodeKind) -> Self {
        Self {
            output_schemas,
            kind,
            ir_node: None,
        }
    }

    pub fn ir_node(&self) -> Option<Node> {
        self.ir_node
    }
```

`Node` is already imported in this file via `use polars_utils::arena::{Arena, Node};`.

- [ ] **Step 3: Fill the field in the description**

In `crates/polars-stream/src/physical_plan/to_description.rs`, replace:

```rust
    while let Some(key) = queue.pop_front() {
        let node = &phys_sm[key];
        let kind = node.kind();
        let (properties, inputs) = phys_props(kind, expr_arena);
        let node = PhysicalNodeDescription {
            id: key.data().as_ffi(),
            input_ids: inputs.iter().map(|k| k.data().as_ffi()).collect(),
            properties,
        };
```

with:

```rust
    while let Some(key) = queue.pop_front() {
        let phys_node = &phys_sm[key];
        let (properties, inputs) = phys_props(phys_node.kind(), expr_arena);
        let node = PhysicalNodeDescription {
            id: key.data().as_ffi(),
            input_ids: inputs.iter().map(|k| k.data().as_ffi()).collect(),
            ir_node_id: phys_node.ir_node().map(|n| n.0),
            properties,
        };
```

- [ ] **Step 4: Format and type-check**

Run: `cargo fmt --all && cargo check -p polars-stream --all-features`

Expected: exits 0 with no warnings from `polars-stream` or `polars-descriptions`.

- [ ] **Step 5: Commit**

```bash
jj describe -m "$(printf 'feat(rust): Add ir_node_id to physical node descriptions\n\nAlways None for now; the lowering builder that fills it follows.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV')" && jj new
```

---

### Task 3: Introduce `PhysPlanBuilder` and thread it through lowering

**Files:**
- Create: `crates/polars-stream/src/physical_plan/builder.rs`
- Modify: `crates/polars-stream/src/physical_plan/mod.rs` (module list at lines 34-40; `build_physical_plan` at lines 955-984)
- Modify: `crates/polars-stream/src/skeleton.rs:86-95` and `:146-157`
- Modify: `crates/polars-stream/src/physical_plan/lower_ir.rs` (imports lines 35-38; `build_slice_stream` line 52; `build_filter_stream` line 81; `build_row_idx_stream` line 137; `lower_ir` lines 163-200; `append_sorted_key_column` line 1899; `lower_subtree_to_inmem_engine` line 1951)
- Modify: `crates/polars-stream/src/physical_plan/lower_expr.rs` (imports lines 28-31; `LowerExprContext` line 56; `lower_exprs` line 2805; `build_select_stream` line 2835; `build_hstack_stream` line 2856)
- Modify: `crates/polars-stream/src/physical_plan/lower_group_by.rs` (imports lines 23-25; `build_group_by_fallback` line 57; `try_lower_agg_input_expr` line 590; `try_build_streaming_group_by` line 749; `try_build_sorted_group_by` line 1101; `build_group_by_stream` line 1280)

**Interfaces:**
- Consumes: `PhysNode::ir_node` private field (Task 2).
- Produces:
  ```rust
  pub struct PhysPlanBuilder { pub current_ir_node: Option<Node>, /* private */ }
  impl PhysPlanBuilder {
      pub fn new(phys_sm: SlotMap<PhysNodeKey, PhysNode>, original_ir_len: usize) -> Self;
      pub fn is_original_ir_node(&self, node: Node) -> bool;
      pub fn insert(&mut self, node: PhysNode) -> PhysNodeKey;
      pub fn into_inner(self) -> SlotMap<PhysNodeKey, PhysNode>;
  }
  impl Deref<Target = SlotMap<PhysNodeKey, PhysNode>> + DerefMut for PhysPlanBuilder;
  pub fn build_physical_plan(root: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>, phys_sm: SlotMap<PhysNodeKey, PhysNode>, ctx: StreamingLowerIRContext<'_>) -> PolarsResult<(PhysNodeKey, SlotMap<PhysNodeKey, PhysNode>)>;
  pub fn lower_ir(.., phys_sm: &mut PhysPlanBuilder, ..) -> PolarsResult<PhysStream>;  // same parameter order as today
  ```
  Every `build_*` / `lower_*` / `try_*` function listed under **Files** takes `phys_sm: &mut PhysPlanBuilder`; `LowerExprContext::phys_sm` is `&'a mut PhysPlanBuilder`. Task 4 relies on `insert_multiplexers` and `split_multiplexers` still taking `&mut SlotMap<PhysNodeKey, PhysNode>` after this task; it changes them itself.

- [ ] **Step 1: Create the builder module**

Create `crates/polars-stream/src/physical_plan/builder.rs`:

```rust
use std::ops::{Deref, DerefMut};

use polars_utils::arena::Node;
use slotmap::SlotMap;

use super::{PhysNode, PhysNodeKey};

/// The physical plan while it is being built from the IR.
///
/// Every physical node is inserted through [`PhysPlanBuilder::insert`], which records the IR
/// node whose lowering created it. Reads and in-place edits go through `Deref` to the slotmap.
pub struct PhysPlanBuilder {
    phys_sm: SlotMap<PhysNodeKey, PhysNode>,
    /// IR node that nodes inserted through [`Self::insert`] are attributed to.
    pub current_ir_node: Option<Node>,
    /// Length of the IR arena before lowering started. Lowering appends temporary IR nodes
    /// beyond this index. They are not part of the plan the query observer sees, so nodes
    /// lowered from them inherit the attribution of the original IR node being lowered.
    original_ir_len: usize,
}

impl PhysPlanBuilder {
    pub fn new(phys_sm: SlotMap<PhysNodeKey, PhysNode>, original_ir_len: usize) -> Self {
        Self {
            phys_sm,
            current_ir_node: None,
            original_ir_len,
        }
    }

    /// Whether `node` was part of the IR plan before lowering started.
    pub fn is_original_ir_node(&self, node: Node) -> bool {
        node.0 < self.original_ir_len
    }

    /// Inserts `node`, attributing it to [`Self::current_ir_node`].
    ///
    /// This shadows `SlotMap::insert` through `Deref`: inherent methods win during method
    /// resolution, so `phys_sm.insert(..)` on a builder always stamps.
    pub fn insert(&mut self, mut node: PhysNode) -> PhysNodeKey {
        node.ir_node = self.current_ir_node;
        self.phys_sm.insert(node)
    }

    pub fn into_inner(self) -> SlotMap<PhysNodeKey, PhysNode> {
        self.phys_sm
    }
}

impl Deref for PhysPlanBuilder {
    type Target = SlotMap<PhysNodeKey, PhysNode>;

    fn deref(&self) -> &Self::Target {
        &self.phys_sm
    }
}

impl DerefMut for PhysPlanBuilder {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.phys_sm
    }
}
```

`node.ir_node` is a private field of `PhysNode` declared in `mod.rs`. Rust makes private items visible to descendant modules, and `builder` is a child of `physical_plan`, so this compiles without widening the field's visibility.

- [ ] **Step 2: Declare the module**

In `crates/polars-stream/src/physical_plan/mod.rs`, replace:

```rust
mod fmt;
mod io;
mod lower_expr;
```

with:

```rust
mod builder;
mod fmt;
mod io;
mod lower_expr;
```

and immediately after the line `pub use fmt::{NodeStyle, visualize_plan};` add:

```rust
pub use builder::PhysPlanBuilder;
```

- [ ] **Step 3: Make `build_physical_plan` own the slotmap through the builder**

In `crates/polars-stream/src/physical_plan/mod.rs`, replace the whole function:

```rust
pub fn build_physical_plan(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
    ctx: StreamingLowerIRContext<'_>,
) -> PolarsResult<PhysNodeKey> {
    let mut schema_cache = PlHashMap::with_capacity(ir_arena.len());
    let mut expr_cache = ExprCache::with_capacity(expr_arena.len());
    let mut cache_nodes = PlHashMap::new();
    let phys_root = lower_ir::lower_ir(
        root,
        ir_arena,
        expr_arena,
        phys_sm,
        &mut schema_cache,
        &mut expr_cache,
        &mut cache_nodes,
        ctx,
        None,
    )?;
    insert_multiplexers(vec![phys_root.node], phys_sm);
    split_multiplexers(vec![phys_root.node], phys_sm);
    fuse_drops(vec![phys_root.node], phys_sm);

    // TODO: remove this after fusing pre-select into group-by node.
    rechunk_group_by_inputs(vec![phys_root.node], phys_sm);

    Ok(phys_root.node)
}
```

with:

```rust
/// Lowers the IR rooted at `root` into `phys_sm` and returns the root physical node together
/// with the filled slotmap.
pub fn build_physical_plan(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: SlotMap<PhysNodeKey, PhysNode>,
    ctx: StreamingLowerIRContext<'_>,
) -> PolarsResult<(PhysNodeKey, SlotMap<PhysNodeKey, PhysNode>)> {
    let mut schema_cache = PlHashMap::with_capacity(ir_arena.len());
    let mut expr_cache = ExprCache::with_capacity(expr_arena.len());
    let mut cache_nodes = PlHashMap::new();
    let mut phys_sm = PhysPlanBuilder::new(phys_sm, ir_arena.len());
    let phys_root = lower_ir::lower_ir(
        root,
        ir_arena,
        expr_arena,
        &mut phys_sm,
        &mut schema_cache,
        &mut expr_cache,
        &mut cache_nodes,
        ctx,
        None,
    )?;
    insert_multiplexers(vec![phys_root.node], &mut phys_sm);
    split_multiplexers(vec![phys_root.node], &mut phys_sm);
    fuse_drops(vec![phys_root.node], &mut phys_sm);

    // TODO: remove this after fusing pre-select into group-by node.
    rechunk_group_by_inputs(vec![phys_root.node], &mut phys_sm);

    Ok((phys_root.node, phys_sm.into_inner()))
}
```

The four passes still take `&mut SlotMap<PhysNodeKey, PhysNode>`; passing `&mut phys_sm` (a `&mut PhysPlanBuilder`) compiles through deref coercion. Task 4 switches the two that insert.

- [ ] **Step 4: Update the two callers in `skeleton.rs`**

In `visualize_physical_plan`, replace:

```rust
    let mut phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());
    let sortedness = IRPlanSorted::resolve(node, ir_arena, expr_arena);

    let ctx = StreamingLowerIRContext {
        prepare_visualization: true,
        sortedness: &sortedness,
    };
    let root_phys_node =
        crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, &mut phys_sm, ctx)?;
```

with:

```rust
    let phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());
    let sortedness = IRPlanSorted::resolve(node, ir_arena, expr_arena);

    let ctx = StreamingLowerIRContext {
        prepare_visualization: true,
        sortedness: &sortedness,
    };
    let (root_phys_node, phys_sm) =
        crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, phys_sm, ctx)?;
```

In `StreamingQuery::build`, replace:

```rust
        let mut phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());
```

with:

```rust
        let phys_sm = SlotMap::with_capacity_and_key(ir_arena.len());
```

and replace:

```rust
        let root_phys_node = crate::physical_plan::build_physical_plan(
            node,
            ir_arena,
            expr_arena,
            &mut phys_sm,
            ctx,
        )?;
```

with:

```rust
        let (root_phys_node, phys_sm) = crate::physical_plan::build_physical_plan(
            node,
            ir_arena,
            expr_arena,
            phys_sm,
            ctx,
        )?;
```

- [ ] **Step 5: Change the lowering parameter types and imports**

In each of `lower_ir.rs`, `lower_expr.rs`, and `lower_group_by.rs`, change every occurrence of

```rust
    phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>,
```

in the functions listed under **Files** to

```rust
    phys_sm: &mut PhysPlanBuilder,
```

In `lower_expr.rs`, change the context field:

```rust
    pub(crate) phys_sm: &'a mut SlotMap<PhysNodeKey, PhysNode>,
```

to:

```rust
    pub(crate) phys_sm: &'a mut PhysPlanBuilder,
```

Imports. In `lower_ir.rs` line 38, change:

```rust
use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysStream};
```

to:

```rust
use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysPlanBuilder, PhysStream};
```

In `lower_expr.rs` line 31, change:

```rust
use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysStream, StreamingLowerIRContext};
```

to:

```rust
use super::{
    PhysNode, PhysNodeKey, PhysNodeKind, PhysPlanBuilder, PhysStream, StreamingLowerIRContext,
};
```

In `lower_group_by.rs` line 25, change:

```rust
use super::{ExprCache, PhysNode, PhysNodeKey, PhysNodeKind, PhysStream, StreamingLowerIRContext};
```

to:

```rust
use super::{
    ExprCache, PhysNode, PhysNodeKey, PhysNodeKind, PhysPlanBuilder, PhysStream,
    StreamingLowerIRContext,
};
```

Each of the three files has a `use slotmap::SlotMap;` line (`lower_ir.rs:35`, `lower_expr.rs:28`, `lower_group_by.rs:23`). After the type changes `cargo check` warns `unused import` for any that no longer has a use; delete exactly those. `PhysNode` and `PhysNodeKey` stay imported in all three files because `PhysNode::new` and `PhysNodeKey` are still used. `cargo fmt` will settle the wrapping of the `use` lines.

Do not touch call sites. `phys_sm.insert(PhysNode::new(..))` now resolves to `PhysPlanBuilder::insert`. `phys_sm[key]`, `phys_sm.get(..)`, `phys_sm.remove(..)`, `stream.output_schema(phys_sm)`, and `stream.output_schema_mut(phys_sm)` all keep compiling through `Deref`/`DerefMut` and deref coercion.

- [ ] **Step 6: Split `lower_ir` into wrapper and inner**

In `lower_ir.rs`, replace the head of the function (already carrying the new parameter type from Step 5):

```rust
#[recursive::recursive]
#[allow(clippy::too_many_arguments)]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut PhysPlanBuilder,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
    expr_cache: &mut ExprCache,
    cache_nodes: &mut PlHashMap<UniqueId, PhysStream>,
    ctx: StreamingLowerIRContext<'_>,
    mut disable_morsel_split: Option<bool>,
) -> PolarsResult<PhysStream> {
    // Helper macro to simplify recursive calls.
    macro_rules! lower_ir {
```

with:

```rust
/// Lowers `node` and everything below it to physical nodes.
///
/// Every physical node inserted while lowering an IR node of the original plan is attributed
/// to that IR node. Lowering also appends temporary IR nodes to the arena and lowers them
/// recursively; those are not part of the plan the query observer sees, so nodes lowered from
/// them inherit the attribution of the original IR node whose lowering created them.
#[recursive::recursive]
#[allow(clippy::too_many_arguments)]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut PhysPlanBuilder,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
    expr_cache: &mut ExprCache,
    cache_nodes: &mut PlHashMap<UniqueId, PhysStream>,
    ctx: StreamingLowerIRContext<'_>,
    disable_morsel_split: Option<bool>,
) -> PolarsResult<PhysStream> {
    let prev_ir_node = phys_sm.current_ir_node;
    if phys_sm.is_original_ir_node(node) {
        phys_sm.current_ir_node = Some(node);
    }
    let result = lower_ir_inner(
        node,
        ir_arena,
        expr_arena,
        phys_sm,
        schema_cache,
        expr_cache,
        cache_nodes,
        ctx,
        disable_morsel_split,
    );
    phys_sm.current_ir_node = prev_ir_node;
    result
}

#[allow(clippy::too_many_arguments)]
fn lower_ir_inner(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut PhysPlanBuilder,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
    expr_cache: &mut ExprCache,
    cache_nodes: &mut PlHashMap<UniqueId, PhysStream>,
    ctx: StreamingLowerIRContext<'_>,
    mut disable_morsel_split: Option<bool>,
) -> PolarsResult<PhysStream> {
    // Helper macro to simplify recursive calls.
    macro_rules! lower_ir {
```

The rest of the old body is now the body of `lower_ir_inner` and is unchanged. The `lower_ir!` macro inside it already calls `lower_ir` by name, which is now the wrapper, so every recursive step sets and restores the current node. `#[recursive::recursive]` stays on the wrapper only: it grows the stack on entry when needed, and every recursion passes through the wrapper before reaching the large inner frame.

- [ ] **Step 7: Format and type-check**

Run: `cargo fmt --all && cargo check -p polars-stream --all-features`

Expected: exits 0, no warnings. Typical errors and what they mean:

- `expected &mut SlotMap<..>, found &mut PhysPlanBuilder` at a call into a function not listed under **Files**: that function also needs the `&mut PhysPlanBuilder` type. Change it the same way and include it in the commit.
- `unused import: SlotMap`: delete that `use slotmap::SlotMap;` line, as described in Step 5.
- `method insert is never used` or `field current_ir_node is never read`: something still bypasses the builder. Every insertion site in `lower_ir.rs`, `lower_expr.rs`, and `lower_group_by.rs` must reach `PhysPlanBuilder::insert` through a `phys_sm: &mut PhysPlanBuilder` binding. Find the site with `grep -n "\.insert(PhysNode" crates/polars-stream/src/physical_plan/*.rs` and trace its `phys_sm` type.

- [ ] **Step 8: Commit**

```bash
jj describe -m "$(printf 'feat(rust): Attribute lowered physical nodes to their IR node\n\nPhysPlanBuilder owns the physical slotmap while build_physical_plan runs\nand stamps the current IR node on every insert. lower_ir sets that node\naround its body, so every physical node created while lowering an original\nIR node records it, and temporary IR nodes added by lowering inherit the\nenclosing original node.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV')" && jj new
```

---

### Task 4: Attribute nodes created by the post-lowering passes and assert the invariant

**Files:**
- Modify: `crates/polars-stream/src/physical_plan/mod.rs` (`insert_multiplexers` line 832; `split_multiplexers` line 859; `build_physical_plan` near the end of the file)

**Interfaces:**
- Consumes: `PhysPlanBuilder` (Task 3), `PhysNode::ir_node()` (Task 2).
- Produces: `fn insert_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut PhysPlanBuilder)`; `fn split_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut PhysPlanBuilder)`. `fuse_drops`, `rechunk_group_by_inputs`, `visit_node_inputs_mut`, `visit_nodes_mut`, and `_visit_nodes_impl` are unchanged and keep taking `&mut SlotMap<PhysNodeKey, PhysNode>`.

- [ ] **Step 1: Make `insert_multiplexers` attribute each multiplexer to its input's IR node**

Replace:

```rust
fn insert_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>) {
    let mut refcount: PlIndexMap<_, usize> = PlIndexMap::new();
    visit_node_inputs_mut(roots.clone(), phys_sm, |i| {
        *refcount.entry(*i).or_insert(0) += 1;
    });

    let mut multiplexer_map: PlHashMap<PhysStream, PhysStream> = refcount
        .into_iter()
        .filter(|(_stream, refcount)| *refcount > 1)
        .map(|(stream, refcount)| {
            let input_schema = Arc::clone(stream.output_schema(phys_sm));
            let multiplexer_node = phys_sm.insert(PhysNode::new_multi_output(
                (0..refcount).map(|_| Arc::clone(&input_schema)).collect(),
                PhysNodeKind::Multiplexer { input: stream },
            ));
            (stream, PhysStream::first(multiplexer_node))
        })
        .collect();
```

with:

```rust
fn insert_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut PhysPlanBuilder) {
    let mut refcount: PlIndexMap<_, usize> = PlIndexMap::new();
    visit_node_inputs_mut(roots.clone(), phys_sm, |i| {
        *refcount.entry(*i).or_insert(0) += 1;
    });

    let mut multiplexer_map: PlHashMap<PhysStream, PhysStream> = refcount
        .into_iter()
        .filter(|(_stream, refcount)| *refcount > 1)
        .map(|(stream, refcount)| {
            let input_schema = Arc::clone(stream.output_schema(phys_sm));
            // A multiplexer only fans out the stream it wraps, so it belongs to the same IR
            // node as that stream's producer.
            phys_sm.current_ir_node = phys_sm[stream.node].ir_node();
            let multiplexer_node = phys_sm.insert(PhysNode::new_multi_output(
                (0..refcount).map(|_| Arc::clone(&input_schema)).collect(),
                PhysNodeKind::Multiplexer { input: stream },
            ));
            (stream, PhysStream::first(multiplexer_node))
        })
        .collect();
```

The trailing `visit_node_inputs_mut(roots, phys_sm, ..)` call in this function is unchanged; `phys_sm` coerces to `&mut SlotMap<..>` there.

- [ ] **Step 2: Make `split_multiplexers` keep the source's attribution on each clone**

Replace:

```rust
fn split_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut SlotMap<PhysNodeKey, PhysNode>) {
```

with:

```rust
fn split_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut PhysPlanBuilder) {
```

and replace:

```rust
    let mut replacements: SecondaryMap<PhysNodeKey, Vec<PhysStream>> = split_map
        .into_iter()
        .map(|(k, n)| {
            let repls = (0..refcount[k]).map(|_| PhysStream::first(phys_sm.insert(n.clone())));
            (k, repls.collect())
        })
        .collect();
```

with:

```rust
    let mut replacements: SecondaryMap<PhysNodeKey, Vec<PhysStream>> = split_map
        .into_iter()
        .map(|(k, n)| {
            // The clones are the same source split per consumer, so they keep its IR node.
            phys_sm.current_ir_node = n.ir_node();
            let repls = (0..refcount[k]).map(|_| PhysStream::first(phys_sm.insert(n.clone())));
            (k, repls.collect())
        })
        .collect();
```

- [ ] **Step 3: Assert the invariant at the end of `build_physical_plan`**

In `build_physical_plan`, replace:

```rust
    // TODO: remove this after fusing pre-select into group-by node.
    rechunk_group_by_inputs(vec![phys_root.node], &mut phys_sm);

    Ok((phys_root.node, phys_sm.into_inner()))
```

with:

```rust
    // TODO: remove this after fusing pre-select into group-by node.
    rechunk_group_by_inputs(vec![phys_root.node], &mut phys_sm);

    debug_assert!(
        phys_sm
            .values()
            .all(|n| n.ir_node().is_some_and(|ir| phys_sm.is_original_ir_node(ir))),
        "every physical node must be attributed to an IR node of the original plan"
    );

    Ok((phys_root.node, phys_sm.into_inner()))
```

This covers orphaned slots too (a filter swapped by `fuse_drops`, a multiplexer replaced by `split_multiplexers`): they were stamped when inserted, so the assertion holds for the whole slotmap, not only the reachable part.

- [ ] **Step 4: Format and type-check**

Run: `cargo fmt --all && cargo check -p polars-stream --all-features`

Expected: exits 0, no warnings. If the compiler reports a borrow conflict on `phys_sm.current_ir_node = phys_sm[stream.node].ir_node();`, split it into `let ir_node = phys_sm[stream.node].ir_node(); phys_sm.current_ir_node = ir_node;`.

- [ ] **Step 5: Commit**

```bash
jj describe -m "$(printf 'feat(rust): Attribute post-lowering physical nodes to their IR node\n\nMultiplexers take the IR node of the stream they fan out and split source\nclones keep the IR node of their source. build_physical_plan now asserts in\ndebug builds that every physical node is attributed to an original IR node.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV')" && jj new
```

---

### Task 5: Verify end to end

**Files:**
- No new edits expected. Fix-ups from this task go into a final commit.

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Full lint on the two crates**

Run:

```bash
cargo fmt --all -- --check && cargo clippy -p polars-stream -p polars-descriptions --all-targets --all-features -- -W clippy::dbg_macro -D warnings
```

Expected: exits 0. Fix any diagnostic in the file it names; do not add `allow` attributes.

- [ ] **Step 2: Rebuild the Python extension with the changes**

Run: `cd py-polars && make build`

Expected: exits 0. This is an incremental build and much faster than the baseline in Task 1.

- [ ] **Step 3: Run the four new tests**

Run:

```bash
cd py-polars && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -k "attributed or lowers_to_many or inherits" -v
```

Expected: 4 PASSED, 0 skipped. A `panicked at ... every physical node must be attributed` message means a node was inserted around the builder; find the call with `grep -rn "\.insert(PhysNode" crates/polars-stream/src/physical_plan/` and check that its `phys_sm` binding has type `&mut PhysPlanBuilder`.

- [ ] **Step 4: Run the whole monitoring file and the lazyframe unit tests**

Run:

```bash
cd py-polars && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -v && ../.venv/bin/pytest tests/unit/lazyframe -q
```

Expected: all pass. The second command exercises `build_physical_plan` on many plans under the debug assertion.

- [ ] **Step 5: Run the fast unit test suite under the debug assertion**

Run: `cd py-polars && make test`

This is the repository's own fast-test target: `pytest -n auto` with the default marker filter from `pyproject.toml`, which already excludes slow, disk-writing, release-only, docs, hypothesis, benchmark, and CI-only tests. Expected: all pass. It is the widest sweep of plan shapes available locally, and every streaming collect in it runs the `debug_assert!`. If it runs longer than about thirty minutes on this machine, stop it and say so in the final report along with how far it got.

- [ ] **Step 6: Commit any fix-ups, then report**

If Steps 1 to 5 required edits:

```bash
jj describe -m "$(printf 'fix(rust): Address clippy and test fallout for IR attribution\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV')" && jj new
```

Otherwise leave the empty working-copy change as is. Then print the branch for the report:

```bash
jj log -r 'trunk()..@' --no-graph -T 'change_id.short() ++ " " ++ description.first_line() ++ "\n"'
```

Expected: an empty change on top, then the four task commits (plus the optional fix-up), the plan commit, and the spec commit.
