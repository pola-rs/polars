# Physical Node IR Provenance (External Map) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the intrusive `PhysNode::ir_node` field and the `PhysSmBuilder` wrapper on branch `louis/attribute_physical_to_ir` with an external `SecondaryMap<PhysNodeKey, Node>` that `lower_ir` fills by diffing the slotmap length around each IR node's lowering, without changing what the query observer receives.

**Architecture:** The physical plan moves from `SlotMap` to `DenseSlotMap`, whose keys iterate in insertion order, so the keys added while lowering one IR node are `phys_sm.keys().skip(len_before)`. `lower_ir` claims those for the innermost *original* IR node (index below the arena length recorded before lowering); temporary IR nodes that lowering appends are skipped so their nodes fall to the enclosing original node. The three post-lowering passes that create or move nodes write the map directly. The new map is first filled *beside* the existing builder with a debug assertion that both agree, and only then is the builder removed, so the branch is green after every task.

**Tech Stack:** Rust (crates `polars-stream`, `polars-descriptions`), `slotmap` 1.1 (`DenseSlotMap`, `SecondaryMap`), Python tests in `py-polars` via pytest and `msgpack`, jujutsu for version control.

**Spec:** `docs/superpowers/specs/2026-09-22-phys-node-ir-provenance-design.md`

## Global Constraints

- All work happens in the jujutsu workspace at `/private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework` (called `$W` in prose; every command below spells the path out). Its `@` sits on top of the spec commit `tprqluoo`, which sits on `louis/attribute_physical_to_ir`. Do not touch `/Volumes/sourcecode/polars` and do not move the bookmark.
- Version control is jujutsu, never git. Commit with `jj commit -m "$(printf '...')"`. Every commit message ends with the two trailer lines `Co-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV`.
- The Python venv lives at `$W/.venv` (one level above `py-polars`, because `py-polars/Makefile` sets `VENV := ../.venv`). Every `../.venv/bin/...` command assumes you are in `$W/py-polars`.
- Tests run against a **debug** build (`make build` in `py-polars`, the dev profile) so `debug_assert!` is active. Never test against a release build.
- Rust verification per task: `cargo fmt --all` then `cargo check -p polars-stream --all-features`, expecting zero warnings from `polars-stream`. Full clippy runs once in Task 4 because it is slow.
- `phys_sm` is `DenseSlotMap<PhysNodeKey, PhysNode>` everywhere; no type alias, no wrapper type.
- The attribution is a bare `SecondaryMap<PhysNodeKey, Node>` named `phys_to_ir`. Only `lower_ir`, `build_physical_plan`, `insert_multiplexers`, `split_multiplexers`, `fuse_drops`, `physical_plan_to_description`, and `StreamingQuery` see it. `build_*_stream`, `LowerExprContext`, and the group-by helpers take the plain `&mut DenseSlotMap<PhysNodeKey, PhysNode>` and know nothing about it.
- `PhysicalNodeDescription::ir_node_id` stays `Option<usize>`. Do not touch `crates/polars-descriptions`.
- The one removal from `phys_sm` during lowering (`simplify_input_streams`, `crates/polars-stream/src/physical_plan/lower_expr.rs:505`) stays; it gets a comment stating the rule. Do not add another removal.
- The six existing tests `test_physical_nodes_attributed_to_ir_nodes`, `test_ir_node_lowers_to_many_physical_nodes`, `test_temporary_ir_node_inherits_attribution`, `test_multiplexer_inherits_input_attribution`, `test_split_source_clones_keep_source_attribution`, `test_fused_drop_keeps_filter_attribution` in `py-polars/tests/unit/lazyframe/test_query_monitoring.py` are the acceptance criteria and must not be edited.

## Review Focus

Inputs the spec implies but the six acceptance tests never exercise. Each gets a test in Task 2, Step 1; all five already pass on the branch (they pin the behaviour the rework must reproduce).

1. **Shared subplan (`collect_all` with common-subplan elimination).** The `Cache` IR node lowers to nothing; a consumer joining metrics must find every `ir_node_id` in the IR payload and must not see a `Cache` id on any physical node. Test: `test_cache_ir_node_owns_no_physical_nodes`.
2. **Range join.** Lowering appends temporary `HStack`/`Sort` IR nodes; their physical nodes must carry the original join's id, not an index the observer cannot resolve. Test: `test_range_join_temporaries_attribute_to_join`.
3. **In-memory fallback (order-preserving keep-last distinct).** The fallback lowers through a throwaway IR arena; its physical node must still belong to the original `Distinct` IR node. Test: `test_in_memory_fallback_attributes_to_its_ir_node`.
4. **Two reductions in one predicate.** `simplify_input_streams` removes two `Reduce` nodes and inserts a merged one inside the same window; the merged node must belong to the `Filter` IR node and nothing may be left unattributed. Test: `test_merged_reductions_attribute_to_filter`.
5. **Sibling inputs (`concat`).** Two inputs lowered back to back must each claim only their own nodes; a window must not bleed into its sibling. Test: `test_concatenated_inputs_attribute_to_their_own_scans`.

---

### Task 1: Store the physical plan in a `DenseSlotMap`

Purely mechanical type change, done while `PhysSmBuilder` still exists (its inner map changes type too). This task also sets up the build environment and takes a green baseline, so any later failure is isolated to code.

**Files:**
- Modify: `crates/polars-stream/src/physical_plan/builder.rs:4,13,24,64,70`
- Modify: `crates/polars-stream/src/physical_plan/mod.rs:54,146,152,595,603-604,611-612,909,960,982,987`
- Modify: `crates/polars-stream/src/physical_plan/to_graph.rs:30,71,81`
- Modify: `crates/polars-stream/src/physical_plan/fmt.rs:14,180,945`
- Modify: `crates/polars-stream/src/physical_plan/to_description.rs:27,33`
- Modify: `crates/polars-stream/src/skeleton.rs:16,104`
- Test: `py-polars/tests/unit/lazyframe/test_query_monitoring.py` (existing tests only)

**Interfaces:**
- Consumes: nothing new.
- Produces: `phys_sm: DenseSlotMap<PhysNodeKey, PhysNode>` in every signature that previously said `SlotMap<PhysNodeKey, PhysNode>`; `PhysSmBuilder::Target = DenseSlotMap<PhysNodeKey, PhysNode>`; `build_physical_plan(..) -> PolarsResult<(PhysNodeKey, DenseSlotMap<PhysNodeKey, PhysNode>)>` (temporary shape, changed in Task 2 and again in Task 3).

- [ ] **Step 1: Create the venv and make sure `msgpack` is installed**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && make requirements
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && .venv/bin/python -c "import msgpack" || .venv/bin/uv pip install --python .venv/bin/python msgpack
```
Expected: both exit 0. Without `msgpack` every attribution test reports SKIPPED instead of FAILED, which hides the signal. If `uv` is missing, use `.venv/bin/python -m pip install msgpack`.

- [ ] **Step 2: Build debug `py-polars` and take a green baseline**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && make build
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -v
```
Expected: the build exits 0 (a from-scratch build takes tens of minutes; later rebuilds are incremental) and `../.venv/bin/python -c "import polars; print(polars.__file__)"` prints a path under `$W`. Every test in the file passes and none is skipped. If anything fails here, stop and report: the environment is wrong, not the code.

- [ ] **Step 3: Rename the type in every signature**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && perl -pi -e 's/\bSlotMap<PhysNodeKey, PhysNode>/DenseSlotMap<PhysNodeKey, PhysNode>/g; s/\bSlotMap::with_capacity_and_key/DenseSlotMap::with_capacity_and_key/g' crates/polars-stream/src/physical_plan/builder.rs crates/polars-stream/src/physical_plan/mod.rs crates/polars-stream/src/physical_plan/to_graph.rs crates/polars-stream/src/physical_plan/fmt.rs crates/polars-stream/src/physical_plan/to_description.rs crates/polars-stream/src/skeleton.rs
```
The `\b` keeps a second run from producing `DenseDenseSlotMap`. Expected: `grep -rnw SlotMap crates/polars-stream/src/physical_plan crates/polars-stream/src/skeleton.rs` now lists only the six `use slotmap::...` lines and the doc comment at `builder.rs:49`.

- [ ] **Step 4: Fix the imports**

Six edits, old line → new line:

`crates/polars-stream/src/physical_plan/builder.rs:4`
```rust
use slotmap::SlotMap;
```
→
```rust
use slotmap::DenseSlotMap;
```

`crates/polars-stream/src/physical_plan/mod.rs:54`
```rust
use slotmap::{SecondaryMap, SlotMap};
```
→
```rust
use slotmap::{DenseSlotMap, SecondaryMap};
```

`crates/polars-stream/src/physical_plan/to_graph.rs:30`
```rust
use slotmap::{SecondaryMap, SlotMap};
```
→
```rust
use slotmap::{DenseSlotMap, SecondaryMap};
```

`crates/polars-stream/src/physical_plan/fmt.rs:14`
```rust
use slotmap::{Key, SecondaryMap, SlotMap};
```
→
```rust
use slotmap::{DenseSlotMap, Key, SecondaryMap};
```

`crates/polars-stream/src/physical_plan/to_description.rs:27`
```rust
use slotmap::{Key, SlotMap};
```
→
```rust
use slotmap::{DenseSlotMap, Key};
```

`crates/polars-stream/src/skeleton.rs:16`
```rust
use slotmap::{SecondaryMap, SlotMap};
```
→
```rust
use slotmap::{DenseSlotMap, SecondaryMap};
```

Leave `builder.rs:49` ("This shadows `SlotMap::insert`") alone; the file is deleted in Task 3.

- [ ] **Step 5: Check the crate compiles cleanly**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && cargo fmt --all && cargo check -p polars-stream --all-features 2>&1 | grep -E "^(warning|error)|Finished"
```
Expected: a single `Finished` line and no `warning`/`error` lines. `DenseSlotMap` has the same `Index`, `get`, `get_mut`, `get_disjoint_mut`, `insert`, `remove`, `iter`, `keys`, `values`, and `with_capacity_and_key` as `SlotMap`, so nothing else should need touching. If `cargo check` names a method that does not exist on `DenseSlotMap`, stop and report it rather than working around it.

- [ ] **Step 6: Rebuild and rerun the acceptance tests**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && make build && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -v
```
Expected: all pass, none skipped.

- [ ] **Step 7: Commit**

```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && jj commit -m "$(printf 'refactor(rust): Store the physical plan in a DenseSlotMap\n\nThe IR attribution that follows finds the nodes lowered for one IR node by\nslicing the keys inserted since lowering it started, which needs keys that\niterate in insertion order.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV')"
```

---

### Task 2: Fill `phys_to_ir` beside the builder and assert the two agree

Introduces the external map and the length-diff rule while the old stamping mechanism is still in place. `physical_plan_to_description` switches to reading the map, so the Python tests exercise the new mechanism, and a transitional `debug_assert!` in `build_physical_plan` checks node by node that the map equals the stamped field. Every streaming collect in the test sweep then verifies equivalence.

**Files:**
- Modify: `py-polars/tests/unit/lazyframe/test_query_monitoring.py` (append five tests after `test_fused_drop_keeps_filter_attribution`, line 487)
- Modify: `crates/polars-stream/src/physical_plan/lower_ir.rs:37,162-241`
- Modify: `crates/polars-stream/src/physical_plan/lower_expr.rs:503-506`
- Modify: `crates/polars-stream/src/physical_plan/mod.rs:843-1016`
- Modify: `crates/polars-stream/src/physical_plan/to_description.rs:25-54`
- Modify: `crates/polars-stream/src/skeleton.rs:63-107,148-160,166-175`

**Interfaces:**
- Consumes: `PhysSmBuilder` (`is_original_ir_node`, `with_ir_node`, `insert`, `into_inner`, `Deref<Target = DenseSlotMap<PhysNodeKey, PhysNode>>`) from the branch; `PhysNode::ir_node() -> Option<Node>`.
- Produces:
  - `lower_ir(node, ir_arena, expr_arena, phys_sm: &mut PhysSmBuilder, phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>, original_ir_len: usize, schema_cache, expr_cache, cache_nodes, ctx, disable_morsel_split)`.
  - `insert_multiplexers(roots, phys_sm: &mut PhysSmBuilder, phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>)`, `split_multiplexers(..same..)`, `fuse_drops(roots, phys_sm: &mut DenseSlotMap<PhysNodeKey, PhysNode>, phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>)`.
  - `build_physical_plan(root, ir_arena, expr_arena, ctx) -> PolarsResult<(PhysNodeKey, DenseSlotMap<PhysNodeKey, PhysNode>, SecondaryMap<PhysNodeKey, Node>)>` (temporary three-tuple; Task 3 gives it its final shape).
  - `physical_plan_to_description(roots, phys_sm: &DenseSlotMap<..>, phys_to_ir: &SecondaryMap<PhysNodeKey, Node>, expr_arena)`.
  - `StreamingQuery.phys_to_ir: SecondaryMap<PhysNodeKey, Node>` (pub field).

- [ ] **Step 1: Add the five Review Focus tests and confirm they pass on the current mechanism**

Append after `test_fused_drop_keeps_filter_attribution` (ends at line 487 with `assert not [n for n in phys if n["properties"]["type"] == "SimpleProjection"]`), before `test_on_query_failed_called`:

```python
def _assert_all_attributed(
    ir: list[dict[str, Any]], phys: list[dict[str, Any]]
) -> None:
    """Every physical node names an IR node that is in the IR payload."""
    ir_ids = {node["id"] for node in ir}
    assert len(phys) > 0
    assert all(node["ir_node_id"] in ir_ids for node in phys)


def test_cache_ir_node_owns_no_physical_nodes() -> None:
    """Common-subplan elimination inserts `Cache` IR nodes that lower to nothing.

    The cached subtree's physical nodes belong to the IR nodes inside it, so no
    physical node carries a `Cache` id, and every id still resolves.
    """
    base = pl.LazyFrame({"a": [1, 2, 3, 4, 5]}).with_columns(b=pl.col("a") * 2)
    observer = _observe_streaming(
        lambda: pl.collect_all(
            [base.select(pl.col("b").sum()), base.filter(pl.col("b") > 4)],
            engine="streaming",
        )
    )
    ir, phys = _planned_payloads(observer)

    cache_ids = {n["id"] for n in ir if n["properties"]["type"] == "Cache"}
    # If this fails, the plan shape no longer triggers CSE: change the plan,
    # not the assertion.
    assert cache_ids
    _assert_all_attributed(ir, phys)
    assert not any(n["ir_node_id"] in cache_ids for n in phys)


def test_range_join_temporaries_attribute_to_join() -> None:
    """Lowering a range join appends temporary `Sort`/`HStack` IR nodes.

    Those never reach the observer, so their physical nodes must be claimed by
    the original join IR node rather than by an id the observer cannot resolve.
    """
    left = pl.LazyFrame({"a": [1, 5, 9]})
    right = pl.LazyFrame({"lo": [0, 4], "hi": [2, 6]})
    lf = left.join_where(
        right, pl.col("a") >= pl.col("lo"), pl.col("a") <= pl.col("hi")
    )
    observer = _observe_streaming(lambda: lf.collect(engine="streaming"))
    ir, phys = _planned_payloads(observer)

    _assert_all_attributed(ir, phys)
    join_ids = {
        n["id"] for n in ir if n["properties"]["type"] in ("Join", "IEJoin")
    }
    assert join_ids
    assert any(n["ir_node_id"] in join_ids for n in phys)


def test_in_memory_fallback_attributes_to_its_ir_node() -> None:
    """An order-preserving keep-last distinct lowers through an in-memory fallback.

    The fallback builds a throwaway IR arena; the physical node it inserts must
    still belong to the original `Distinct` IR node.
    """
    lf = pl.LazyFrame({"a": [1, 1, 2, 2, 3]}).unique(keep="last", maintain_order=True)
    observer = _observe_streaming(lambda: lf.collect(engine="streaming"))
    ir, phys = _planned_payloads(observer)

    _assert_all_attributed(ir, phys)
    (distinct_id,) = [n["id"] for n in ir if n["properties"]["type"] == "Distinct"]
    assert any(n["ir_node_id"] == distinct_id for n in phys)


def test_merged_reductions_attribute_to_filter() -> None:
    """Two reductions over the same input are folded into one `Reduce` node.

    Folding removes the per-reduction nodes from the plan while the filter is
    being lowered; the surviving node must still belong to the `Filter` IR node.
    """
    lf = pl.LazyFrame({"a": [1, 2, 3, 4, 5]}).filter(
        (pl.col("a") > pl.col("a").mean()) & (pl.col("a") < pl.col("a").max())
    )
    observer = _observe_streaming(lambda: lf.collect(engine="streaming"))
    ir, phys = _planned_payloads(observer)

    _assert_all_attributed(ir, phys)
    (filter_id,) = [n["id"] for n in ir if n["properties"]["type"] == "Filter"]
    reduces = [n for n in phys if n["properties"]["type"] == "Reduce"]
    assert len(reduces) == 1
    assert reduces[0]["ir_node_id"] == filter_id


def test_concatenated_inputs_attribute_to_their_own_scans() -> None:
    """Sibling inputs lowered back to back must not claim each other's nodes."""
    lf = pl.concat([pl.LazyFrame({"a": [1, 2]}), pl.LazyFrame({"a": [3, 4]})]).select(
        pl.col("a") * 2
    )
    observer = _observe_streaming(lambda: lf.collect(engine="streaming"))
    ir, phys = _planned_payloads(observer)

    _assert_all_attributed(ir, phys)
    scan_ids = [n["id"] for n in ir if n["properties"]["type"] == "DataFrameScan"]
    assert len(scan_ids) == 2
    for scan_id in scan_ids:
        assert sum(n["ir_node_id"] == scan_id for n in phys) == 1
```

`Any` is already imported at the top of the file (`from typing import TYPE_CHECKING, Any`), and `_observe_streaming` / `_planned_payloads` are the helpers defined just above the six acceptance tests.

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && ../.venv/bin/ruff format tests/unit/lazyframe/test_query_monitoring.py && ../.venv/bin/ruff check tests/unit/lazyframe/test_query_monitoring.py && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -v
```
Expected: ruff clean; every test passes. These five pass against the builder mechanism already; they pin what the rest of this task and Task 3 must reproduce. If one of them fails here, the plan shape it uses does not lower the way the spec assumes: report which one and the payload it saw, do not weaken the assertion.

- [ ] **Step 2: Thread `phys_to_ir` and `original_ir_len` through `lower_ir`**

In `crates/polars-stream/src/physical_plan/lower_ir.rs`:

Line 37, old:
```rust
use super::{PhysNode, PhysNodeKind, PhysSmBuilder, PhysStream};
```
new:
```rust
use slotmap::SecondaryMap;

use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysSmBuilder, PhysStream};
```
(`cargo fmt` will move the `slotmap` import up into the external-crate group; fine.)

Replace lines 162–208 (the doc comment, `lower_ir` signature, and body through the closing brace) with:
```rust
/// Lowers `node` and everything below it to physical nodes.
///
/// Every physical node inserted while lowering an IR node of the original plan is attributed
/// to that IR node in `phys_to_ir`. Lowering also appends temporary IR nodes to the arena and
/// lowers them recursively; those are not part of the plan the query observer sees, so nodes
/// lowered from them are attributed to the original IR node whose lowering created them.
#[recursive::recursive]
#[allow(clippy::too_many_arguments)]
pub fn lower_ir(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut PhysSmBuilder,
    phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>,
    original_ir_len: usize,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
    expr_cache: &mut ExprCache,
    cache_nodes: &mut PlHashMap<UniqueId, PhysStream>,
    ctx: StreamingLowerIRContext<'_>,
    disable_morsel_split: Option<bool>,
) -> PolarsResult<PhysStream> {
    // Every key at or beyond this position was inserted by lowering `node` or one of its
    // inputs. This relies on the slotmap being append-only while lowering: `DenseSlotMap::remove`
    // swaps the last key into the removed position, so removing a key below `len_before` would
    // hide a new node from this window. The one removal, in `simplify_input_streams`, only
    // removes nodes inserted during the current `lower_ir` call; see the comment there.
    let len_before = phys_sm.len();
    let out = if phys_sm.is_original_ir_node(node) {
        phys_sm.with_ir_node(node, |phys_sm| {
            lower_ir_inner(
                node,
                ir_arena,
                expr_arena,
                phys_sm,
                phys_to_ir,
                original_ir_len,
                schema_cache,
                expr_cache,
                cache_nodes,
                ctx,
                disable_morsel_split,
            )
        })
    } else {
        lower_ir_inner(
            node,
            ir_arena,
            expr_arena,
            phys_sm,
            phys_to_ir,
            original_ir_len,
            schema_cache,
            expr_cache,
            cache_nodes,
            ctx,
            disable_morsel_split,
        )
    };
    let out = out?;
    // Temporary IR nodes are not in the IR the observer sees; the physical nodes lowered from
    // them are claimed by the enclosing original node's window instead.
    if node.0 < original_ir_len {
        for key in phys_sm.keys().skip(len_before) {
            // A nested `lower_ir` call already claimed its own nodes, so the innermost
            // original IR node wins.
            if !phys_to_ir.contains_key(key) {
                phys_to_ir.insert(key, node);
            }
        }
    }
    Ok(out)
}
```

Then in `lower_ir_inner` (starts at the old line 210) add the two parameters after `phys_sm` and forward them in the macro. Old:
```rust
fn lower_ir_inner(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut PhysSmBuilder,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
```
new:
```rust
fn lower_ir_inner(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut PhysSmBuilder,
    phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>,
    original_ir_len: usize,
    schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
```
and in the `lower_ir!` macro, old:
```rust
            lower_ir(
                $input,
                ir_arena,
                expr_arena,
                phys_sm,
                schema_cache,
```
new:
```rust
            lower_ir(
                $input,
                ir_arena,
                expr_arena,
                phys_sm,
                phys_to_ir,
                original_ir_len,
                schema_cache,
```
The macro is the only recursive call site (`grep -n "lower_ir(" crates/polars-stream/src/physical_plan/lower_ir.rs` shows the definition and the macro body only), so nothing else in the file changes.

- [ ] **Step 3: State the removal rule at the one `remove`**

In `crates/polars-stream/src/physical_plan/lower_expr.rs`, old (lines 503–506):
```rust
                if *inner == orig_input {
                    combined_exprs.extend(exprs.iter().cloned());
                    ctx.phys_sm.remove(input_stream.node);
                    return false;
```
new:
```rust
                if *inner == orig_input {
                    combined_exprs.extend(exprs.iter().cloned());
                    // The only removal from `phys_sm` during lowering. The IR attribution in
                    // `lower_ir` slices the keys inserted since it started lowering a node, and
                    // `DenseSlotMap::remove` swaps the last key into the removed position, so a
                    // removal may only touch nodes inserted during the current `lower_ir` call.
                    // This one does: `lower_reduce_node` inserted these while lowering the same
                    // expression. Removing an older node here would silently attribute a new
                    // node to an outer IR node.
                    ctx.phys_sm.remove(input_stream.node);
                    return false;
```

- [ ] **Step 4: Write the map from the post-lowering passes and build the plan with it**

In `crates/polars-stream/src/physical_plan/mod.rs`:

`insert_multiplexers` (old lines 843–875): change the signature and add one map insert after the multiplexer is created. Old:
```rust
fn insert_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut PhysSmBuilder) {
```
new:
```rust
fn insert_multiplexers(
    roots: Vec<PhysNodeKey>,
    phys_sm: &mut PhysSmBuilder,
    phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>,
) {
```
and old:
```rust
            let multiplexer_node = phys_sm.with_ir_node(ir_node, |phys_sm| {
                phys_sm.insert(PhysNode::new_multi_output(
                    (0..refcount).map(|_| Arc::clone(&input_schema)).collect(),
                    PhysNodeKind::Multiplexer { input: stream },
                ))
            });
            (stream, PhysStream::first(multiplexer_node))
```
new:
```rust
            let multiplexer_node = phys_sm.with_ir_node(ir_node, |phys_sm| {
                phys_sm.insert(PhysNode::new_multi_output(
                    (0..refcount).map(|_| Arc::clone(&input_schema)).collect(),
                    PhysNodeKind::Multiplexer { input: stream },
                ))
            });
            let source_ir_node = phys_to_ir[stream.node];
            phys_to_ir.insert(multiplexer_node, source_ir_node);
            (stream, PhysStream::first(multiplexer_node))
```
Keep the existing `let ir_node = phys_sm[stream.node].ir_node().expect(..)` line: the old mechanism keeps deriving its value from the field, the new one from the map, so the assertion below compares two independent sources.

`split_multiplexers` (old lines 877–907): record the source's IR node next to the cloned node. Old:
```rust
fn split_multiplexers(roots: Vec<PhysNodeKey>, phys_sm: &mut PhysSmBuilder) {
    let mut refcount: SecondaryMap<PhysNodeKey, usize> = SecondaryMap::new();
    visit_node_inputs_mut(roots.clone(), phys_sm, |i| {
        *refcount.entry(i.node).unwrap().or_insert(0) += 1;
    });

    let mut split_map: SecondaryMap<PhysNodeKey, PhysNode> = SecondaryMap::new();
    for (k, n) in phys_sm.iter() {
        if let PhysNodeKind::Multiplexer { input } = n.kind {
            if let PhysNodeKind::InMemorySource { .. } = phys_sm[input.node].kind {
                split_map.insert(k, phys_sm[input.node].clone());
            }
        }
    }

    let mut replacements: SecondaryMap<PhysNodeKey, Vec<PhysStream>> = split_map
        .into_iter()
        .map(|(k, n)| {
            // The clones are the same source split per consumer; `insert` keeps the IR node
            // they already carry.
            let repls = (0..refcount[k]).map(|_| PhysStream::first(phys_sm.insert(n.clone())));
            (k, repls.collect())
        })
        .collect();
```
new:
```rust
fn split_multiplexers(
    roots: Vec<PhysNodeKey>,
    phys_sm: &mut PhysSmBuilder,
    phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>,
) {
    let mut refcount: SecondaryMap<PhysNodeKey, usize> = SecondaryMap::new();
    visit_node_inputs_mut(roots.clone(), phys_sm, |i| {
        *refcount.entry(i.node).unwrap().or_insert(0) += 1;
    });

    // The in-memory source to clone for each multiplexer, with the IR node it was lowered from.
    let mut split_map: SecondaryMap<PhysNodeKey, (PhysNode, Node)> = SecondaryMap::new();
    for (k, n) in phys_sm.iter() {
        if let PhysNodeKind::Multiplexer { input } = n.kind {
            if let PhysNodeKind::InMemorySource { .. } = phys_sm[input.node].kind {
                split_map.insert(k, (phys_sm[input.node].clone(), phys_to_ir[input.node]));
            }
        }
    }

    let mut replacements: SecondaryMap<PhysNodeKey, Vec<PhysStream>> = split_map
        .into_iter()
        .map(|(k, (n, source_ir_node))| {
            // The clones are the same source split per consumer, so each keeps its IR node.
            let repls = (0..refcount[k]).map(|_| {
                let clone = phys_sm.insert(n.clone());
                phys_to_ir.insert(clone, source_ir_node);
                PhysStream::first(clone)
            });
            (k, repls.collect())
        })
        .collect();
```
The rest of the function (the `visit_node_inputs_mut` that pops replacements) is unchanged.

`fuse_drops` (old lines 909–954): swap the map entries with the nodes. Old signature:
```rust
fn fuse_drops(roots: Vec<PhysNodeKey>, phys_sm: &mut DenseSlotMap<PhysNodeKey, PhysNode>) {
```
new:
```rust
fn fuse_drops(
    roots: Vec<PhysNodeKey>,
    phys_sm: &mut DenseSlotMap<PhysNodeKey, PhysNode>,
    phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>,
) {
```
and old (the last statement of the closure):
```rust
        if !has_rename && simple_proj_node.output_schemas.len() == 1 {
            std::mem::swap(simple_proj_node, input_node);
        }
```
new:
```rust
        if !has_rename && simple_proj_node.output_schemas.len() == 1 {
            std::mem::swap(simple_proj_node, input_node);
            // The filter now lives in the projection's slot; move the attribution with it so
            // the surviving node keeps the `Filter` IR node.
            let proj_ir_node = phys_to_ir[key];
            let filter_ir_node = phys_to_ir[input];
            phys_to_ir.insert(key, filter_ir_node);
            phys_to_ir.insert(input, proj_ir_node);
        }
```
(`input` was rebound to `input.node`, a `PhysNodeKey`, earlier in the closure.)

`build_physical_plan` (old lines 975–1016), replace whole function:
```rust
/// Lowers the IR rooted at `root` into a physical plan and returns the root physical node
/// together with the slotmap holding the plan and the IR node each physical node was lowered
/// from.
pub fn build_physical_plan(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    ctx: StreamingLowerIRContext<'_>,
) -> PolarsResult<(
    PhysNodeKey,
    DenseSlotMap<PhysNodeKey, PhysNode>,
    SecondaryMap<PhysNodeKey, Node>,
)> {
    // IR nodes at or beyond this index are added by lowering itself and are not part of the
    // plan the query observer sees.
    let original_ir_len = ir_arena.len();
    let mut schema_cache = PlHashMap::with_capacity(ir_arena.len());
    let mut expr_cache = ExprCache::with_capacity(expr_arena.len());
    let mut cache_nodes = PlHashMap::new();
    let mut phys_sm = PhysSmBuilder::new(
        DenseSlotMap::with_capacity_and_key(ir_arena.len()),
        original_ir_len,
    );
    let mut phys_to_ir: SecondaryMap<PhysNodeKey, Node> =
        SecondaryMap::with_capacity(ir_arena.len());
    let phys_root = lower_ir::lower_ir(
        root,
        ir_arena,
        expr_arena,
        &mut phys_sm,
        &mut phys_to_ir,
        original_ir_len,
        &mut schema_cache,
        &mut expr_cache,
        &mut cache_nodes,
        ctx,
        None,
    )?;
    insert_multiplexers(vec![phys_root.node], &mut phys_sm, &mut phys_to_ir);
    split_multiplexers(vec![phys_root.node], &mut phys_sm, &mut phys_to_ir);
    fuse_drops(vec![phys_root.node], &mut phys_sm, &mut phys_to_ir);

    // TODO: remove this after fusing pre-select into group-by node.
    rechunk_group_by_inputs(vec![phys_root.node], &mut phys_sm);

    // Transitional, removed with `PhysSmBuilder`: the external map must agree with the IR node
    // stamped on every physical node.
    debug_assert!(
        phys_sm.iter().all(|(key, node)| {
            node.ir_node()
                .is_some_and(|ir| phys_sm.is_original_ir_node(ir))
                && phys_to_ir.get(key).copied() == node.ir_node()
        }),
        "phys_to_ir must match the IR node stamped on every physical node"
    );

    Ok((phys_root.node, phys_sm.into_inner(), phys_to_ir))
}
```

- [ ] **Step 5: Read the map in `physical_plan_to_description` and carry it on `StreamingQuery`**

`crates/polars-stream/src/physical_plan/to_description.rs`, old (lines 25–35 and 46–54):
```rust
use polars_utils::arena::Arena;
use polars_utils::index::idxsize_to_u64;
use slotmap::{DenseSlotMap, Key};

use crate::{PhysNode, PhysNodeKey, PhysNodeKind};

pub fn physical_plan_to_description(
    roots: &[PhysNodeKey],
    phys_sm: &DenseSlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
) -> Vec<PhysicalNodeDescription> {
```
new:
```rust
use polars_utils::arena::{Arena, Node};
use polars_utils::index::idxsize_to_u64;
use slotmap::{DenseSlotMap, Key, SecondaryMap};

use crate::{PhysNode, PhysNodeKey, PhysNodeKind};

/// Describes the physical nodes reachable from `roots`. `phys_to_ir` maps each physical node
/// to the IR node it was lowered from; a node missing from it is described with no `ir_node_id`.
pub fn physical_plan_to_description(
    roots: &[PhysNodeKey],
    phys_sm: &DenseSlotMap<PhysNodeKey, PhysNode>,
    phys_to_ir: &SecondaryMap<PhysNodeKey, Node>,
    expr_arena: &Arena<AExpr>,
) -> Vec<PhysicalNodeDescription> {
```
and old:
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
new:
```rust
    while let Some(key) = queue.pop_front() {
        let (properties, inputs) = phys_props(phys_sm[key].kind(), expr_arena);
        let node = PhysicalNodeDescription {
            id: key.data().as_ffi(),
            input_ids: inputs.iter().map(|k| k.data().as_ffi()).collect(),
            ir_node_id: phys_to_ir.get(key).map(|n| n.0),
            properties,
        };
```

`crates/polars-stream/src/skeleton.rs`:

`to_planned_query`, old (lines 70–71):
```rust
        let physical =
            physical_plan_to_description(&[self.root_phys_node], &self.phys_sm, expr_arena);
```
new:
```rust
        let physical = physical_plan_to_description(
            &[self.root_phys_node],
            &self.phys_sm,
            &self.phys_to_ir,
            expr_arena,
        );
```

`visualize_physical_plan`, old (lines 92–93):
```rust
    let (root_phys_node, phys_sm) =
        crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, ctx)?;
```
new:
```rust
    let (root_phys_node, phys_sm, _phys_to_ir) =
        crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, ctx)?;
```

`StreamingQuery` struct, old (lines 103–105):
```rust
    pub root_phys_node: PhysNodeKey,
    pub phys_sm: DenseSlotMap<PhysNodeKey, PhysNode>,
    pub phys_to_graph: SecondaryMap<PhysNodeKey, GraphNodeKey>,
```
new:
```rust
    pub root_phys_node: PhysNodeKey,
    pub phys_sm: DenseSlotMap<PhysNodeKey, PhysNode>,
    /// The IR node each physical node in `phys_sm` was lowered from.
    pub phys_to_ir: SecondaryMap<PhysNodeKey, Node>,
    pub phys_to_graph: SecondaryMap<PhysNodeKey, GraphNodeKey>,
```

`StreamingQuery::build`, old (lines 148–149):
```rust
        let (root_phys_node, phys_sm) =
            crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, ctx)?;
```
new:
```rust
        let (root_phys_node, phys_sm, phys_to_ir) =
            crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, ctx)?;
```
and the struct literal a few lines below, old:
```rust
        let out = StreamingQuery {
            top_ir,
            graph,
            root_phys_node,
            phys_sm,
            phys_to_graph,
            metrics,
        };
```
new:
```rust
        let out = StreamingQuery {
            top_ir,
            graph,
            root_phys_node,
            phys_sm,
            phys_to_ir,
            phys_to_graph,
            metrics,
        };
```

`StreamingQuery::execute` destructures every field; old:
```rust
        let StreamingQuery {
            top_ir,
            mut graph,
            root_phys_node,
            phys_sm,
            phys_to_graph,
            metrics,
        } = self;
```
new:
```rust
        let StreamingQuery {
            top_ir,
            mut graph,
            root_phys_node,
            phys_sm,
            phys_to_ir: _,
            phys_to_graph,
            metrics,
        } = self;
```

- [ ] **Step 6: Check the crate compiles cleanly**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && cargo fmt --all && cargo check -p polars-stream --all-features 2>&1 | grep -E "^(warning|error)|Finished"
```
Expected: a single `Finished` line and no `warning`/`error` lines. The likely mistakes are a missed `phys_to_ir, original_ir_len,` in the `lower_ir!` macro (error: wrong number of arguments) and a forgotten `phys_to_ir: _` in the `execute` destructure (error: pattern does not mention field).

- [ ] **Step 7: Rebuild, run the attribution tests, then the sweep that exercises the equivalence assertion**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && make build && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -v
```
Expected: all eleven attribution tests (six acceptance, five Review Focus) and the rest of the file pass.

Then:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && ../.venv/bin/pytest tests/unit/lazyframe tests/unit/streaming -n auto -q
```
Expected: all pass. Every streaming collect in these directories runs `build_physical_plan` and therefore the node-by-node equivalence `debug_assert!`; a mismatch panics with `phys_to_ir must match the IR node stamped on every physical node`. If it does, report the failing test and do not proceed to Task 3: the length diff and the builder disagree for that plan shape, and the difference has to be understood first.

- [ ] **Step 8: Commit**

```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && jj commit -m "$(printf 'feat(rust): Fill an external IR attribution map during lowering\n\nlower_ir records the slotmap length before lowering a node and attributes\nevery key added since to the innermost original IR node. The post-lowering\npasses write the map directly. The map is filled beside the existing builder\nfor now, and build_physical_plan asserts in debug builds that the two agree.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV')"
```

---

### Task 3: Drop the intrusive attribution

Delete `PhysSmBuilder` and `PhysNode::ir_node`, give every lowering signature the plain `DenseSlotMap` back, and give `build_physical_plan` its final shape.

**Files:**
- Delete: `crates/polars-stream/src/physical_plan/builder.rs`
- Modify: `crates/polars-stream/src/physical_plan/mod.rs:34,43,78-115,843-1016`
- Modify: `crates/polars-stream/src/physical_plan/lower_ir.rs:37,55,84,140,162-241,1951,2002`
- Modify: `crates/polars-stream/src/physical_plan/lower_expr.rs:30-32,61,2810,2840,2861`
- Modify: `crates/polars-stream/src/physical_plan/lower_group_by.rs:24-26,67,596,759,1111,1290`
- Modify: `crates/polars-stream/src/skeleton.rs:80-98,138-160`

**Interfaces:**
- Consumes: everything Task 2 produced.
- Produces (final):
  - `lower_ir(node, ir_arena, expr_arena, phys_sm: &mut DenseSlotMap<PhysNodeKey, PhysNode>, phys_to_ir: &mut SecondaryMap<PhysNodeKey, Node>, original_ir_len: usize, schema_cache, expr_cache, cache_nodes, ctx, disable_morsel_split)`.
  - `build_physical_plan(root, ir_arena, expr_arena, phys_sm: &mut DenseSlotMap<PhysNodeKey, PhysNode>, ctx) -> PolarsResult<(PhysNodeKey, SecondaryMap<PhysNodeKey, Node>)>`.
  - `PhysNode` with only `output_schemas` and `kind`; no `ir_node()`.
  - No `PhysSmBuilder` anywhere.

- [ ] **Step 1: Delete the builder and its re-export**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && rm crates/polars-stream/src/physical_plan/builder.rs
```
In `crates/polars-stream/src/physical_plan/mod.rs` delete line 34 (`mod builder;`) and line 43 (`pub use builder::PhysSmBuilder;`).

- [ ] **Step 2: Remove the field and getter from `PhysNode`**

In `crates/polars-stream/src/physical_plan/mod.rs`, old (lines 80–115):
```rust
pub struct PhysNode {
    output_schemas: UnitVec<Arc<Schema>>,
    kind: PhysNodeKind,
    /// The IR node whose lowering created this node. Set by `PhysSmBuilder::insert`; always
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

    pub fn output_schema(&self, port_idx: usize) -> &Arc<Schema> {
```
new:
```rust
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

    pub fn output_schema(&self, port_idx: usize) -> &Arc<Schema> {
```
This is exactly `main`'s `PhysNode`.

- [ ] **Step 3: Give every lowering signature the plain slotmap back**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && perl -pi -e "s/(&(?:'a )?mut) PhysSmBuilder\b/\$1 DenseSlotMap<PhysNodeKey, PhysNode>/g" crates/polars-stream/src/physical_plan/lower_ir.rs crates/polars-stream/src/physical_plan/lower_expr.rs crates/polars-stream/src/physical_plan/lower_group_by.rs crates/polars-stream/src/physical_plan/mod.rs
```
This rewrites `&mut PhysSmBuilder` (18 signatures across the three lowering files plus `insert_multiplexers`/`split_multiplexers`) and the `&'a mut PhysSmBuilder` on `LowerExprContext::phys_sm`. Expected: `grep -rn PhysSmBuilder crates/polars-stream/src` now lists only the three `use super::{..}` lines, `mod.rs`'s `PhysSmBuilder::new(` in `build_physical_plan`, and the transitional comment in `build_physical_plan`.

Then fix the imports:

`crates/polars-stream/src/physical_plan/lower_ir.rs`, old:
```rust
use slotmap::SecondaryMap;

use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysSmBuilder, PhysStream};
```
new:
```rust
use slotmap::{DenseSlotMap, SecondaryMap};

use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysStream};
```
(`cargo fmt` may have already moved the `slotmap` line up into the external-crate group after Task 2; edit it wherever it is.)

`crates/polars-stream/src/physical_plan/lower_expr.rs`, old (lines 30–32):
```rust
use super::{
    PhysNode, PhysNodeKey, PhysNodeKind, PhysSmBuilder, PhysStream, StreamingLowerIRContext,
};
```
new:
```rust
use slotmap::DenseSlotMap;

use super::{PhysNode, PhysNodeKey, PhysNodeKind, PhysStream, StreamingLowerIRContext};
```

`crates/polars-stream/src/physical_plan/lower_group_by.rs`, old (lines 24–26):
```rust
use super::{
    ExprCache, PhysNode, PhysNodeKind, PhysSmBuilder, PhysStream, StreamingLowerIRContext,
};
```
new:
```rust
use slotmap::DenseSlotMap;

use super::{ExprCache, PhysNode, PhysNodeKey, PhysNodeKind, PhysStream, StreamingLowerIRContext};
```
(`PhysNodeKey` comes back because the type now names it.)

- [ ] **Step 4: Simplify the `lower_ir` wrapper**

In `crates/polars-stream/src/physical_plan/lower_ir.rs` replace the body of `lower_ir` written in Task 2 — from the `// Every key at or beyond this position ...` comment above `let len_before = phys_sm.len();` through `Ok(out)` — with:
```rust
    // Every key at or beyond this position was inserted by lowering `node` or one of its
    // inputs. This relies on the slotmap being append-only while lowering: `DenseSlotMap::remove`
    // swaps the last key into the removed position, so removing a key below `len_before` would
    // hide a new node from this window. The one removal, in `simplify_input_streams`, only
    // removes nodes inserted during the current `lower_ir` call; see the comment there.
    let len_before = phys_sm.len();
    let out = lower_ir_inner(
        node,
        ir_arena,
        expr_arena,
        phys_sm,
        phys_to_ir,
        original_ir_len,
        schema_cache,
        expr_cache,
        cache_nodes,
        ctx,
        disable_morsel_split,
    )?;
    // Temporary IR nodes are not in the IR the observer sees; the physical nodes lowered from
    // them are claimed by the enclosing original node's window instead.
    if node.0 < original_ir_len {
        for key in phys_sm.keys().skip(len_before) {
            // A nested `lower_ir` call already claimed its own nodes, so the innermost
            // original IR node wins.
            if !phys_to_ir.contains_key(key) {
                phys_to_ir.insert(key, node);
            }
        }
    }
    Ok(out)
```
The signature and doc comment from Task 2 stay; only `phys_sm`'s type changed in Step 3.

- [ ] **Step 5: Stop stamping in `insert_multiplexers` and finish `build_physical_plan`**

In `crates/polars-stream/src/physical_plan/mod.rs`, `insert_multiplexers`, old:
```rust
            let input_schema = Arc::clone(stream.output_schema(phys_sm));
            // A multiplexer only fans out the stream it wraps, so it belongs to the same IR
            // node as that stream's producer.
            let ir_node = phys_sm[stream.node]
                .ir_node()
                .expect("lowered physical nodes are attributed to an IR node");
            let multiplexer_node = phys_sm.with_ir_node(ir_node, |phys_sm| {
                phys_sm.insert(PhysNode::new_multi_output(
                    (0..refcount).map(|_| Arc::clone(&input_schema)).collect(),
                    PhysNodeKind::Multiplexer { input: stream },
                ))
            });
            let source_ir_node = phys_to_ir[stream.node];
            phys_to_ir.insert(multiplexer_node, source_ir_node);
            (stream, PhysStream::first(multiplexer_node))
```
new:
```rust
            let input_schema = Arc::clone(stream.output_schema(phys_sm));
            let multiplexer_node = phys_sm.insert(PhysNode::new_multi_output(
                (0..refcount).map(|_| Arc::clone(&input_schema)).collect(),
                PhysNodeKind::Multiplexer { input: stream },
            ));
            // A multiplexer only fans out the stream it wraps, so it belongs to the same IR
            // node as that stream's producer.
            let source_ir_node = phys_to_ir[stream.node];
            phys_to_ir.insert(multiplexer_node, source_ir_node);
            (stream, PhysStream::first(multiplexer_node))
```

`build_physical_plan`, replace the whole function from Task 2 with its final form:
```rust
/// Lowers the IR rooted at `root` into physical nodes in `phys_sm` and returns the root
/// physical node together with the IR node each physical node was lowered from.
pub fn build_physical_plan(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    phys_sm: &mut DenseSlotMap<PhysNodeKey, PhysNode>,
    ctx: StreamingLowerIRContext<'_>,
) -> PolarsResult<(PhysNodeKey, SecondaryMap<PhysNodeKey, Node>)> {
    // IR nodes at or beyond this index are added by lowering itself and are not part of the
    // plan the query observer sees.
    let original_ir_len = ir_arena.len();
    let mut schema_cache = PlHashMap::with_capacity(ir_arena.len());
    let mut expr_cache = ExprCache::with_capacity(expr_arena.len());
    let mut cache_nodes = PlHashMap::new();
    let mut phys_to_ir: SecondaryMap<PhysNodeKey, Node> =
        SecondaryMap::with_capacity(ir_arena.len());
    let phys_root = lower_ir::lower_ir(
        root,
        ir_arena,
        expr_arena,
        phys_sm,
        &mut phys_to_ir,
        original_ir_len,
        &mut schema_cache,
        &mut expr_cache,
        &mut cache_nodes,
        ctx,
        None,
    )?;
    insert_multiplexers(vec![phys_root.node], phys_sm, &mut phys_to_ir);
    split_multiplexers(vec![phys_root.node], phys_sm, &mut phys_to_ir);
    fuse_drops(vec![phys_root.node], phys_sm, &mut phys_to_ir);

    // TODO: remove this after fusing pre-select into group-by node.
    rechunk_group_by_inputs(vec![phys_root.node], phys_sm);

    // Guards the passes above: a pass that inserts a node without attributing it fails here.
    // It cannot detect a node claimed by the wrong `lower_ir` window, since the root's window
    // starts at 0 and claims anything left over.
    debug_assert!(
        phys_sm
            .keys()
            .all(|key| phys_to_ir.get(key).is_some_and(|ir| ir.0 < original_ir_len)),
        "every physical node must be attributed to an IR node of the original plan"
    );

    Ok((phys_root.node, phys_to_ir))
}
```

- [ ] **Step 6: Let the callers own the slotmap again**

In `crates/polars-stream/src/skeleton.rs`:

`visualize_physical_plan`, old:
```rust
    let (root_phys_node, phys_sm, _phys_to_ir) =
        crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, ctx)?;
```
new:
```rust
    let mut phys_sm = DenseSlotMap::with_capacity_and_key(ir_arena.len());
    let (root_phys_node, _phys_to_ir) = crate::physical_plan::build_physical_plan(
        node,
        ir_arena,
        expr_arena,
        &mut phys_sm,
        ctx,
    )?;
```

`StreamingQuery::build`, old:
```rust
        let (root_phys_node, phys_sm, phys_to_ir) =
            crate::physical_plan::build_physical_plan(node, ir_arena, expr_arena, ctx)?;
```
new:
```rust
        let mut phys_sm = DenseSlotMap::with_capacity_and_key(ir_arena.len());
        let (root_phys_node, phys_to_ir) = crate::physical_plan::build_physical_plan(
            node,
            ir_arena,
            expr_arena,
            &mut phys_sm,
            ctx,
        )?;
```
This is `main`'s shape with the map added to the return.

- [ ] **Step 7: Confirm nothing intrusive is left and the crate compiles cleanly**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && grep -rn "PhysSmBuilder\|with_ir_node\|is_original_ir_node\|\.ir_node()\|ir_node: " crates/polars-stream/src crates/polars-descriptions/src; ls crates/polars-stream/src/physical_plan/builder.rs
```
Expected: the grep prints only `crates/polars-stream/src/skeleton.rs:65:        ir_node: Node,` (the parameter of `to_planned_query`, unrelated), and `ls` reports no such file.

Then:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && cargo fmt --all && cargo check -p polars-stream --all-features 2>&1 | grep -E "^(warning|error)|Finished"
```
Expected: a single `Finished` line and no `warning`/`error` lines. An `unused import` warning for `Node` or `DenseSlotMap` in one of the lowering files means a `use` line in Step 3 is wrong for that file; fix the import, do not add `#[allow]`.

- [ ] **Step 8: Rebuild and run the attribution tests**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && make build && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -v
```
Expected: all pass, none skipped. The eleven attribution tests now run against the external map alone.

- [ ] **Step 9: Commit**

```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && jj commit -m "$(printf 'refactor(rust): Drop the intrusive IR attribution from PhysNode\n\nPhysSmBuilder and PhysNode::ir_node are gone. Lowering takes the plain\nDenseSlotMap again, build_physical_plan takes it from the caller as before\nand returns phys_to_ir next to the root, and the map is the only attribution.\n\nCo-Authored-By: Claude Fable 5.1 <noreply@anthropic.com>\nClaude-Session: https://claude.ai/code/session_01XwgYakUsCvyAZYjWXjbAZV')"
```

---

### Task 4: Full verification

No code changes unless clippy or the sweep finds something.

**Files:**
- Modify: only if a fix is needed, in the files Tasks 1–3 touched.

- [ ] **Step 1: Clippy and format check with the repository's flags**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && cargo fmt --all -- --check && cargo clippy -p polars-stream -p polars-descriptions --all-targets --all-features -- -W clippy::dbg_macro -D warnings
```
Expected: exits 0. If clippy flags something in code this plan wrote, fix it in place, rerun, and commit as `fix(rust): Address clippy fallout for IR attribution` with the standard trailers. Do not silence lints with `#[allow]`.

- [ ] **Step 2: The wider Python sweep**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework/py-polars && ../.venv/bin/pytest tests/unit/lazyframe/test_query_monitoring.py -v && ../.venv/bin/pytest tests/unit/lazyframe tests/unit/streaming tests/unit/operations -n auto -q
```
Expected: all pass. Every streaming collect runs the final `debug_assert!` in `build_physical_plan`. If it runs longer than about thirty minutes, stop it and say so in the final report along with how far it got.

- [ ] **Step 3: Confirm the branch shape and the spec's verification list**

Run:
```bash
cd /private/tmp/claude-501/-Volumes-sourcecode-polars/21e902e2-9137-42c9-bec9-0ec8301a7813/scratchpad/ws-provenance-rework && jj log -r 'louis/attribute_physical_to_ir..@' --no-graph -T 'change_id.short() ++ " " ++ description.first_line() ++ "\n"' && jj diff --from louis/attribute_physical_to_ir --to @- --stat
```
Expected: four described commits on top of the bookmark (spec, Task 1, Task 2, Task 3, plus a clippy fix if Step 1 needed one) and `@` empty. The stat shows `builder.rs` deleted and no change under `crates/polars-descriptions`.

Report the exact test counts from Steps 1–2 and the `jj log` output. Do not move the bookmark `louis/attribute_physical_to_ir`; that is a decision for the human partner after review.
