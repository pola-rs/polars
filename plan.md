
## Context

`polars-sql` (crates/polars-sql) is a SQL transpiler: queries are parsed with sqlparser-rs 0.62 and translated to LazyFrames — there is no separate SQL engine. Coverage is already substantial (joins incl. semi/anti/inequality, set ops, non-recursive CTEs, DISTINCT ON, QUALIFY, ~99 functions), but well short of the full SELECT surface, and correctness is only checked by hand-written tests plus a home-grown Python harness (`assert_sql_matches` in `py-polars/tests/unit/sql/asserts.py`, ~257 uses) that cross-checks results against in-memory SQLite/DuckDB. There is no external conformance suite in the repo.

**Agreed scope** (per user): full SELECT/read-query surface only — no INSERT/UPDATE/MERGE, expanded DDL, or transactions. Conformance testing via **sqllogictest** (the corpus is not in this repo; vendor a subset). Where dialects diverge, align semantics with **PostgreSQL**.

**Strategy**: stand up the sqllogictest harness *first*, so every gap-closing PR is measured by conformance pass-rate and the backlog is ranked empirically rather than by intuition.

Key verified facts that shape the plan:
- Scalar-subquery plumbing already exists: `Expr::SubPlan` + `SQLContext::process_subqueries` (context.rs:2014-2056) rewrite subplans via `first()` + broadcast hconcat. Comparisons like `x > (SELECT …)` are blocked only by an explicit bail in `visit_binary_op` (sql_expr.rs:567-577).
- Window frames are blocked only by `validate_window_frame` (functions.rs:1860-1900); Polars already has `rolling_*` with `RollingOptionsFixedWindow` and value-based `rolling_*_by`, enough to express bounded ROWS and numeric RANGE frames. The documented ROWS-vs-RANGE default-semantics mismatch is at functions.rs:1902-1924.
- sqlparser 0.62 already *parses* everything needed (RANGE/GROUPS frames, GROUPING SETS, RECURSIVE, LATERAL); all gaps are in translation.
- The SQLite sqllogictest corpus sets up data via CREATE TABLE + INSERT — the harness needs a small test-only INSERT shim so product scope stays read-only.

---

## Phase 1 — sqllogictest harness (M) — do first

New unpublished workspace member `crates/polars-sqllogictest/` (mirrors DataFusion's layout; keeps the corpus out of the per-PR `cargo test -p polars-sql` path):

```
crates/polars-sqllogictest/
  src/main.rs        # runner binary ([[test]] harness = false), filters, --bless
  src/engine.rs      # impl sqllogictest::DB for PolarsEngine (sync trait; Polars is sync)
  src/setup.rs       # test-only CREATE TABLE / INSERT INTO … VALUES / DROP shim
                     #   (builds eager DataFrames, SQLContext::register)
  src/output.rs      # DataFrame → DBOutput: int/bool→I, float/decimal→R, else→T;
                     #   NULL, (empty), %.3f floats per sqllogictest convention
  src/expected.rs    # expected-failures baseline: load/compare/ratchet
  slt/polars/        # own-authored .slt corpus (port feature areas from py tests)
  slt/sqlite/        # Phase 2: vendored sqlite subset + UPSTREAM pin file
  expected_failures.txt
```

- Queries route through `SQLContext::execute(sql)?.collect()`; setup statements route to the shim. `rowsort`/hashing handled by the sqllogictest Runner.
- **Ratchet CI policy**: failure not in baseline → red; baseline entry now passing → red ("remove it"); print `passed / expected-fail / total` pass-rate every run. `skipif polars` reserved for *intentional* dialect divergences (e.g. integer division), baseline file for *not-yet-implemented*.
- CI: add a step/job to `.github/workflows/test-rust.yml` running `cargo run -p polars-sqllogictest --release`. No network needed (vendored corpus). Optional nightly job fetches the full external corpus (pinned URL + sha256) and runs report-only.

## Phase 2 — corpus import & baseline (M) — DONE

Vendored 13 files (`select1–4`, `in1`/`in2`, SELECT-relevant `slt_lang_*`, 4 truncated `random/` samples) from `gregrahn/sqllogictest` @ `c67f97b`; provenance + all vendoring modifications in `slt/sqlite/UPSTREAM`. Baseline: **4829 passed / 5200 expected-fail / 10029 total (48.2%)**; sqlite corpus alone 47.6%. 160 `skipif polars` records documented as intentional PostgreSQL-alignment divergences (integer division #27391, boolean rendering, empty `IN ()`). `select5.test` dropped: N-way implicit joins (4–64 tables) OOM — see 3h.

Empirical ranking (failing records): subqueries ~2900 · self-join duplicate-column collisions 1309 (new, → 3h) · chained set ops 878 · DISTINCT-in-aggregates 25 · small correctness bugs (→ 3h) · window/GROUPING SETS ≈ 0 corpus weight (keep, but urgency drops).

## Phase 3 — gap closing (parallelizable after Phase 2)

Items are **[transpiler]** (polars-sql only) unless marked **[engine]** (cross-crate, longer lead time — raise early with polars-plan owners).

Empirical priority order (by Phase 2 failure count): 3a subqueries → 3h join-column collisions → 3e chained set ops → everything else.

### 3a. Subqueries (M staged; general case XL)
- Scalar-subquery comparisons — **DONE**: bail removed, subquery operands route through `Expr::SubPlan` (`SubqueryRestriction::SingleValue`), `process_subqueries` covers SELECT-list/HAVING.
- PostgreSQL ">1 row is an error" guard [S] — **not deferred after all**: replace the `first()` reduction in the scalar-subquery path with `Expr::item(allow_empty: true)` (`AggExpr::Item`, polars-plan/src/dsl/mod.rs:192), which errors on multiple values and yields NULL for zero rows — exactly PostgreSQL scalar-subquery semantics. Swap both the `SingleValue` reduce in `visit_subquery` and the placeholder rewrite in `process_subqueries`; update the multi-row test that currently pins first()-behavior to expect an error.
- Subquery with its own WITH — **DONE** (visit_subquery uses the CTE-aware execute path).
- Correlated scalar subqueries, Stage 1 [M]: extend the equi-correlation decorrelation in `subquery.rs` (today IN/EXISTS only) to scalar subqueries → `group_by(outer keys).agg` + left join. Stage 2 (general decorrelation / dependent join in polars-plan IR) is **[engine, XL]** — its own tracked project, not blocking v1.

### 3b. Window functions (L; one engine item)
All in `functions.rs` unless noted:
- Bounded ROWS frames [M]: `n PRECEDING`/`m FOLLOWING`/`UNBOUNDED FOLLOWING` via `rolling_*(window_size, min_periods=1)` + `shift`/`reverse` composites inside `.over_with_options`. Enable rolling feature flags in polars-sql/Cargo.toml.
- Correct default RANGE peer semantics (fixes the documented mismatch) [M]: cumulative result then broadcast last value per peer group: `.last().over(partition ++ order_keys)`.
- RANGE with numeric offsets [M]: map to `rolling_*_by` (`RollingOptionsDynamicWindow`). GROUPS frames [M/L]: dense-rank of order keys + rolling over the rank; do last (low corpus weight).
- Mixed ASC/DESC in OVER ORDER BY **[engine, M]**: `over_with_options` (polars-plan/src/dsl/mod.rs:825) takes a single `SortOptions` for all keys → needs per-key `SortMultipleOptions` through DSL/IR + polars-expr window impl.
- New functions [S each]: `ntile`, `nth_value`, `percent_rank`, `cume_dist` — all expressible with existing rank/len/row_number exprs (match PostgreSQL bucket semantics).
- `IGNORE/RESPECT NULLS` [M] via fill compositions; `FILTER (WHERE …) OVER` [S] via `when(pred).then(arg)` pre-agg; `WITHIN GROUP` for percentile_cont/disc [S/M] via `quantile`.

### 3c. Function library expansion (L total; embarrassingly parallel)
Additive batches in `functions.rs` (enum + aliases + visitor arm), each landing with its own `.slt` file, prioritized by Phase 2 failure counts. Confirmed missing: `regexp_replace`/`regexp_extract`, `date_trunc`, `to_char`, `generate_series` (as table function in `table_functions.rs` → `int_range`/`date_range`), `make_date`, `age`, `translate`, `to_hex`, `gcd`/`lcm`, `random`, etc. — all map onto existing Polars exprs.

### 3d. GROUPING SETS / ROLLUP / CUBE (M)
In `context.rs` group-by path: expand to grouping-set list → one aggregation per set over a shared `lf.cache()` input → diagonal `concat` with null-filled keys; `GROUPING()` as per-set constant bitmask. Make the diagonal-concat feature non-optional for polars-sql.

### 3e. Small semantics items (S each; set-ops promoted to M by corpus weight)
- Chained/mixed set operations (3+ `UNION`/`EXCEPT`/`INTERSECT` terms, 878 corpus failures) and `EXCEPT`/`INTERSECT` at all (currently rejected): occurrence-index column (`cum_count().over(all cols)`) added to the anti/semi join keys in `process_except_intersect` (context.rs:533-576).
- `FETCH … WITH TIES`: rank-based filter after sort; `FETCH … PERCENT`: `row_number <= ceil(len * p / 100)`.
- Expression gaps in `sql_expr.rs`: LIKE `ESCAPE`, array slicing (`Subscript::Slice` → `list().slice`), TRIM custom charset, CAST FORMAT (common templates → strftime), negative interval strings, chained field access.
- `IN`/`NOT IN` three-valued logic: NULL in the list with no match must yield NULL, not false/true (~7 corpus records). IN-list elements that are expressions, not literals (~8).
- `TIME WITH TIME ZONE`: keep rejecting with a clear message; document as intentional (PostgreSQL itself discourages it).

### 3f. Recursive CTEs (M/L)
In `register_ctes` (context.rs): split anchor/recursive terms; iterate eagerly (collect anchor → re-register → collect recursive term → append; unique+fixpoint for UNION, iteration cap for safety); register materialized result. Document that this executes at planning time (unavoidable without engine-level iteration).

### 3h. New from Phase 2 triage
- Unaliased-expression output-name collisions in the SELECT list [M] — the cluster Phase 2 triage mislabeled as "self-join column collisions" (self-joins/multi-table FROM actually work; the random/select corpus has zero baseline entries). Real repro: `SELECT a-b, a FROM t1` and `SELECT a, abs(a) FROM t1` fail with "duplicate name passed to with_columns" because unaliased expressions take their left operand/argument as output name. Valid SQL allows repeated/derived output names. Fix in context.rs SELECT-list projection: assign unique internal names to colliding unaliased expressions (positional disambiguation), keeping first-occurrence display names. Huge share of the select1–4 baseline (`SELECT a-b, abs(a), a+b*2 …` patterns throughout).
- Correlated scalar subqueries are far bigger than ranked: `(SELECT count(*) FROM t1 AS x WHERE x.b<t1.b)`-style records appear 1784× in select3, 501× in select1, 475× in select2 — 3a Stage 1 (equi-correlation decorrelation) is likely the single largest pass-rate lever and should be pulled forward.
- Correctness bugs [S each, do early — they're wrong answers, not missing features]:
  - `-COUNT(*)` unsigned wraparound (renders `4294967293` for `-3`); negation of unsigned agg outputs must cast.
  - `SUM(<literal>)` returns the literal instead of literal × row count.
  - `SUM`/`MIN` over empty or all-NULL group must return NULL (currently `0`/placeholder).
- DISTINCT inside aggregates (`SUM(DISTINCT x)` etc., 25 corpus failures) — fold into 3c batches.
- N-way implicit join lowering **[engine, L]**: `FROM t1,…,tN WHERE <equalities>` lowers to filtered cross product; ≥~20 tables OOMs (why select5.test is dropped). Needs join-reordering / predicate-pushdown-into-join at planning; raise with polars-plan owners alongside the other engine items.

### 3g. Deferred / own track (L–XL)
- LATERAL Stage 1: table functions + `UNNEST … WITH ORDINALITY` (extend existing CROSS JOIN UNNEST path; ordinality = row-index column). Stage 2 (general LATERAL) = same dependent-join engine problem as 3a Stage 2 — defer jointly **[engine, XL]**.
- TABLESAMPLE: BERNOULLI via seeded `random() < p` filter; low priority.

## Phase 4 — Maintenance (ongoing)

- `expected_failures.txt` line count is the burn-down metric; ratchet enforced by CI (unexpected pass = red). Baseline growth in a PR requires justification.
- Runner emits JSON + GitHub step-summary (pass-rate overall and per directory); nightly full-corpus job uploads it as an artifact.
- Every Phase 3 PR ships `.slt` coverage for its feature. Keep `assert_sql_matches` as the Python-side semantic oracle; add `duckdb` to `requirements-dev.txt` so the DuckDB comparisons run locally, not just in CI.
- Vendored-corpus / sqlparser bumps are deliberate re-baselining PRs (UPSTREAM pin file records provenance).

## Sequencing & effort

| Phase | Size | Notes |
|---|---|---|
| 1 Harness + CI | M | **DONE** (96-record own corpus, ratchet, CI job) |
| 2 Corpus + triage | M | **DONE** (48.2% baseline, ranked backlog below) |
| 3a Subqueries (staged) | M | ~2900 corpus failures — top priority |
| 3h Join-column collisions + correctness bugs | M | 1309 failures + 3 wrong-answer bugs — second priority |
| 3e Small semantics (set ops first) | M + ~5×S | Chained set ops = 878 failures |
| 3c Function batches | L | Many independent S items; fully parallel |
| 3b Window frames/fns | L | ~0 corpus weight; keep for SELECT-surface completeness |
| 3d GROUPING SETS | M | ~0 corpus weight; context.rs |
| 3f Recursive CTEs | M/L | context.rs |
| 3g LATERAL/general decorrelation + N-way join reordering | XL | Engine track; not blocking v1 |

Only hard ordering: 1 → 2 → 3. Within Phase 3 the work splits cleanly by file (functions.rs vs context.rs vs sql_expr.rs+subquery.rs) with low conflict surface. Start the two engine dependencies (per-key OVER ordering; dependent join) early — their lead time dominates.

## Verification

- Phase 1 done when: `cargo run -p polars-sqllogictest --release` runs the own-authored `slt/polars/` corpus green in CI, and deliberately breaking a query turns CI red.
- Phase 2 done when: vendored sqlite subset runs with a committed baseline and a printed pass-rate.
- Each Phase 3 item done when: its `.slt` tests pass, corresponding `expected_failures.txt` entries are removed (pass-rate strictly increases), new Rust tests in `crates/polars-sql/tests/` and Python tests in `py-polars/tests/unit/sql/` (using `assert_sql_matches` against DuckDB/SQLite where semantics allow) pass, and `cargo test -p polars-sql --all-features` + `pytest py-polars/tests/unit/sql` stay green.
- Semantic spot-checks against PostgreSQL for divergence-prone areas (window peer semantics, division, NULL ordering) via the comparison harness or documented manual checks.





