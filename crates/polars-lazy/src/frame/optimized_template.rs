use polars_core::prelude::*;
use polars_expr::state::ExecutionState;
use polars_mem_engine::create_physical_plan;
use polars_utils::aliases::PlHashMap;
use polars_utils::pl_str::PlSmallStr;

use super::BUILD_STREAMING_EXECUTOR;
use crate::prelude::*;

/// Cached information about a single PlaceholderScan node in the optimized IR.
#[derive(Clone)]
struct PlaceholderInfo {
    /// Arena index where this PlaceholderScan lives.
    node: Node,
    /// The full schema declared by the placeholder.
    schema: SchemaRef,
    /// The projected output schema (set by projection pushdown), or None if all columns are needed.
    output_schema: Option<SchemaRef>,
}

/// An optimized query plan template containing PlaceholderScan nodes.
///
/// Created by calling [`LazyFrame::optimize_template()`] on a LazyFrame that contains
/// PlaceholderScan leaves. The plan is optimized once at creation time, and can be bound
/// to different concrete data sources repeatedly without re-running optimization.
pub struct OptimizedTemplate {
    /// The optimized IR plan (contains PlaceholderScan nodes as leaves).
    ir_plan: IRPlan,
    /// Map from placeholder name to info about each occurrence in the IR arena.
    placeholder_info: PlHashMap<PlSmallStr, Vec<PlaceholderInfo>>,
}

/// Find all PlaceholderScan nodes in the IR arena.
fn find_placeholder_nodes(arena: &Arena<IR>) -> PlHashMap<PlSmallStr, Vec<PlaceholderInfo>> {
    let mut result: PlHashMap<PlSmallStr, Vec<PlaceholderInfo>> = PlHashMap::new();
    for idx in 0..arena.len() {
        let node = Node(idx);
        if let IR::PlaceholderScan {
            name,
            schema,
            output_schema,
        } = arena.get(node)
        {
            result
                .entry(name.clone())
                .or_default()
                .push(PlaceholderInfo {
                    node,
                    schema: schema.clone(),
                    output_schema: output_schema.clone(),
                });
        }
    }
    result
}

/// Validate that the binding's schema is compatible with the placeholder's declared schema.
fn validate_schema_compatible(
    placeholder_name: &str,
    placeholder_schema: &Schema,
    binding_schema: &Schema,
) -> PolarsResult<()> {
    for (name, dtype) in placeholder_schema.iter() {
        match binding_schema.get(name) {
            None => {
                polars_bail!(
                    SchemaMismatch:
                    "binding for placeholder '{}' is missing column '{}' declared in placeholder schema",
                    placeholder_name,
                    name
                );
            },
            Some(binding_dtype) if binding_dtype != dtype => {
                polars_bail!(
                    SchemaMismatch:
                    "binding for placeholder '{}': column '{}' has type {} but placeholder declares {}",
                    placeholder_name,
                    name,
                    binding_dtype,
                    dtype
                );
            },
            _ => {},
        }
    }
    Ok(())
}

impl OptimizedTemplate {
    /// Create an `OptimizedTemplate` from an already-optimized `IRPlan`.
    ///
    /// Returns an error if no PlaceholderScan nodes are found.
    pub(crate) fn new(ir_plan: IRPlan) -> PolarsResult<Self> {
        let placeholder_info = find_placeholder_nodes(&ir_plan.lp_arena);
        polars_ensure!(
            !placeholder_info.is_empty(),
            InvalidOperation:
            "optimize_template() called on a LazyFrame with no PlaceholderScan nodes"
        );
        Ok(Self {
            ir_plan,
            placeholder_info,
        })
    }

    /// Bind concrete LazyFrames to placeholders and collect immediately.
    ///
    /// This is the fast path: clones the optimized IR, replaces PlaceholderScan nodes
    /// with the bindings' IR, then goes directly to physical planning + execution,
    /// skipping optimization entirely.
    pub fn bind_and_collect(
        &self,
        bindings: PlHashMap<PlSmallStr, LazyFrame>,
    ) -> PolarsResult<DataFrame> {
        let mut ir_plan = self.bind_ir(bindings)?;
        ir_plan.ensure_root_node_is_sink();

        let mut physical_plan = create_physical_plan(
            ir_plan.lp_top,
            &mut ir_plan.lp_arena,
            &mut ir_plan.expr_arena,
            BUILD_STREAMING_EXECUTOR,
        )?;
        let mut state = ExecutionState::new();
        physical_plan.execute(&mut state)
    }

    /// Bind concrete LazyFrames to placeholders and return a new `LazyFrame`.
    ///
    /// The returned LazyFrame wraps the already-optimized IR. Calling `.collect()` on it
    /// will re-run optimization passes (which should be fast since the plan is already
    /// optimized). For maximum performance, prefer [`bind_and_collect()`].
    pub fn bind(&self, bindings: PlHashMap<PlSmallStr, LazyFrame>) -> PolarsResult<LazyFrame> {
        let ir_plan = self.bind_ir(bindings)?;
        let version = ir_plan.lp_arena.version();
        let node = ir_plan.lp_top;

        let lf = LazyFrame::from_inner(
            DslPlan::IR {
                dsl: Arc::new(DslPlan::default()),
                version,
                node: Some(node),
            },
            Default::default(),
            Default::default(),
        );
        lf.set_cached_arena(ir_plan.lp_arena, ir_plan.expr_arena);
        Ok(lf)
    }

    /// Core IR-level bind implementation.
    ///
    /// Clones the template arenas, converts each binding's DslPlan to IR in the
    /// cloned arenas, then replaces PlaceholderScan nodes in-place.
    fn bind_ir(&self, bindings: PlHashMap<PlSmallStr, LazyFrame>) -> PolarsResult<IRPlan> {
        // 1. Clone arenas (template stays reusable).
        let mut lp_arena = self.ir_plan.lp_arena.clone();
        let mut expr_arena = self.ir_plan.expr_arena.clone();
        let lp_top = self.ir_plan.lp_top;

        // 2. Validate all placeholders have bindings.
        for name in self.placeholder_info.keys() {
            polars_ensure!(
                bindings.contains_key(name),
                InvalidOperation:
                "OptimizedTemplate placeholder '{}' has no binding",
                name
            );
        }

        // 3. Replace each placeholder occurrence.
        for (name, infos) in &self.placeholder_info {
            let binding_lf = bindings.get(name).unwrap();

            for info in infos {
                // Convert the binding's DslPlan to IR nodes in the cloned arena.
                let binding_node = to_alp(
                    binding_lf.logical_plan.clone(),
                    &mut expr_arena,
                    &mut lp_arena,
                    &mut OptFlags::schema_only(),
                )?;

                // Validate schema compatibility.
                let binding_schema = lp_arena.get(binding_node).schema(&lp_arena);
                validate_schema_compatible(name.as_str(), &info.schema, &binding_schema)?;

                // Determine the replacement IR.
                let replacement_ir = if let Some(output_schema) = &info.output_schema {
                    // Projection pushdown narrowed the columns. If the binding has more
                    // columns, insert a SimpleProjection to select only what's needed.
                    if output_schema.as_ref() != binding_schema.as_ref().as_ref() {
                        IR::SimpleProjection {
                            input: binding_node,
                            columns: output_schema.clone(),
                        }
                    } else {
                        lp_arena.take(binding_node)
                    }
                } else {
                    lp_arena.take(binding_node)
                };

                // Replace the PlaceholderScan node in-place.
                lp_arena.replace(info.node, replacement_ir);
            }
        }

        Ok(IRPlan::new(lp_top, lp_arena, expr_arena))
    }

    /// Get the names of all placeholders in this template.
    pub fn placeholder_names(&self) -> Vec<PlSmallStr> {
        self.placeholder_info.keys().cloned().collect()
    }
}
