mod dot;
mod format;
pub mod inputs;
mod schema;
pub(crate) mod tree_format;
#[cfg(feature = "ir_visualization")]
pub mod visualization;

use std::borrow::Cow;
use std::fmt;

pub use dot::{EscapeLabel, IRDotDisplay, PathsDisplay, ScanSourcesDisplay};
pub use format::{ExprIRDisplay, IRDisplay, write_group_by, write_ir_non_recursive};
use polars_core::prelude::*;
use polars_utils::idx_vec::UnitVec;
use polars_utils::unique_id::UniqueId;
#[cfg(feature = "ir_serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use self::hive::HivePartitionsDf;
use crate::prelude::*;

#[cfg_attr(feature = "ir_serde", derive(serde::Serialize, serde::Deserialize))]
pub struct IRPlan {
    pub lp_top: Node,
    pub lp_arena: Arena<IR>,
    pub expr_arena: Arena<AExpr>,
}

#[derive(Clone, Copy)]
pub struct IRPlanRef<'a> {
    pub lp_top: Node,
    pub lp_arena: &'a Arena<IR>,
    pub expr_arena: &'a Arena<AExpr>,
}

/// [`IR`] is a representation of [`DslPlan`] with [`Node`]s which are allocated in an [`Arena`]
/// In this IR the logical plan has access to the full dataset.
#[derive(Clone, Debug, Default, IntoStaticStr)]
#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum IR {
    #[cfg(feature = "python")]
    PythonScan {
        options: PythonOptions,
    },
    Slice {
        input: Node,
        offset: i64,
        len: IdxSize,
    },
    Filter {
        input: Node,
        predicate: ExprIR,
    },
    Scan {
        sources: ScanSources,
        file_info: FileInfo,
        hive_parts: Option<HivePartitionsDf>,
        predicate: Option<ExprIR>,
        /// * None: No skipping
        /// * Some(v): Files were skipped (filtered out), where:
        ///   * v @ true: Filter was fully applied (e.g. refers only to hive parts), so does not need to be applied at execution.
        ///   * v @ false: Filter still needs to be applied on remaining data.
        predicate_file_skip_applied: Option<bool>,
        /// schema of the projected file
        output_schema: Option<SchemaRef>,
        scan_type: Box<FileScanIR>,
        /// generic options that can be used for all file types.
        unified_scan_args: Box<UnifiedScanArgs>,
    },
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        // Schema of the projected file
        // If `None`, no projection is applied
        output_schema: Option<SchemaRef>,
    },
    /// Placeholder for data source when serializing templates.
    /// Used for serializing transformation logic without actual data.
    PlaceholderScan {
        id: usize,
        schema: SchemaRef,
        output_schema: Option<SchemaRef>,
    },
    // Only selects columns (semantically only has row access).
    // This is a more restricted operation than `Select`.
    SimpleProjection {
        input: Node,
        columns: SchemaRef,
    },
    // Polars' `select` operation. This may access full materialized data.
    Select {
        input: Node,
        expr: Vec<ExprIR>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Sort {
        input: Node,
        by_column: Vec<ExprIR>,
        slice: Option<(i64, usize)>,
        sort_options: SortMultipleOptions,
    },
    Cache {
        input: Node,
        /// This holds the `Arc<DslPlan>` to guarantee uniqueness.
        id: UniqueId,
    },
    GroupBy {
        input: Node,
        keys: Vec<ExprIR>,
        aggs: Vec<ExprIR>,
        schema: SchemaRef,
        maintain_order: bool,
        options: Arc<GroupbyOptions>,
        apply: Option<PlanCallback<DataFrame, DataFrame>>,
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: SchemaRef,
        left_on: Vec<ExprIR>,
        right_on: Vec<ExprIR>,
        options: Arc<JoinOptionsIR>,
    },
    HStack {
        input: Node,
        exprs: Vec<ExprIR>,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Distinct {
        input: Node,
        options: DistinctOptionsIR,
    },
    MapFunction {
        input: Node,
        function: FunctionIR,
    },
    Union {
        inputs: Vec<Node>,
        options: UnionOptions,
    },
    /// Horizontal concatenation
    /// - Invariant: the names will be unique
    HConcat {
        inputs: Vec<Node>,
        schema: SchemaRef,
        options: HConcatOptions,
    },
    ExtContext {
        input: Node,
        contexts: Vec<Node>,
        schema: SchemaRef,
    },
    Sink {
        input: Node,
        payload: SinkTypeIR,
    },
    /// Node that allows for multiple plans to be executed in parallel with common subplan
    /// elimination and everything.
    SinkMultiple {
        inputs: Vec<Node>,
    },
    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        input_left: Node,
        input_right: Node,
        key: PlSmallStr,
    },
    #[default]
    Invalid,
}

impl IRPlan {
    pub fn new(top: Node, ir_arena: Arena<IR>, expr_arena: Arena<AExpr>) -> Self {
        Self {
            lp_top: top,
            lp_arena: ir_arena,
            expr_arena,
        }
    }

    pub fn root(&self) -> &IR {
        self.lp_arena.get(self.lp_top)
    }

    pub fn as_ref(&self) -> IRPlanRef<'_> {
        IRPlanRef {
            lp_top: self.lp_top,
            lp_arena: &self.lp_arena,
            expr_arena: &self.expr_arena,
        }
    }

    pub fn describe(&self) -> String {
        self.as_ref().describe()
    }

    pub fn describe_tree_format(&self) -> String {
        self.as_ref().describe_tree_format()
    }

    pub fn display(&self) -> format::IRDisplay<'_> {
        self.as_ref().display()
    }

    pub fn display_dot(&self) -> dot::IRDotDisplay<'_> {
        self.as_ref().display_dot()
    }

    /// Convert an IR plan to a template by replacing all data sources with PlaceholderScan nodes.
    ///
    /// This traverses the IR tree and replaces:
    /// - `IR::DataFrameScan` → `IR::PlaceholderScan`
    /// - `IR::Scan` (CSV, Parquet, etc.) → `IR::PlaceholderScan`
    /// - `IR::PythonScan` → `IR::PlaceholderScan`
    ///
    /// All other IR nodes (Filter, Select, Join, etc.) are preserved as-is.
    ///
    /// # Important: Only Use with Unoptimized Plans
    ///
    /// This method should only be called on **unoptimized** IR plans where:
    /// - Predicates remain as separate `IR::Filter` nodes
    /// - Projections remain as separate `IR::Select` nodes
    /// - No optimizations have pushed these into Scan nodes
    ///
    /// **Why:** `PlaceholderScan` only preserves `schema` and `output_schema`. If a Scan
    /// node contains pushed-down predicates or projections, those will be silently lost.
    ///
    /// The safe usage pattern is:
    /// ```ignore
    /// // DSL → IR (no optimization) → template
    /// let ir_plan = lf.to_alp()?;  // No optimization
    /// let template = ir_plan.to_template();  // Safe
    /// ```
    ///
    /// **Unsafe pattern** (would lose predicates):
    /// ```ignore
    /// // DSL → optimized IR → template
    /// let ir_plan = lf.to_alp_optimized()?;  // Predicates pushed into Scans
    /// let template = ir_plan.to_template();  // Predicates would be lost!
    /// ```
    pub fn to_template(&self) -> Self {
        let mut new_arena = Arena::with_capacity(self.lp_arena.len());
        let mut placeholder_id = 0;
        let new_top = Self::convert_to_placeholder(
            self.lp_top,
            &self.lp_arena,
            &mut new_arena,
            &mut placeholder_id,
        );
        Self {
            lp_top: new_top,
            lp_arena: new_arena,
            expr_arena: self.expr_arena.clone(),
        }
    }

    #[recursive::recursive]
    fn convert_to_placeholder(
        node: Node,
        old_arena: &Arena<IR>,
        new_arena: &mut Arena<IR>,
        placeholder_id: &mut usize,
    ) -> Node {
        let ir = old_arena.get(node);
        let new_ir = match ir {
            IR::DataFrameScan {
                schema,
                output_schema,
                ..
            } => {
                let id = *placeholder_id;
                *placeholder_id += 1;
                IR::PlaceholderScan {
                    id,
                    schema: schema.clone(),
                    output_schema: output_schema.clone(),
                }
            },
            IR::Scan {
                file_info,
                output_schema,
                ..
            } => {
                let id = *placeholder_id;
                *placeholder_id += 1;
                IR::PlaceholderScan {
                    id,
                    schema: file_info.schema.clone(),
                    output_schema: output_schema.clone(),
                }
            },
            #[cfg(feature = "python")]
            IR::PythonScan { options } => {
                let id = *placeholder_id;
                *placeholder_id += 1;
                IR::PlaceholderScan {
                    id,
                    schema: options.schema.clone(),
                    output_schema: options.output_schema.clone(),
                }
            },
            IR::Select {
                input,
                expr,
                schema,
                options,
            } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::Select {
                    input: new_input,
                    expr: expr.clone(),
                    schema: schema.clone(),
                    options: *options,
                }
            },
            IR::Filter { input, predicate } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::Filter {
                    input: new_input,
                    predicate: predicate.clone(),
                }
            },
            IR::Slice { input, offset, len } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::Slice {
                    input: new_input,
                    offset: *offset,
                    len: *len,
                }
            },
            IR::GroupBy {
                input,
                keys,
                aggs,
                schema,
                maintain_order,
                options,
                apply,
            } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::GroupBy {
                    input: new_input,
                    keys: keys.clone(),
                    aggs: aggs.clone(),
                    schema: schema.clone(),
                    maintain_order: *maintain_order,
                    options: options.clone(),
                    apply: apply.clone(),
                }
            },
            IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } => {
                let new_left =
                    Self::convert_to_placeholder(*input_left, old_arena, new_arena, placeholder_id);
                let new_right = Self::convert_to_placeholder(
                    *input_right,
                    old_arena,
                    new_arena,
                    placeholder_id,
                );
                IR::Join {
                    input_left: new_left,
                    input_right: new_right,
                    schema: schema.clone(),
                    left_on: left_on.clone(),
                    right_on: right_on.clone(),
                    options: options.clone(),
                }
            },
            IR::HStack {
                input,
                exprs,
                schema,
                options,
            } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::HStack {
                    input: new_input,
                    exprs: exprs.clone(),
                    schema: schema.clone(),
                    options: *options,
                }
            },
            IR::SimpleProjection { input, columns } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::SimpleProjection {
                    input: new_input,
                    columns: columns.clone(),
                }
            },
            IR::Sort {
                input,
                by_column,
                slice,
                sort_options,
            } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::Sort {
                    input: new_input,
                    by_column: by_column.clone(),
                    slice: *slice,
                    sort_options: sort_options.clone(),
                }
            },
            IR::Distinct { input, options } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::Distinct {
                    input: new_input,
                    options: options.clone(),
                }
            },
            IR::MapFunction { input, function } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::MapFunction {
                    input: new_input,
                    function: function.clone(),
                }
            },
            IR::Cache { input, id } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::Cache {
                    input: new_input,
                    id: *id,
                }
            },
            IR::Union { inputs, options } => {
                let new_inputs: Vec<_> = inputs
                    .iter()
                    .map(|&input| {
                        Self::convert_to_placeholder(input, old_arena, new_arena, placeholder_id)
                    })
                    .collect();
                IR::Union {
                    inputs: new_inputs,
                    options: *options,
                }
            },
            IR::HConcat {
                inputs,
                schema,
                options,
            } => {
                let new_inputs: Vec<_> = inputs
                    .iter()
                    .map(|&input| {
                        Self::convert_to_placeholder(input, old_arena, new_arena, placeholder_id)
                    })
                    .collect();
                IR::HConcat {
                    inputs: new_inputs,
                    schema: schema.clone(),
                    options: *options,
                }
            },
            IR::ExtContext {
                input,
                contexts,
                schema,
            } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                let new_contexts: Vec<_> = contexts
                    .iter()
                    .map(|&ctx| {
                        Self::convert_to_placeholder(ctx, old_arena, new_arena, placeholder_id)
                    })
                    .collect();
                IR::ExtContext {
                    input: new_input,
                    contexts: new_contexts,
                    schema: schema.clone(),
                }
            },
            IR::Sink { input, payload } => {
                let new_input =
                    Self::convert_to_placeholder(*input, old_arena, new_arena, placeholder_id);
                IR::Sink {
                    input: new_input,
                    payload: payload.clone(),
                }
            },
            IR::SinkMultiple { inputs } => {
                let new_inputs: Vec<_> = inputs
                    .iter()
                    .map(|&input| {
                        Self::convert_to_placeholder(input, old_arena, new_arena, placeholder_id)
                    })
                    .collect();
                IR::SinkMultiple { inputs: new_inputs }
            },
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left,
                input_right,
                key,
            } => {
                let new_left =
                    Self::convert_to_placeholder(*input_left, old_arena, new_arena, placeholder_id);
                let new_right = Self::convert_to_placeholder(
                    *input_right,
                    old_arena,
                    new_arena,
                    placeholder_id,
                );
                IR::MergeSorted {
                    input_left: new_left,
                    input_right: new_right,
                    key: key.clone(),
                }
            },
            // Nodes without inputs - just clone them
            IR::PlaceholderScan { .. } => ir.clone(),
            IR::Invalid => ir.clone(),
        };
        new_arena.add(new_ir)
    }

    pub fn bind_data(
        &self,
        data_map: PlHashMap<usize, Node>,
        data_arena: &Arena<IR>,
    ) -> PolarsResult<Self> {
        let mut new_arena = Arena::with_capacity(self.lp_arena.len());
        let new_top = Self::replace_placeholder(
            self.lp_top,
            &data_map,
            data_arena,
            &self.lp_arena,
            &mut new_arena,
        )?;
        Ok(Self {
            lp_top: new_top,
            lp_arena: new_arena,
            expr_arena: self.expr_arena.clone(),
        })
    }

    pub fn bind_to_df(&self, df: Arc<DataFrame>) -> PolarsResult<Self> {
        // Validate that the template has exactly one placeholder
        let placeholder_count = self.count_placeholders();
        if placeholder_count != 1 {
            polars_bail!(ComputeError:
                "bind_to_df expects a template with exactly 1 placeholder, but found {}",
                placeholder_count
            );
        }

        let schema = df.schema().clone();
        let mut data_arena = Arena::with_capacity(1);
        let data_node = data_arena.add(IR::DataFrameScan {
            df,
            schema,
            output_schema: None,
        });
        let mut data_map = PlHashMap::new();
        data_map.insert(0, data_node);
        self.bind_data(data_map, &data_arena)
    }

    /// Bind multiple DataFrames to a template containing multiple placeholders.
    ///
    /// # Ordering Requirement
    ///
    /// **IMPORTANT:** DataFrames must be passed in the same order as data sources appear
    /// during a depth-first traversal of the IR tree. This is the order in which placeholder
    /// IDs are assigned (0, 1, 2, ...) when `to_template()` converts data sources.
    ///
    /// The i-th DataFrame in the vector will be mapped to placeholder ID `i`. If the template
    /// was created from a query with multiple data sources, you must provide DataFrames in
    /// the exact order those sources were encountered during traversal (typically from bottom
    /// to top of the query, following input dependencies).
    ///
    /// # Example
    ///
    /// ```ignore
    /// // Original query with two data sources (encountered in order: df1, df2)
    /// let query = df1.lazy().join(df2.lazy(), ...).filter(...);
    /// let template = query.to_template();
    ///
    /// // Must pass DataFrames in the same order: [df1_new, df2_new]
    /// let result = template.bind_to_dfs(vec![df1_new, df2_new])?;
    /// ```
    pub fn bind_to_dfs(&self, dfs: Vec<Arc<DataFrame>>) -> PolarsResult<Self> {
        if dfs.is_empty() {
            polars_bail!(ComputeError: "bind_to_dfs requires at least one DataFrame");
        }

        // Validate that the number of DataFrames matches the number of placeholders
        let placeholder_count = self.count_placeholders();
        if dfs.len() != placeholder_count {
            polars_bail!(ComputeError:
                "DataFrame count mismatch: template has {} placeholder{}, but {} DataFrame{} provided",
                placeholder_count,
                if placeholder_count == 1 { "" } else { "s" },
                dfs.len(),
                if dfs.len() == 1 { "" } else { "s" }
            );
        }

        let mut data_arena = Arena::with_capacity(dfs.len());
        let mut data_map = PlHashMap::new();

        for (id, df) in dfs.iter().enumerate() {
            let schema = df.schema().clone();
            let data_node = data_arena.add(IR::DataFrameScan {
                df: df.clone(),
                schema,
                output_schema: None,
            });
            data_map.insert(id, data_node);
        }

        self.bind_data(data_map, &data_arena)
    }

    /// Count the number of PlaceholderScan nodes in the IR plan.
    ///
    /// Placeholders are encountered and assigned IDs (0, 1, 2, ...) during a depth-first
    /// traversal of the IR tree in `to_template()`. This same traversal order determines
    /// the mapping between DataFrame positions in `bind_to_dfs()` and placeholder IDs.
    fn count_placeholders(&self) -> usize {
        Self::count_placeholders_recursive(self.lp_top, &self.lp_arena)
    }

    /// Recursively count PlaceholderScan nodes in the IR tree.
    ///
    /// This traversal follows the same depth-first order used when assigning placeholder IDs,
    /// ensuring consistent ordering across template creation and data binding operations.
    fn count_placeholders_recursive(node: Node, arena: &Arena<IR>) -> usize {
        let ir = arena.get(node);
        match ir {
            IR::PlaceholderScan { .. } => 1,
            _ => {
                // Sum placeholder counts from all input nodes
                ir.inputs()
                    .map(|input| Self::count_placeholders_recursive(input, arena))
                    .sum()
            },
        }
    }

    #[recursive::recursive]
    fn replace_placeholder(
        node: Node,
        data_map: &PlHashMap<usize, Node>,
        data_arena: &Arena<IR>,
        template_arena: &Arena<IR>,
        new_arena: &mut Arena<IR>,
    ) -> PolarsResult<Node> {
        let ir = template_arena.get(node);
        let new_ir = match ir {
            IR::PlaceholderScan { id, schema, .. } => {
                let data_node = data_map.get(id).ok_or_else(
                    || polars_err!(ComputeError: "Placeholder ID {} not found in data map", id),
                )?;

                let data_ir = data_arena.get(*data_node);
                let data_schema = match data_ir {
                    IR::DataFrameScan {
                        schema: data_schema,
                        ..
                    } => data_schema,
                    IR::Scan { file_info, .. } => &file_info.schema,
                    #[cfg(feature = "python")]
                    IR::PythonScan { options } => &options.schema,
                    _ => polars_bail!(ComputeError:
                        "bind_data requires data to be a data-source node (DataFrameScan, Scan, or PythonScan)"
                    ),
                };

                // Allow empty schemas to bind to any data (generic templates)
                if !schema.is_empty() {
                    if schema.len() != data_schema.len() {
                        polars_bail!(SchemaMismatch:
                            "Schema mismatch: template expects {} columns, data has {}",
                            schema.len(),
                            data_schema.len()
                        );
                    }
                    // Validate column names and types
                    for (col_name, dtype) in schema.iter() {
                        match data_schema.get(col_name) {
                            Some(data_dtype) if data_dtype == dtype => {},
                            Some(data_dtype) => polars_bail!(SchemaMismatch:
                                "Column '{}' type mismatch: template expects {:?}, data has {:?}",
                                col_name, dtype, data_dtype
                            ),
                            None => polars_bail!(SchemaMismatch:
                                "Column '{}' not found in data schema",
                                col_name
                            ),
                        }
                    }
                }

                return Ok(new_arena.add(data_ir.clone()));
            },
            IR::Select {
                input,
                expr,
                schema,
                options,
            } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::Select {
                    input: new_input,
                    expr: expr.clone(),
                    schema: schema.clone(),
                    options: *options,
                }
            },
            IR::Filter { input, predicate } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::Filter {
                    input: new_input,
                    predicate: predicate.clone(),
                }
            },
            IR::Slice { input, offset, len } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::Slice {
                    input: new_input,
                    offset: *offset,
                    len: *len,
                }
            },
            IR::GroupBy {
                input,
                keys,
                aggs,
                schema,
                maintain_order,
                options,
                apply,
            } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::GroupBy {
                    input: new_input,
                    keys: keys.clone(),
                    aggs: aggs.clone(),
                    schema: schema.clone(),
                    maintain_order: *maintain_order,
                    options: options.clone(),
                    apply: apply.clone(),
                }
            },
            IR::Join {
                input_left,
                input_right,
                schema,
                left_on,
                right_on,
                options,
            } => {
                let new_left = Self::replace_placeholder(
                    *input_left,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                let new_right = Self::replace_placeholder(
                    *input_right,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::Join {
                    input_left: new_left,
                    input_right: new_right,
                    schema: schema.clone(),
                    left_on: left_on.clone(),
                    right_on: right_on.clone(),
                    options: options.clone(),
                }
            },
            IR::HStack {
                input,
                exprs,
                schema,
                options,
            } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::HStack {
                    input: new_input,
                    exprs: exprs.clone(),
                    schema: schema.clone(),
                    options: *options,
                }
            },
            IR::SimpleProjection { input, columns } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::SimpleProjection {
                    input: new_input,
                    columns: columns.clone(),
                }
            },
            IR::Sort {
                input,
                by_column,
                slice,
                sort_options,
            } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::Sort {
                    input: new_input,
                    by_column: by_column.clone(),
                    slice: *slice,
                    sort_options: sort_options.clone(),
                }
            },
            IR::Distinct { input, options } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::Distinct {
                    input: new_input,
                    options: options.clone(),
                }
            },
            IR::MapFunction { input, function } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::MapFunction {
                    input: new_input,
                    function: function.clone(),
                }
            },
            IR::Cache { input, id } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::Cache {
                    input: new_input,
                    id: *id,
                }
            },
            IR::Union { inputs, options } => {
                let new_inputs: Vec<_> = inputs
                    .iter()
                    .map(|&input| {
                        Self::replace_placeholder(
                            input,
                            data_map,
                            data_arena,
                            template_arena,
                            new_arena,
                        )
                    })
                    .collect::<PolarsResult<_>>()?;
                IR::Union {
                    inputs: new_inputs,
                    options: *options,
                }
            },
            IR::HConcat {
                inputs,
                schema,
                options,
            } => {
                let new_inputs: Vec<_> = inputs
                    .iter()
                    .map(|&input| {
                        Self::replace_placeholder(
                            input,
                            data_map,
                            data_arena,
                            template_arena,
                            new_arena,
                        )
                    })
                    .collect::<PolarsResult<_>>()?;
                IR::HConcat {
                    inputs: new_inputs,
                    schema: schema.clone(),
                    options: *options,
                }
            },
            IR::ExtContext {
                input,
                contexts,
                schema,
            } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                let new_contexts: Vec<_> = contexts
                    .iter()
                    .map(|&ctx| {
                        Self::replace_placeholder(
                            ctx,
                            data_map,
                            data_arena,
                            template_arena,
                            new_arena,
                        )
                    })
                    .collect::<PolarsResult<_>>()?;
                IR::ExtContext {
                    input: new_input,
                    contexts: new_contexts,
                    schema: schema.clone(),
                }
            },
            IR::Sink { input, payload } => {
                let new_input = Self::replace_placeholder(
                    *input,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::Sink {
                    input: new_input,
                    payload: payload.clone(),
                }
            },
            IR::SinkMultiple { inputs } => {
                let new_inputs: Vec<_> = inputs
                    .iter()
                    .map(|&input| {
                        Self::replace_placeholder(
                            input,
                            data_map,
                            data_arena,
                            template_arena,
                            new_arena,
                        )
                    })
                    .collect::<PolarsResult<_>>()?;
                IR::SinkMultiple { inputs: new_inputs }
            },
            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted {
                input_left,
                input_right,
                key,
            } => {
                let new_left = Self::replace_placeholder(
                    *input_left,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                let new_right = Self::replace_placeholder(
                    *input_right,
                    data_map,
                    data_arena,
                    template_arena,
                    new_arena,
                )?;
                IR::MergeSorted {
                    input_left: new_left,
                    input_right: new_right,
                    key: key.clone(),
                }
            },
            // Nodes without inputs - clone as-is
            // Note: DataFrameScan/Scan/PythonScan shouldn't appear in templates (they're replaced by PlaceholderScan),
            // but we handle them explicitly for exhaustiveness checking
            IR::DataFrameScan { .. } => ir.clone(),
            IR::Scan { .. } => ir.clone(),
            #[cfg(feature = "python")]
            IR::PythonScan { .. } => ir.clone(),
            IR::Invalid => ir.clone(),
        };
        Ok(new_arena.add(new_ir))
    }
}

impl<'a> IRPlanRef<'a> {
    pub fn root(self) -> &'a IR {
        self.lp_arena.get(self.lp_top)
    }

    pub fn with_root(self, root: Node) -> Self {
        Self {
            lp_top: root,
            lp_arena: self.lp_arena,
            expr_arena: self.expr_arena,
        }
    }

    pub fn display(self) -> format::IRDisplay<'a> {
        format::IRDisplay::new(self)
    }

    pub fn display_dot(self) -> dot::IRDotDisplay<'a> {
        dot::IRDotDisplay::new(self)
    }

    pub fn describe(self) -> String {
        self.display().to_string()
    }

    pub fn describe_tree_format(self) -> String {
        let mut visitor = tree_format::TreeFmtVisitor::default();
        tree_format::TreeFmtNode::root_logical_plan(self).traverse(&mut visitor);
        format!("{visitor:#?}")
    }
}

impl fmt::Debug for IRPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <format::IRDisplay as fmt::Display>::fmt(&self.display(), f)
    }
}

impl fmt::Debug for IRPlanRef<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        <format::IRDisplay as fmt::Display>::fmt(&self.display(), f)
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // skipped for now
    #[ignore]
    #[test]
    fn test_alp_size() {
        assert!(size_of::<IR>() <= 152);
    }
}
