#[cfg(feature = "ipc")]
use crate::logical_plan::IpcScanOptions;
#[cfg(feature = "parquet")]
use crate::logical_plan::ParquetOptions;
use crate::logical_plan::{det_melt_schema, Context, CsvParserOptions};
use crate::prelude::*;
use crate::utils::{aexprs_to_schema, PushNode};
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};
#[cfg(any(feature = "csv-file", feature = "parquet"))]
use std::path::PathBuf;
use std::sync::Arc;

// ALogicalPlan is a representation of LogicalPlan with Nodes which are allocated in an Arena
#[derive(Clone, Debug)]
pub enum ALogicalPlan {
    Melt {
        input: Node,
        id_vars: Arc<Vec<String>>,
        value_vars: Arc<Vec<String>>,
        schema: SchemaRef,
    },
    Slice {
        input: Node,
        offset: i64,
        len: u32,
    },
    Selection {
        input: Node,
        predicate: Node,
    },
    #[cfg(feature = "csv-file")]
    CsvScan {
        path: PathBuf,
        // schema of the complete file
        schema: SchemaRef,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        options: CsvParserOptions,
        predicate: Option<Node>,
        aggregate: Vec<Node>,
    },
    #[cfg(feature = "ipc")]
    IpcScan {
        path: PathBuf,
        schema: SchemaRef,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        options: IpcScanOptions,
        predicate: Option<Node>,
        aggregate: Vec<Node>,
    },
    #[cfg(feature = "parquet")]
    ParquetScan {
        path: PathBuf,
        // schema of the complete file
        schema: SchemaRef,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        predicate: Option<Node>,
        aggregate: Vec<Node>,
        options: ParquetOptions,
    },
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        projection: Option<Vec<Node>>,
        selection: Option<Node>,
    },
    Projection {
        input: Node,
        expr: Vec<Node>,
        schema: SchemaRef,
    },
    LocalProjection {
        expr: Vec<Node>,
        input: Node,
        schema: SchemaRef,
    },
    Sort {
        input: Node,
        by_column: Vec<Node>,
        args: SortArguments,
    },
    Explode {
        input: Node,
        columns: Vec<String>,
    },
    Cache {
        input: Node,
    },
    Aggregate {
        input: Node,
        keys: Vec<Node>,
        aggs: Vec<Node>,
        schema: SchemaRef,
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        options: GroupbyOptions,
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: SchemaRef,
        left_on: Vec<Node>,
        right_on: Vec<Node>,
        options: JoinOptions,
    },
    HStack {
        input: Node,
        exprs: Vec<Node>,
        schema: SchemaRef,
    },
    Distinct {
        input: Node,
        options: DistinctOptions,
    },
    Udf {
        input: Node,
        function: Arc<dyn DataFrameUdf>,
        options: LogicalPlanUdfOptions,
        schema: Option<SchemaRef>,
    },
    Union {
        inputs: Vec<Node>,
        options: UnionOptions,
    },
}

impl Default for ALogicalPlan {
    fn default() -> Self {
        // the lp is should not be valid. By choosing a max value we'll likely panic indicating
        // a programming error early.
        ALogicalPlan::Selection {
            input: Node(usize::max_value()),
            predicate: Node(usize::max_value()),
        }
    }
}

impl ALogicalPlan {
    pub(crate) fn schema<'a>(&'a self, arena: &'a Arena<ALogicalPlan>) -> &'a SchemaRef {
        use ALogicalPlan::*;
        match self {
            Union { inputs, .. } => arena.get(inputs[0]).schema(arena),
            Cache { input } => arena.get(*input).schema(arena),
            Sort { input, .. } => arena.get(*input).schema(arena),
            Explode { input, .. } => arena.get(*input).schema(arena),
            #[cfg(feature = "parquet")]
            ParquetScan {
                schema,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(schema),
            #[cfg(feature = "ipc")]
            IpcScan {
                schema,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(schema),
            DataFrameScan { schema, .. } => schema,
            Selection { input, .. } => arena.get(*input).schema(arena),
            #[cfg(feature = "csv-file")]
            CsvScan {
                schema,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(schema),
            Projection { schema, .. } => schema,
            LocalProjection { schema, .. } => schema,
            Aggregate { schema, .. } => schema,
            Join { schema, .. } => schema,
            HStack { schema, .. } => schema,
            Distinct { input, .. } => arena.get(*input).schema(arena),
            Slice { input, .. } => arena.get(*input).schema(arena),
            Melt { schema, .. } => schema,
            Udf { input, schema, .. } => match schema {
                Some(schema) => schema,
                None => arena.get(*input).schema(arena),
            },
        }
    }
}

impl ALogicalPlan {
    /// Takes the expressions of an LP node and the inputs of that node and reconstruct
    pub fn with_exprs_and_input(&self, mut exprs: Vec<Node>, inputs: Vec<Node>) -> ALogicalPlan {
        use ALogicalPlan::*;

        match self {
            Union { options, .. } => Union {
                inputs,
                options: *options,
            },
            Melt {
                id_vars,
                value_vars,
                schema,
                ..
            } => Melt {
                input: inputs[0],
                id_vars: id_vars.clone(),
                value_vars: value_vars.clone(),
                schema: schema.clone(),
            },
            Slice { offset, len, .. } => Slice {
                input: inputs[0],
                offset: *offset,
                len: *len,
            },
            Selection { .. } => Selection {
                input: inputs[0],
                predicate: exprs[0],
            },
            LocalProjection { schema, .. } => LocalProjection {
                input: inputs[0],
                expr: exprs,
                schema: schema.clone(),
            },
            Projection { schema, .. } => Projection {
                input: inputs[0],
                expr: exprs,
                schema: schema.clone(),
            },
            Aggregate {
                keys,
                schema,
                apply,
                maintain_order,
                options: dynamic_options,
                ..
            } => Aggregate {
                input: inputs[0],
                keys: exprs[..keys.len()].to_vec(),
                aggs: exprs[keys.len()..].to_vec(),
                schema: schema.clone(),
                apply: apply.clone(),
                maintain_order: *maintain_order,
                options: dynamic_options.clone(),
            },
            Join {
                schema,
                left_on,
                options,
                ..
            } => Join {
                input_left: inputs[0],
                input_right: inputs[1],
                schema: schema.clone(),
                left_on: exprs[..left_on.len()].to_vec(),
                right_on: exprs[left_on.len()..].to_vec(),
                options: options.clone(),
            },
            Sort {
                by_column, args, ..
            } => Sort {
                input: inputs[0],
                by_column: by_column.clone(),
                args: args.clone(),
            },
            Explode { columns, .. } => Explode {
                input: inputs[0],
                columns: columns.clone(),
            },
            Cache { .. } => Cache { input: inputs[0] },
            Distinct { options, .. } => Distinct {
                input: inputs[0],
                options: options.clone(),
            },
            HStack { schema, .. } => HStack {
                input: inputs[0],
                exprs,
                schema: schema.clone(),
            },
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                schema,
                output_schema,
                options,
                predicate,
                ..
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }

                IpcScan {
                    path: path.clone(),
                    schema: schema.clone(),
                    output_schema: output_schema.clone(),
                    predicate: new_predicate,
                    aggregate: exprs,
                    options: options.clone(),
                }
            }

            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                output_schema,
                predicate,
                options,
                ..
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }

                ParquetScan {
                    path: path.clone(),
                    schema: schema.clone(),
                    output_schema: output_schema.clone(),
                    predicate: new_predicate,
                    aggregate: exprs,
                    options: options.clone(),
                }
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                schema,
                output_schema,
                predicate,
                options,
                ..
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }
                CsvScan {
                    path: path.clone(),
                    schema: schema.clone(),
                    output_schema: output_schema.clone(),
                    options: options.clone(),
                    predicate: new_predicate,
                    aggregate: exprs,
                }
            }
            DataFrameScan {
                df,
                schema,
                projection,
                selection,
            } => {
                let mut new_selection = None;
                if selection.is_some() {
                    new_selection = exprs.pop()
                }
                let mut new_projection = None;
                if projection.is_some() {
                    new_projection = Some(exprs)
                }

                DataFrameScan {
                    df: df.clone(),
                    schema: schema.clone(),
                    projection: new_projection,
                    selection: new_selection,
                }
            }
            Udf {
                function,
                options,
                schema,
                ..
            } => Udf {
                input: inputs[0],
                function: function.clone(),
                options: *options,
                schema: schema.clone(),
            },
        }
    }

    /// Copy the exprs in this LP node to an existing container.
    pub fn copy_exprs(&self, container: &mut Vec<Node>) {
        use ALogicalPlan::*;
        match self {
            Melt { .. }
            | Slice { .. }
            | Sort { .. }
            | Explode { .. }
            | Cache { .. }
            | Distinct { .. }
            | Union { .. }
            | Udf { .. } => {}
            Selection { predicate, .. } => container.push(*predicate),
            Projection { expr, .. } => container.extend_from_slice(expr),
            LocalProjection { expr, .. } => container.extend_from_slice(expr),
            Aggregate { keys, aggs, .. } => {
                let iter = keys.iter().copied().chain(aggs.iter().copied());
                container.extend(iter)
            }
            Join {
                left_on, right_on, ..
            } => {
                let iter = left_on.iter().copied().chain(right_on.iter().copied());
                container.extend(iter)
            }
            HStack { exprs, .. } => container.extend_from_slice(exprs),
            #[cfg(feature = "parquet")]
            ParquetScan {
                predicate,
                aggregate,
                ..
            } => {
                container.extend_from_slice(aggregate);
                if let Some(node) = predicate {
                    container.push(*node)
                }
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                predicate,
                aggregate,
                ..
            } => {
                container.extend_from_slice(aggregate);
                if let Some(node) = predicate {
                    container.push(*node)
                }
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                predicate,
                aggregate,
                ..
            } => {
                container.extend_from_slice(aggregate);
                if let Some(node) = predicate {
                    container.push(*node)
                }
            }
            DataFrameScan {
                projection,
                selection,
                ..
            } => {
                if let Some(expr) = projection {
                    container.extend_from_slice(expr)
                }
                if let Some(expr) = selection {
                    container.push(*expr)
                }
            }
        }
    }

    /// Get expressions in this node.
    pub fn get_exprs(&self) -> Vec<Node> {
        let mut exprs = Vec::new();
        self.copy_exprs(&mut exprs);
        exprs
    }

    /// Push inputs of the LP in of this node to an existing container.
    /// Most plans have typically one input. A join has two and a scan (CsvScan)
    /// or an in-memory DataFrame has none. A Union has multiple.
    pub(crate) fn copy_inputs<T>(&self, container: &mut T)
    where
        T: PushNode,
    {
        use ALogicalPlan::*;
        let input = match self {
            Union { inputs, .. } => {
                for node in inputs {
                    container.push_node(*node);
                }
                return;
            }
            Melt { input, .. } => *input,
            Slice { input, .. } => *input,
            Selection { input, .. } => *input,
            Projection { input, .. } => *input,
            LocalProjection { input, .. } => *input,
            Sort { input, .. } => *input,
            Explode { input, .. } => *input,
            Cache { input, .. } => *input,
            Aggregate { input, .. } => *input,
            Join {
                input_left,
                input_right,
                ..
            } => {
                container.push_node(*input_left);
                container.push_node(*input_right);
                return;
            }
            HStack { input, .. } => *input,
            Distinct { input, .. } => *input,
            Udf { input, .. } => *input,
            #[cfg(feature = "parquet")]
            ParquetScan { .. } => return,
            #[cfg(feature = "ipc")]
            IpcScan { .. } => return,
            #[cfg(feature = "csv-file")]
            CsvScan { .. } => return,
            DataFrameScan { .. } => return,
        };
        container.push_node(input)
    }

    pub fn get_inputs(&self) -> Vec<Node> {
        let mut inputs = Vec::new();
        self.copy_inputs(&mut inputs);
        inputs
    }
}

pub struct ALogicalPlanBuilder<'a> {
    root: Node,
    expr_arena: &'a mut Arena<AExpr>,
    lp_arena: &'a mut Arena<ALogicalPlan>,
}

impl<'a> ALogicalPlanBuilder<'a> {
    pub(crate) fn new(
        root: Node,
        expr_arena: &'a mut Arena<AExpr>,
        lp_arena: &'a mut Arena<ALogicalPlan>,
    ) -> Self {
        ALogicalPlanBuilder {
            root,
            expr_arena,
            lp_arena,
        }
    }

    pub(crate) fn from_lp(
        lp: ALogicalPlan,
        expr_arena: &'a mut Arena<AExpr>,
        lp_arena: &'a mut Arena<ALogicalPlan>,
    ) -> Self {
        let root = lp_arena.add(lp);
        ALogicalPlanBuilder {
            root,
            expr_arena,
            lp_arena,
        }
    }

    pub fn melt(self, id_vars: Arc<Vec<String>>, value_vars: Arc<Vec<String>>) -> Self {
        let schema = det_melt_schema(&id_vars, &value_vars, self.schema());

        let lp = ALogicalPlan::Melt {
            input: self.root,
            id_vars,
            value_vars,
            schema,
        };
        let node = self.lp_arena.add(lp);
        ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
    }

    pub fn project_local(self, exprs: Vec<Node>) -> Self {
        let input_schema = self.lp_arena.get(self.root).schema(self.lp_arena);
        let schema = aexprs_to_schema(&exprs, input_schema, Context::Default, self.expr_arena);
        let lp = ALogicalPlan::LocalProjection {
            expr: exprs,
            input: self.root,
            schema: Arc::new(schema),
        };
        let node = self.lp_arena.add(lp);
        ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
    }

    pub fn project(self, exprs: Vec<Node>) -> Self {
        let input_schema = self.lp_arena.get(self.root).schema(self.lp_arena);
        let schema = aexprs_to_schema(&exprs, input_schema, Context::Default, self.expr_arena);

        // if len == 0, no projection has to be done. This is a select all operation.
        if !exprs.is_empty() {
            let lp = ALogicalPlan::Projection {
                expr: exprs,
                input: self.root,
                schema: Arc::new(schema),
            };
            let node = self.lp_arena.add(lp);
            ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
        } else {
            self
        }
    }

    pub fn build(self) -> ALogicalPlan {
        if self.root.0 == self.lp_arena.len() {
            self.lp_arena.pop().unwrap()
        } else {
            self.lp_arena.take(self.root)
        }
    }

    pub(crate) fn schema(&self) -> &Schema {
        self.lp_arena.get(self.root).schema(self.lp_arena)
    }

    pub(crate) fn with_columns(self, exprs: Vec<Node>) -> Self {
        let schema = self.schema();
        let mut new_schema = (*schema).clone();

        for e in &exprs {
            let field = self
                .expr_arena
                .get(*e)
                .to_field(schema, Context::Default, self.expr_arena)
                .unwrap();

            new_schema.with_column(field.name().clone(), field.data_type().clone());
        }

        let lp = ALogicalPlan::HStack {
            input: self.root,
            exprs,
            schema: Arc::new(new_schema),
        };
        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }

    pub fn groupby(
        self,
        keys: Vec<Node>,
        aggs: Vec<Node>,
        apply: Option<Arc<dyn DataFrameUdf>>,
        maintain_order: bool,
        options: GroupbyOptions,
    ) -> Self {
        let current_schema = self.schema();
        // TODO! add this line if LogicalPlan is dropped in favor of ALogicalPlan
        // let aggs = rewrite_projections(aggs, current_schema);

        let mut schema = aexprs_to_schema(&keys, current_schema, Context::Default, self.expr_arena);
        let other = aexprs_to_schema(&aggs, current_schema, Context::Aggregation, self.expr_arena);
        schema.merge(other);

        let index_columns = &[
            options
                .rolling
                .as_ref()
                .map(|options| &options.index_column),
            options
                .dynamic
                .as_ref()
                .map(|options| &options.index_column),
        ];
        for &name in index_columns.iter().flatten() {
            let dtype = current_schema.get(name).unwrap();
            schema.with_column(name.clone(), dtype.clone());
        }

        let lp = ALogicalPlan::Aggregate {
            input: self.root,
            keys,
            aggs,
            schema: Arc::new(schema),
            apply,
            maintain_order,
            options,
        };
        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }

    pub fn join(
        self,
        other: Node,
        left_on: Vec<Node>,
        right_on: Vec<Node>,
        options: JoinOptions,
    ) -> Self {
        let schema_left = self.schema();
        let schema_right = self.lp_arena.get(other).schema(self.lp_arena);

        // column names of left table
        let mut names: PlHashSet<&str> =
            PlHashSet::with_capacity(schema_left.len() + schema_right.len());
        let mut new_schema = Schema::with_capacity(schema_left.len() + schema_right.len());

        for (name, dtype) in schema_left.iter() {
            names.insert(name.as_str());
            new_schema.with_column(name.to_string(), dtype.clone())
        }

        let right_names: PlHashSet<_> = right_on
            .iter()
            .map(|e| match self.expr_arena.get(*e) {
                AExpr::Alias(_, name) => name.clone(),
                AExpr::Column(name) => name.clone(),
                _ => panic!("could not determine join column names"),
            })
            .collect();

        for (name, dtype) in schema_right.iter() {
            if !right_names.contains(name.as_str()) {
                if names.contains(name.as_str()) {
                    let new_name = format!("{}{}", name, options.suffix.as_ref());
                    new_schema.with_column(new_name, dtype.clone());
                } else {
                    new_schema.with_column(name.to_string(), dtype.clone());
                }
            }
        }

        let schema = Arc::new(new_schema);

        let lp = ALogicalPlan::Join {
            input_left: self.root,
            input_right: other,
            schema,
            left_on,
            right_on,
            options,
        };
        drop(names);
        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }
}
