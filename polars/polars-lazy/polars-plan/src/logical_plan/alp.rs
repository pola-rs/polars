use std::borrow::Cow;
#[cfg(any(feature = "ipc", feature = "csv-file", feature = "parquet"))]
use std::path::PathBuf;
use std::sync::Arc;

#[cfg(feature = "parquet")]
use polars_core::cloud::CloudOptions;
use polars_core::frame::explode::MeltArgs;
use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};

use crate::logical_plan::functions::FunctionNode;
use crate::logical_plan::schema::{det_join_schema, FileInfo};
#[cfg(feature = "csv-file")]
use crate::logical_plan::CsvParserOptions;
#[cfg(feature = "ipc")]
use crate::logical_plan::IpcScanOptionsInner;
#[cfg(feature = "parquet")]
use crate::logical_plan::ParquetOptions;
use crate::logical_plan::{det_melt_schema, Context};
use crate::prelude::*;
use crate::utils::{aexprs_to_schema, PushNode};

/// ALogicalPlan is a representation of LogicalPlan with Nodes which are allocated in an Arena
#[derive(Clone, Debug)]
pub enum ALogicalPlan {
    AnonymousScan {
        function: Arc<dyn AnonymousScan>,
        file_info: FileInfo,
        output_schema: Option<SchemaRef>,
        predicate: Option<Node>,
        options: AnonymousScanOptions,
    },
    #[cfg(feature = "python")]
    PythonScan {
        options: PythonOptions,
        predicate: Option<Node>,
    },
    Melt {
        input: Node,
        args: Arc<MeltArgs>,
        schema: SchemaRef,
    },
    Slice {
        input: Node,
        offset: i64,
        len: IdxSize,
    },
    Selection {
        input: Node,
        predicate: Node,
    },
    #[cfg(feature = "csv-file")]
    CsvScan {
        path: PathBuf,
        file_info: FileInfo,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        options: CsvParserOptions,
        predicate: Option<Node>,
    },
    #[cfg(feature = "ipc")]
    IpcScan {
        path: PathBuf,
        file_info: FileInfo,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        options: IpcScanOptionsInner,
        predicate: Option<Node>,
    },
    #[cfg(feature = "parquet")]
    ParquetScan {
        path: PathBuf,
        file_info: FileInfo,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        predicate: Option<Node>,
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
    },
    DataFrameScan {
        df: Arc<DataFrame>,
        schema: SchemaRef,
        // schema of the projected file
        output_schema: Option<SchemaRef>,
        projection: Option<Arc<Vec<String>>>,
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
        schema: SchemaRef,
    },
    Cache {
        input: Node,
        id: usize,
        count: usize,
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
    MapFunction {
        input: Node,
        function: FunctionNode,
    },
    Union {
        inputs: Vec<Node>,
        options: UnionOptions,
    },
    ExtContext {
        input: Node,
        contexts: Vec<Node>,
        schema: SchemaRef,
    },
    FileSink {
        input: Node,
        payload: FileSinkOptions,
    },
}

impl Default for ALogicalPlan {
    fn default() -> Self {
        // the lp is should not be valid. By choosing a max value we'll likely panic indicating
        // a programming error early.
        ALogicalPlan::Selection {
            input: Node(usize::MAX),
            predicate: Node(usize::MAX),
        }
    }
}

impl ALogicalPlan {
    /// Get the schema of the logical plan node but don't take projections into account at the scan
    /// level. This ensures we can apply the predicate
    pub(crate) fn scan_schema(&self) -> &SchemaRef {
        use ALogicalPlan::*;
        match self {
            #[cfg(feature = "python")]
            PythonScan { options, .. } => &options.schema,
            #[cfg(feature = "csv-file")]
            CsvScan { file_info, .. } => &file_info.schema,
            #[cfg(feature = "parquet")]
            ParquetScan { file_info, .. } => &file_info.schema,
            #[cfg(feature = "ipc")]
            IpcScan { file_info, .. } => &file_info.schema,
            AnonymousScan { file_info, .. } => &file_info.schema,
            _ => unreachable!(),
        }
    }

    /// Get the schema of the logical plan node.
    pub fn schema<'a>(&'a self, arena: &'a Arena<ALogicalPlan>) -> Cow<'a, SchemaRef> {
        use ALogicalPlan::*;
        let schema = match self {
            #[cfg(feature = "python")]
            PythonScan { options, .. } => &options.schema,
            Union { inputs, .. } => return arena.get(inputs[0]).schema(arena),
            Cache { input, .. } => return arena.get(*input).schema(arena),
            Sort { input, .. } => return arena.get(*input).schema(arena),
            Explode { schema, .. } => schema,
            #[cfg(feature = "parquet")]
            ParquetScan {
                file_info,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(&file_info.schema),
            #[cfg(feature = "ipc")]
            IpcScan {
                file_info,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(&file_info.schema),
            DataFrameScan {
                schema,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(schema),
            AnonymousScan {
                file_info,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(&file_info.schema),
            Selection { input, .. } => return arena.get(*input).schema(arena),
            #[cfg(feature = "csv-file")]
            CsvScan {
                file_info,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(&file_info.schema),
            Projection { schema, .. } => schema,
            LocalProjection { schema, .. } => schema,
            Aggregate { schema, .. } => schema,
            Join { schema, .. } => schema,
            HStack { schema, .. } => schema,
            Distinct { input, .. } | FileSink { input, .. } => {
                return arena.get(*input).schema(arena)
            }
            Slice { input, .. } => return arena.get(*input).schema(arena),
            Melt { schema, .. } => schema,
            MapFunction { input, function } => {
                let input_schema = arena.get(*input).schema(arena);

                return match input_schema {
                    Cow::Owned(schema) => {
                        Cow::Owned(function.schema(&schema).unwrap().into_owned())
                    }
                    Cow::Borrowed(schema) => function.schema(schema).unwrap(),
                };
            }
            ExtContext { schema, .. } => schema,
        };
        Cow::Borrowed(schema)
    }
}

impl ALogicalPlan {
    /// Takes the expressions of an LP node and the inputs of that node and reconstruct
    pub fn with_exprs_and_input(
        &self,
        mut exprs: Vec<Node>,
        mut inputs: Vec<Node>,
    ) -> ALogicalPlan {
        use ALogicalPlan::*;

        match self {
            #[cfg(feature = "python")]
            PythonScan { options, predicate } => PythonScan {
                options: options.clone(),
                predicate: *predicate,
            },
            Union { options, .. } => Union {
                inputs,
                options: *options,
            },
            Melt { args, schema, .. } => Melt {
                input: inputs[0],
                args: args.clone(),
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
            Explode {
                columns, schema, ..
            } => Explode {
                input: inputs[0],
                columns: columns.clone(),
                schema: schema.clone(),
            },
            Cache { id, count, .. } => Cache {
                input: inputs[0],
                id: *id,
                count: *count,
            },
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
                file_info,
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
                    file_info: file_info.clone(),
                    output_schema: output_schema.clone(),
                    predicate: new_predicate,
                    options: options.clone(),
                }
            }

            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                file_info,
                output_schema,
                predicate,
                options,
                cloud_options,
                ..
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }

                ParquetScan {
                    path: path.clone(),
                    file_info: file_info.clone(),
                    output_schema: output_schema.clone(),
                    predicate: new_predicate,
                    options: options.clone(),
                    cloud_options: cloud_options.clone(),
                }
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                file_info,
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
                    file_info: file_info.clone(),
                    output_schema: output_schema.clone(),
                    options: options.clone(),
                    predicate: new_predicate,
                }
            }
            DataFrameScan {
                df,
                schema,
                output_schema,
                projection,
                selection,
            } => {
                let mut new_selection = None;
                if selection.is_some() {
                    new_selection = exprs.pop()
                }

                DataFrameScan {
                    df: df.clone(),
                    schema: schema.clone(),
                    output_schema: output_schema.clone(),
                    projection: projection.clone(),
                    selection: new_selection,
                }
            }
            AnonymousScan {
                function,
                file_info,
                output_schema,
                predicate,
                options,
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }

                AnonymousScan {
                    function: function.clone(),
                    file_info: file_info.clone(),
                    output_schema: output_schema.clone(),
                    predicate: new_predicate,
                    options: options.clone(),
                }
            }
            MapFunction { function, .. } => MapFunction {
                input: inputs[0],
                function: function.clone(),
            },
            ExtContext { schema, .. } => ExtContext {
                input: inputs.pop().unwrap(),
                contexts: inputs,
                schema: schema.clone(),
            },
            FileSink { payload, .. } => FileSink {
                input: inputs.pop().unwrap(),
                payload: payload.clone(),
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
            | MapFunction { .. } => {}
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
            ParquetScan { predicate, .. } => {
                if let Some(node) = predicate {
                    container.push(*node)
                }
            }
            #[cfg(feature = "ipc")]
            IpcScan { predicate, .. } => {
                if let Some(node) = predicate {
                    container.push(*node)
                }
            }
            #[cfg(feature = "csv-file")]
            CsvScan { predicate, .. } => {
                if let Some(node) = predicate {
                    container.push(*node)
                }
            }
            DataFrameScan { selection, .. } => {
                if let Some(expr) = selection {
                    container.push(*expr)
                }
            }
            #[cfg(feature = "python")]
            PythonScan { .. } => {}
            AnonymousScan { predicate, .. } => {
                if let Some(node) = predicate {
                    container.push(*node)
                }
            }
            ExtContext { .. } | FileSink { .. } => {}
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
    pub fn copy_inputs<T>(&self, container: &mut T)
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
            MapFunction { input, .. } => *input,
            FileSink { input, .. } => *input,
            ExtContext {
                input, contexts, ..
            } => {
                for n in contexts {
                    container.push_node(*n)
                }
                *input
            }
            #[cfg(feature = "parquet")]
            ParquetScan { .. } => return,
            #[cfg(feature = "ipc")]
            IpcScan { .. } => return,
            #[cfg(feature = "csv-file")]
            CsvScan { .. } => return,
            DataFrameScan { .. } => return,
            AnonymousScan { .. } => return,
            #[cfg(feature = "python")]
            PythonScan { .. } => return,
        };
        container.push_node(input)
    }

    pub fn get_inputs(&self) -> Vec<Node> {
        let mut inputs = Vec::new();
        self.copy_inputs(&mut inputs);
        inputs
    }
    /// panics if more than one input
    #[cfg(any(
        all(feature = "strings", feature = "concat_str"),
        feature = "streaming"
    ))]
    pub(crate) fn get_input(&self) -> Option<Node> {
        let mut inputs = [None, None];
        self.copy_inputs(&mut inputs);
        inputs[0]
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

    pub fn melt(self, args: Arc<MeltArgs>) -> Self {
        let schema = det_melt_schema(&args, &self.schema());

        let lp = ALogicalPlan::Melt {
            input: self.root,
            args,
            schema,
        };
        let node = self.lp_arena.add(lp);
        ALogicalPlanBuilder::new(node, self.expr_arena, self.lp_arena)
    }

    pub fn project_local(self, exprs: Vec<Node>) -> Self {
        let input_schema = self.lp_arena.get(self.root).schema(self.lp_arena);
        let schema = aexprs_to_schema(&exprs, &input_schema, Context::Default, self.expr_arena);
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
        let schema = aexprs_to_schema(&exprs, &input_schema, Context::Default, self.expr_arena);

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

    pub(crate) fn schema(&'a self) -> Cow<'a, SchemaRef> {
        self.lp_arena.get(self.root).schema(self.lp_arena)
    }

    pub(crate) fn with_columns(self, exprs: Vec<Node>) -> Self {
        let schema = self.schema();
        let mut new_schema = (**schema).clone();

        for e in &exprs {
            let field = self
                .expr_arena
                .get(*e)
                .to_field(&schema, Context::Default, self.expr_arena)
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

        let mut schema =
            aexprs_to_schema(&keys, &current_schema, Context::Default, self.expr_arena);
        let other = aexprs_to_schema(
            &aggs,
            &current_schema,
            Context::Aggregation,
            self.expr_arena,
        );
        schema.merge(other);

        #[cfg(feature = "dynamic_groupby")]
        {
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

        let left_on_exprs = left_on
            .iter()
            .map(|node| node_to_expr(*node, self.expr_arena))
            .collect::<Vec<_>>();
        let right_on_exprs = right_on
            .iter()
            .map(|node| node_to_expr(*node, self.expr_arena))
            .collect::<Vec<_>>();

        let schema = det_join_schema(
            &schema_left,
            &schema_right,
            &left_on_exprs,
            &right_on_exprs,
            &options,
        )
        .unwrap();

        let lp = ALogicalPlan::Join {
            input_left: self.root,
            input_right: other,
            schema,
            left_on,
            right_on,
            options,
        };

        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }
}
