use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::Arc;

use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unitvec;

use super::projection_expr::*;
use crate::logical_plan::functions::FunctionNode;
use crate::logical_plan::schema::FileInfo;
use crate::logical_plan::FileScan;
use crate::prelude::*;
use crate::utils::PushNode;

/// [`ALogicalPlan`] is a representation of [`LogicalPlan`] with [`Node`]s which are allocated in an [`Arena`]
#[derive(Clone, Debug)]
pub enum ALogicalPlan {
    #[cfg(feature = "python")]
    PythonScan {
        options: PythonOptions,
        predicate: Option<Node>,
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
    Scan {
        paths: Arc<[PathBuf]>,
        file_info: FileInfo,
        predicate: Option<Node>,
        /// schema of the projected file
        output_schema: Option<SchemaRef>,
        scan_type: FileScan,
        /// generic options that can be used for all file types.
        file_options: FileScanOptions,
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
        expr: ProjectionExprs,
        schema: SchemaRef,
        options: ProjectionOptions,
    },
    Sort {
        input: Node,
        by_column: Vec<Node>,
        args: SortArguments,
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
        options: Arc<GroupbyOptions>,
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: SchemaRef,
        left_on: Vec<Node>,
        right_on: Vec<Node>,
        options: Arc<JoinOptions>,
    },
    HStack {
        input: Node,
        exprs: ProjectionExprs,
        schema: SchemaRef,
        options: ProjectionOptions,
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
        payload: SinkType,
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
            Scan { file_info, .. } => &file_info.schema,
            #[cfg(feature = "python")]
            PythonScan { options, .. } => &options.schema,
            _ => unreachable!(),
        }
    }

    pub fn name(&self) -> &'static str {
        use ALogicalPlan::*;
        match self {
            Scan { scan_type, .. } => scan_type.into(),
            #[cfg(feature = "python")]
            PythonScan { .. } => "python_scan",
            Slice { .. } => "slice",
            Selection { .. } => "selection",
            DataFrameScan { .. } => "df",
            Projection { .. } => "projection",
            Sort { .. } => "sort",
            Cache { .. } => "cache",
            Aggregate { .. } => "aggregate",
            Join { .. } => "join",
            HStack { .. } => "hstack",
            Distinct { .. } => "distinct",
            MapFunction { .. } => "map_function",
            Union { .. } => "union",
            HConcat { .. } => "hconcat",
            ExtContext { .. } => "ext_context",
            Sink { payload, .. } => match payload {
                SinkType::Memory => "sink (memory)",
                SinkType::File { .. } => "sink (file)",
                #[cfg(feature = "cloud")]
                SinkType::Cloud { .. } => "sink (cloud)",
            },
        }
    }

    /// Get the schema of the logical plan node.
    pub fn schema<'a>(&'a self, arena: &'a Arena<ALogicalPlan>) -> Cow<'a, SchemaRef> {
        use ALogicalPlan::*;
        let schema = match self {
            #[cfg(feature = "python")]
            PythonScan { options, .. } => options.output_schema.as_ref().unwrap_or(&options.schema),
            Union { inputs, .. } => return arena.get(inputs[0]).schema(arena),
            HConcat { schema, .. } => schema,
            Cache { input, .. } => return arena.get(*input).schema(arena),
            Sort { input, .. } => return arena.get(*input).schema(arena),
            Scan {
                output_schema,
                file_info,
                ..
            } => output_schema.as_ref().unwrap_or(&file_info.schema),
            DataFrameScan {
                schema,
                output_schema,
                ..
            } => output_schema.as_ref().unwrap_or(schema),
            Selection { input, .. } => return arena.get(*input).schema(arena),
            Projection { schema, .. } => schema,
            Aggregate { schema, .. } => schema,
            Join { schema, .. } => schema,
            HStack { schema, .. } => schema,
            Distinct { input, .. } | Sink { input, .. } => return arena.get(*input).schema(arena),
            Slice { input, .. } => return arena.get(*input).schema(arena),
            MapFunction { input, function } => {
                let input_schema = arena.get(*input).schema(arena);

                return match input_schema {
                    Cow::Owned(schema) => {
                        Cow::Owned(function.schema(&schema).unwrap().into_owned())
                    },
                    Cow::Borrowed(schema) => function.schema(schema).unwrap(),
                };
            },
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
            HConcat {
                schema, options, ..
            } => HConcat {
                inputs,
                schema: schema.clone(),
                options: *options,
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
            Projection {
                schema, options, ..
            } => Projection {
                input: inputs[0],
                expr: ProjectionExprs::new(exprs),
                schema: schema.clone(),
                options: *options,
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
            Cache { id, count, .. } => Cache {
                input: inputs[0],
                id: *id,
                count: *count,
            },
            Distinct { options, .. } => Distinct {
                input: inputs[0],
                options: options.clone(),
            },
            HStack {
                schema, options, ..
            } => HStack {
                input: inputs[0],
                exprs: ProjectionExprs::new(exprs),
                schema: schema.clone(),
                options: *options,
            },
            Scan {
                paths,
                file_info,
                output_schema,
                predicate,
                file_options: options,
                scan_type,
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }
                Scan {
                    paths: paths.clone(),
                    file_info: file_info.clone(),
                    output_schema: output_schema.clone(),
                    file_options: options.clone(),
                    predicate: new_predicate,
                    scan_type: scan_type.clone(),
                }
            },
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
            },
            MapFunction { function, .. } => MapFunction {
                input: inputs[0],
                function: function.clone(),
            },
            ExtContext { schema, .. } => ExtContext {
                input: inputs.pop().unwrap(),
                contexts: inputs,
                schema: schema.clone(),
            },
            Sink { payload, .. } => Sink {
                input: inputs.pop().unwrap(),
                payload: payload.clone(),
            },
        }
    }

    /// Copy the exprs in this LP node to an existing container.
    pub fn copy_exprs(&self, container: &mut Vec<Node>) {
        use ALogicalPlan::*;
        match self {
            Slice { .. } | Cache { .. } | Distinct { .. } | Union { .. } | MapFunction { .. } => {},
            Sort { by_column, .. } => container.extend_from_slice(by_column),
            Selection { predicate, .. } => container.push(*predicate),
            Projection { expr, .. } => container.extend_from_slice(expr),
            Aggregate { keys, aggs, .. } => {
                let iter = keys.iter().copied().chain(aggs.iter().copied());
                container.extend(iter)
            },
            Join {
                left_on, right_on, ..
            } => {
                let iter = left_on.iter().copied().chain(right_on.iter().copied());
                container.extend(iter)
            },
            HStack { exprs, .. } => container.extend_from_slice(exprs),
            Scan { predicate, .. } => {
                if let Some(node) = predicate {
                    container.push(*node)
                }
            },
            DataFrameScan { selection, .. } => {
                if let Some(expr) = selection {
                    container.push(*expr)
                }
            },
            #[cfg(feature = "python")]
            PythonScan { .. } => {},
            HConcat { .. } => {},
            ExtContext { .. } | Sink { .. } => {},
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
            },
            HConcat { inputs, .. } => {
                for node in inputs {
                    container.push_node(*node);
                }
                return;
            },
            Slice { input, .. } => *input,
            Selection { input, .. } => *input,
            Projection { input, .. } => *input,
            Sort { input, .. } => *input,
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
            },
            HStack { input, .. } => *input,
            Distinct { input, .. } => *input,
            MapFunction { input, .. } => *input,
            Sink { input, .. } => *input,
            ExtContext {
                input, contexts, ..
            } => {
                for n in contexts {
                    container.push_node(*n)
                }
                *input
            },
            Scan { .. } => return,
            DataFrameScan { .. } => return,
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
        feature = "streaming",
        feature = "fused"
    ))]
    pub(crate) fn get_input(&self) -> Option<Node> {
        let mut inputs: UnitVec<Node> = unitvec!();
        self.copy_inputs(&mut inputs);
        inputs.first().copied()
    }
}

#[cfg(test)]
mod test {
    use super::*;

    // skipped for now
    #[ignore]
    #[test]
    fn test_alp_size() {
        assert!(std::mem::size_of::<ALogicalPlan>() <= 152);
    }
}
