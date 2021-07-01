use crate::logical_plan::{det_melt_schema, Context, CsvParserOptions};
use crate::prelude::*;
use crate::utils::{aexprs_to_schema, PushNode};
use ahash::RandomState;
use polars_core::frame::hash_join::JoinType;
use polars_core::prelude::*;
use polars_core::utils::{Arena, Node};
use std::collections::HashSet;
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
        len: usize,
    },
    Selection {
        input: Node,
        predicate: Node,
    },
    #[cfg(feature = "csv-file")]
    CsvScan {
        path: PathBuf,
        schema: SchemaRef,
        options: CsvParserOptions,
        predicate: Option<Node>,
        aggregate: Vec<Node>,
    },
    #[cfg(feature = "parquet")]
    ParquetScan {
        path: PathBuf,
        schema: SchemaRef,
        with_columns: Option<Vec<String>>,
        predicate: Option<Node>,
        aggregate: Vec<Node>,
        stop_after_n_rows: Option<usize>,
        cache: bool,
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
        reverse: Vec<bool>,
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
    },
    Join {
        input_left: Node,
        input_right: Node,
        schema: SchemaRef,
        how: JoinType,
        left_on: Vec<Node>,
        right_on: Vec<Node>,
        allow_par: bool,
        force_par: bool,
    },
    HStack {
        input: Node,
        exprs: Vec<Node>,
        schema: SchemaRef,
    },
    Distinct {
        input: Node,
        maintain_order: bool,
        subset: Arc<Option<Vec<String>>>,
    },
    Udf {
        input: Node,
        function: Arc<dyn DataFrameUdf>,
        ///  allow predicate pushdown optimizations
        predicate_pd: bool,
        ///  allow projection pushdown optimizations
        projection_pd: bool,
        schema: Option<SchemaRef>,
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
    pub(crate) fn schema<'a>(&'a self, arena: &'a Arena<ALogicalPlan>) -> &'a Schema {
        use ALogicalPlan::*;
        match self {
            Cache { input } => arena.get(*input).schema(arena),
            Sort { input, .. } => arena.get(*input).schema(arena),
            Explode { input, .. } => arena.get(*input).schema(arena),
            #[cfg(feature = "parquet")]
            ParquetScan { schema, .. } => schema,
            DataFrameScan { schema, .. } => schema,
            Selection { input, .. } => arena.get(*input).schema(arena),
            #[cfg(feature = "csv-file")]
            CsvScan { schema, .. } => schema,
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

    /// Check ALogicalPlan equality. The nodes may differ.
    ///
    /// For instance: there can be two columns "foo" in the memory arena. These are equal,
    /// but would have different node values.
    #[cfg(feature = "private")]
    pub(crate) fn eq(
        node_left: Node,
        node_right: Node,
        lp_arena: &Arena<ALogicalPlan>,
        expr_arena: &Arena<AExpr>,
    ) -> bool {
        use crate::logical_plan::iterator::ArenaLpIter;
        use std::fs::canonicalize;

        let cmp = |(node_left, node_right)| {
            use ALogicalPlan::*;
            match (lp_arena.get(node_left), lp_arena.get(node_right)) {
                #[cfg(feature = "csv-file")]
                (CsvScan { path: path_a, .. }, CsvScan { path: path_b, .. }) => {
                    canonicalize(path_a).unwrap() == canonicalize(path_b).unwrap()
                }
                #[cfg(feature = "parquet")]
                (ParquetScan { path: path_a, .. }, ParquetScan { path: path_b, .. }) => {
                    canonicalize(path_a).unwrap() == canonicalize(path_b).unwrap()
                }
                (DataFrameScan { df: df_a, .. }, DataFrameScan { df: df_b, .. }) => {
                    df_a.ptr_equal(df_b)
                }
                // the following don't affect the schema, but do affect the # of rows or the row order.
                (Selection { predicate: l, .. }, Selection { predicate: r, .. }) => {
                    AExpr::eq(*l, *r, expr_arena)
                }
                (
                    Sort {
                        by_column: l,
                        reverse: r_l,
                        ..
                    },
                    Sort {
                        by_column: r,
                        reverse: r_r,
                        ..
                    },
                ) => l == r && r_l == r_r,
                (Explode { columns: l, .. }, Explode { columns: r, .. }) => l == r,
                (
                    Distinct {
                        maintain_order: l1,
                        subset: l2,
                        ..
                    },
                    Distinct {
                        maintain_order: r1,
                        subset: r2,
                        ..
                    },
                ) => l1 == r1 && l2 == r2,
                (a, b) => {
                    std::mem::discriminant(a) == std::mem::discriminant(b)
                        && a.schema(lp_arena) == b.schema(lp_arena)
                }
            }
        };

        lp_arena
            .iter(node_left)
            .zip(lp_arena.iter(node_right))
            .map(|(tpll, tplr)| (tpll.0, tplr.0))
            .all(cmp)
    }
}

impl ALogicalPlan {
    /// Takes the expressions of an LP node and the inputs of that node and reconstruct
    pub fn from_exprs_and_input(&self, mut exprs: Vec<Node>, inputs: Vec<Node>) -> ALogicalPlan {
        use ALogicalPlan::*;

        match self {
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
                ..
            } => Aggregate {
                input: inputs[0],
                keys: exprs[..keys.len()].to_vec(),
                aggs: exprs[keys.len()..].to_vec(),
                schema: schema.clone(),
                apply: apply.clone(),
            },
            Join {
                schema,
                how,
                left_on,
                allow_par,
                force_par,
                ..
            } => Join {
                input_left: inputs[0],
                input_right: inputs[1],
                schema: schema.clone(),
                how: *how,
                left_on: exprs[..left_on.len()].to_vec(),
                right_on: exprs[left_on.len()..].to_vec(),
                allow_par: *allow_par,
                force_par: *force_par,
            },
            Sort {
                by_column, reverse, ..
            } => Sort {
                input: inputs[0],
                by_column: by_column.clone(),
                reverse: reverse.clone(),
            },
            Explode { columns, .. } => Explode {
                input: inputs[0],
                columns: columns.clone(),
            },
            Cache { .. } => Cache { input: inputs[0] },
            Distinct {
                maintain_order,
                subset,
                ..
            } => Distinct {
                input: inputs[0],
                maintain_order: *maintain_order,
                subset: subset.clone(),
            },
            HStack { schema, .. } => HStack {
                input: inputs[0],
                exprs,
                schema: schema.clone(),
            },
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                with_columns,
                predicate,
                stop_after_n_rows,
                cache,
                ..
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }

                ParquetScan {
                    path: path.clone(),
                    schema: schema.clone(),
                    with_columns: with_columns.clone(),
                    predicate: new_predicate,
                    aggregate: exprs,
                    stop_after_n_rows: *stop_after_n_rows,
                    cache: *cache,
                }
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                schema,
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
                predicate_pd,
                projection_pd,
                schema,
                ..
            } => Udf {
                input: inputs[0],
                function: function.clone(),
                predicate_pd: *predicate_pd,
                projection_pd: *projection_pd,
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
    /// or an in-memory DataFrame has none.
    pub(crate) fn copy_inputs<T>(&self, container: &mut T)
    where
        T: PushNode,
    {
        use ALogicalPlan::*;
        let input = match self {
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

    pub fn melt(self, id_vars: Arc<Vec<String>>, value_vars: Arc<Vec<String>>) -> Self {
        let schema = det_melt_schema(&value_vars, self.schema());

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
        if !exprs.is_empty() {
            let lp = ALogicalPlan::LocalProjection {
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
        // current schema
        let schema = self.schema();

        let mut new_fields = schema.fields().clone();

        for e in &exprs {
            let field = self
                .expr_arena
                .get(*e)
                .to_field(schema, Context::Default, self.expr_arena)
                .unwrap();
            match schema.index_of(field.name()) {
                Ok(idx) => {
                    new_fields[idx] = field;
                }
                Err(_) => new_fields.push(field),
            }
        }

        let new_schema = Schema::new(new_fields);

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
    ) -> Self {
        debug_assert!(!keys.is_empty());
        let current_schema = self.schema();
        // TODO! add this line if LogicalPlan is dropped in favor of ALogicalPlan
        // let aggs = rewrite_projections(aggs, current_schema);

        let schema1 = aexprs_to_schema(&keys, current_schema, Context::Default, self.expr_arena);
        let schema2 =
            aexprs_to_schema(&aggs, current_schema, Context::Aggregation, self.expr_arena);

        let schema = Schema::try_merge(&[schema1, schema2]).unwrap();

        let lp = ALogicalPlan::Aggregate {
            input: self.root,
            keys,
            aggs,
            schema: Arc::new(schema),
            apply,
        };
        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }

    pub fn join(
        self,
        other: Node,
        how: JoinType,
        left_on: Vec<Node>,
        right_on: Vec<Node>,
        allow_par: bool,
        force_par: bool,
    ) -> Self {
        let schema_left = self.schema();
        let schema_right = self.lp_arena.get(other).schema(self.lp_arena);

        // column names of left table
        let mut names: HashSet<&String, RandomState> = HashSet::with_capacity_and_hasher(
            schema_left.len() + schema_right.len(),
            Default::default(),
        );
        // fields of new schema
        let mut fields = Vec::with_capacity(schema_left.len() + schema_right.len());

        for f in schema_left.fields() {
            names.insert(f.name());
            fields.push(f.clone());
        }

        let right_names: HashSet<_, RandomState> = right_on
            .iter()
            .map(|e| match self.expr_arena.get(*e) {
                AExpr::Alias(_, name) => name.clone(),
                AExpr::Column(name) => name.clone(),
                _ => panic!("could not determine join column names"),
            })
            .collect();

        for f in schema_right.fields() {
            let name = f.name();
            if !right_names.contains(name) {
                if names.contains(name) {
                    let new_name = format!("{}_right", name);
                    let field = Field::new(&new_name, f.data_type().clone());
                    fields.push(field)
                } else {
                    fields.push(f.clone())
                }
            }
        }

        let schema = Arc::new(Schema::new(fields));

        let lp = ALogicalPlan::Join {
            input_left: self.root,
            input_right: other,
            how,
            schema,
            left_on,
            right_on,
            allow_par,
            force_par,
        };
        let root = self.lp_arena.add(lp);
        Self::new(root, self.expr_arena, self.lp_arena)
    }
}
