pub mod edge;

use std::ops::ControlFlow;
use std::sync::Arc;

use edge::Edge;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, DataType, PlHashMap, ScratchIndexMap, ScratchIndexSet};
use polars_core::schema::Schema;
use polars_io::RowIndex;
use polars_ops::frame::{JoinCoalesce, JoinType};
use polars_utils::arena::{Arena, Node};
use polars_utils::format_pl_smallstr;
use polars_utils::itertools::Itertools as _;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::scratch_vec::ScratchVec;

use crate::dsl::{FileScanIR, JoinTypeOptionsIR, PredicateFileSkip};
use crate::plans::optimizer::ir_traversal::ir_graph_traversal;
use crate::plans::optimizer::ir_traversal::storage::IRTraversalStorage;
use crate::plans::optimizer::projection_pushdown2::edge::{
    GetParentKeyAndPort as _, GetProjectionState, ParentKeyAndPort, Projection, ProjectionState,
};
use crate::plans::projection_height::{ExprProjectionHeight, aexpr_projection_height_rec};
use crate::plans::{
    AExpr, ArenaExprIter, ExprIR, ExprOrigin, FunctionIR, IR, IRAggExpr, IRBuilder, OutputName,
    det_join_schema,
};
use crate::prelude::{DistinctOptionsIR, ProjectionOptions};
use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};
use crate::utils::{aexpr_to_leaf_names_iter, rename_columns};

pub fn projection_pushdown(root: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) {
    let schema_cache = &mut PlHashMap::default();

    // Put a simple projection on top so that the root node has a valid ParentKeyAndPort to point to.
    let root_ir = ir_arena.take(root);
    let optimize_root = ir_arena.add(root_ir);

    ir_arena.replace(
        root,
        IR::SimpleProjection {
            input: optimize_root,
            columns: IR::schema_with_cache(optimize_root, ir_arena, schema_cache),
        },
    );

    ir_graph_traversal(
        optimize_root,
        &mut ProjectionPushdownVisitor {
            expr_arena,
            schema_cache,
            ae_nodes_scratch: &mut ScratchVec::default(),
            ae_height_scratch: &mut ScratchVec::default(),
            names_set_scratch: &mut ScratchIndexSet::default(),
            names_set_scratch2: &mut ScratchIndexSet::default(),
            names_set_scratch3: &mut ScratchIndexSet::default(),
            rename_map: &mut ScratchIndexMap::default(),
            default_edge: Edge::new(
                Projection::All,
                None,
                ParentKeyAndPort { node: root, idx: 0 },
            ),
            maintain_errors: false,
        },
        &mut vec![],
        &mut vec![],
        IRTraversalStorage {
            arena: ir_arena,
            skip_subtree: |_| false,
        },
    )
    .continue_value()
    .unwrap();

    // Assign optimized plan back to root node.
    let IR::SimpleProjection { input, columns: _ } = ir_arena.take(root) else {
        unreachable!()
    };

    ir_arena.swap(root, input);
}

pub struct ProjectionPushdownVisitor<'a, 'arena> {
    expr_arena: &'arena mut Arena<AExpr>,
    schema_cache: &'a mut PlHashMap<Node, Arc<Schema>>,
    ae_nodes_scratch: &'a mut ScratchVec<Node>,
    ae_height_scratch: &'a mut ScratchVec<ExprProjectionHeight>,
    names_set_scratch: &'a mut ScratchIndexSet<PlSmallStr>,
    names_set_scratch2: &'a mut ScratchIndexSet<PlSmallStr>,
    names_set_scratch3: &'a mut ScratchIndexSet<PlSmallStr>,
    rename_map: &'a mut ScratchIndexMap<PlSmallStr, PlSmallStr>,
    default_edge: Edge,
    maintain_errors: bool,
}

impl<'a, 'arena> NodeVisitor for ProjectionPushdownVisitor<'a, 'arena> {
    type Edge = Edge;
    type BreakValue = ();
    type Key = Node;
    type Storage = IRTraversalStorage<'arena>;

    fn default_edge(
        &mut self,
        _key: Self::Key,
        parent_key_and_port: Option<(Self::Key, usize)>,
    ) -> Self::Edge {
        let mut edge = self.default_edge.clone();

        if let Some((node, idx)) = parent_key_and_port {
            *edge.parent_key_and_port_mut() = ParentKeyAndPort { node, idx }
        }

        edge
    }

    fn is_deleted_edge(&mut self, edge: &Self::Edge) -> Option<bool> {
        Some(edge.parent_key_and_port().is_deleted())
    }

    fn pre_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn NodeEdgesProvider<Self::Edge>,
    ) -> std::ops::ControlFlow<Self::BreakValue, crate::traversal::visitor::SubtreeVisit> {
        self.pushdown(key, storage, edges);
        ControlFlow::Continue(SubtreeVisit::Visit)
    }

    fn post_visit(
        &mut self,
        key: Self::Key,
        storage: &mut Self::Storage,
        edges: &mut dyn crate::traversal::edge_provider::NodeEdgesProvider<Self::Edge>,
    ) -> std::ops::ControlFlow<Self::BreakValue> {
        let out_edge = &mut edges.outputs()[0];

        // This node was unlinked. We skip post-visit but remove the deletion mark,
        // as otherwise the parent node will not be visited.
        if out_edge.parent_key_and_port().is_deleted() {
            out_edge.parent_key_and_port_mut().set_deleted(false);
            return ControlFlow::Continue(());
        }

        let parent_key_and_port = out_edge.parent_key_and_port();

        'patch_ext_context: {
            let IR::ExtContext { schema, .. } = storage.get(key) else {
                break 'patch_ext_context;
            };

            let schema = match storage.get(parent_key_and_port.node) {
                // Replace simple-projection added from pre-visit
                IR::SimpleProjection { columns, .. } => columns.clone(),
                // Wrap in `Select {}` if it is the root node, otherwise it only returns cols from first input.
                _ if parent_key_and_port.node == self.default_edge.parent_key_and_port().node => {
                    schema.clone()
                },
                _ => break 'patch_ext_context,
            };

            let mut exprs = Vec::with_capacity(schema.len());
            let schema = schema.clone();
            exprs.extend(
                schema
                    .iter_names_cloned()
                    .map(|name| ExprIR::from_column_name(name, self.expr_arena)),
            );

            let ext_ctx_ir = storage.take(key);
            let new_key = storage.add(ext_ctx_ir);

            storage.replace(
                key,
                IR::Select {
                    input: new_key,
                    expr: exprs,
                    schema,
                    options: ProjectionOptions {
                        run_parallel: false,
                        duplicate_check: false,
                        should_broadcast: true,
                    },
                },
            );
        }

        ControlFlow::Continue(())
    }
}

impl ProjectionPushdownVisitor<'_, '_> {
    fn pushdown(
        &mut self,
        key: <Self as NodeVisitor>::Key,
        storage: &mut <Self as NodeVisitor>::Storage,
        edges: &mut dyn NodeEdgesProvider<<Self as NodeVisitor>::Edge>,
    ) {
        use std::mem;

        let num_input_edges = edges.inputs().len();

        fn unlink_current_node(
            edges: &mut dyn NodeEdgesProvider<<ProjectionPushdownVisitor as NodeVisitor>::Edge>,
            input_node: Node,
            storage: &mut Arena<IR>,
        ) {
            assert!(edges.outputs().len() == 1 && edges.inputs().len() == 1);
            let out_edge = &mut edges.outputs()[0];
            let parent_key_and_port = out_edge.parent_key_and_port();

            *storage
                .get_mut(parent_key_and_port.node)
                .inputs_mut()
                .nth(parent_key_and_port.idx)
                .unwrap() = input_node;

            edges.swap_input_output(0, 0);
            edges.outputs()[0]
                .parent_key_and_port_mut()
                .set_deleted(true);
        }

        macro_rules! unlink_current_node_and_return {
            ($input_node:expr) => {{
                unlink_current_node(edges, $input_node, storage);
                return;
            }};
        }

        let current_node_schema = IR::schema_with_cache(key, storage, self.schema_cache);
        if !matches!(storage.get(key), IR::Cache { .. }) {
            assert_eq!(edges.outputs().len(), 1);
        }
        let out_edge: &mut Edge = &mut edges.outputs()[0];

        /// Note: Materializes `::Len` to a name.
        macro_rules! projected_names_subset_or_return {
            () => {{
                if out_edge
                    .compute_projected_names_subset(&current_node_schema)
                    .is_none()
                {
                    // Names could be in different order
                    if let Some(names) = out_edge.compute_projected_names(&current_node_schema)
                        && let Some(schema) = compute_simple_projection_schema(
                            names.as_slice(),
                            &current_node_schema,
                            false,
                        )
                    {
                        out_edge
                            .parent_key_and_port_mut()
                            .attach_simple_projection(Arc::new(schema), storage);
                    }

                    return;
                };

                if out_edge.projection() == Projection::Len {
                    *out_edge.projection_mut() = Projection::Names;
                }

                (out_edge.names_mut(), &current_node_schema)
            }};
        }

        fn pushdown_with_added_names(
            key: Node,
            edges: &mut dyn NodeEdgesProvider<<ProjectionPushdownVisitor as NodeVisitor>::Edge>,
            len_before_added_names: usize,
            storage: &mut Arena<IR>,
            current_node_schema: &Schema,
            schema_cache: &mut PlHashMap<Node, Arc<Schema>>,
        ) {
            let out_edge = &mut edges.outputs()[0];
            let projected_names = out_edge.names();

            if projected_names.len() != len_before_added_names {
                let schema = Arc::new(
                    compute_simple_projection_schema(
                        &projected_names[..len_before_added_names],
                        current_node_schema,
                        false,
                    )
                    .unwrap(),
                );
                out_edge
                    .parent_key_and_port_mut()
                    .attach_simple_projection(schema, storage);
            }

            let mut projected_names = Some(out_edge.take_names().unwrap());
            let num_input_edges = edges.inputs().len();

            for (i, node) in (0..edges.inputs().len()).zip_eq(storage.get(key).inputs()) {
                let input_schema = IR::schema_with_cache(node, storage, schema_cache);

                let mut names = if i + 1 == num_input_edges {
                    projected_names.take()
                } else {
                    projected_names.clone()
                }
                .unwrap();

                names.retain(|name| input_schema.contains(name));

                *edges.inputs()[i].projection_state_mut() = ProjectionState {
                    projection: Projection::Names,
                    names: Some(names),
                };
            }
        }

        /// Pushdown names to all inputs. If names were added by the current node,
        /// a projection is added on top of the current node to drop them.
        macro_rules! pushdown_with_added_names {
            ($len_before_added_names:expr) => {
                pushdown_with_added_names(
                    key,
                    edges,
                    $len_before_added_names,
                    storage,
                    &current_node_schema,
                    self.schema_cache,
                )
            };
        }

        /// Names will not be pushed past the current node.
        fn post_project(
            edges: &mut dyn NodeEdgesProvider<<ProjectionPushdownVisitor as NodeVisitor>::Edge>,
            current_node_schema: &Schema,
            storage: &mut Arena<IR>,
        ) {
            let out_edge = &mut edges.outputs()[0];

            if let Some(names) = out_edge.compute_projected_names(current_node_schema)
                && let Some(schema) =
                    compute_simple_projection_schema(names.as_slice(), current_node_schema, false)
            {
                out_edge
                    .parent_key_and_port_mut()
                    .attach_simple_projection(Arc::new(schema), storage);
            }

            reuse_names_alloc(edges);
        }

        macro_rules! post_project_and_return {
            () => {{
                post_project(edges, &current_node_schema, storage);
                return;
            }};
        }

        match storage.get_mut(key) {
            IR::SimpleProjection { input, columns } => {
                match out_edge.projection() {
                    Projection::All => {
                        let names = out_edge.names_mut();
                        names.clear();
                        names.extend(columns.iter_names_cloned());
                        *out_edge.projection_mut() = Projection::Names;
                    },
                    Projection::Names => {
                        let initial_len = out_edge.names().len();
                        out_edge.names_mut().retain(|name| columns.contains(name));
                        // We should raise ColumnNotFound if this fails, but this should already
                        // be checked in DSL->IR.
                        assert_eq!(out_edge.names().len(), initial_len);
                    },
                    Projection::Len => {},
                }

                let input = *input;
                unlink_current_node_and_return!(input)
            },

            IR::Select {
                input,
                expr: exprs,
                schema,
                ..
            } => {
                use ExprProjectionHeight as EH;
                let input_node = *input;

                'len_propagate: {
                    // lf.select(a + 1).select(len()) -> lf.select(len())

                    if out_edge.projection() != Projection::Len {
                        break 'len_propagate;
                    }

                    let mut has_column = false;
                    for e in exprs.iter() {
                        match aexpr_projection_height_rec(
                            e.node(),
                            self.expr_arena,
                            self.ae_nodes_scratch,
                            self.ae_height_scratch,
                        ) {
                            EH::Unknown => break 'len_propagate,
                            EH::Column => has_column = true,
                            EH::Scalar => {},
                        }
                    }

                    if !has_column {
                        break 'len_propagate;
                    }

                    unlink_current_node_and_return!(input_node)
                }

                if let Some(projected_names) = out_edge.compute_projected_names(schema) {
                    // Sort exprs to projection order. Non-projected exprs to the end.
                    exprs.sort_by_key(|e| {
                        projected_names
                            .get_index_of(e.output_name())
                            .unwrap_or(usize::MAX)
                    });

                    let mut truncate_len = projected_names.len();

                    // If exprs has any column-height output, at least 1 of them must be projected.
                    if let Some(column_height_idx) = exprs.iter().position(|e| {
                        matches!(
                            aexpr_projection_height_rec(
                                e.node(),
                                self.expr_arena,
                                self.ae_nodes_scratch,
                                self.ae_height_scratch,
                            ),
                            EH::Column
                        )
                    }) && column_height_idx >= truncate_len
                    {
                        exprs.swap(column_height_idx, projected_names.len());
                        truncate_len += 1;
                    }

                    // Project all unknown heights to catch length mismatch errors.
                    if self.maintain_errors {
                        let range = truncate_len..exprs.len();
                        for i in range {
                            match aexpr_projection_height_rec(
                                exprs[i].node(),
                                self.expr_arena,
                                self.ae_nodes_scratch,
                                self.ae_height_scratch,
                            ) {
                                EH::Scalar | EH::Column => {},
                                EH::Unknown => {
                                    exprs.swap(i, truncate_len);
                                    truncate_len += 1;
                                },
                            }
                        }
                    }

                    exprs.truncate(truncate_len);

                    let schema_arc = schema;

                    // Update schema if changed.
                    if !iters_eq(
                        schema_arc.iter_names(),
                        exprs.iter().map(|e| e.output_name()),
                    ) {
                        *schema_arc = Arc::new(Schema::from_iter(exprs.iter().map(|e| {
                            (
                                e.output_name().clone(),
                                schema_arc.get(e.output_name()).unwrap().clone(),
                            )
                        })));
                    }

                    // If exprs length is equal, they are already in order. Otherwise, we need to attach
                    // a simple projection.
                    if exprs.len() != projected_names.len() {
                        let schema = Arc::new(
                            compute_simple_projection_schema(
                                projected_names.as_slice(),
                                schema_arc,
                                true,
                            )
                            .unwrap(),
                        );
                        out_edge
                            .parent_key_and_port_mut()
                            .attach_simple_projection(schema, storage);
                    }
                }

                let IR::Select { expr: exprs, .. } = storage.get_mut(key) else {
                    unreachable!()
                };

                if exprs.len() == 1
                    && match self.expr_arena.get(exprs[0].node()) {
                        AExpr::Len => true,
                        // select(col(a).len()) -> select(len().alias(a))
                        AExpr::Agg(IRAggExpr::Count {
                            input,
                            include_nulls: true,
                        }) if matches!(
                            aexpr_projection_height_rec(
                                *input,
                                self.expr_arena,
                                self.ae_nodes_scratch,
                                self.ae_height_scratch
                            ),
                            EH::Column
                        ) =>
                        {
                            self.expr_arena.replace(exprs[0].node(), AExpr::Len);
                            true
                        },
                        _ => false,
                    }
                {
                    let name = exprs[0].output_name().clone();
                    exprs[0].set_alias(name);
                    *edges.inputs()[0].projection_mut() = Projection::Len;
                    reuse_names_alloc(edges);
                } else {
                    // Determine names referenced by exprs as the projection to pushdown.
                    let input_names_projection = self.names_set_scratch.get();
                    let mut has_non_simple_projection = false;
                    let mut has_window_expr = false;

                    for e in exprs {
                        input_names_projection
                            .extend(aexpr_to_leaf_names_iter(e.node(), self.expr_arena).cloned());

                        has_window_expr = has_window_expr
                            || self
                                .expr_arena
                                .iter(e.node())
                                .any(|(_, e)| matches!(e, AExpr::Over { .. }));

                        match self.expr_arena.get(e.node()) {
                            AExpr::Column(name) if name == e.output_name() => {},
                            _ => has_non_simple_projection = true,
                        }
                    }

                    let input_schema =
                        IR::schema_with_cache(input_node, storage, self.schema_cache);

                    // E.g. `select(<literal>.sum().over(<literal>))`, we must project at least 1 input
                    // column for height.
                    if has_window_expr && input_names_projection.is_empty() {
                        input_names_projection
                            .extend(min_dtype_size_col(input_schema.iter()).cloned())
                    }

                    let prune = !has_non_simple_projection;

                    if input_names_projection.len() != input_schema.len()
                        || (prune
                            && !iters_eq(input_names_projection.iter(), input_schema.iter_names()))
                    {
                        mem::swap(out_edge.names_mut(), input_names_projection);
                        let names = out_edge.take_names();
                        *edges.inputs()[0].projection_state_mut() = ProjectionState {
                            projection: Projection::Names,
                            names,
                        };
                    } else {
                        reuse_names_alloc(edges)
                    }

                    // All column-projections; unlink this `select()`.
                    if prune {
                        let out_edge = &mut edges.outputs()[0];
                        let parent_key_and_port = out_edge.parent_key_and_port().clone();

                        *storage
                            .get_mut(parent_key_and_port.node)
                            .inputs_mut()
                            .nth(parent_key_and_port.idx)
                            .unwrap() = input_node;

                        let in_edge = &mut edges.inputs()[0];
                        *in_edge.parent_key_and_port_mut() = parent_key_and_port;
                        edges.outputs()[0]
                            .parent_key_and_port_mut()
                            .set_deleted(true);
                    }
                }
            },

            IR::HStack { input, .. } => {
                use ExprProjectionHeight as EH;

                let input = *input;

                if out_edge.projection() == Projection::Len {
                    unlink_current_node_and_return!(input)
                }

                let original_schema = current_node_schema;
                let input_schema = IR::schema_with_cache(input, storage, self.schema_cache);

                let IR::HStack {
                    exprs,
                    schema: output_schema,
                    ..
                } = storage.get_mut(key)
                else {
                    unreachable!()
                };

                let expr_output_names = self.names_set_scratch.get();
                let input_names_projection = self.names_set_scratch2.get();

                let opt_projected_names = out_edge.compute_projected_names(output_schema);

                // Prune exprs whose output names are unused.
                let orig_exprs_len = exprs.len();

                if let Some(projected_names) = &opt_projected_names {
                    exprs.retain(|e| {
                        projected_names.contains(e.output_name())
                            || (self.maintain_errors
                                && matches!(
                                    aexpr_projection_height_rec(
                                        e.node(),
                                        self.expr_arena,
                                        self.ae_nodes_scratch,
                                        self.ae_height_scratch
                                    ),
                                    EH::Unknown
                                ))
                    });
                }

                if exprs.is_empty() {
                    unlink_current_node_and_return!(input)
                }

                input_names_projection.reserve(input_schema.len());

                // * Add all column names referenced by projected exprs
                // * At the same time, build a hashset lookup for all expr output names.
                for e in exprs.iter() {
                    expr_output_names.insert(e.output_name().clone());
                    input_names_projection
                        .extend(aexpr_to_leaf_names_iter(e.node(), self.expr_arena).cloned());
                }

                // Directly passed to output (i.e. do not originate from exprs) that are being
                // referenced. These do not exist in the expr output names hashset.
                if let Some(projected_names) = &opt_projected_names {
                    for name in projected_names.iter() {
                        if !expr_output_names.contains(name) {
                            input_names_projection.insert(name.clone());
                        }
                    }
                } else {
                    for name in output_schema.iter_names() {
                        if !expr_output_names.contains(name) {
                            input_names_projection.insert(name.clone());
                        }
                    }
                }

                // Must project at least 1 column for height.
                if input_names_projection.is_empty() {
                    input_names_projection.extend(min_dtype_size_col(input_schema.iter()).cloned());
                }

                let output_schema_arc = output_schema;

                if exprs.len() != orig_exprs_len
                    || input_names_projection.len() != input_schema.len()
                {
                    let output_schema = Arc::make_mut(output_schema_arc);
                    let mut orig_schema = mem::take(output_schema);

                    for name in input_names_projection.iter() {
                        output_schema
                            .extend(orig_schema.remove(name).map(|dtype| (name.clone(), dtype)));
                    }

                    for name in expr_output_names.iter() {
                        output_schema
                            .extend(orig_schema.remove(name).map(|dtype| (name.clone(), dtype)));
                    }
                }

                // Generate again against updated output schema.
                let opt_projected_names = out_edge.compute_projected_names(output_schema_arc);

                if let Some(schema) = opt_projected_names
                    .and_then(|projected_names| {
                        compute_simple_projection_schema(
                            projected_names.as_slice(),
                            output_schema_arc,
                            false,
                        )
                        .map(Arc::new)
                    })
                    .or_else(|| {
                        (out_edge.projection() == Projection::All
                            && input_names_projection.len() != input_schema.len())
                        .then_some(original_schema)
                    })
                {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(schema, storage);
                }

                if input_names_projection.len() != input_schema.len() {
                    mem::swap(out_edge.names_mut(), input_names_projection);
                    let names = out_edge.take_names();
                    *edges.inputs()[0].projection_state_mut() = ProjectionState {
                        projection: Projection::Names,
                        names,
                    };
                } else {
                    reuse_names_alloc(edges)
                }
            },

            IR::Filter { input, predicate } => {
                let predicate_node = predicate.node();

                'len_to_predicate_sum: {
                    // .filter(x == 3).select(pl.len()) -> .select((x == 3).sum())
                    if out_edge.projection() != Projection::Len {
                        break 'len_to_predicate_sum;
                    }

                    let input = *input;

                    if !matches!(
                        aexpr_projection_height_rec(
                            predicate_node,
                            self.expr_arena,
                            self.ae_nodes_scratch,
                            self.ae_height_scratch
                        ),
                        ExprProjectionHeight::Column
                    ) {
                        break 'len_to_predicate_sum;
                    }

                    let Some(expr) = extract_select_len_expr(
                        storage.get_mut(out_edge.parent_key_and_port().node),
                        self.expr_arena,
                    ) else {
                        break 'len_to_predicate_sum;
                    };

                    let sum_expr = self
                        .expr_arena
                        .add(AExpr::Agg(IRAggExpr::Sum(predicate_node)));

                    expr.set_node(sum_expr);

                    *out_edge.projection_mut() = Projection::Names;
                    let names = out_edge.names_mut();
                    names.clear();
                    names.extend(aexpr_to_leaf_names_iter(sum_expr, self.expr_arena).cloned());

                    unlink_current_node_and_return!(input)
                }

                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();

                projected_names
                    .extend(aexpr_to_leaf_names_iter(predicate_node, self.expr_arena).cloned());

                pushdown_with_added_names!(len_before_added_names)
            },

            IR::Join {
                input_left,
                input_right,
                ..
            } => {
                assert_eq!(num_input_edges, 2);

                let input_left = *input_left;
                let input_right = *input_right;
                let input_schema_left =
                    IR::schema_with_cache(input_left, storage, self.schema_cache);
                let input_schema_right =
                    IR::schema_with_cache(input_right, storage, self.schema_cache);

                let IR::Join {
                    schema: output_schema_arc,
                    left_on,
                    right_on,
                    options,
                    ..
                } = storage.get_mut(key)
                else {
                    unreachable!()
                };

                let opt_projected_names = out_edge.compute_projected_names(output_schema_arc);

                let is_projected_in_output = |name: &str| {
                    if let Some(projected_names) = &opt_projected_names {
                        projected_names.contains(name)
                    } else {
                        output_schema_arc.contains(name)
                    }
                };

                let project_left = self.names_set_scratch.get();
                let project_right = self.names_set_scratch2.get();

                project_left.reserve(input_schema_left.len());
                project_right.reserve(input_schema_right.len());

                let coalesced_to_right = self.names_set_scratch3.get();

                if options.args.should_coalesce()
                    && let JoinType::Right = &options.args.how
                {
                    coalesced_to_right.extend(left_on.iter().map(|expr| {
                        let node = match self.expr_arena.get(expr.node()) {
                            AExpr::Cast {
                                expr,
                                dtype: _,
                                options: _,
                            } => *expr,

                            _ => expr.node(),
                        };

                        let AExpr::Column(name) = self.expr_arena.get(node) else {
                            // All keys should be columns when coalesce=True
                            unreachable!()
                        };

                        name.clone()
                    }))
                }

                let mut pred_used_names_iter = None;
                let mut has_cross_filter = false;

                if let Some(JoinTypeOptionsIR::CrossAndFilter { predicate }) = &options.options {
                    pred_used_names_iter =
                        Some(aexpr_to_leaf_names_iter(predicate.node(), self.expr_arena));
                    has_cross_filter = true;
                }

                // Add accumulated projections
                for output_name in output_schema_arc
                    .iter_names()
                    .filter(|name| is_projected_in_output(name))
                    .chain(pred_used_names_iter.into_iter().flatten())
                {
                    match ExprOrigin::get_column_origin(
                        output_name,
                        &input_schema_left,
                        &input_schema_right,
                        options.args.suffix(),
                        Some(&|name| coalesced_to_right.contains(name)),
                    )
                    .unwrap()
                    {
                        ExprOrigin::None => {},
                        ExprOrigin::Left => {
                            project_left.insert(output_name.clone());
                        },
                        ExprOrigin::Right => {
                            let name = if !input_schema_right.contains(output_name.as_str()) {
                                PlSmallStr::from_str(
                                    output_name
                                        .strip_suffix(options.args.suffix().as_str())
                                        .unwrap(),
                                )
                            } else {
                                output_name.clone()
                            };

                            debug_assert!(input_schema_right.contains(name.as_str()));

                            project_right.insert(name);
                        },
                        ExprOrigin::Both => unreachable!(),
                    }
                }

                // Add projections required by the join itself
                for expr_ir in left_on.as_slice() {
                    project_left
                        .extend(aexpr_to_leaf_names_iter(expr_ir.node(), self.expr_arena).cloned())
                }

                for expr_ir in right_on.as_slice() {
                    project_right
                        .extend(aexpr_to_leaf_names_iter(expr_ir.node(), self.expr_arena).cloned())
                }

                #[cfg(feature = "asof_join")]
                if let JoinType::AsOf(asof_options) = &options.args.how {
                    if let Some(left_by) = asof_options.left_by.as_deref() {
                        for name in left_by {
                            project_left.insert(name.clone());
                        }
                    }

                    if let Some(right_by) = asof_options.right_by.as_deref() {
                        for name in right_by {
                            project_right.insert(name.clone());
                        }
                    }
                }

                // Turn on coalesce if non-coalesced keys are not included in projection. Reduces materialization.
                if !options.args.should_coalesce()
                    && matches!(options.args.how, JoinType::Inner | JoinType::Left)
                    && left_on
                        .iter()
                        .all(|e| matches!(self.expr_arena.get(e.node()), AExpr::Column(_)))
                    && right_on.iter().all(|e| {
                        let AExpr::Column(name) = self.expr_arena.get(e.node()) else {
                            return false;
                        };

                        let projected = if input_schema_left.contains(name.as_str()) {
                            let name = format_pl_smallstr!("{}{}", name, options.args.suffix());
                            is_projected_in_output(&name)
                        } else {
                            is_projected_in_output(name)
                        };

                        !projected
                    })
                {
                    Arc::make_mut(options).args.coalesce = JoinCoalesce::CoalesceColumns;
                }

                let new_input_schema_left = if project_left.len() == input_schema_left.len() {
                    input_schema_left.clone()
                } else {
                    Arc::new(input_schema_left.try_project(project_left.iter()).unwrap())
                };

                let new_input_schema_right = if project_right.len() == input_schema_right.len() {
                    input_schema_right.clone()
                } else {
                    Arc::new(
                        input_schema_right
                            .try_project(project_right.iter())
                            .unwrap(),
                    )
                };

                let new_output_schema = det_join_schema(
                    &new_input_schema_left,
                    &new_input_schema_right,
                    left_on,
                    right_on,
                    options,
                    self.expr_arena,
                )
                .unwrap();

                if project_left.len() != input_schema_left.len() {
                    *edges.inputs()[0].projection_state_mut() = ProjectionState {
                        projection: Projection::Names,
                        names: Some(Box::new(mem::take(project_left))),
                    };
                }

                if project_right.len() != input_schema_right.len() {
                    *edges.inputs()[1].projection_state_mut() = ProjectionState {
                        projection: Projection::Names,
                        names: Some(Box::new(mem::take(project_right))),
                    };
                }

                let out_edge = &mut edges.outputs()[0];
                let opt_projected_names = out_edge.compute_projected_names(output_schema_arc);

                *output_schema_arc = new_output_schema;

                if let Some(projected_names) = &opt_projected_names {
                    let orig_to_new_name_map = self.rename_map.get();

                    for name in projected_names.iter() {
                        if !output_schema_arc.contains(name) {
                            let new_output_name = PlSmallStr::from_str(
                                name.strip_suffix(options.args.suffix().as_str()).unwrap(),
                            );

                            orig_to_new_name_map.insert(name.clone(), new_output_name);
                        }
                    }

                    if !orig_to_new_name_map.is_empty() {
                        if has_cross_filter {
                            let Some(JoinTypeOptionsIR::CrossAndFilter { predicate }) =
                                &mut Arc::make_mut(options).options
                            else {
                                unreachable!()
                            };

                            predicate.set_node(rename_columns(
                                predicate.node(),
                                self.expr_arena,
                                orig_to_new_name_map,
                            ));
                        }

                        let post_project = opt_projected_names
                            .unwrap()
                            .iter()
                            .map(|name| {
                                if let Some(new_name) = orig_to_new_name_map.get(name) {
                                    ExprIR::new(
                                        self.expr_arena.add(AExpr::Column(new_name.clone())),
                                        OutputName::Alias(name.clone()),
                                    )
                                } else {
                                    ExprIR::from_column_name(name.clone(), self.expr_arena)
                                }
                            })
                            .collect();

                        let post_project_node = IRBuilder::new(key, self.expr_arena, storage)
                            .project(
                                post_project,
                                ProjectionOptions {
                                    run_parallel: false,
                                    duplicate_check: false,
                                    should_broadcast: false,
                                },
                            )
                            .node();

                        let out_edge = &mut edges.outputs()[0];
                        let parent_key_and_port = out_edge.parent_key_and_port_mut();

                        *storage
                            .get_mut(parent_key_and_port.node)
                            .inputs_mut()
                            .nth(parent_key_and_port.idx)
                            .unwrap() = post_project_node;

                        *parent_key_and_port = ParentKeyAndPort {
                            node: post_project_node,
                            idx: 0,
                        };
                    } else if let Some(schema) = compute_simple_projection_schema(
                        projected_names.as_slice(),
                        output_schema_arc,
                        false,
                    ) {
                        edges.outputs()[0]
                            .parent_key_and_port_mut()
                            .attach_simple_projection(Arc::new(schema), storage);
                    }
                }
            },

            IR::GroupBy { apply: Some(_), .. } => {
                post_project_and_return!()
            },

            IR::GroupBy {
                input, apply: None, ..
            } => {
                // TODO: We could rewrite to Distinct for non-dynamic groupby.
                let is_len = out_edge.projection() == Projection::Len;
                let input_schema = IR::schema_with_cache(*input, storage, self.schema_cache);

                let IR::GroupBy {
                    aggs,
                    schema: output_schema_arc,
                    ..
                } = storage.get_mut(key)
                else {
                    unreachable!()
                };

                if let Some(projected_names) =
                    out_edge.compute_projected_names_subset(output_schema_arc)
                {
                    let removed_names = self.names_set_scratch.get();
                    aggs.retain(|e| {
                        let remove = !projected_names.contains(e.output_name()) || is_len;

                        if remove {
                            removed_names.insert(e.output_name().clone());
                        }

                        !remove
                    });

                    if !removed_names.is_empty() {
                        Arc::make_mut(output_schema_arc)
                            .retain(|name, _| !removed_names.contains(name))
                    }
                }

                if let Some(projected_names) = out_edge.compute_projected_names(output_schema_arc)
                    && let Some(schema) = compute_simple_projection_schema(
                        projected_names.as_slice(),
                        output_schema_arc,
                        false,
                    )
                {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(Arc::new(schema), storage);
                }

                let input_names_projection = self.names_set_scratch.get();

                for e in storage.get(key).exprs() {
                    input_names_projection
                        .extend(aexpr_to_leaf_names_iter(e.node(), self.expr_arena).cloned())
                }

                let IR::GroupBy { options, .. } = storage.get(key) else {
                    unreachable!()
                };

                #[cfg(feature = "dynamic_group_by")]
                if let Some(options) = &options.dynamic {
                    input_names_projection.insert(options.index_column.clone());
                }

                #[cfg(feature = "dynamic_group_by")]
                if let Some(options) = &options.rolling {
                    input_names_projection.insert(options.index_column.clone());
                }

                if input_names_projection.len() != input_schema.len() {
                    mem::swap(out_edge.names_mut(), input_names_projection);
                    let names = out_edge.take_names();

                    *edges.inputs()[0].projection_state_mut() = ProjectionState {
                        projection: Projection::Names,
                        names,
                    };
                } else {
                    reuse_names_alloc(edges)
                }
            },

            IR::Distinct {
                options: DistinctOptionsIR { subset: None, .. },
                ..
            } => {
                post_project_and_return!()
            },

            IR::Distinct {
                options:
                    DistinctOptionsIR {
                        subset: Some(subset),
                        ..
                    },
                ..
            } => {
                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();
                projected_names.extend(subset.iter().cloned());
                pushdown_with_added_names!(len_before_added_names)
            },

            ir @ IR::Union { .. } | ir @ IR::Slice { .. } | ir @ IR::Sort { .. } => {
                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();

                for e in ir.exprs() {
                    projected_names
                        .extend(aexpr_to_leaf_names_iter(e.node(), self.expr_arena).cloned())
                }

                pushdown_with_added_names!(len_before_added_names)
            },

            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted { key, .. } => {
                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();
                projected_names.insert(key.clone());
                pushdown_with_added_names!(len_before_added_names)
            },

            IR::ExtContext { schema, .. } => {
                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();
                Arc::make_mut(schema).retain(|name, _| projected_names.contains(name));
                pushdown_with_added_names!(len_before_added_names)
            },

            IR::HConcat {
                inputs, options, ..
            } => {
                let (..) = projected_names_subset_or_return!();

                let mut inputs = mem::take(inputs);
                let projected_names = out_edge.take_names().unwrap();
                let strict = options.strict;
                let hconcat_projected_names = self.names_set_scratch.get();

                assert_eq!(num_input_edges, inputs.len());

                let mut idx: usize = 0;
                let mut deleted: usize = 0;

                inputs.retain(|input_node| {
                    idx += 1;
                    let idx = idx - 1;

                    let input_node = *input_node;
                    let input_schema_arc =
                        IR::schema_with_cache(input_node, storage, self.schema_cache);
                    let base_new_names_len = hconcat_projected_names.len();
                    let in_edge = &mut edges.inputs()[idx];

                    let mut keep = false;

                    'set_keep: {
                        if input_schema_arc.is_empty() {
                            break 'set_keep;
                        }

                        hconcat_projected_names.extend(
                            input_schema_arc
                                .iter_names()
                                .filter(|name| projected_names.contains(*name))
                                .cloned(),
                        );

                        if hconcat_projected_names.len() == base_new_names_len {
                            if strict && !self.maintain_errors {
                                break 'set_keep;
                            }

                            hconcat_projected_names.insert(
                                min_dtype_size_col(input_schema_arc.iter()).unwrap().clone(),
                            );
                        }

                        let names_this_input = &hconcat_projected_names.as_slice()
                            [base_new_names_len..hconcat_projected_names.len()];

                        if names_this_input.len() != input_schema_arc.len() {
                            let mut names = mem::take(self.names_set_scratch2.get());
                            names.extend(names_this_input.iter().cloned());

                            *in_edge.projection_state_mut() = ProjectionState {
                                projection: Projection::Names,
                                names: Some(Box::new(names)),
                            };
                        }

                        keep = true;
                    };

                    let in_port = in_edge.parent_key_and_port_mut();
                    assert_eq!(in_port.idx, idx);

                    if !keep {
                        in_port.set_deleted(true);
                        deleted += 1;
                    } else if deleted != 0 {
                        in_port.idx = idx - deleted;
                    }

                    keep
                });

                let new_inputs = inputs;
                let IR::HConcat { inputs, schema, .. } = storage.get_mut(key) else {
                    unreachable!()
                };
                *inputs = new_inputs;

                Arc::make_mut(schema).retain(|name, _| hconcat_projected_names.contains(name));

                if hconcat_projected_names.len() != projected_names.len() {
                    edges.outputs()[0]
                        .parent_key_and_port_mut()
                        .attach_simple_projection(
                            Arc::new(
                                compute_simple_projection_schema(
                                    projected_names.as_slice(),
                                    schema,
                                    false,
                                )
                                .unwrap(),
                            ),
                            storage,
                        );
                }
            },

            IR::Scan { .. } => {
                let scan_schema = current_node_schema;

                let [
                    IR::Scan {
                        scan_type,
                        predicate,
                        predicate_file_skip_applied,
                        unified_scan_args,
                        ..
                    },
                    parent_ir,
                ] = storage.get_disjoint_mut([key, out_edge.parent_key_and_port().node])
                else {
                    unreachable!()
                };

                // Convert to fast-count MapFunction for CSV
                // New-streaming is generally on par for all except CSV (see https://github.com/pola-rs/polars/pull/22363).
                // In the future we can potentially remove the dedicated count codepaths.
                #[cfg(feature = "csv")]
                if out_edge.projection() == Projection::Len
                    && let FileScanIR::Csv { .. } = scan_type.as_ref()
                    && (predicate.is_none()
                        || matches!(
                            predicate_file_skip_applied,
                            Some(PredicateFileSkip {
                                no_residual_predicate: true,
                                ..
                            })
                        ))
                    && (match std::env::var("POLARS_NO_FAST_FILE_COUNT").as_deref() {
                        Ok("0") | Err(_) => true,
                        Ok("1") => false,
                        Ok(v) => {
                            panic!("POLARS_NO_FAST_FILE_COUNT must be one of ('0', '1'), got: {v}")
                        },
                    })
                    && let Some(name) = extract_select_len_expr(parent_ir, self.expr_arena)
                        .map(|e| e.output_name().clone())
                {
                    // Replace the scan with empty DF scan as input to the MapFunction.
                    let dummy_ir = IR::DataFrameScan {
                        df: Arc::new(Default::default()),
                        schema: Arc::new(Default::default()),
                        output_schema: None,
                    };

                    let IR::Scan {
                        sources,
                        scan_type,
                        unified_scan_args,
                        ..
                    } = storage.replace(key, dummy_ir)
                    else {
                        unreachable!()
                    };

                    storage.replace(
                        out_edge.parent_key_and_port().node,
                        IR::MapFunction {
                            input: key,
                            function: FunctionIR::FastCount {
                                sources,
                                scan_type,
                                alias: Some(name),
                                cloud_options: unified_scan_args.cloud_options,
                            },
                        },
                    );

                    return;
                }

                let projected_schema = if out_edge.projection() == Projection::Len
                    && !matches!(scan_type.as_ref(), FileScanIR::Anonymous { .. })
                {
                    // Streaming sources support 0-width projection with correct height.
                    Default::default()
                } else {
                    let Some(names) = out_edge.compute_projected_names(&scan_schema) else {
                        return;
                    };

                    let Some(projected_schema) =
                        compute_simple_projection_schema(names.as_slice(), &scan_schema, false)
                    else {
                        return;
                    };

                    projected_schema
                };

                let projected_schema = Arc::new(projected_schema);
                let mut scan_projected_schema = projected_schema.clone();

                if let Some(predicate) = predicate
                    && !matches!(
                        predicate_file_skip_applied,
                        Some(PredicateFileSkip {
                            no_residual_predicate: true,
                            ..
                        })
                    )
                {
                    for name in aexpr_to_leaf_names_iter(predicate.node(), self.expr_arena) {
                        Arc::make_mut(&mut scan_projected_schema)
                            .insert(name.clone(), scan_schema.get(name).unwrap().clone());
                    }
                }

                if match scan_type.as_ref() {
                    #[cfg(feature = "csv")]
                    FileScanIR::Csv { .. } => true,
                    _ => false,
                } {
                    Arc::make_mut(&mut scan_projected_schema)
                        .sort_by_key(|name, _| scan_schema.index_of(name));
                }

                if let Some(RowIndex { name, offset: _ }) = &unified_scan_args.row_index
                    && let Some(idx) = scan_projected_schema.index_of(name)
                    && idx != 0
                {
                    let schema = Arc::make_mut(&mut scan_projected_schema);
                    let (entry_name, dtype) = schema.shift_remove_index(idx).unwrap();
                    debug_assert_eq!(entry_name, name);
                    schema.insert_at_index(0, entry_name, dtype).unwrap();
                }

                if !iters_eq(
                    scan_projected_schema.iter_names(),
                    projected_schema.iter_names(),
                ) {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(projected_schema, storage);
                }

                set_scan_projection(storage.get_mut(key), scan_projected_schema);
            },

            IR::DataFrameScan {
                df,
                schema,
                output_schema,
            } => {
                if out_edge.projection() == Projection::Len {
                    if df.width() != 0 {
                        let new_df = unsafe {
                            DataFrame::new_unchecked(
                                df.height(),
                                df.columns()
                                    .iter()
                                    .map(|c| small_dummy_column(c.name().clone(), c.len()))
                                    .collect(),
                            )
                        };

                        *df = Arc::new(new_df);
                        *schema = df.schema().clone();
                        output_schema.replace(Arc::new(Schema::default()));
                    }

                    return;
                }

                let (projected_names, _) = projected_names_subset_or_return!();

                if let Some(projected_schema) =
                    compute_simple_projection_schema(projected_names.as_slice(), df.schema(), false)
                {
                    // Turn all non-projected columns into 0-field struct to drop the columns
                    // while keeping the total column count for the explain() output.
                    let new_df = unsafe {
                        DataFrame::new_unchecked(
                            df.height(),
                            df.columns()
                                .iter()
                                .map(|c| {
                                    if projected_names.contains(c.name()) {
                                        c.clone()
                                    } else {
                                        small_dummy_column(c.name().clone(), c.len())
                                    }
                                })
                                .collect(),
                        )
                    };

                    *df = Arc::new(new_df);
                    *schema = df.schema().clone();
                    output_schema.replace(Arc::new(projected_schema));
                }
            },

            #[cfg(feature = "python")]
            IR::PythonScan { options } => {
                use crate::plans::PythonPredicate;

                let (projected_names, _) = projected_names_subset_or_return!();

                let len_before_predicate_names = projected_names.len();

                if let PythonPredicate::Polars(pred) = &options.predicate {
                    projected_names
                        .extend(aexpr_to_leaf_names_iter(pred.node(), self.expr_arena).cloned())
                };

                let mut new_output_schema = current_node_schema.as_ref();

                if projected_names.len() != current_node_schema.len() {
                    let with_columns: Arc<[_]> = options
                        .schema
                        .iter_names()
                        .filter(|name| projected_names.contains(*name))
                        .cloned()
                        .collect();

                    options.output_schema.replace(Arc::new(
                        with_columns
                            .iter()
                            .map(|name| {
                                (name.clone(), current_node_schema.get(name).unwrap().clone())
                            })
                            .collect(),
                    ));
                    new_output_schema = options.output_schema.as_deref().unwrap();

                    options.with_columns = Some(with_columns)
                }

                if let Some(schema) = compute_simple_projection_schema(
                    &projected_names.as_slice()[..len_before_predicate_names],
                    new_output_schema,
                    false,
                ) {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(Arc::new(schema), storage);
                }
            },

            IR::Sink { .. } | IR::SinkMultiple { .. } => {
                debug_assert!(
                    edges
                        .outputs()
                        .get(0)
                        .is_none_or(|e| e.projection() == Projection::All)
                );
                reuse_names_alloc(edges)
            },

            IR::MapFunction {
                input,
                function: function @ FunctionIR::Hint(_),
                ..
            } => {
                let input = *input;
                let (projected_names, _) = projected_names_subset_or_return!();

                function.clear_cached_schema();

                let FunctionIR::Hint(hint) = function else {
                    unreachable!()
                };

                if !hint.retain_names(|name| projected_names.contains(name)) {
                    unlink_current_node_and_return!(input)
                }

                let [in_, out] = edges.get_input_output_mut(0, 0);
                mem::swap(out.projection_state_mut(), in_.projection_state_mut());
            },

            IR::MapFunction {
                input,
                function: function @ FunctionIR::RowIndex { .. },
            } => {
                let (projected_names, _) = projected_names_subset_or_return!();
                function.clear_cached_schema();

                let FunctionIR::RowIndex { name, .. } = function else {
                    unreachable!()
                };

                if !projected_names.shift_remove(name) {
                    unlink_current_node_and_return!(*input)
                }

                let names = out_edge.take_names();
                *edges.inputs()[0].projection_state_mut() = ProjectionState {
                    projection: Projection::Names,
                    names,
                };
            },

            #[cfg(feature = "pivot")]
            IR::MapFunction {
                function: FunctionIR::Unpivot { .. },
                ..
            } => {
                let schema = current_node_schema;

                if let Some(names) = out_edge.compute_projected_names(&schema)
                    && let Some(schema) =
                        compute_simple_projection_schema(names.as_slice(), &schema, false)
                {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(Arc::new(schema), storage);
                }

                let IR::MapFunction { function, .. } = storage.get_mut(key) else {
                    unreachable!()
                };

                function.clear_cached_schema();

                let FunctionIR::Unpivot { args, .. } = function else {
                    unreachable!()
                };

                let names = out_edge.names_mut();
                names.clear();
                names.extend(args.index.iter().cloned());
                names.extend(args.on.iter().cloned());

                let [in_edge, out_edge] = edges.get_input_output_mut(0, 0);
                *in_edge.projection_state_mut() = ProjectionState {
                    projection: Projection::Names,
                    names: out_edge.take_names(),
                };
            },

            IR::MapFunction {
                input,
                function: function @ FunctionIR::Unnest { .. },
            } => {
                let schema = current_node_schema;

                let Some(names) = out_edge.compute_projected_names(&schema) else {
                    return;
                };

                function.clear_cached_schema();

                let input = *input;
                let input_schema = IR::schema_with_cache(input, storage, self.schema_cache);

                if let Some(schema) =
                    compute_simple_projection_schema(names.as_slice(), &schema, false)
                {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(Arc::new(schema), storage);
                }

                let IR::MapFunction {
                    function: FunctionIR::Unnest { columns, .. },
                    ..
                } = storage.get_mut(key)
                else {
                    unreachable!()
                };

                let names = out_edge.names_mut();
                names.retain(|name| input_schema.contains(name));
                names.extend(columns.iter().cloned());
                let names = out_edge.take_names();

                let in_edge = &mut edges.inputs()[0];
                *in_edge.projection_state_mut() = ProjectionState {
                    projection: Projection::Names,
                    names,
                };
            },

            IR::MapFunction {
                input, function, ..
            } => {
                if !function.allow_projection_pd() {
                    post_project_and_return!();
                }

                let input = *input;
                let (projected_names, _) = projected_names_subset_or_return!();
                let input_schema = IR::schema_with_cache(input, storage, self.schema_cache);

                let IR::MapFunction { function, .. } = storage.get(key) else {
                    unreachable!()
                };

                function.clear_cached_schema();

                let input_projected_names = self.names_set_scratch.get();

                input_projected_names.extend(
                    projected_names
                        .iter()
                        .filter(|name| input_schema.contains(name))
                        .cloned(),
                );

                for name in function.additional_projection_pd_columns().as_ref() {
                    input_projected_names.insert(name.clone());
                }

                if let Some(schema) = compute_simple_projection_schema(
                    projected_names.as_slice(),
                    &current_node_schema,
                    false,
                ) {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(Arc::new(schema), storage);
                }

                mem::swap(out_edge.names_mut(), input_projected_names);
                let names = out_edge.take_names();

                *edges.inputs()[0].projection_state_mut() = ProjectionState {
                    projection: Projection::Names,
                    names,
                }
            },

            IR::Cache { input, .. } => {
                let input = *input;

                if edges.outputs().len() == 1 {
                    unlink_current_node_and_return!(input)
                }

                let schema = current_node_schema;
                let mut schema = schema.as_ref();
                let updated_schema;

                'push_common_subset: {
                    let mut projected_names = mem::take(self.names_set_scratch.get());

                    for i in 0..edges.outputs().len() {
                        let e = &mut edges.outputs()[i];
                        let Some(names) = e.compute_projected_names_subset(schema) else {
                            break 'push_common_subset;
                        };

                        projected_names.extend(names.iter().cloned());

                        if projected_names.len() == schema.len() {
                            break 'push_common_subset;
                        }
                    }

                    updated_schema = Some(schema.try_project(projected_names.iter()).unwrap());
                    schema = updated_schema.as_ref().unwrap();

                    *edges.inputs()[0].projection_state_mut() = ProjectionState {
                        projection: Projection::Names,
                        names: Some(Box::new(projected_names)),
                    };
                }

                for i in 0..edges.outputs().len() {
                    let e = &mut edges.outputs()[i];

                    if let Some(names) = e.compute_projected_names(schema)
                        && let Some(schema) =
                            compute_simple_projection_schema(names.as_slice(), schema, false)
                    {
                        e.parent_key_and_port_mut()
                            .attach_simple_projection(Arc::new(schema), storage);
                    }
                }
            },

            IR::UnoptimizedDispatch { .. } => {
                post_project_and_return!()
            },

            IR::Invalid => unreachable!(),
        };
    }
}

fn min_dtype_size_col<'a, I, K>(iter: I) -> Option<K>
where
    I: IntoIterator<Item = (K, &'a DataType)>,
{
    iter.into_iter()
        .min_by_key(|(_, dtype)| match () {
            _ if dtype.is_null() => 0,
            _ if dtype.is_bool() => 1,
            _ if !dtype.is_nested() => 2,
            _ => 32 + dtype.nesting_level(),
        })
        .map(|(k, _)| k)
}

/// Returns `None` if `projected_names` match `input_schema`.
fn compute_simple_projection_schema(
    projected_names: &indexmap::set::Slice<PlSmallStr>,
    input_schema: &Schema,
    allow_order_mismatch: bool,
) -> Option<Schema> {
    if projected_names.len() == input_schema.len() && {
        if allow_order_mismatch {
            projected_names
                .iter()
                .all(|name| input_schema.contains(name))
        } else {
            projected_names
                .iter()
                .zip(input_schema.iter_names())
                .all(|(l, r)| l == r)
        }
    } {
        return None;
    }

    Some(Schema::from_iter(projected_names.iter().map(|name| {
        let dtype = input_schema.get(name).unwrap().clone();
        (name.clone(), dtype)
    })))
}

/// Returns true if both iterators have the same length, and the items at each
/// index are equal.
fn iters_eq<L, R, T, U>(left: L, right: R) -> bool
where
    L: IntoIterator<Item = T>,
    R: IntoIterator<Item = U>,
    T: PartialEq<U>,
    L::IntoIter: ExactSizeIterator,
    R::IntoIter: ExactSizeIterator,
{
    let left = left.into_iter();
    let right = right.into_iter();
    left.len() == right.len() && left.zip(right).all(|(l, r)| l == r)
}

fn set_scan_projection(scan_ir: &mut IR, projection_schema: Arc<Schema>) {
    let IR::Scan {
        file_info,
        output_schema,
        unified_scan_args,
        ..
    } = scan_ir
    else {
        panic!()
    };

    if let Some(RowIndex { name, .. }) = unified_scan_args
        .row_index
        .take_if(|ri| !projection_schema.contains(&ri.name))
    {
        Arc::make_mut(&mut file_info.schema).shift_remove(&name);
    }

    if let Some(RowIndex { name, .. }) = &unified_scan_args.row_index {
        assert_eq!(projection_schema.index_of(name), Some(0));
    }

    if let Some(name) = unified_scan_args
        .include_file_paths
        .take_if(|name| !projection_schema.contains(name))
    {
        Arc::make_mut(&mut file_info.schema).shift_remove(&name);
    }

    unified_scan_args.projection = Some(
        projection_schema
            .iter_names()
            .filter(|name| {
                if let Some(ri) = &unified_scan_args.row_index
                    && ri.name == name
                {
                    return false;
                }

                if let Some(file_path_col) = &unified_scan_args.include_file_paths
                    && file_path_col == name.as_str()
                {
                    return false;
                }

                true
            })
            .cloned()
            .collect(),
    );

    *output_schema = Some(projection_schema);
}

/// Create a dummy column with small memory footprint.
fn small_dummy_column(name: PlSmallStr, height: usize) -> Column {
    // Prefer 0-field struct if possible, as it doesn't need validity allocation.
    #[cfg(feature = "dtype-struct")]
    {
        use arrow::array::StructArray;
        use arrow::datatypes::ArrowDataType;
        use polars_core::prelude::{IntoColumn, StructChunked};

        unsafe {
            StructChunked::from_chunks(
                name,
                vec![StructArray::new(ArrowDataType::Struct(vec![]), height, vec![], None).boxed()],
            )
        }
        .into_column()
    }
    // Null column if we don't have struct available. For <=67108864 rows it uses a global zero
    // buffer, otherwise it will allocate for validity.
    #[cfg(not(feature = "dtype-struct"))]
    {
        Column::full_null(name, height, &DataType::Null)
    }
}

/// Returns `Some(ExprIR)` if `ir` is `select(len())`.
fn extract_select_len_expr<'a>(
    ir: &'a mut IR,
    expr_arena: &Arena<AExpr>,
) -> Option<&'a mut ExprIR> {
    if let IR::Select { expr, .. } = ir
        && expr.len() == 1
        && matches!(expr_arena.get(expr[0].node()), AExpr::Len)
    {
        Some(&mut expr[0])
    } else {
        None
    }
}

fn reuse_names_alloc(edges: &mut dyn NodeEdgesProvider<Edge>) {
    for i in 0..usize::min(edges.inputs().len(), edges.outputs().len()) {
        edges.inputs()[i].projection_state_mut().names =
            edges.outputs()[i].take_names().map(|mut x| {
                x.clear();
                x
            });
    }
}
