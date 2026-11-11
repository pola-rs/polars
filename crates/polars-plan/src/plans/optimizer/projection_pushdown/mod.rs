pub mod edge;
mod field_pruning;

use std::ops::ControlFlow;
use std::sync::Arc;

use edge::Edge;
use polars_core::chunked_array::cast::CastOptions;
use polars_core::frame::DataFrame;
#[cfg(feature = "dtype-struct")]
use polars_core::prelude::StructFields;
use polars_core::prelude::{Column, DataType, IdxSize, PlIndexMap, ScratchIndexMap};
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_defs::join::{JoinCoalesce, JoinType};
use polars_io::RowIndex;
#[allow(clippy::disallowed_types)]
use polars_utils::aliases::{PlHashMap, PlHashSet};
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools as _;
use polars_utils::itertools::iters_eq::iters_eq;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::scratch_vec::ScratchVec;
use polars_utils::unique_id::UniqueId;
use polars_utils::{UnitVec, format_pl_smallstr, unitvec};

use crate::constants::get_len_name;
use crate::dsl::{FieldSelection, FileScanIR, PredicateFileSkip, ScanFieldSelection, UnionOptions};
use crate::plans::optimizer::ir_traversal::ir_graph_traversal;
use crate::plans::optimizer::projection_pushdown::edge::{
    GetParentKeyAndPort as _, GetProjectionState, ParentKeyAndPort, Projection, ProjectionState,
};
use crate::plans::optimizer::projection_pushdown::field_pruning::{
    ColumnSelection, ColumnSelectionExt, SelectionWalker, pruning_select, refresh_schemas,
};
use crate::plans::optimizer::recursive_caches::caches_shared_across_rounds;
use crate::plans::options::JoinTypeOptionsIR;
use crate::plans::projection_height::{ExprProjectionHeight, aexpr_projection_height_rec};
use crate::plans::{
    AExpr, ArenaExprIter, ArenaLpIter, ExprIR, ExprOrigin, FunctionIR, IR, IRAggExpr, IRBuilder,
    IRFunctionExpr, OutputName, det_join_schema,
};
use crate::prelude::{DistinctOptionsIR, Operator, ProjectionOptions};
use crate::traversal::edge_provider::NodeEdgesProvider;
use crate::traversal::visitor::{NodeVisitor, SubtreeVisit};
use crate::utils::rename_columns;

pub fn projection_pushdown(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    has_recursive: bool,
    prune_struct_fields: bool,
) {
    let schema_cache = &mut Default::default();
    #[allow(clippy::disallowed_types)]
    let mut dtypes_changed_nodes = PlHashSet::default();

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

    #[allow(clippy::disallowed_types)]
    let mut optimized_cache_nodes = PlHashMap::default();
    let caches_shared_across_rounds = if has_recursive {
        caches_shared_across_rounds(optimize_root, ir_arena)
    } else {
        Default::default()
    };

    ir_graph_traversal(
        optimize_root,
        &mut ProjectionPushdownVisitor {
            expr_arena,
            schema_cache,
            ae_nodes_scratch: &mut ScratchVec::default(),
            ae_height_scratch: &mut ScratchVec::default(),
            names_set_scratch: &mut ScratchIndexMap::default(),
            names_set_scratch2: &mut ScratchIndexMap::default(),
            names_set_scratch3: &mut ScratchIndexMap::default(),
            names_vec_scratch: &mut ScratchVec::default(),
            rename_map: &mut ScratchIndexMap::default(),
            optimized_cache_nodes: &mut optimized_cache_nodes,
            caches_shared_across_rounds: &caches_shared_across_rounds,
            default_edge: Edge::new(
                Projection::All,
                None,
                ParentKeyAndPort { node: root, idx: 0 },
            ),
            maintain_errors: false,
            prune_struct_fields,
            dtypes_changed_nodes: &mut dtypes_changed_nodes,
        },
        &mut vec![],
        &mut vec![],
        ir_arena,
    )
    .continue_value()
    .unwrap();

    // Ensure cache nodes for an ID all point to same input Node.
    let cache_nodes = Vec::from_iter(
        ir_arena
            .iter(root)
            .filter_map(|(node, ir)| matches!(ir, IR::Cache { .. }).then_some(node)),
    );

    for node in cache_nodes {
        let IR::Cache { id, .. } = ir_arena.get(node) else {
            unreachable!()
        };
        if let Some(optimized_cache) = optimized_cache_nodes.get(id) {
            let IR::Cache { input, .. } = ir_arena.get(*optimized_cache) else {
                unreachable!()
            };
            let optimized_input = *input;
            let IR::Cache { input, .. } = ir_arena.get_mut(node) else {
                unreachable!()
            };
            *input = optimized_input;
        }
    }

    // Assign optimized plan back to root node after resolving the recorded cache node keys.
    let IR::SimpleProjection { input, columns: _ } = ir_arena.take(root) else {
        unreachable!()
    };

    ir_arena.swap(root, input);

    // Pruned struct fields narrowed the dtypes of some nodes: update the schemas above them.
    if !dtypes_changed_nodes.is_empty() {
        refresh_schemas(root, ir_arena, expr_arena, &dtypes_changed_nodes);
    }
}

pub struct ProjectionPushdownVisitor<'a, 'arena> {
    expr_arena: &'arena mut Arena<AExpr>,
    #[allow(clippy::disallowed_types)] // We don't iterate over cache.
    schema_cache: &'a mut PlHashMap<Node, Arc<Schema>>,
    ae_nodes_scratch: &'a mut ScratchVec<Node>,
    ae_height_scratch: &'a mut ScratchVec<ExprProjectionHeight>,
    names_set_scratch: &'a mut ScratchIndexMap<PlSmallStr, FieldSelection>,
    names_set_scratch2: &'a mut ScratchIndexMap<PlSmallStr, FieldSelection>,
    names_set_scratch3: &'a mut ScratchIndexMap<PlSmallStr, FieldSelection>,
    names_vec_scratch: &'a mut ScratchVec<PlSmallStr>,
    rename_map: &'a mut ScratchIndexMap<PlSmallStr, PlSmallStr>,
    #[allow(clippy::disallowed_types)] // We don't iterate over cache.
    optimized_cache_nodes: &'a mut PlHashMap<UniqueId, Node>,
    #[allow(clippy::disallowed_types)] // We don't iterate over it.
    caches_shared_across_rounds: &'a PlHashSet<UniqueId>,
    default_edge: Edge,
    maintain_errors: bool,
    /// Whether to prune struct fields, at scans and in expressions.
    prune_struct_fields: bool,
    /// Nodes whose output dtypes narrowed because of pruned struct fields.
    #[allow(clippy::disallowed_types)] // We don't iterate over it.
    dtypes_changed_nodes: &'a mut PlHashSet<Node>,
}

impl<'a, 'arena> NodeVisitor for ProjectionPushdownVisitor<'a, 'arena> {
    type Edge = Edge;
    type BreakValue = ();
    type Key = Node;
    type Storage = Arena<IR>;

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
        let parent_key_and_port = out_edge.parent_key_and_port();

        // This node was unlinked. We skip post-visit but remove the deletion mark,
        // as otherwise the parent node will not be called for post_visit.
        if parent_key_and_port.is_deleted() {
            out_edge.parent_key_and_port_mut().set_deleted(false);
            return ControlFlow::Continue(());
        }

        match storage.get(key) {
            IR::HConcat { inputs, .. } => {
                debug_assert_eq!(inputs.len(), edges.inputs().len());
            },

            IR::Cache { id, .. } => {
                // Nested caches can still modify this cache's input after post_visit.
                self.optimized_cache_nodes.insert(*id, key);
            },

            _ => {},
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

                    reuse_names_alloc(edges);

                    return;
                };

                if out_edge.projection() == Projection::Len {
                    *out_edge.projection_mut() = Projection::Names;
                }

                (out_edge.names_mut(), &current_node_schema)
            }};
        }

        #[allow(clippy::disallowed_types)] // We don't iterate over cache.
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
                        &projected_names.as_slice()[..len_before_added_names],
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

                names.retain(|name, _| input_schema.contains(name));
                names.normalize(&input_schema);

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
                        names.extend_whole(columns.iter_names_cloned());
                        *out_edge.projection_mut() = Projection::Names;
                    },
                    Projection::Names => {
                        let initial_len = out_edge.names().len();
                        out_edge
                            .names_mut()
                            .retain(|name, _| columns.contains(name));
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
                            EH::Unknown | EH::Range => break 'len_propagate,
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

                    if projected_names.len() != exprs.len() {
                        let mut truncate_len = projected_names.len();
                        let mut height_expr_idx = usize::MAX;

                        for (i, e) in exprs.iter().enumerate() {
                            match aexpr_projection_height_rec(
                                e.node(),
                                self.expr_arena,
                                self.ae_nodes_scratch,
                                self.ae_height_scratch,
                            ) {
                                EH::Column => {
                                    height_expr_idx = i;
                                    break;
                                },
                                EH::Unknown | EH::Range => height_expr_idx = i,
                                EH::Scalar => height_expr_idx = usize::min(i, height_expr_idx),
                            }
                        }

                        if height_expr_idx != usize::MAX && height_expr_idx >= truncate_len {
                            exprs.swap(height_expr_idx, projected_names.len());
                            truncate_len += 1;
                        }

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
                                    EH::Unknown | EH::Range => {
                                        exprs.swap(i, truncate_len);
                                        truncate_len += 1;
                                    },
                                }
                            }
                        }

                        exprs.truncate(truncate_len);
                    }

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

                'select_len_pushdown: {
                    let IR::Select { expr: exprs, .. } = storage.get(key) else {
                        unreachable!()
                    };

                    let mut len_nodes: Vec<Node> = Vec::new();

                    for e in exprs.iter() {
                        if !matches!(
                            aexpr_projection_height_rec(
                                e.node(),
                                self.expr_arena,
                                self.ae_nodes_scratch,
                                self.ae_height_scratch
                            ),
                            EH::Scalar
                        ) || !extract_len_nodes(
                            e.node(),
                            self.expr_arena,
                            &mut len_nodes,
                            self.ae_nodes_scratch,
                            self.ae_height_scratch,
                        ) {
                            break 'select_len_pushdown;
                        }
                    }

                    if len_nodes.is_empty() {
                        break 'select_len_pushdown;
                    }

                    let insert_select_len_ir =
                        if exprs.len() != 1 || len_nodes[0] != exprs[0].node() {
                            for node in len_nodes {
                                self.expr_arena.replace(node, AExpr::Column(get_len_name()));
                            }

                            Some(storage.add(IR::Select {
                                input: input_node,
                                expr: vec![ExprIR::new(
                                    self.expr_arena.add(AExpr::Len),
                                    OutputName::Alias(get_len_name()),
                                )],
                                schema: Arc::new(Schema::from_iter([(
                                    get_len_name(),
                                    DataType::IDX_DTYPE,
                                )])),
                                options: ProjectionOptions::default(),
                            }))
                        } else {
                            // select(col(a).len()) -> select(len().alias(a))
                            self.expr_arena.replace(len_nodes[0], AExpr::Len);

                            None
                        };

                    let IR::Select {
                        input, expr: exprs, ..
                    } = storage.get_mut(key)
                    else {
                        unreachable!()
                    };

                    // Force alias, this way downstream doesn't need to handle name when mutating
                    // the AExpr nodes.
                    for e in exprs.iter_mut() {
                        let name = e.output_name().clone();
                        e.set_alias(name);
                    }

                    let in_edge = &mut edges.inputs()[0];

                    if let Some(node) = insert_select_len_ir {
                        *input = node;
                        *in_edge.parent_key_and_port_mut() = ParentKeyAndPort { node, idx: 0 };
                    }

                    *in_edge.projection_mut() = Projection::Len;
                    reuse_names_alloc(edges);

                    return;
                }

                let input_schema = IR::schema_with_cache(input_node, storage, self.schema_cache);

                let IR::Select { expr: exprs, .. } = storage.get_mut(key) else {
                    unreachable!()
                };

                // Determine names referenced by exprs as the projection to pushdown.
                let input_names_projection = self.names_set_scratch.get();
                let mut has_non_simple_projection = false;
                let mut has_window_expr = false;
                let mut walker = SelectionWalker::new(
                    self.expr_arena,
                    self.prune_struct_fields,
                    Some(input_schema.as_ref()),
                );

                for e in exprs {
                    let selection = output_selection(out_edge, e.output_name());
                    walker.walk_expr_ir(e, &selection, input_names_projection, false);

                    has_window_expr = has_window_expr
                        || walker
                            .expr_arena
                            .iter(e.node())
                            .any(|(_, e)| matches!(e, AExpr::Over { .. }));

                    match walker.expr_arena.get(e.node()) {
                        AExpr::Column(name) if name == e.output_name() => {},
                        _ => has_non_simple_projection = true,
                    }
                }

                if walker.rewrote {
                    self.dtypes_changed_nodes.insert(key);
                }

                // E.g. `select(<literal>.sum().over(<literal>))`, we must project at least 1 input
                // column for height.
                if has_window_expr && input_names_projection.is_empty() {
                    input_names_projection
                        .extend_whole(min_dtype_size_col(input_schema.iter()).cloned())
                }

                let prune = !has_non_simple_projection;
                input_names_projection.normalize(&input_schema);
                if !prune {
                    sort_full_projection(input_names_projection, &input_schema);
                }

                if input_names_projection.len() != input_schema.len()
                    || input_names_projection.has_field_selections()
                    || (prune
                        && !iters_eq(input_names_projection.keys(), input_schema.iter_names()))
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

                    // A rewrite made it a projection of columns (a `with_fields` whose new fields
                    // nobody reads): its consumer now sees the narrowed dtype.
                    if self.dtypes_changed_nodes.remove(&key) {
                        self.dtypes_changed_nodes.insert(parent_key_and_port.node);
                    }

                    let in_edge = &mut edges.inputs()[0];
                    *in_edge.parent_key_and_port_mut() = parent_key_and_port;
                    edges.outputs()[0]
                        .parent_key_and_port_mut()
                        .set_deleted(true);
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
                        projected_names.contains_key(e.output_name())
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
                let mut walker = SelectionWalker::new(
                    self.expr_arena,
                    self.prune_struct_fields,
                    Some(input_schema.as_ref()),
                );
                for e in exprs.iter_mut() {
                    expr_output_names.insert_whole(e.output_name().clone());
                    let selection = opt_projected_names
                        .and_then(|names| names.get(e.output_name()))
                        .cloned()
                        .unwrap_or_default();
                    walker.walk_expr_ir(e, &selection, input_names_projection, true);
                }
                if walker.rewrote {
                    self.dtypes_changed_nodes.insert(key);
                }

                // Directly passed to output (i.e. do not originate from exprs) that are being
                // referenced. These do not exist in the expr output names hashset.
                if let Some(projected_names) = &opt_projected_names {
                    for (name, selection) in projected_names.iter() {
                        if !expr_output_names.contains_key(name) {
                            input_names_projection.insert_merged(name.clone(), selection.clone());
                        }
                    }
                } else {
                    for name in output_schema.iter_names() {
                        if !expr_output_names.contains_key(name) {
                            input_names_projection.insert_whole(name.clone());
                        }
                    }
                }

                // Must project at least 1 column for height.
                if input_names_projection.is_empty() {
                    input_names_projection
                        .extend_whole(min_dtype_size_col(input_schema.iter()).cloned());
                }

                input_names_projection.normalize(&input_schema);
                let has_field_selections = input_names_projection.has_field_selections();

                let output_schema_arc = output_schema;

                let has_dropped_input_column = input_names_projection.len() != input_schema.len();

                if exprs.len() != orig_exprs_len || has_dropped_input_column {
                    input_names_projection.sort_by_cached_key(|name, _| {
                        input_schema.index_of(name).unwrap_or(usize::MAX)
                    });

                    let output_schema = Arc::make_mut(output_schema_arc);
                    let mut orig_schema = mem::take(output_schema);

                    for name in input_names_projection.keys() {
                        output_schema
                            .extend(orig_schema.remove(name).map(|dtype| (name.clone(), dtype)));
                    }

                    for name in expr_output_names.keys() {
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
                    .or(
                        (has_dropped_input_column && out_edge.projection() == Projection::All)
                            .then_some(original_schema),
                    )
                {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(schema, storage);
                }

                if has_dropped_input_column || has_field_selections {
                    // Every column, pushed for its field selections: keep the input order, which
                    // the output schema assumes.
                    sort_full_projection(input_names_projection, &input_schema);
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

                    let Some(select_len_ae_node) = extract_select_len_expr(
                        storage.get_mut(out_edge.parent_key_and_port().node),
                        self.expr_arena,
                    )
                    .map(|eir| eir.node()) else {
                        break 'len_to_predicate_sum;
                    };

                    self.expr_arena.replace(
                        select_len_ae_node,
                        AExpr::Agg(IRAggExpr::Sum {
                            input: predicate_node,
                            null_on_empty: false,
                        }),
                    );

                    *out_edge.projection_mut() = Projection::Names;
                    let names = out_edge.names_mut();
                    names.clear();
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        predicate_node,
                        names,
                    );
                    names.normalize(&current_node_schema);

                    unlink_current_node_and_return!(input)
                }

                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();

                SelectionWalker::collect(
                    self.expr_arena,
                    self.prune_struct_fields,
                    predicate_node,
                    projected_names,
                );

                pushdown_with_added_names!(len_before_added_names)
            },

            IR::Gather {
                null_on_oob, idxs, ..
            } => {
                assert_eq!(num_input_edges, 2);

                // `lf.gather(idxs, null_on_oob=True).select(len())` -> `idxs.select(len())`
                if out_edge.projection() == Projection::Len && *null_on_oob {
                    let idxs_node = *idxs;
                    let [idxs_edge, out_edge] = edges.get_input_output_mut(1, 0);

                    let parent_key_and_port = out_edge.parent_key_and_port();

                    *storage
                        .get_mut(parent_key_and_port.node)
                        .inputs_mut()
                        .nth(parent_key_and_port.idx)
                        .unwrap() = idxs_node;

                    mem::swap(idxs_edge, out_edge);
                    out_edge.parent_key_and_port_mut().set_deleted(true);
                    return;
                }

                let (..) = projected_names_subset_or_return!();

                *edges.inputs()[0].projection_state_mut() = ProjectionState {
                    projection: Projection::Names,
                    names: out_edge.take_names(),
                };
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
                    options,
                    ..
                } = storage.get_mut(key)
                else {
                    unreachable!()
                };

                let opt_projected_names = out_edge.compute_projected_names(output_schema_arc);

                let is_projected_in_output = |name: &str| {
                    if let Some(projected_names) = &opt_projected_names {
                        projected_names.contains_key(name)
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
                    coalesced_to_right.extend_whole(options.options.left_on().map(|expr| {
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

                let has_cross_filter =
                    matches!(options.options, JoinTypeOptionsIR::CrossAndFilter { .. });

                // A join predicate reads output columns the final projection may not ask for.
                let mut predicate_names = ColumnSelection::default();
                if let Some(predicate) = match &options.options {
                    JoinTypeOptionsIR::CrossAndFilter { predicate } => Some(predicate),
                    options => options.fused_predicate(),
                } {
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        predicate.node(),
                        &mut predicate_names,
                    );
                }
                let reads_in_predicate = |name: &PlSmallStr| predicate_names.contains_key(name);
                // What is needed of an output column, by the consumers and the predicate.
                let output_selection_of = |name: &str| {
                    let mut selection = match &opt_projected_names {
                        Some(projected_names) => projected_names
                            .get(name)
                            .cloned()
                            .unwrap_or_else(FieldSelection::validity_only),
                        None => FieldSelection::All,
                    };
                    if let Some(predicate_selection) = predicate_names.get(name) {
                        selection.union(predicate_selection.clone());
                    }
                    selection
                };

                // Add accumulated projections
                for output_name in output_schema_arc
                    .iter_names()
                    .filter(|name| is_projected_in_output(name) || reads_in_predicate(name))
                {
                    match ExprOrigin::get_column_origin(
                        output_name,
                        &input_schema_left,
                        &input_schema_right,
                        options.args.suffix(),
                        Some(&|name| coalesced_to_right.contains_key(name)),
                    )
                    .unwrap()
                    {
                        ExprOrigin::None => {},
                        ExprOrigin::Left => {
                            project_left.insert_merged(
                                output_name.clone(),
                                output_selection_of(output_name),
                            );
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

                            project_right.insert_merged(name, output_selection_of(output_name));
                        },
                        ExprOrigin::Both => unreachable!(),
                    }
                }

                // Add projections required by the join itself
                for expr_ir in options.options.left_on() {
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        expr_ir.node(),
                        project_left,
                    );
                }

                for expr_ir in options.options.right_on() {
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        expr_ir.node(),
                        project_right,
                    );
                }

                #[cfg(feature = "asof_join")]
                if let JoinType::AsOf(asof_options) = &options.args.how {
                    if let Some(left_by) = asof_options.left_by.as_deref() {
                        for name in left_by {
                            project_left.insert_whole(name.clone());
                        }
                    }

                    if let Some(right_by) = asof_options.right_by.as_deref() {
                        for name in right_by {
                            project_right.insert_whole(name.clone());
                        }
                    }
                }

                // Turn on coalesce if non-coalesced keys are not included in projection. Reduces materialization.
                if !options.args.should_coalesce()
                    && matches!(options.args.how, JoinType::Inner | JoinType::Left)
                    && matches!(options.options, JoinTypeOptionsIR::Equi { .. })
                    && options
                        .options
                        .left_on()
                        .all(|e| matches!(self.expr_arena.get(e.node()), AExpr::Column(_)))
                    && options.options.right_on().all(|e| {
                        let AExpr::Column(name) = self.expr_arena.get(e.node()) else {
                            return false;
                        };

                        // Coalescing drops the right key, so a predicate reading it counts
                        // as a use.
                        let projected = if input_schema_left.contains(name.as_str()) {
                            let name = format_pl_smallstr!("{}{}", name, options.args.suffix());
                            is_projected_in_output(&name) || reads_in_predicate(&name)
                        } else {
                            is_projected_in_output(name) || reads_in_predicate(name)
                        };

                        !projected
                    })
                {
                    Arc::make_mut(options).args.coalesce = JoinCoalesce::CoalesceColumns;
                }

                project_left.normalize(&input_schema_left);
                project_right.normalize(&input_schema_right);

                // Every column, pushed for its field selections: keep the input order, which the
                // join schema assumes below.
                sort_full_projection(project_left, &input_schema_left);
                sort_full_projection(project_right, &input_schema_right);

                let new_input_schema_left = if project_left.len() == input_schema_left.len() {
                    input_schema_left.clone()
                } else {
                    Arc::new(input_schema_left.try_project(project_left.keys()).unwrap())
                };

                let new_input_schema_right = if project_right.len() == input_schema_right.len() {
                    input_schema_right.clone()
                } else {
                    Arc::new(
                        input_schema_right
                            .try_project(project_right.keys())
                            .unwrap(),
                    )
                };

                let new_output_schema =
                    det_join_schema(&new_input_schema_left, &new_input_schema_right, options)
                        .unwrap();

                if project_left.len() != input_schema_left.len()
                    || project_left.has_field_selections()
                {
                    *edges.inputs()[0].projection_state_mut() = ProjectionState {
                        projection: Projection::Names,
                        names: Some(Box::new(mem::take(project_left))),
                    };
                }

                if project_right.len() != input_schema_right.len()
                    || project_right.has_field_selections()
                {
                    *edges.inputs()[1].projection_state_mut() = ProjectionState {
                        projection: Projection::Names,
                        names: Some(Box::new(mem::take(project_right))),
                    };
                }

                let out_edge = &mut edges.outputs()[0];
                let opt_projected_names = out_edge.compute_projected_names(output_schema_arc);

                *output_schema_arc = new_output_schema;

                // Narrowing an input can remove a name collision, dropping the suffix
                // from the right column. The map below covers only projected names, so a
                // column read solely by the predicate is renamed here.
                if !predicate_names.is_empty() {
                    let mut renames: PlIndexMap<PlSmallStr, PlSmallStr> = PlIndexMap::default();

                    for name in predicate_names.keys() {
                        if output_schema_arc.contains(name) {
                            continue;
                        }
                        let Some(stripped) = name.strip_suffix(options.args.suffix().as_str())
                        else {
                            continue;
                        };
                        renames.insert(name.clone(), PlSmallStr::from_str(stripped));
                    }

                    if !renames.is_empty() {
                        let predicate = match &mut Arc::make_mut(options).options {
                            JoinTypeOptionsIR::Equi {
                                fused_predicate: Some(predicate),
                                ..
                            }
                            | JoinTypeOptionsIR::CrossAndFilter { predicate } => predicate,
                            _ => unreachable!(),
                        };

                        predicate.set_node(rename_columns(
                            predicate.node(),
                            self.expr_arena,
                            &renames,
                        ));
                    }
                }

                if let Some(projected_names) = &opt_projected_names {
                    let orig_to_new_name_map = self.rename_map.get();

                    for name in projected_names.keys() {
                        if !output_schema_arc.contains(name) {
                            let new_output_name = PlSmallStr::from_str(
                                name.strip_suffix(options.args.suffix().as_str()).unwrap(),
                            );

                            orig_to_new_name_map.insert(name.clone(), new_output_name);
                        }
                    }

                    if !orig_to_new_name_map.is_empty() {
                        if has_cross_filter {
                            let JoinTypeOptionsIR::CrossAndFilter { predicate } =
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
                            .keys()
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
                                    maintain_dataframe_height: false,
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

            // Windows are created after projection pushdown.
            IR::GroupBy { apply: Some(_), .. } | IR::Window { .. } => {
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

                'prune_aggs: {
                    if aggs.is_empty() {
                        break 'prune_aggs;
                    }

                    let removed_names = if is_len {
                        let removed_names = self.names_set_scratch.get();

                        removed_names.extend_whole(
                            aggs.drain(..)
                                .map(|eir| eir.into_inner().1.into_inner().unwrap()),
                        );

                        removed_names
                    } else if let Some(projected_names) =
                        out_edge.compute_projected_names_subset(output_schema_arc)
                    {
                        let removed_names = self.names_set_scratch.get();

                        aggs.retain(|e| {
                            let keep = projected_names.contains_key(e.output_name());
                            if !keep {
                                removed_names.insert_whole(e.output_name().clone());
                            }
                            keep
                        });

                        removed_names
                    } else {
                        break 'prune_aggs;
                    };

                    if !removed_names.is_empty() {
                        Arc::make_mut(output_schema_arc)
                            .retain(|name, _| !removed_names.contains_key(name))
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

                // The keys are needed whole. The aggregations are walked with what is needed of
                // their outputs, unrewritten: a narrowed input narrows their dtypes only, which
                // `refresh_schemas` recomputes.
                let IR::GroupBy { keys, aggs, .. } = storage.get(key) else {
                    unreachable!()
                };
                for e in keys {
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        e.node(),
                        input_names_projection,
                    );
                }
                let mut walker =
                    SelectionWalker::new(self.expr_arena, self.prune_struct_fields, None);
                for e in aggs {
                    let selection = output_selection(out_edge, e.output_name());
                    let new_node = walker.walk(e.node(), &selection, input_names_projection);
                    debug_assert_eq!(new_node, e.node());
                }

                let IR::GroupBy { options, .. } = storage.get(key) else {
                    unreachable!()
                };

                #[cfg(feature = "dynamic_group_by")]
                if let Some(options) = &options.dynamic {
                    input_names_projection.insert_whole(options.index_column.clone());
                }

                #[cfg(feature = "dynamic_group_by")]
                if let Some(options) = &options.rolling {
                    input_names_projection.insert_whole(options.index_column.clone());
                }

                input_names_projection.normalize(&input_schema);
                sort_full_projection(input_names_projection, &input_schema);

                if input_names_projection.len() != input_schema.len()
                    || input_names_projection.has_field_selections()
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
                projected_names.extend_whole(subset.iter().cloned());
                pushdown_with_added_names!(len_before_added_names)
            },

            IR::Union {
                inputs,
                options:
                    UnionOptions {
                        slice,
                        rows: _,
                        parallel: _,
                        from_partitioned_ds: _,
                        flattened_by_opt: _,
                        rechunk,
                        maintain_order,
                    },
            } => {
                // concat(a, b, ..).select(len())
                //   -> concat(a.select(len()), b.select(len()), ..).select(col('len').cast(UInt128).sum().cast(IDX_DTYPE))
                // A slice of the union would apply to the per-input lengths instead of the rows,
                // so a slice with a non-negative offset is instead applied to the summed length.
                if out_edge.projection() == Projection::Len
                    && !inputs.is_empty()
                    && slice.is_none_or(|(offset, _)| offset >= 0)
                {
                    let slice = slice.take();
                    *rechunk = false;
                    *maintain_order = false;
                    let len_schema =
                        Arc::new(Schema::from_iter([(get_len_name(), DataType::IDX_DTYPE)]));

                    let new_inputs: Vec<Node> = mem::take(inputs)
                        .into_iter()
                        .enumerate()
                        .map(|(i, node)| {
                            let node = storage.add(IR::Select {
                                input: node,
                                expr: vec![ExprIR::new(
                                    self.expr_arena.add(AExpr::Len),
                                    OutputName::Alias(get_len_name()),
                                )],
                                schema: Arc::clone(&len_schema),
                                options: ProjectionOptions::default(),
                            });

                            let edge = &mut edges.inputs()[i];

                            *edge.parent_key_and_port_mut() = ParentKeyAndPort { node, idx: 0 };
                            *edge.projection_mut() = Projection::Len;

                            node
                        })
                        .collect();

                    let IR::Union { inputs, .. } = storage.get_mut(key) else {
                        unreachable!()
                    };
                    *inputs = new_inputs;

                    let sum_to_len = {
                        let mut e = AExpr::Column(get_len_name());
                        e = AExpr::Cast {
                            expr: self.expr_arena.add(e),
                            dtype: DataType::UInt128,
                            options: CastOptions::Overflowing,
                        };
                        e = AExpr::Agg(IRAggExpr::Sum {
                            input: self.expr_arena.add(e),
                            null_on_empty: false,
                        });
                        e = AExpr::Cast {
                            expr: self.expr_arena.add(e),
                            dtype: DataType::IDX_DTYPE,
                            options: CastOptions::Strict,
                        };
                        let mut len = self.expr_arena.add(e);

                        // slice(offset, length).len() == min(max(len, offset) - offset, length)
                        if let Some((offset, length)) = slice {
                            let arena = &mut *self.expr_arena;
                            let mut lit = |v: usize| {
                                let v = IdxSize::try_from(v).unwrap_or(IdxSize::MAX);
                                arena.add(AExpr::Literal(Scalar::new_idxsize(v).into()))
                            };
                            let (offset_lit, length_lit) = (lit(offset as usize), lit(length));
                            let gt = |arena: &mut Arena<AExpr>, left, right| {
                                arena.add(AExpr::BinaryExpr {
                                    left,
                                    op: Operator::Gt,
                                    right,
                                })
                            };

                            if offset != 0 {
                                let predicate = gt(arena, len, offset_lit);
                                let max = arena.add(AExpr::Ternary {
                                    predicate,
                                    truthy: len,
                                    falsy: offset_lit,
                                });
                                len = arena.add(AExpr::BinaryExpr {
                                    left: max,
                                    op: Operator::Minus,
                                    right: offset_lit,
                                });
                            }

                            let predicate = gt(arena, len, length_lit);
                            len = arena.add(AExpr::Ternary {
                                predicate,
                                truthy: length_lit,
                                falsy: len,
                            });
                        }

                        len
                    };

                    extract_select_len_expr(
                        storage.get_mut(edges.outputs()[0].parent_key_and_port().node),
                        self.expr_arena,
                    )
                    .unwrap()
                    .set_node(sum_to_len);

                    reuse_names_alloc(edges);
                    return;
                }

                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();
                pushdown_with_added_names!(len_before_added_names)
            },

            ir @ IR::Slice { .. } | ir @ IR::Sort { .. } => {
                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();
                for e in ir.exprs() {
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        e.node(),
                        projected_names,
                    );
                }
                pushdown_with_added_names!(len_before_added_names)
            },

            #[cfg(feature = "merge_sorted")]
            IR::MergeSorted { key, .. } => {
                let (projected_names, _) = projected_names_subset_or_return!();
                let len_before_added_names = projected_names.len();
                for key in key.iter() {
                    projected_names.insert_whole(key.clone());
                }
                pushdown_with_added_names!(len_before_added_names)
            },

            IR::HConcat { .. } => {
                let [
                    IR::HConcat {
                        inputs, options, ..
                    },
                    parent_ir,
                ] = storage.get_disjoint_mut([key, out_edge.parent_key_and_port().node])
                else {
                    unreachable!()
                };

                let strict = options.strict;

                // concat_horizontal([lf, ..]).select(len())
                // -> concat_horizontal([lf.select(len().alias(i)) for i, lf in enumerate(..)]).select(max_horizontal('*'))
                // When unit-length inputs are broadcast, the height is not the max: a height of
                // 0 and a height of 1 give 0.
                if out_edge.projection() == Projection::Len
                    && !strict
                    && !options.broadcast_unit_length
                    && let Some(select_len_ae_node) =
                        extract_select_len_expr(parent_ir, self.expr_arena).map(|eir| eir.node())
                {
                    let mut max_horizontal_inputs = Vec::with_capacity(inputs.len());
                    let mut new_output_schema = Schema::with_capacity(inputs.len());
                    let mut new_inputs = mem::take(inputs);

                    for (i, input_ir_node) in new_inputs.iter_mut().enumerate() {
                        let name = format_pl_smallstr!("__POLARS_{:010x}", i);

                        let exprs = vec![ExprIR::new(
                            self.expr_arena.add(AExpr::Len),
                            OutputName::Alias(name.clone()),
                        )];
                        max_horizontal_inputs
                            .push(ExprIR::from_column_name(name.clone(), self.expr_arena));
                        new_output_schema.insert(name.clone(), DataType::IDX_DTYPE);

                        let select_ir_node = storage.add(IR::Select {
                            input: *input_ir_node,
                            expr: exprs,
                            schema: Arc::new(Schema::from_iter([(name, DataType::IDX_DTYPE)])),
                            options: ProjectionOptions {
                                run_parallel: false,
                                duplicate_check: false,
                                should_broadcast: false,
                                maintain_dataframe_height: false,
                            },
                        });

                        *input_ir_node = select_ir_node;

                        let in_edge = &mut edges.inputs()[i];
                        *in_edge.projection_mut() = Projection::Len;
                        *in_edge.parent_key_and_port_mut() = ParentKeyAndPort {
                            node: select_ir_node,
                            idx: 0,
                        };

                        debug_assert!(
                            extract_select_len_expr(
                                storage.get_mut(in_edge.parent_key_and_port().node),
                                self.expr_arena
                            )
                            .is_some()
                        );
                    }

                    let IR::HConcat { inputs, schema, .. } = storage.get_mut(key) else {
                        unreachable!()
                    };

                    *inputs = new_inputs;
                    *schema = Arc::new(new_output_schema);

                    let function = IRFunctionExpr::MaxHorizontal;
                    let options = function.function_options();

                    self.expr_arena.replace(
                        select_len_ae_node,
                        AExpr::Function {
                            input: max_horizontal_inputs,
                            function,
                            options,
                        },
                    );

                    reuse_names_alloc(edges);
                    return;
                }

                if out_edge
                    .compute_projected_names(&current_node_schema)
                    .is_none()
                {
                    return;
                }

                let projected_names = out_edge.take_names().unwrap();

                let mut inputs = mem::take(inputs);
                let strict = options.strict;
                let hconcat_projected_names = self.names_vec_scratch.get();

                assert_eq!(num_input_edges, inputs.len());

                let mut idx: usize = 0;
                let mut deleted: usize = 0;
                let mut last_kept_input: usize = usize::MAX;

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
                                .filter(|name| projected_names.contains_key(*name))
                                .cloned(),
                        );

                        if hconcat_projected_names.len() == base_new_names_len {
                            if strict
                                && polars_config::config()
                                    .projection_pushdown_prune_strict_hconcat_inputs()
                                && !self.maintain_errors
                            {
                                break 'set_keep;
                            }

                            hconcat_projected_names
                                .push(min_dtype_size_col(input_schema_arc.iter()).unwrap().clone());
                        }

                        let names_this_input =
                            &mut hconcat_projected_names.as_mut_slice()[base_new_names_len..];

                        names_this_input.sort_by_key(|name| {
                            projected_names.get_index_of(name).unwrap_or(usize::MAX)
                        });

                        let has_field_selections = names_this_input
                            .iter()
                            .any(|name| projected_names.get(name).is_some_and(|s| !s.is_all()));

                        if has_field_selections
                            || !iters_eq(names_this_input.iter(), input_schema_arc.iter_names())
                        {
                            let mut names = mem::take(self.names_set_scratch2.get());
                            names.extend_merged(names_this_input.iter().map(|name| {
                                (
                                    name.clone(),
                                    projected_names.get(name).cloned().unwrap_or_default(),
                                )
                            }));
                            names.normalize(&input_schema_arc);

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
                    } else {
                        last_kept_input = idx;

                        if deleted != 0 {
                            in_port.idx = idx - deleted;
                        }
                    }

                    keep
                });

                let new_inputs = inputs;

                if new_inputs.len() == 1 {
                    let input_node = new_inputs.into_iter().next().unwrap();
                    let [in_edge, out_edge] = edges.get_input_output_mut(last_kept_input, 0);
                    let parent_key_and_port = out_edge.parent_key_and_port();

                    *storage
                        .get_mut(parent_key_and_port.node)
                        .inputs_mut()
                        .nth(parent_key_and_port.idx)
                        .unwrap() = input_node;

                    // Only update parent node info; projection is already set from above.
                    mem::swap(
                        in_edge.parent_key_and_port_mut(),
                        out_edge.parent_key_and_port_mut(),
                    );

                    edges.outputs()[0]
                        .parent_key_and_port_mut()
                        .set_deleted(true);
                    return;
                }

                let IR::HConcat { inputs, schema, .. } = storage.get_mut(key) else {
                    unreachable!()
                };
                *inputs = new_inputs;

                let new_schema = hconcat_projected_names
                    .drain(..)
                    .map(|name| {
                        let dtype = schema.get(&name).unwrap().clone();
                        (name, dtype)
                    })
                    .collect();

                *schema = Arc::new(new_schema);

                if let Some(projected_schema) =
                    compute_simple_projection_schema(projected_names.as_slice(), schema, false)
                {
                    edges.outputs()[0]
                        .parent_key_and_port_mut()
                        .attach_simple_projection(Arc::new(projected_schema), storage);
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
                        file_info,
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
                    && unified_scan_args.pre_slice.is_none()
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
                                cloud_options: Box::new(unified_scan_args.cloud_options),
                            },
                        },
                    );

                    return;
                }

                // The columns read, with what is needed of each: only collected if the scan
                // can read a field selection.
                let select_fields = self.prune_struct_fields
                    && scan_supports_field_selection(scan_type, unified_scan_args);
                let scan_selection = self.names_set_scratch3.get();

                let projected_schema = if out_edge.projection() == Projection::Len
                    && !matches!(scan_type.as_ref(), FileScanIR::Anonymous { .. })
                {
                    // Streaming sources support 0-width projection with correct height.
                    Default::default()
                } else {
                    let Some(names) = out_edge.compute_projected_names(&scan_schema) else {
                        return;
                    };

                    let projected_schema = match compute_simple_projection_schema(
                        names.as_slice(),
                        &scan_schema,
                        false,
                    ) {
                        Some(projected_schema) => projected_schema,
                        // Every column, but not every struct field.
                        None if names.has_field_selections() => scan_schema.as_ref().clone(),
                        None => return,
                    };

                    if select_fields {
                        scan_selection
                            .extend_merged(names.iter().map(|(n, s)| (n.clone(), s.clone())));
                    }

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
                    let predicate_selection = self.names_set_scratch.get();
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        predicate.node(),
                        predicate_selection,
                    );

                    for (name, selection) in predicate_selection.drain(..) {
                        Arc::make_mut(&mut scan_projected_schema)
                            .insert(name.clone(), scan_schema.get(&name).unwrap().clone());
                        if select_fields {
                            scan_selection.insert_merged(name, selection);
                        }
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

                let field_selection = select_fields
                    .then(|| {
                        scan_field_selection(
                            scan_selection,
                            &file_info.schema,
                            unified_scan_args.field_selection.as_deref(),
                        )
                    })
                    .flatten();

                if let Some(field_selection) = &field_selection {
                    let schema = Arc::make_mut(&mut scan_projected_schema);
                    for (name, selection) in field_selection.0.iter() {
                        if let Some(dtype) = schema.get_mut(name) {
                            *dtype = selection.prune_dtype(file_info.schema.get(name).unwrap());
                        }
                    }
                    self.dtypes_changed_nodes.insert(key);
                }

                if !iters_eq(
                    scan_projected_schema.iter_names(),
                    projected_schema.iter_names(),
                ) {
                    out_edge
                        .parent_key_and_port_mut()
                        .attach_simple_projection(projected_schema, storage);
                }

                set_scan_projection(storage.get_mut(key), scan_projected_schema, field_selection);
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
                                    if projected_names.contains_key(c.name()) {
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
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        pred.node(),
                        projected_names,
                    );
                };

                let mut new_output_schema = current_node_schema.as_ref();

                if projected_names.len() != current_node_schema.len() {
                    let with_columns: Arc<[_]> = options
                        .schema
                        .iter_names()
                        .filter(|name| projected_names.contains_key(*name))
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

                if !hint.retain_names(|name| projected_names.contains_key(name)) {
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

                let Some(row_index_idx) = projected_names.get_index_of(name) else {
                    unlink_current_node_and_return!(*input)
                };

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
                } else {
                    debug_assert_eq!(row_index_idx, 0);
                }

                let mut projected_names = out_edge.take_names().unwrap();
                projected_names.shift_remove_index(row_index_idx);

                *edges.inputs()[0].projection_state_mut() = ProjectionState {
                    projection: Projection::Names,
                    names: Some(projected_names),
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
                names.extend_whole(args.index.iter().cloned());
                names.extend_whole(args.on.iter().cloned());

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
                    function: FunctionIR::Unnest { columns, separator },
                    ..
                } = storage.get_mut(key)
                else {
                    unreachable!()
                };

                // Each struct is needed for the fields that are needed of its unnested columns.
                let unnested: Vec<(PlSmallStr, FieldSelection)> = columns
                    .iter()
                    .map(|column| {
                        let selection = match input_schema.get(column) {
                            Some(dtype) if self.prune_struct_fields => {
                                field_pruning::unnested_struct_selection(
                                    column,
                                    dtype,
                                    separator.as_ref(),
                                    out_edge.names(),
                                )
                            },
                            _ => FieldSelection::All,
                        };
                        (column.clone(), selection)
                    })
                    .collect();

                let names = out_edge.names_mut();
                names.retain(|name, _| input_schema.contains(name) && !columns.contains(name));
                names.extend_merged(unnested);
                names.normalize(&input_schema);
                // The unnest keeps the order of its input, which the schema above it follows.
                names.sort_by_cached_key(|name, _| {
                    input_schema.index_of(name).unwrap_or(usize::MAX)
                });
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

                // The row count of an explode is that of an exploded column, read for its offsets
                // and validity: count with it rather than with another column, or with it whole
                // when it is the only one.
                if out_edge.projection() == Projection::Len
                    && let FunctionIR::Explode { columns, .. } = function
                    && let Some(name) = columns.first()
                {
                    *out_edge.projection_mut() = Projection::Names;
                    out_edge
                        .names_mut()
                        .insert_merged(name.clone(), FieldSelection::validity_only());
                }

                let (projected_names, _) = projected_names_subset_or_return!();
                let input_schema = IR::schema_with_cache(input, storage, self.schema_cache);

                let IR::MapFunction { function, .. } = storage.get(key) else {
                    unreachable!()
                };

                function.clear_cached_schema();

                let input_projected_names = self.names_set_scratch.get();

                // Only these pass the values of the other columns through unchanged.
                let keep_selections =
                    matches!(function, FunctionIR::Rechunk | FunctionIR::Explode { .. });

                input_projected_names.extend_merged(
                    projected_names
                        .iter()
                        .filter(|(name, _)| input_schema.contains(name))
                        .map(|(name, selection)| {
                            let selection = if keep_selections {
                                selection.clone()
                            } else {
                                FieldSelection::All
                            };
                            (name.clone(), selection)
                        }),
                );

                if let FunctionIR::Explode { columns, .. } = function {
                    // An exploded column is needed for the selection of its element, plus its
                    // offsets and validity.
                    for name in columns.iter() {
                        let element = projected_names
                            .get(name)
                            .cloned()
                            .unwrap_or_else(FieldSelection::validity_only);
                        match input_projected_names.get_mut(name) {
                            Some(selection) => *selection = FieldSelection::element(element),
                            None => {
                                input_projected_names
                                    .insert(name.clone(), FieldSelection::element(element));
                            },
                        }
                    }
                } else {
                    for name in function.additional_projection_pd_columns().as_ref() {
                        input_projected_names.insert_whole(name.clone());
                    }
                }

                input_projected_names.normalize(&input_schema);

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

            IR::Cache { input, id } => {
                let input = *input;

                // A single reader in the step of a recursive query can run more than once.
                if edges.outputs().len() == 1 && !self.caches_shared_across_rounds.contains(id) {
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

                        projected_names
                            .extend_merged(names.iter().map(|(n, s)| (n.clone(), s.clone())));

                        if projected_names.len() == schema.len()
                            && !projected_names.has_field_selections()
                        {
                            break 'push_common_subset;
                        }
                    }

                    projected_names.normalize(schema);
                    updated_schema = Some(schema.try_project(projected_names.keys()).unwrap());
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

            // Each round reads all columns of the round before.
            IR::UnoptimizedDispatch { .. } | IR::Recursive { .. } | IR::WorkingTableScan { .. } => {
                post_project_and_return!()
            },

            IR::Resolver {
                projection,
                filters,
                filter_drop_columns_idx,
                resolved_ir,
                ..
            } => {
                if resolved_ir.is_some() {
                    edges.swap_input_output(0, 0);
                    return;
                }

                // Resolved plans preserve height with zero-width projections, so a count
                // does not need to materialize a column before resolving the plan.
                if out_edge.projection() == Projection::Len {
                    *out_edge.projection_mut() = Projection::Names;
                    out_edge.names_mut().clear();
                }

                let (projected_names, _) = projected_names_subset_or_return!();

                let len_before_added_names = projected_names.len();

                *filter_drop_columns_idx = Some(len_before_added_names);

                for eir in filters.iter() {
                    SelectionWalker::collect(
                        self.expr_arena,
                        self.prune_struct_fields,
                        eir.node(),
                        projected_names,
                    );
                }

                // The plan is resolved later: its columns are read whole.
                projected_names.select_all_whole();
                *projection = Some(projected_names.keys().cloned().collect());

                reuse_names_alloc(edges);
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
            _ if dtype.is_nested() => 32 + dtype.nesting_level(),
            _ if dtype.is_string() || dtype.is_binary() || dtype.is_object() => 3,
            _ => 2,
        })
        .map(|(k, _)| k)
}

/// Sorts `names` in the order of `schema` if they are all of its columns. A projection of every
/// column is only pushed down for its field selections, and must not reorder the columns.
fn sort_full_projection(names: &mut ColumnSelection, schema: &Schema) {
    if names.len() == schema.len() && !iters_eq(names.keys(), schema.iter_names()) {
        names.sort_by_cached_key(|name, _| schema.index_of(name).unwrap_or(usize::MAX));
    }
}

/// What is needed of output column `name` of the node of `edge`.
fn output_selection(edge: &Edge, name: &str) -> FieldSelection {
    if edge.projection() == Projection::All {
        return FieldSelection::All;
    }

    edge.names().get(name).cloned().unwrap_or_default()
}

/// Returns `None` if `projected_names` match `input_schema`.
fn compute_simple_projection_schema(
    projected_names: &indexmap::map::Slice<PlSmallStr, FieldSelection>,
    input_schema: &Schema,
    allow_order_mismatch: bool,
) -> Option<Schema> {
    if projected_names.len() == input_schema.len() && {
        if allow_order_mismatch {
            projected_names
                .keys()
                .all(|name| input_schema.contains(name))
        } else {
            projected_names
                .keys()
                .zip(input_schema.iter_names())
                .all(|(l, r)| l == r)
        }
    } {
        return None;
    }

    Some(Schema::from_iter(projected_names.keys().map(|name| {
        let dtype = input_schema.get(name).unwrap().clone();
        (name.clone(), dtype)
    })))
}

/// Whether the scan reads only the selected struct fields when given a field selection.
fn scan_supports_field_selection(
    scan_type: &FileScanIR,
    unified_scan_args: &crate::dsl::UnifiedScanArgs,
) -> bool {
    match scan_type {
        // Also with a column mapping (Iceberg field IDs): the selection, made on the names of the
        // table, is mapped to the fields of each file by ID.
        #[cfg(feature = "parquet")]
        FileScanIR::Parquet { .. } => true,
        // Skips the buffers of unselected fields.
        #[cfg(feature = "ipc")]
        FileScanIR::Ipc { .. } => unified_scan_args.column_mapping.is_none(),
        // Parses with the pruned schema.
        #[cfg(feature = "json")]
        FileScanIR::NDJson { .. } => unified_scan_args.column_mapping.is_none(),
        // Expanded after projection pushdown (`pl.scan_delta`, `pl.scan_iceberg`), keeping the
        // field selection; an expansion that can't read it is narrowed after the scan
        // (`narrow_expanded_dataset_scan`).
        #[cfg(feature = "python")]
        FileScanIR::PythonDataset { .. } => true,
        _ => false,
    }
}

/// After a dataset scan at `node` is expanded: if the expanded source can't read the field
/// selection made for it (e.g. a Python scan fallback), the selection is cleared and the scan's
/// output narrowed by a `select` above it, keeping the pruned dtypes the plan was made with.
#[cfg(feature = "python")]
pub(super) fn narrow_expanded_dataset_scan(
    node: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) {
    let IR::Scan {
        scan_type,
        unified_scan_args,
        file_info,
        output_schema,
        ..
    } = ir_arena.get_mut(node)
    else {
        return;
    };
    if unified_scan_args.field_selection.is_none()
        || (!matches!(scan_type.as_ref(), FileScanIR::PythonDataset { .. })
            && scan_supports_field_selection(scan_type, unified_scan_args))
    {
        return;
    }

    let field_selection = unified_scan_args.field_selection.take().unwrap();
    let pruned_schema = output_schema.clone().unwrap();
    let mut full_schema = pruned_schema.as_ref().clone();
    for name in field_selection.0.keys() {
        if let (Some(dtype), Some(full)) = (full_schema.get_mut(name), file_info.schema.get(name)) {
            *dtype = full.clone();
        }
    }
    let full_schema = Arc::new(full_schema);
    *output_schema = Some(full_schema.clone());

    let scan = ir_arena.take(node);
    let scan = ir_arena.add(scan);
    let select = pruning_select(scan, &full_schema, pruned_schema, expr_arena);
    ir_arena.replace(node, select);
}

/// The field selection of a scan: the columns of `selection` that are not read whole, relative to
/// the dtypes of `file_schema`. `previous`: the field selection the scan already has, within which
/// `selection` was made.
fn scan_field_selection(
    selection: &ColumnSelection,
    file_schema: &Schema,
    previous: Option<&ScanFieldSelection>,
) -> Option<ScanFieldSelection> {
    let mut out = PlIndexMap::default();

    for (name, selection) in selection.iter() {
        let previous = previous.map_or(&FieldSelection::All, |p| p.get(name));
        if selection.is_all() && previous.is_all() {
            continue;
        }
        let Some(dtype) = file_schema.get(name) else {
            continue;
        };

        let selection = previous.intersect(selection).normalize(dtype);
        if !selection.is_all() {
            out.insert(name.clone(), selection);
        }
    }

    (!out.is_empty()).then_some(ScanFieldSelection(out))
}

fn set_scan_projection(
    scan_ir: &mut IR,
    projection_schema: Arc<Schema>,
    field_selection: Option<ScanFieldSelection>,
) {
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

    unified_scan_args.field_selection = field_selection.map(Arc::new);

    *output_schema = Some(projection_schema);
}

/// Create a dummy column with small memory footprint.
fn small_dummy_column(name: PlSmallStr, height: usize) -> Column {
    // Prefer 0-field struct if possible, as it doesn't need validity allocation when
    // materialized.
    #[cfg(feature = "dtype-struct")]
    let dtype = DataType::Struct(StructFields::default());
    #[cfg(not(feature = "dtype-struct"))]
    let dtype = DataType::Null;

    Column::full_null(name, height, &dtype)
}

/// Extends `len_nodes` with the arena nodes of the sub-expressions of `node` that evaluate to the
/// height of the input frame, returning `false` if the expression references the input frame in
/// any other way.
///
/// Note that this says nothing about the projection height of the expression itself.
fn extract_len_nodes(
    node: Node,
    expr_arena: &Arena<AExpr>,
    len_nodes: &mut Vec<Node>,
    ae_nodes_scratch: &mut ScratchVec<Node>,
    ae_height_scratch: &mut ScratchVec<ExprProjectionHeight>,
) -> bool {
    let mut stack: UnitVec<Node> = unitvec![node];

    while let Some(node) = stack.pop() {
        match expr_arena.get(node) {
            AExpr::Len => len_nodes.push(node),

            // select(col(a).len()) -> select(len().alias(a))
            //
            // Note: `col(a)` is only used for its height, so we don't traverse into it.
            AExpr::Agg(IRAggExpr::Count {
                input,
                include_nulls: true,
            }) if matches!(
                aexpr_projection_height_rec(
                    *input,
                    expr_arena,
                    ae_nodes_scratch,
                    ae_height_scratch
                ),
                ExprProjectionHeight::Column
            ) =>
            {
                len_nodes.push(node)
            },

            // References the data of the input frame.
            AExpr::Column(_) | AExpr::Element => return false,
            #[cfg(feature = "dtype-struct")]
            AExpr::StructField(_) => return false,

            // Nested contexts, within which `len()` does not refer to the height of the input
            // frame.
            AExpr::Eval { .. } | AExpr::Over { .. } => return false,
            #[cfg(feature = "dtype-struct")]
            AExpr::StructEval { .. } => return false,
            #[cfg(feature = "dynamic_group_by")]
            AExpr::Rolling { .. } => return false,

            // Evaluated against the same frame, so a nested `len()` keeps its meaning.
            ae @ (AExpr::Literal(_)
            | AExpr::Explode { .. }
            | AExpr::BinaryExpr { .. }
            | AExpr::Cast { .. }
            | AExpr::Sort { .. }
            | AExpr::Gather { .. }
            | AExpr::SortBy { .. }
            | AExpr::Filter { .. }
            | AExpr::Agg(_)
            | AExpr::Ternary { .. }
            | AExpr::AnonymousAgg { .. }
            | AExpr::AnonymousFunction { .. }
            | AExpr::Function { .. }
            | AExpr::Slice { .. }) => ae.children_rev(&mut stack),
        }
    }

    true
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
