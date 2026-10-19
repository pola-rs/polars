//! Moves `over` windows out of `Select` and `HStack` nodes into [`IR::Window`] nodes.
//!
//! ```text
//! SELECT [a, (x - x.mean().over(g)) as b]
//!   input
//! ```
//! becomes
//! ```text
//! SELECT [a, (x - col(tmp)) as b]
//!   WINDOW PARTITION BY [g]: [x.mean().over(g) as tmp]
//!     input
//! ```
//!
//! Only windows that are reached from an output through elementwise expressions are moved.
//! Keys that are not plain columns are computed in an `HStack` below the windows. Every window
//! spec gets its own node.
//!
//! If all windows of an unordered spec are reductions of numeric inputs, or elementwise
//! expressions of such reductions, the streaming engine can reduce them without keeping the
//! groups. Then every reduction gets its own window, e.g. `(x.max() - x.min()).over(g)` becomes
//! `x.max().over(g) - x.min().over(g)`, and reduction inputs that are not plain columns are
//! computed below the windows.

use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use recursive::recursive;

use crate::plans::{
    AExpr, ArenaExprIter, CanonicalExprId, CanonicalExprMap, ExprIR, IR, IRAggExpr, OutputName,
    ToFieldContext, is_elementwise_rec, is_length_preserving_ae, is_splittable,
};
use crate::prelude::{ProjectionOptions, WindowMapping};
use crate::unique_column_name;

pub(super) fn extract_windows(root: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) {
    let mut visited = PlIndexSet::new();
    let mut stack = vec![root];
    let mut projections = Vec::new();

    while let Some(node) = stack.pop() {
        if !visited.insert(node) {
            continue;
        }
        let ir = ir_arena.get(node);
        if matches!(ir, IR::Select { .. } | IR::HStack { .. }) {
            projections.push(node);
        }
        stack.extend(ir.inputs());
    }

    for node in projections {
        extract_from_projection(node, ir_arena, expr_arena);
    }
}

#[derive(PartialEq)]
struct WindowSpec {
    partition_by: Vec<CanonicalExprId>,
    order_by: Option<(CanonicalExprId, SortOptions)>,
}

struct ExtractedWindow {
    name: PlSmallStr,
    function: Node,
    id: CanonicalExprId,
    spec: usize,
}

/// A window expression of an `IR::Window` node.
struct WindowPart {
    name: PlSmallStr,
    function: Node,
    /// The input of the reduction `function` if it is computed below the window.
    input: Option<CanonicalExprId>,
}

struct Extractor {
    canonical: CanonicalExprMap,
    specs: Vec<WindowSpec>,
    windows: Vec<ExtractedWindow>,
    /// The column of each deterministic window function per spec.
    window_names: PlIndexMap<(CanonicalExprId, usize), PlSmallStr>,
    memo: PlIndexMap<Node, Node>,
}

impl Extractor {
    #[recursive]
    fn rebuild(&mut self, node: Node, expr_arena: &mut Arena<AExpr>) -> Node {
        if let Some(new_node) = self.memo.get(&node) {
            return *new_node;
        }

        let new_node = if let Some(spec) = self.window_spec(node, expr_arena) {
            let AExpr::Over { function, .. } = expr_arena.get(node) else {
                unreachable!()
            };
            self.add_window(*function, spec, expr_arena)
        } else {
            let ae = expr_arena.get(node);
            let mut inputs_rev = UnitVec::new();
            ae.inputs_rev(&mut inputs_rev);

            let descend =
                !matches!(ae, AExpr::Eval { .. }) && is_splittable(ae, &inputs_rev, expr_arena);

            if descend {
                let inputs = inputs_rev
                    .iter()
                    .rev()
                    .map(|input| self.rebuild(*input, expr_arena))
                    .collect::<Vec<_>>();

                if inputs.iter().eq(inputs_rev.iter().rev()) {
                    node
                } else {
                    let ae = expr_arena.get(node).clone();
                    expr_arena.add(ae.replace_inputs(&inputs))
                }
            } else {
                node
            }
        };

        self.memo.insert(node, new_node);
        new_node
    }

    /// Adds the window `function` and returns the column that holds its values.
    fn add_window(&mut self, function: Node, spec: usize, expr_arena: &mut Arena<AExpr>) -> Node {
        let id = self.canonical.resolve(function, expr_arena);
        let deterministic = !self.canonical.is_nondeterministic(id);
        if deterministic && let Some(name) = self.window_names.get(&(id, spec)) {
            return expr_arena.add(AExpr::Column(name.clone()));
        }

        let name = unique_column_name();
        if deterministic {
            self.window_names.insert((id, spec), name.clone());
        }
        self.windows.push(ExtractedWindow {
            name: name.clone(),
            function,
            id,
            spec,
        });
        expr_arena.add(AExpr::Column(name))
    }

    /// The window expressions of the node of `spec`. Split windows are added to `split_windows` as
    /// the expression that computes them from the window columns, if `allow_split`.
    fn spec_windows(
        &mut self,
        spec: usize,
        schema: &Schema,
        allow_split: bool,
        split_windows: &mut PlIndexMap<PlSmallStr, Node>,
        expr_arena: &mut Arena<AExpr>,
    ) -> Vec<WindowPart> {
        let windows = self
            .windows
            .iter()
            .filter(|w| w.spec == spec)
            .map(|w| (w.name.clone(), w.function, w.id))
            .collect::<Vec<_>>();

        let split = allow_split
            && self.specs[spec].order_by.is_none()
            && windows.iter().all(|(_, function, id)| {
                !self.canonical.is_nondeterministic(*id)
                    && is_reducible(*function, schema, expr_arena)
            });
        if !split {
            let parts = windows
                .into_iter()
                .map(|(name, function, _)| WindowPart {
                    name,
                    function,
                    input: None,
                })
                .collect();
            return parts;
        }

        let mut parts = Vec::with_capacity(windows.len());
        let mut part_names = PlIndexMap::new();
        // Reductions first, so that the split windows can reuse them.
        for (name, function, _) in &windows {
            if is_reduction(*function, schema, expr_arena) {
                self.add_part(
                    *function,
                    Some(name.clone()),
                    &mut parts,
                    &mut part_names,
                    expr_arena,
                );
            }
        }
        for (name, function, _) in windows {
            if !is_reduction(function, schema, expr_arena) {
                let node = self.split(function, &mut parts, &mut part_names, schema, expr_arena);
                split_windows.insert(name, node);
            }
        }
        parts
    }

    /// Adds the reduction `function` as a window, unless an equal one exists, and returns the name
    /// of its column.
    fn add_part(
        &mut self,
        function: Node,
        name: Option<PlSmallStr>,
        parts: &mut Vec<WindowPart>,
        part_names: &mut PlIndexMap<CanonicalExprId, PlSmallStr>,
        expr_arena: &Arena<AExpr>,
    ) -> PlSmallStr {
        let id = self.canonical.resolve(function, expr_arena);
        if let Some(name) = part_names.get(&id) {
            return name.clone();
        }
        let name = name.unwrap_or_else(unique_column_name);
        let input = match expr_arena.get(function) {
            AExpr::Agg(agg) => {
                let input = agg.get_input().first();
                (!matches!(expr_arena.get(input), AExpr::Column(_)))
                    .then(|| self.canonical.resolve(input, expr_arena))
            },
            _ => None,
        };
        part_names.insert(id, name.clone());
        parts.push(WindowPart {
            name: name.clone(),
            function,
            input,
        });
        name
    }

    /// Rebuilds `node`, for which [`is_reducible`] holds, with every reduction replaced by the
    /// column of its window.
    #[recursive]
    fn split(
        &mut self,
        node: Node,
        parts: &mut Vec<WindowPart>,
        part_names: &mut PlIndexMap<CanonicalExprId, PlSmallStr>,
        schema: &Schema,
        expr_arena: &mut Arena<AExpr>,
    ) -> Node {
        if is_reduction(node, schema, expr_arena) {
            let name = self.add_part(node, None, parts, part_names, expr_arena);
            return expr_arena.add(AExpr::Column(name));
        }

        let mut inputs = UnitVec::new();
        expr_arena.get(node).inputs_rev(&mut inputs);
        for input in inputs.iter_mut() {
            *input = self.split(*input, parts, part_names, schema, expr_arena);
        }
        inputs.reverse();
        let ae = expr_arena.get(node).clone();
        expr_arena.add(ae.replace_inputs(&inputs))
    }

    /// The index of the spec of `node` if it is a window that can be moved.
    fn window_spec(&mut self, node: Node, expr_arena: &Arena<AExpr>) -> Option<usize> {
        let AExpr::Over {
            function: _,
            partition_by,
            order_by,
            mapping: WindowMapping::GroupsToRows,
        } = expr_arena.get(node)
        else {
            return None;
        };

        let is_valid_key = |key: Node| {
            !matches!(expr_arena.get(key), AExpr::Literal(_))
                && is_length_preserving_ae(key, expr_arena)
                && !expr_arena.iter(key).any(|(_, ae)| is_window(ae))
        };

        if partition_by.is_empty()
            || !partition_by.iter().all(|key| is_valid_key(*key))
            || !order_by.as_ref().is_none_or(|(key, _)| is_valid_key(*key))
        {
            return None;
        }

        // Repeated keys do not change the partitions.
        let mut partition_ids = Vec::with_capacity(partition_by.len());
        for key in partition_by {
            let id = self.canonical.resolve(*key, expr_arena);
            if !partition_ids.contains(&id) {
                partition_ids.push(id);
            }
        }

        let spec = WindowSpec {
            partition_by: partition_ids,
            order_by: order_by
                .as_ref()
                .map(|(key, options)| (self.canonical.resolve(*key, expr_arena), *options)),
        };

        Some(match self.specs.iter().position(|s| *s == spec) {
            Some(idx) => idx,
            None => {
                self.specs.push(spec);
                self.specs.len() - 1
            },
        })
    }
}

/// Whether `node` is a reduction of a numeric, length preserving, elementwise expression that the
/// streaming engine computes without keeping the groups.
fn is_reduction(node: Node, schema: &Schema, expr_arena: &Arena<AExpr>) -> bool {
    let input = match expr_arena.get(node) {
        AExpr::Len => return true,
        AExpr::Agg(
            IRAggExpr::Sum { input, .. }
            | IRAggExpr::Min { input, .. }
            | IRAggExpr::Max { input, .. }
            | IRAggExpr::Count { input, .. }
            | IRAggExpr::Mean(input)
            | IRAggExpr::Var(input, _)
            | IRAggExpr::Std(input, _),
        ) => *input,
        _ => return false,
    };
    is_elementwise_rec(input, expr_arena)
        && is_length_preserving_ae(input, expr_arena)
        && expr_arena
            .get(input)
            .to_dtype(&ToFieldContext::new(expr_arena, schema))
            .is_ok_and(|dtype| dtype.is_primitive_numeric())
}

/// Whether `node` is a reduction, or an elementwise expression of reductions and scalar literals
/// with at least one reduction.
fn is_reducible(node: Node, schema: &Schema, expr_arena: &Arena<AExpr>) -> bool {
    #[recursive]
    fn only_reductions(node: Node, schema: &Schema, expr_arena: &Arena<AExpr>) -> bool {
        if is_reduction(node, schema, expr_arena) {
            return true;
        }
        let ae = expr_arena.get(node);
        let mut inputs_rev = UnitVec::new();
        ae.inputs_rev(&mut inputs_rev);
        !matches!(ae, AExpr::Eval { .. })
            && is_splittable(ae, &inputs_rev, expr_arena)
            && inputs_rev
                .iter()
                .all(|input| only_reductions(*input, schema, expr_arena))
    }

    only_reductions(node, schema, expr_arena)
        && expr_arena
            .iter(node)
            .any(|(node, _)| is_reduction(node, schema, expr_arena))
}

fn is_window(ae: &AExpr) -> bool {
    match ae {
        AExpr::Over { .. } => true,
        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling { .. } => true,
        _ => false,
    }
}

fn extract_from_projection(node: Node, ir_arena: &mut Arena<IR>, expr_arena: &mut Arena<AExpr>) {
    let (input, exprs, options) = match ir_arena.get(node) {
        IR::Select {
            input,
            expr,
            options,
            ..
        } => (*input, expr, *options),
        IR::HStack {
            input,
            exprs,
            options,
            ..
        } => (*input, exprs, *options),
        _ => unreachable!(),
    };

    let mut extractor = Extractor {
        canonical: CanonicalExprMap::new(),
        specs: Vec::new(),
        windows: Vec::new(),
        window_names: Default::default(),
        memo: Default::default(),
    };

    let exprs = exprs.clone();
    let new_exprs = exprs
        .iter()
        .map(|e| {
            let new_node = extractor.rebuild(e.node(), expr_arena);
            if new_node == e.node() {
                e.clone()
            } else {
                ExprIR::new(new_node, OutputName::Alias(e.output_name().clone()))
            }
        })
        .collect::<Vec<_>>();

    if extractor.windows.is_empty() {
        return;
    }

    let input_schema = ir_arena.get(input).schema(ir_arena).into_owned();
    // The streaming engine cannot gather objects, so window nodes with objects are not reduced.
    let is_object = |node: Node| {
        expr_arena
            .get(node)
            .to_dtype(&ToFieldContext::new(expr_arena, &input_schema))
            .is_ok_and(|dtype| dtype.contains_objects())
    };
    let has_objects = input_schema
        .iter_values()
        .any(|dtype| dtype.contains_objects())
        || extractor.windows.iter().any(|w| is_object(w.function))
        || extractor.specs.iter().any(|s| {
            s.partition_by
                .iter()
                .chain(s.order_by.as_ref().map(|(id, _)| id))
                .any(|id| is_object(extractor.canonical.representative(*id)))
        });
    let mut split_windows = PlIndexMap::new();
    let spec_parts = (0..extractor.specs.len())
        .map(|spec| {
            extractor.spec_windows(
                spec,
                &input_schema,
                !has_objects,
                &mut split_windows,
                expr_arena,
            )
        })
        .collect::<Vec<_>>();
    let new_exprs = if split_windows.is_empty() {
        new_exprs
    } else {
        let mut memo = PlIndexMap::new();
        new_exprs
            .into_iter()
            .map(|e| {
                let node = inline_columns(e.node(), &split_windows, &mut memo, expr_arena);
                ExprIR::new(node, e.output_name_inner().clone())
            })
            .collect()
    };

    let Extractor {
        canonical, specs, ..
    } = extractor;

    // Keys and reduction inputs that are not plain columns are computed below the windows.
    let mut column_names: PlIndexMap<CanonicalExprId, PlSmallStr> = PlIndexMap::new();
    let mut hstack_exprs = Vec::new();
    let key_ids = specs.iter().flat_map(|s| {
        s.partition_by
            .iter()
            .copied()
            .chain(s.order_by.as_ref().map(|(id, _)| *id))
    });
    let input_ids = spec_parts.iter().flatten().filter_map(|p| p.input);
    for id in key_ids.chain(input_ids) {
        if column_names.contains_key(&id) {
            continue;
        }
        let expr = canonical.representative(id);
        let name = match expr_arena.get(expr) {
            AExpr::Column(name) => name.clone(),
            _ => {
                let name = unique_column_name();
                hstack_exprs.push(ExprIR::new(expr, OutputName::Alias(name.clone())));
                name
            },
        };
        column_names.insert(id, name);
    }

    let mut top = input;
    if !hstack_exprs.is_empty() {
        top = add_hstack(
            top,
            hstack_exprs,
            ProjectionOptions::default(),
            ir_arena,
            expr_arena,
        );
    }

    for (spec, parts) in specs.iter().zip(spec_parts) {
        let partition_by = spec
            .partition_by
            .iter()
            .map(|id| column_names[id].clone())
            .collect::<Vec<_>>();
        let order_by = spec
            .order_by
            .as_ref()
            .map(|(id, options)| (column_names[id].clone(), *options));

        let window_exprs = parts
            .into_iter()
            .map(|p| {
                let function = match p.input {
                    Some(id) => {
                        let input = expr_arena.add(AExpr::Column(column_names[&id].clone()));
                        let ae = expr_arena.get(p.function).clone();
                        expr_arena.add(ae.replace_inputs(&[input]))
                    },
                    None => p.function,
                };
                let over = AExpr::Over {
                    function,
                    partition_by: partition_by
                        .iter()
                        .map(|name| expr_arena.add(AExpr::Column(name.clone())))
                        .collect(),
                    order_by: order_by.as_ref().map(|(name, options)| {
                        (expr_arena.add(AExpr::Column(name.clone())), *options)
                    }),
                    mapping: WindowMapping::GroupsToRows,
                };
                ExprIR::new(expr_arena.add(over), OutputName::Alias(p.name))
            })
            .collect::<Vec<_>>();

        let input_schema = ir_arena.get(top).schema(ir_arena).into_owned();
        let mut schema = input_schema.as_ref().clone();
        for e in &window_exprs {
            let field = expr_arena
                .get(e.node())
                .to_field(&ToFieldContext::new(expr_arena, &input_schema))
                .unwrap();
            let dtype = field.dtype.materialize_unknown(true).unwrap();
            schema.with_column(e.output_name().clone(), dtype);
        }

        top = ir_arena.add(IR::Window {
            input: top,
            partition_by,
            order_by,
            exprs: window_exprs,
            schema: Arc::new(schema),
            maintain_order: true,
            ordered_eval: true,
        });
    }

    let new_ir = match ir_arena.get(node) {
        IR::Select { schema, .. } => IR::Select {
            input: top,
            expr: new_exprs,
            schema: schema.clone(),
            options,
        },
        IR::HStack { schema, .. } => {
            let schema = schema.clone();
            let hstack = add_hstack(top, new_exprs, options, ir_arena, expr_arena);
            IR::SimpleProjection {
                input: hstack,
                columns: schema,
            }
        },
        _ => unreachable!(),
    };
    ir_arena.replace(node, new_ir);
}

/// Replaces the columns in `exprs` by their expressions.
#[recursive]
fn inline_columns(
    node: Node,
    exprs: &PlIndexMap<PlSmallStr, Node>,
    memo: &mut PlIndexMap<Node, Node>,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    if let Some(new_node) = memo.get(&node) {
        return *new_node;
    }

    let ae = expr_arena.get(node);
    let new_node = if let AExpr::Column(name) = ae {
        exprs.get(name).copied().unwrap_or(node)
    } else {
        let mut inputs = UnitVec::new();
        ae.inputs_rev(&mut inputs);
        let mut changed = false;
        for input in inputs.iter_mut() {
            let new_input = inline_columns(*input, exprs, memo, expr_arena);
            changed |= new_input != *input;
            *input = new_input;
        }
        if changed {
            inputs.reverse();
            let ae = expr_arena.get(node).clone();
            expr_arena.add(ae.replace_inputs(&inputs))
        } else {
            node
        }
    };

    memo.insert(node, new_node);
    new_node
}

fn add_hstack(
    input: Node,
    exprs: Vec<ExprIR>,
    options: ProjectionOptions,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> Node {
    crate::plans::IRBuilder::new(input, expr_arena, ir_arena)
        .with_columns(exprs, options)
        .node()
}
