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
}

/// A window expression of an `IR::Window` node.
struct WindowPart {
    name: PlSmallStr,
    function: Node,
    /// The input of the reduction `function` if it is computed below the window.
    input: Option<CanonicalExprId>,
    /// Whether the column is only read by split windows.
    temporary: bool,
}

/// The windows of the `IR::Window` node of a spec.
struct SpecWindows {
    parts: Vec<WindowPart>,
    /// The expressions that compute the split windows from the columns of `parts`.
    split: PlIndexMap<PlSmallStr, Node>,
}

/// The column of a reduction, and the node that reads it once one is needed.
type PartColumn = (PlSmallStr, Option<Node>);

/// The node that reads `column`.
fn column_node(column: &mut PartColumn, expr_arena: &mut Arena<AExpr>) -> Node {
    *column
        .1
        .get_or_insert_with(|| expr_arena.add(AExpr::Column(column.0.clone())))
}

struct Extractor {
    canonical: CanonicalExprMap,
    specs: Vec<WindowSpec>,
    /// The windows of each spec.
    windows: Vec<Vec<ExtractedWindow>>,
    /// The column of each deterministic window function per spec.
    window_names: PlIndexMap<(CanonicalExprId, usize), PlSmallStr>,
    memo: PlIndexMap<Node, Node>,
    /// Whether each node is a reduction, see [`Extractor::is_reduction`].
    reductions: PlIndexMap<Node, bool>,
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
        self.windows[spec].push(ExtractedWindow {
            name: name.clone(),
            function,
            id,
        });
        expr_arena.add(AExpr::Column(name))
    }

    /// The windows of the node of `spec`. If `allow_split`, windows that combine reductions are
    /// split into a window per reduction.
    fn spec_windows(
        &mut self,
        spec: usize,
        windows: Vec<ExtractedWindow>,
        schema: &Schema,
        allow_split: bool,
        expr_arena: &mut Arena<AExpr>,
    ) -> SpecWindows {
        let mut split = PlIndexMap::new();
        let can_split = allow_split
            && self.specs[spec].order_by.is_none()
            && windows.iter().all(|w| {
                !self.canonical.is_nondeterministic(w.id)
                    && self.has_reduction(w.function, schema, expr_arena) == Some(true)
            });
        if !can_split {
            let parts = windows
                .into_iter()
                .map(|w| WindowPart {
                    name: w.name,
                    function: w.function,
                    input: None,
                    temporary: false,
                })
                .collect();
            return SpecWindows { parts, split };
        }

        let mut parts = Vec::with_capacity(windows.len());
        let mut part_columns = PlIndexMap::new();
        for w in windows {
            let node = if self.is_reduction(w.function, schema, expr_arena) {
                let column = self.add_part(
                    w.function,
                    Some(w.name.clone()),
                    &mut parts,
                    &mut part_columns,
                    expr_arena,
                );
                if column.0 == w.name {
                    continue;
                }
                column_node(column, expr_arena)
            } else {
                self.split(
                    w.function,
                    &mut parts,
                    &mut part_columns,
                    schema,
                    expr_arena,
                )
            };
            split.insert(w.name, node);
        }
        SpecWindows { parts, split }
    }

    /// Adds the reduction `function` as a window, unless an equal one exists, and returns its
    /// column.
    fn add_part<'a>(
        &mut self,
        function: Node,
        name: Option<PlSmallStr>,
        parts: &mut Vec<WindowPart>,
        part_columns: &'a mut PlIndexMap<CanonicalExprId, PartColumn>,
        expr_arena: &Arena<AExpr>,
    ) -> &'a mut PartColumn {
        let id = self.canonical.resolve(function, expr_arena);
        if let Some(idx) = part_columns.get_index_of(&id) {
            return &mut part_columns[idx];
        }
        let temporary = name.is_none();
        let name = name.unwrap_or_else(unique_column_name);
        let input = match expr_arena.get(function) {
            AExpr::Agg(agg) => {
                let input = agg.get_input().first();
                (!matches!(expr_arena.get(input), AExpr::Column(_)))
                    .then(|| self.canonical.resolve(input, expr_arena))
            },
            _ => None,
        };
        parts.push(WindowPart {
            name: name.clone(),
            function,
            input,
            temporary,
        });
        let (idx, _) = part_columns.insert_full(id, (name, None));
        &mut part_columns[idx]
    }

    /// Rebuilds `node`, for which [`Extractor::has_reduction`] holds, with every reduction
    /// replaced by the column of its window.
    #[recursive]
    fn split(
        &mut self,
        node: Node,
        parts: &mut Vec<WindowPart>,
        part_columns: &mut PlIndexMap<CanonicalExprId, PartColumn>,
        schema: &Schema,
        expr_arena: &mut Arena<AExpr>,
    ) -> Node {
        if self.is_reduction(node, schema, expr_arena) {
            let column = self.add_part(node, None, parts, part_columns, expr_arena);
            return column_node(column, expr_arena);
        }

        let mut inputs = UnitVec::new();
        expr_arena.get(node).inputs_rev(&mut inputs);
        let mut changed = false;
        for input in inputs.iter_mut() {
            let new_input = self.split(*input, parts, part_columns, schema, expr_arena);
            changed |= new_input != *input;
            *input = new_input;
        }
        if !changed {
            return node;
        }
        inputs.reverse();
        let ae = expr_arena.get(node).clone();
        expr_arena.add(ae.replace_inputs(&inputs))
    }

    /// Whether `node` is a scalar window reduction of a numeric, length preserving, elementwise
    /// expression.
    fn is_reduction(&mut self, node: Node, schema: &Schema, expr_arena: &Arena<AExpr>) -> bool {
        *self.reductions.entry(node).or_insert_with(|| {
            is_scalar_window_reduction(node, expr_arena, |input| {
                is_elementwise_rec(input, expr_arena)
                    && is_length_preserving_ae(input, expr_arena)
                    && expr_arena
                        .get(input)
                        .to_dtype(&ToFieldContext::new(expr_arena, schema))
                        .is_ok_and(|dtype| dtype.is_primitive_numeric())
            })
        })
    }

    /// Whether `node` has a reduction, or `None` if it is not a reduction or a numeric, boolean or
    /// null elementwise expression of reductions, columns and scalar literals.
    #[recursive]
    fn has_reduction(
        &mut self,
        node: Node,
        schema: &Schema,
        expr_arena: &Arena<AExpr>,
    ) -> Option<bool> {
        if self.is_reduction(node, schema, expr_arena) {
            return Some(true);
        }
        let ae = expr_arena.get(node);
        let mut inputs_rev = UnitVec::new();
        ae.inputs_rev(&mut inputs_rev);
        let is_valid = !matches!(ae, AExpr::Eval { .. })
            && is_splittable(ae, &inputs_rev, expr_arena)
            && ae
                .to_dtype(&ToFieldContext::new(expr_arena, schema))
                .is_ok_and(|dtype| {
                    dtype.is_primitive_numeric() || dtype.is_bool() || dtype.is_null()
                });
        if !is_valid {
            return None;
        }
        let mut has = false;
        for input in inputs_rev {
            has |= self.has_reduction(input, schema, expr_arena)?;
        }
        Some(has)
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
                self.windows.push(Vec::new());
                self.specs.len() - 1
            },
        })
    }
}

/// Whether `node` is a reduction that the streaming engine computes without keeping the groups,
/// given that `is_valid_input` holds for its input.
pub fn is_scalar_window_reduction(
    node: Node,
    expr_arena: &Arena<AExpr>,
    is_valid_input: impl Fn(Node) -> bool,
) -> bool {
    match expr_arena.get(node) {
        AExpr::Len => true,
        AExpr::Agg(
            IRAggExpr::Sum { input, .. }
            | IRAggExpr::Min { input, .. }
            | IRAggExpr::Max { input, .. }
            | IRAggExpr::Count { input, .. }
            | IRAggExpr::Mean(input)
            | IRAggExpr::Var(input, _)
            | IRAggExpr::Std(input, _),
        ) => is_valid_input(*input),
        _ => false,
    }
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
        reductions: Default::default(),
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

    if extractor.specs.is_empty() {
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
        || extractor
            .windows
            .iter()
            .flatten()
            .any(|w| is_object(w.function))
        || extractor.specs.iter().any(|s| {
            s.partition_by
                .iter()
                .chain(s.order_by.as_ref().map(|(id, _)| id))
                .any(|id| is_object(extractor.canonical.representative(*id)))
        });
    let mut spec_windows = std::mem::take(&mut extractor.windows)
        .into_iter()
        .enumerate()
        .map(|(spec, windows)| {
            extractor.spec_windows(spec, windows, &input_schema, !has_objects, expr_arena)
        })
        .collect::<Vec<_>>();

    // Split windows of the last spec that are read once are computed in the output expressions.
    let mut inlined = PlIndexMap::new();
    let last = spec_windows.last_mut().unwrap();
    if !last.split.is_empty() {
        let mut uses = PlIndexMap::<PlSmallStr, usize>::new();
        for e in &new_exprs {
            for (_, ae) in expr_arena.iter(e.node()) {
                if let AExpr::Column(name) = ae
                    && last.split.contains_key(name)
                {
                    *uses.entry(name.clone()).or_default() += 1;
                }
            }
        }
        last.split.retain(|name, node| {
            let is_inlined = uses.get(name) == Some(&1);
            if is_inlined {
                inlined.insert(name.clone(), *node);
            }
            !is_inlined
        });
    }
    let new_exprs = if inlined.is_empty() {
        new_exprs
    } else {
        let mut memo = PlIndexMap::new();
        new_exprs
            .into_iter()
            .map(|e| {
                let node = inline_columns(e.node(), &inlined, &mut memo, expr_arena);
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
    let input_ids = spec_windows
        .iter()
        .flat_map(|w| w.parts.iter().filter_map(|p| p.input));
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

    let n_specs = specs.len();
    for (idx, (spec, windows)) in specs.iter().zip(spec_windows).enumerate() {
        let SpecWindows { parts, split } = windows;
        // The columns that later windows do not have to keep.
        let temporary = if idx + 1 < n_specs && !split.is_empty() {
            parts
                .iter()
                .filter(|p| p.temporary)
                .map(|p| p.name.clone())
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };

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

        if !split.is_empty() {
            let exprs = split
                .into_iter()
                .map(|(name, node)| ExprIR::new(node, OutputName::Alias(name)))
                .collect();
            top = add_hstack(
                top,
                exprs,
                ProjectionOptions::default(),
                ir_arena,
                expr_arena,
            );
        }
        if !temporary.is_empty() {
            let columns = ir_arena
                .get(top)
                .schema(ir_arena)
                .iter()
                .filter(|(name, _)| !temporary.contains(name))
                .map(|(name, dtype)| (name.clone(), dtype.clone()))
                .collect();
            top = ir_arena.add(IR::SimpleProjection {
                input: top,
                columns: Arc::new(columns),
            });
        }
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
