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

use polars_core::prelude::*;
use polars_utils::arena::{Arena, Node};
use polars_utils::idx_vec::UnitVec;
use polars_utils::unique_column_name;
use recursive::recursive;

use crate::plans::{
    AExpr, ArenaExprIter, CanonicalExprId, CanonicalExprMap, ExprIR, IR, OutputName,
    ToFieldContext, is_splittable,
};
use crate::prelude::{ProjectionOptions, WindowMapping};

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
    spec: usize,
}

struct Extractor {
    canonical: CanonicalExprMap,
    specs: Vec<WindowSpec>,
    windows: Vec<ExtractedWindow>,
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
            let name = unique_column_name();
            self.windows.push(ExtractedWindow {
                name: name.clone(),
                function: *function,
                spec,
            });
            expr_arena.add(AExpr::Column(name))
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
                && !expr_arena.iter(key).any(|(_, ae)| is_window(ae))
        };

        if partition_by.is_empty()
            || !partition_by.iter().all(|key| is_valid_key(*key))
            || !order_by.as_ref().is_none_or(|(key, _)| is_valid_key(*key))
        {
            return None;
        }

        let spec = WindowSpec {
            partition_by: partition_by
                .iter()
                .map(|key| self.canonical.resolve(*key, expr_arena))
                .collect(),
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

    let Extractor {
        canonical,
        specs,
        windows,
        ..
    } = extractor;

    // Keys that are not plain columns are computed below the windows.
    let mut key_names: PlIndexMap<CanonicalExprId, PlSmallStr> = PlIndexMap::new();
    let mut key_exprs = Vec::new();
    let key_ids = specs.iter().flat_map(|s| {
        s.partition_by
            .iter()
            .copied()
            .chain(s.order_by.as_ref().map(|(id, _)| *id))
    });
    for id in key_ids {
        if key_names.contains_key(&id) {
            continue;
        }
        let key = canonical.representative(id);
        let name = match expr_arena.get(key) {
            AExpr::Column(name) => name.clone(),
            _ => {
                let name = unique_column_name();
                key_exprs.push(ExprIR::new(key, OutputName::Alias(name.clone())));
                name
            },
        };
        key_names.insert(id, name);
    }

    let mut top = input;
    if !key_exprs.is_empty() {
        top = add_hstack(
            top,
            key_exprs,
            ProjectionOptions::default(),
            ir_arena,
            expr_arena,
        );
    }

    for (spec_idx, spec) in specs.iter().enumerate() {
        let partition_by = spec
            .partition_by
            .iter()
            .map(|id| key_names[id].clone())
            .collect::<Vec<_>>();
        let order_by = spec
            .order_by
            .as_ref()
            .map(|(id, options)| (key_names[id].clone(), *options));

        let window_exprs = windows
            .iter()
            .filter(|w| w.spec == spec_idx)
            .map(|w| {
                let over = AExpr::Over {
                    function: w.function,
                    partition_by: partition_by
                        .iter()
                        .map(|name| expr_arena.add(AExpr::Column(name.clone())))
                        .collect(),
                    order_by: order_by.as_ref().map(|(name, options)| {
                        (expr_arena.add(AExpr::Column(name.clone())), *options)
                    }),
                    mapping: WindowMapping::GroupsToRows,
                };
                ExprIR::new(expr_arena.add(over), OutputName::Alias(w.name.clone()))
            })
            .collect::<Vec<_>>();

        let input_schema = ir_arena.get(top).schema(ir_arena).into_owned();
        let mut schema = input_schema.as_ref().clone();
        for e in &window_exprs {
            let field = expr_arena
                .get(e.node())
                .to_field(&ToFieldContext::new(expr_arena, &input_schema))
                .unwrap();
            schema.with_column(e.output_name().clone(), field.dtype);
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
