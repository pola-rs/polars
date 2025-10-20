use std::sync::Arc;

use polars_core::prelude::PlHashMap;
use polars_utils::UnitVec;
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;

use crate::plans::{AExpr, FunctionIR, HintIR, IR, Sorted};

type IRSorted = Arc<[Arc<[Sorted]>]>;

pub fn propagate_sortedness(
    roots: &[Node],
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
) -> PlHashMap<Node, IRSorted> {
    let mut sortedness = PlHashMap::<Node, IRSorted>::default();
    let mut cache_proxy = PlHashMap::<UniqueId, IRSorted>::default();

    for root in roots {
        sortedness_rec(
            *root,
            ir_arena,
            expr_arena,
            &mut sortedness,
            &mut cache_proxy,
        );
    }

    sortedness
}

#[recursive::recursive]
fn sortedness_rec(
    root: Node,
    ir_arena: &mut Arena<IR>,
    expr_arena: &mut Arena<AExpr>,
    sortedness: &mut PlHashMap<Node, IRSorted>,
    cache_proxy: &mut PlHashMap<UniqueId, IRSorted>,
) -> IRSorted {
    if let Some(s) = sortedness.get(&root) {
        return s.clone();
    }

    fn empty() -> IRSorted {
        [].into()
    }

    macro_rules! rec {
        ($node:expr) => {{ sortedness_rec($node, ir_arena, expr_arena, sortedness, cache_proxy) }};
    }

    // @NOTE: Most of the below implementations are very very conservative.
    let sorted = match ir_arena.get(root) {
        IR::PythonScan { .. } => empty(),
        IR::Slice {
            input,
            offset: _,
            len: _,
        } => rec!(*input),
        IR::Filter {
            input,
            predicate: _,
        } => rec!(*input),
        IR::Scan { .. } => empty(),
        IR::DataFrameScan { .. } => empty(),
        IR::SimpleProjection { input, columns } => {
            rec!(*input);
            empty()
        },
        IR::Select { input, .. } => {
            rec!(*input);
            empty()
        },
        IR::Sort {
            input,
            by_column,
            slice: _,
            sort_options,
        } => {
            rec!(*input);
            empty()
        },
        IR::Cache { input, id } => {
            let (input, id) = (*input, *id);
            if let Some(s) = cache_proxy.get(&id) {
                s.clone()
            } else {
                let s = rec!(input);
                cache_proxy.insert(id, s.clone());
                s
            }
        },
        IR::GroupBy { input, .. } => {
            rec!(*input);
            empty()
        },
        IR::Join {
            input_left,
            input_right,
            ..
        } => {
            let (input_left, input_right) = (*input_left, *input_right);
            rec!(input_left);
            rec!(input_right);
            empty()
        },
        IR::HStack {
            input,
            exprs,
            schema,
            options,
        } => {
            rec!(*input);
            empty()
        },
        IR::Distinct { input, options: _ } => {
            rec!(*input);
            empty()
        },
        IR::MapFunction { input, function } => {
            let function = function.clone();
            let mut input = rec!(*input);
            match function {
                FunctionIR::Hint(hint) => match hint {
                    HintIR::Sorted(v) => input
                        .iter()
                        .cloned()
                        .chain(std::iter::once(v.clone()))
                        .collect(),
                    _ => input,
                },
                _ => empty(),
            }
        },
        IR::Union { inputs, options: _ } => {
            let inputs = inputs.clone();
            for i in inputs {
                rec!(i);
            }
            empty()
        },
        IR::HConcat { inputs, .. } => {
            let inputs = inputs.clone();
            for i in inputs {
                rec!(i);
            }
            empty()
        },
        IR::ExtContext {
            input,
            contexts,
            schema: _,
        } => {
            let input = *input;
            let contexts = contexts.clone();
            rec!(input);
            for c in contexts {
                rec!(c);
            }
            empty()
        },
        IR::Sink { input, payload: _ } => {
            rec!(*input);
            empty()
        },
        IR::SinkMultiple { inputs } => {
            let inputs = inputs.clone();
            for i in inputs {
                rec!(i);
            }
            empty()
        },
        IR::MergeSorted {
            input_left,
            input_right,
            key: _,
        } => {
            let (input_left, input_right) = (*input_left, *input_right);
            rec!(input_left);
            rec!(input_right);
            empty()
        },
        IR::Invalid => unreachable!(),
    };

    sortedness.insert(root, sorted.clone());
    sorted
}
