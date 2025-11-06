use std::sync::Arc;

use polars_core::prelude::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;

use crate::plans::{AExpr, FunctionIR, HintIR, IR, Sorted, into_column};

pub type IRSorted = Arc<[Sorted]>;

#[expect(unused)]
pub fn is_sorted(root: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<IRSorted> {
    let mut sortedness = PlHashMap::default();
    let mut cache_proxy = PlHashMap::default();

    is_sorted_rec(
        root,
        ir_arena,
        expr_arena,
        &mut sortedness,
        &mut cache_proxy,
    )
}

#[recursive::recursive]
fn is_sorted_rec(
    root: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    sortedness: &mut PlHashMap<Node, Option<IRSorted>>,
    cache_proxy: &mut PlHashMap<UniqueId, Option<IRSorted>>,
) -> Option<IRSorted> {
    if let Some(s) = sortedness.get(&root) {
        return s.clone();
    }

    macro_rules! rec {
        ($node:expr) => {{ is_sorted_rec($node, ir_arena, expr_arena, sortedness, cache_proxy) }};
    }

    // @NOTE: Most of the below implementations are very very conservative.
    let sorted = match ir_arena.get(root) {
        #[cfg(feature = "python")]
        IR::PythonScan { .. } => None,
        IR::Slice {
            input,
            offset: _,
            len: _,
        } => rec!(*input),
        IR::Filter {
            input,
            predicate: _,
        } => rec!(*input),
        IR::Scan { .. } => None,
        IR::DataFrameScan { .. } => None,
        IR::SimpleProjection { .. } => None,
        IR::Select { .. } => None,
        IR::Sort {
            input: _,
            by_column,
            slice: _,
            sort_options,
        } => {
            let mut s = by_column
                .iter()
                .map_while(|e| {
                    into_column(e.node(), expr_arena).map(|c| Sorted {
                        column: c.clone(),
                        descending: false,
                        nulls_last: false,
                    })
                })
                .collect::<Vec<_>>();
            if sort_options.descending.len() != 1 {
                s.iter_mut()
                    .zip(sort_options.descending.iter())
                    .for_each(|(s, &d)| s.descending = d);
            } else if sort_options.descending[0] {
                s.iter_mut().for_each(|s| s.descending = true);
            }
            if sort_options.nulls_last.len() != 1 {
                s.iter_mut()
                    .zip(sort_options.nulls_last.iter())
                    .for_each(|(s, &d)| s.nulls_last = d);
            } else if sort_options.nulls_last[0] {
                s.iter_mut().for_each(|s| s.nulls_last = true);
            }

            Some(s.into())
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
        IR::GroupBy { .. } => None,
        IR::Join { .. } => None,
        IR::HStack { .. } => None,
        IR::MapFunction { input, function } => match function {
            FunctionIR::Hint(hint) => match hint {
                HintIR::Sorted(v) => Some(v.clone()),
                #[expect(unreachable_patterns)]
                _ => rec!(*input),
            },
            _ => None,
        },
        IR::Union { .. } => None,
        IR::HConcat { .. } => None,
        IR::ExtContext { .. } => None,
        IR::Sink { .. } => None,
        IR::SinkMultiple { .. } => None,
        #[cfg(feature = "merge_sorted")]
        IR::MergeSorted { .. } => None,
        IR::Distinct { .. } => None,
        IR::Invalid => unreachable!(),
        IR::PlaceholderScan { .. } => unreachable!(),
    };

    sortedness.insert(root, sorted.clone());
    sorted
}
