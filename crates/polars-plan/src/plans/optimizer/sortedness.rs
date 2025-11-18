use std::sync::Arc;

use polars_core::prelude::PlHashMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::unique_id::UniqueId;

use crate::plans::{AExpr, ExprIR, FunctionIR, HintIR, IR, Sorted, into_column};

#[derive(Debug, Clone)]
pub struct IRSorted(pub Arc<[Sorted]>);

impl IRSorted {
    /// Is the data in any way sorted by the keys?
    pub fn is_sorted_any(&self, keys: &[ExprIR], expr_arena: &Arena<AExpr>) -> bool {
        if keys.len() > self.0.len() {
            return false;
        }

        keys.iter()
            .zip(self.0.iter())
            .all(|(k, s)| into_column(k.node(), expr_arena).is_some_and(|k| k == &s.column))
    }
}

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
        IR::SimpleProjection { input, columns } => {
            let (input, columns) = (*input, columns.clone());
            match rec!(input) {
                None => None,
                Some(v) => {
                    let num_keys = v.0.iter().filter(|v| columns.contains(&v.column)).count();
                    if num_keys == 0 {
                        None
                    } else if num_keys == v.0.len() {
                        Some(v)
                    } else {
                        Some(IRSorted(
                            v.0.iter()
                                .filter(|v| columns.contains(&v.column))
                                .cloned()
                                .collect(),
                        ))
                    }
                },
            }
        },
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

            Some(IRSorted(s.into()))
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
                HintIR::Sorted(v) => Some(IRSorted(v.clone())),
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
    };

    sortedness.insert(root, sorted.clone());
    sorted
}
