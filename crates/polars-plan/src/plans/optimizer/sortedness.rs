use std::sync::Arc;

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::{FillNullStrategy, PlHashMap};
use polars_core::schema::Schema;
use polars_core::series::IsSorted;
use polars_utils::aliases::ScratchMap;
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[cfg(all(feature = "strings", feature = "concat_str"))]
use crate::plans::IRStringFunction;
use crate::plans::ir_traversal::edge_provider::EdgesProvider;
use crate::plans::ir_traversal::ir_node_key::IRNodeKey;
use crate::plans::ir_traversal::pullup_traversal::ir_pullup_traversal_rec;
use crate::plans::partitioning::frame::FramePartitioning;
use crate::plans::{
    AExpr, ExprIR, FunctionIR, HintIR, IR, IRFunctionExpr, Sorted, ToFieldContext,
    constant_evaluate, into_column,
};

/// Container for sortedness state at each stage in an IR plan.
#[derive(Debug)]
pub struct IRPlanSorted(PlHashMap<IRNodeKey, FramePartitioning>);

impl IRPlanSorted {
    pub fn resolve(root: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Self {
        let mut cache = PlHashMap::default();
        let mut column_names_map = ScratchMap::default();

        ir_pullup_traversal_rec(
            root,
            &mut |current_ir_node, ir_arena, expr_arena, edge_provider| {
                pullup_sorted_single(
                    current_ir_node,
                    ir_arena,
                    expr_arena,
                    edge_provider,
                    &mut column_names_map,
                );
                Ok(())
            },
            ir_arena,
            expr_arena,
            &mut cache,
            false,
        )
        .unwrap();

        Self(cache)
    }

    pub fn get(&self, node_key: &IRNodeKey) -> &FramePartitioning {
        self.0
            .get(node_key)
            .unwrap_or(FramePartitioning::empty_static())
    }

    pub fn is_expr_sorted(
        &self,
        at: &IRNodeKey,
        expr: &ExprIR,
        expr_arena: &Arena<AExpr>,
        input_schema: &Schema,
    ) -> Option<AExprSorted> {
        expr_is_sorted(self.get(at), expr, expr_arena, input_schema)
    }

    pub fn are_keys_sorted_any(
        &self,
        at: &IRNodeKey,
        keys: &[ExprIR],
        expr_arena: &Arena<AExpr>,
        input_schema: &Schema,
    ) -> Option<Vec<AExprSorted>> {
        are_keys_sorted_any(self.get(at), keys, expr_arena, input_schema)
    }
}

#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
#[derive(Debug, Default, PartialEq, Clone, Copy, Hash)]
pub struct AExprSorted {
    /// If `Some(true)`, the expression is sorted in descending order.
    /// If `Some(false)`, the expression is sorted in ascending order.
    /// If `None`, the sorting order is unknown.
    pub descending: Option<bool>,
    /// If `Some(true)`, null values (if any) are at the end of the expression result.
    /// If `Some(false)`, null values (if any) are at the beginning of the expression result.
    /// If `None`, the null value position is unknown or there are no nulls.
    pub nulls_last: Option<bool>,
}

impl AExprSorted {
    pub fn reverse(self) -> Self {
        Self {
            descending: self.descending.map(|x| !x),
            nulls_last: self.nulls_last.map(|x| !x),
        }
    }

    pub fn with_desc(mut self, desc: Option<bool>) -> Self {
        self.descending = desc;
        self
    }

    pub fn with_nulls_last(mut self, nulls_last: Option<bool>) -> Self {
        self.nulls_last = nulls_last;
        self
    }

    pub fn is_asc(&self) -> bool {
        matches!(self.descending, Some(false))
    }

    pub fn is_desc(&self) -> bool {
        matches!(self.descending, Some(true))
    }

    pub fn is_nulls_first(&self) -> bool {
        matches!(self.nulls_last, Some(false))
    }

    pub fn is_nulls_last(&self) -> bool {
        matches!(self.nulls_last, Some(true))
    }
}

impl From<AExprSorted> for IsSorted {
    fn from(val: AExprSorted) -> Self {
        match val.descending {
            Some(false) => IsSorted::Ascending,
            Some(true) => IsSorted::Descending,
            None => IsSorted::Not,
        }
    }
}

#[derive(Debug, Clone)]
pub struct IRSorted(pub Arc<[Sorted]>);

/// Are the keys together sorted in any way?
///
/// Returns the way in which the keys are sorted, if they are sorted.
pub fn are_keys_sorted_any(
    ir_sorted: &FramePartitioning,
    keys: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
) -> Option<Vec<AExprSorted>> {
    let mut sortedness = Vec::with_capacity(keys.len());
    for eir in keys.iter() {
        let s = aexpr_sortedness(
            expr_arena.get(eir.node()),
            expr_arena,
            input_schema,
            ir_sorted,
        )?;
        sortedness.push(s);
    }
    Some(sortedness)
}

/// Is this expression sorted given the sortedness of the input dataframe?
///
/// Returns the way in which the expression is sorted, if it is sorted.
pub fn expr_is_sorted(
    ir_sorted: &FramePartitioning,
    expr: &ExprIR,
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
) -> Option<AExprSorted> {
    aexpr_sortedness(
        expr_arena.get(expr.node()),
        expr_arena,
        input_schema,
        ir_sorted,
    )
}

pub fn is_sorted(root: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> FramePartitioning {
    let mut cache = PlHashMap::default();
    let mut column_names_map = ScratchMap::default();

    ir_pullup_traversal_rec(
        root,
        &mut |current_ir_node, ir_arena, expr_arena, edge_provider| {
            pullup_sorted_single(
                current_ir_node,
                ir_arena,
                expr_arena,
                edge_provider,
                &mut column_names_map,
            );
            Ok(())
        },
        ir_arena,
        expr_arena,
        &mut cache,
        false,
    )
    .unwrap()
}

pub fn pullup_sorted_single<EP: EdgesProvider<FramePartitioning>>(
    current_ir_node: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    edges_provider: &mut EP,
    column_names_map: &mut ScratchMap<PlSmallStr, Option<PlSmallStr>>,
) {
    macro_rules! unpack_edges {
        ($total:literal) => {{
            let (l, r) = edges_provider.unpack_edges_mut::<_, _, $total>().unwrap();
            let l: [FramePartitioning; _] = l.map(|x| x.clone());
            (l, r)
        }};
    }

    match ir_arena.get(current_ir_node) {
        #[cfg(feature = "python")]
        IR::PythonScan { .. } => {},

        IR::Filter { .. } | IR::Slice { .. } => {
            let ([partitioning], [out_edge]) = unpack_edges!(2);
            *out_edge = partitioning.clone();
        },

        IR::Scan { .. } => {},

        IR::DataFrameScan { df, .. } => {
            let ([], [out_edge]) = unpack_edges!(1);

            let sorted_cols = df
                .columns()
                .iter()
                .filter_map(|c| match c.is_sorted_flag() {
                    IsSorted::Not => None,
                    IsSorted::Ascending => Some(Sorted {
                        column: c.name().clone(),
                        descending: Some(false),
                        nulls_last: Some(c.get(0).is_ok_and(|v| !v.is_null())),
                    }),
                    IsSorted::Descending => Some(Sorted {
                        column: c.name().clone(),
                        descending: Some(true),
                        nulls_last: Some(c.get(0).is_ok_and(|v| !v.is_null())),
                    }),
                })
                .collect_vec();

            *out_edge = FramePartitioning::from_iter(sorted_cols);
        },

        IR::SimpleProjection { columns, .. } => {
            let ([mut partitioning], [out_edge]) = unpack_edges!(2);

            if let Some(i) = partitioning.keys().position(|name| !columns.contains(name)) {
                partitioning.truncate(i);
            }

            *out_edge = partitioning;
        },

        IR::Select {
            input, expr: exprs, ..
        }
        | IR::HStack { input, exprs, .. } => {
            let column_names_map = column_names_map.get();

            if matches!(ir_arena.get(current_ir_node), IR::HStack { .. }) {
                for name in ir_arena.get(*input).schema(ir_arena).iter_names() {
                    column_names_map.insert(name.clone(), None);
                }
            }

            let ([input_partitioning], [out_edge]) = unpack_edges!(2);
            let mut partitioning = input_partitioning.clone();

            for eir in exprs.iter() {
                match expr_arena.get(eir.node()) {
                    AExpr::Column(name) => {
                        column_names_map.insert(
                            name.clone(),
                            (name != eir.output_name()).then(|| eir.output_name().clone()),
                        );
                    },
                    _ => {
                        column_names_map.remove(eir.output_name());
                    },
                }
            }

            let mut i: usize = 0;

            while i < partitioning.len() {
                let name = &partitioning[i].column;

                match column_names_map.get(name) {
                    None => break,
                    Some(None) => {},
                    Some(Some(new_name)) => {
                        partitioning.make_mut()[i].column = new_name.clone();
                    },
                }

                i += 1;
            }

            partitioning.truncate(i);

            if partitioning.is_empty()
                && let Some(sorted) = first_expr_ir_sorted(
                    exprs,
                    expr_arena,
                    &ir_arena.get(*input).schema(ir_arena),
                    &input_partitioning,
                )
            {
                partitioning = FramePartitioning::from_iter(vec![sorted])
            }

            *out_edge = partitioning;
        },

        IR::Sort {
            input,
            by_column,
            slice,
            sort_options,
        } => {
            let mut s = by_column
                .iter()
                .map_while(|e| {
                    into_column(e.node(), expr_arena).map(|c| Sorted {
                        column: c.clone(),
                        descending: Some(false),
                        nulls_last: Some(false),
                    })
                })
                .collect::<Vec<_>>();
            if sort_options.descending.len() != 1 {
                s.iter_mut()
                    .zip(sort_options.descending.iter())
                    .for_each(|(s, &d)| s.descending = Some(d));
            } else if sort_options.descending[0] {
                s.iter_mut().for_each(|s| s.descending = Some(true));
            }
            if sort_options.nulls_last.len() != 1 {
                s.iter_mut()
                    .zip(sort_options.nulls_last.iter())
                    .for_each(|(s, &d)| s.nulls_last = Some(d));
            } else if sort_options.nulls_last[0] {
                s.iter_mut().for_each(|s| s.nulls_last = Some(true));
            }

            *edges_provider.get_out_edge_mut(0) = FramePartitioning::from_iter(s);
        },

        IR::Cache { .. } => {
            let partitioning = edges_provider.get_in_edge_mut(0).clone();

            edges_provider
                .map_out_edges_mut(|e| *e = partitioning.clone())
                .for_each(|_| ());
        },

        IR::GroupBy {
            input: _,
            keys,
            aggs: _,
            schema: _,
            maintain_order,
            options,
            apply: _,
        } => {
            let ([input_partitioning], [out_edge]) = unpack_edges!(2);

            #[cfg(feature = "dynamic_group_by")]
            if let Some(rolling_options) = &options.rolling {
                *out_edge = FramePartitioning::from_iter(vec![Sorted {
                    column: rolling_options.index_column.clone(),
                    descending: None,
                    nulls_last: None,
                }]);

                return;
            }

            #[cfg(feature = "dynamic_group_by")]
            if let Some(dynamic_options) = &options.dynamic {
                *out_edge = FramePartitioning::from_iter(Some(Sorted {
                    column: dynamic_options.index_column.clone(),
                    descending: None,
                    nulls_last: None,
                }));

                return;
            };

            let mut in_common_prefix = *maintain_order;

            *out_edge = FramePartitioning::from_iter(keys.iter().map(|eir| {
                let mut out = Sorted {
                    column: eir.output_name().clone(),
                    descending: None,
                    nulls_last: None,
                };

                if in_common_prefix
                    && let AExpr::Column(input_name) = expr_arena.get(eir.node())
                    && let Some(sorted) = input_partitioning.get(input_name)
                {
                    out.descending = sorted.descending;
                    out.nulls_last = sorted.nulls_last;
                } else {
                    in_common_prefix = false;
                }

                out
            }));
        },

        // TODO: Order-maintaining joins
        IR::Join { .. } => {},

        IR::MapFunction { input: _, function } => match function {
            FunctionIR::Hint(hint) => match hint {
                HintIR::Sorted(v) => {
                    *edges_provider.get_out_edge_mut(0) =
                        FramePartitioning::from_iter(v.iter().cloned());
                },
            },
            _ => {},
        },

        IR::Union { .. } | IR::HConcat { .. } | IR::ExtContext { .. } => {},

        IR::Sink { .. } | IR::SinkMultiple { .. } => {
            let partitioning = edges_provider.get_in_edge_mut(0).clone();

            edges_provider
                .map_out_edges_mut(|e| *e = partitioning.clone())
                .for_each(|_| ());
        },

        #[cfg(feature = "merge_sorted")]
        IR::MergeSorted { key, .. } => {
            *edges_provider.get_out_edge_mut(0) = FramePartitioning::from_iter(Some(Sorted {
                column: key.clone(),
                descending: None,
                nulls_last: None,
            }))
        },

        IR::Distinct { input: _, options } => {
            let ([mut partitioning], [out_edge]) = unpack_edges!(2);

            if !options.maintain_order {
                for v in partitioning.make_mut().values_mut() {
                    v.descending = None;
                    v.nulls_last = None;
                }
            }

            if let Some(subset) = options.subset.as_deref() {
                partitioning.make_mut().extend(subset.iter().map(|name| {
                    (
                        name.clone(),
                        Sorted {
                            column: name.clone(),
                            descending: None,
                            nulls_last: None,
                        },
                    )
                }))
            } else {
                partitioning.make_mut().extend(
                    ir_arena
                        .get(current_ir_node)
                        .schema(ir_arena)
                        .iter_names()
                        .map(|name| {
                            (
                                name.clone(),
                                Sorted {
                                    column: name.clone(),
                                    descending: None,
                                    nulls_last: None,
                                },
                            )
                        }),
                )
            }

            *out_edge = partitioning;
        },

        IR::Invalid => unreachable!(),
    }
}

fn first_expr_ir_sorted(
    exprs: &[ExprIR],
    arena: &Arena<AExpr>,
    schema: &Schema,
    input_sorted: &FramePartitioning,
) -> Option<Sorted> {
    exprs.iter().find_map(|e| {
        aexpr_sortedness(arena.get(e.node()), arena, schema, input_sorted).map(|s| Sorted {
            column: e.output_name().clone(),
            descending: s.descending,
            nulls_last: s.nulls_last,
        })
    })
}

#[recursive::recursive]
pub fn aexpr_sortedness(
    aexpr: &AExpr,
    arena: &Arena<AExpr>,
    schema: &Schema,
    input_sorted: &FramePartitioning,
) -> Option<AExprSorted> {
    match aexpr {
        AExpr::Element => None,
        AExpr::Explode { .. } => None,
        AExpr::Column(col) => {
            let sorted = input_sorted.get(col)?;
            Some(AExprSorted {
                descending: sorted.descending,
                nulls_last: sorted.nulls_last,
            })
        },
        #[cfg(feature = "dtype-struct")]
        AExpr::StructField(_) => None,
        AExpr::Literal(lv) if lv.is_scalar() => Some(AExprSorted {
            descending: Some(false),
            nulls_last: Some(false),
        }),
        AExpr::Literal(_) => None,

        AExpr::Len => Some(AExprSorted {
            descending: Some(false),
            nulls_last: Some(false),
        }),
        AExpr::Cast {
            expr,
            dtype,
            options: CastOptions::Strict,
        } if dtype.is_integer() => {
            let expr = arena.get(*expr);
            let expr_sortedness = aexpr_sortedness(expr, arena, schema, input_sorted)?;
            let input_dtype = expr.to_dtype(&ToFieldContext::new(arena, schema)).ok()?;
            if !input_dtype.is_integer() {
                return None;
            }
            Some(expr_sortedness)
        },
        AExpr::Cast { .. } => None, // @TODO: More casts are allowed
        AExpr::Sort { expr: _, options } => Some(AExprSorted {
            descending: Some(options.descending),
            nulls_last: Some(options.nulls_last),
        }),
        AExpr::Function {
            input,
            function,
            options: _,
        } => function_expr_sortedness(function, input, arena, schema, input_sorted),
        AExpr::Filter { input, by: _ }
        | AExpr::Slice {
            input,
            offset: _,
            length: _,
        } => aexpr_sortedness(arena.get(*input), arena, schema, input_sorted),

        AExpr::BinaryExpr { .. }
        | AExpr::Gather { .. }
        | AExpr::SortBy { .. }
        | AExpr::Agg(_)
        | AExpr::Ternary { .. }
        | AExpr::AnonymousAgg { .. }
        | AExpr::AnonymousFunction { .. }
        | AExpr::Eval { .. }
        | AExpr::Over { .. } => None,

        #[cfg(feature = "dtype-struct")]
        AExpr::StructEval { .. } => None,

        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling { .. } => None,
    }
}

pub fn function_expr_sortedness(
    function: &IRFunctionExpr,
    inputs: &[ExprIR],
    arena: &Arena<AExpr>,
    schema: &Schema,
    input_sorted: &FramePartitioning,
) -> Option<AExprSorted> {
    macro_rules! rec_ae {
        ($node:expr) => {{ aexpr_sortedness(arena.get($node), arena, schema, input_sorted) }};
    }

    match function {
        #[cfg(feature = "rle")]
        IRFunctionExpr::RLEID => Some(AExprSorted {
            descending: Some(false),
            nulls_last: Some(false),
        }),
        IRFunctionExpr::SetSortedFlag(sortedness) => match sortedness.descending {
            Some(false) => Some(AExprSorted {
                descending: Some(false),
                nulls_last: None,
            }),
            Some(true) => Some(AExprSorted {
                descending: Some(true),
                nulls_last: None,
            }),
            None => None,
        },

        IRFunctionExpr::Unique(true)
        | IRFunctionExpr::DropNulls
        | IRFunctionExpr::DropNans
        | IRFunctionExpr::FillNullWithStrategy(
            FillNullStrategy::Forward(None) | FillNullStrategy::Backward(None),
        ) => {
            let [e] = inputs else {
                return None;
            };

            rec_ae!(e.node())
        },
        #[cfg(feature = "mode")]
        IRFunctionExpr::Mode {
            maintain_order: true,
        } => {
            let [e] = inputs else {
                return None;
            };

            rec_ae!(e.node())
        },

        #[cfg(feature = "range")]
        IRFunctionExpr::Range(range) => {
            use crate::plans::IRRangeFunction as R;
            match range {
                // `int_range(0, ..., step=1, dtype=UNSIGNED)`
                R::IntRange { step: 1, dtype }
                    if dtype.is_unsigned_integer()
                        && constant_evaluate(inputs[0].node(), arena, schema, 0)??
                            .extract_i64()
                            .is_ok_and(|v| v == 0) =>
                {
                    Some(AExprSorted {
                        descending: Some(false),
                        nulls_last: Some(false),
                    })
                },

                _ => None,
            }
        },

        IRFunctionExpr::Reverse => {
            let [e] = inputs else {
                return None;
            };

            let mut sortedness = rec_ae!(e.node())?;

            if let Some(d) = &mut sortedness.descending {
                *d = !*d;
            }
            if let Some(n) = &mut sortedness.nulls_last {
                *n ^= !*n;
            }
            Some(sortedness)
        },

        #[cfg(all(feature = "strings", feature = "concat_str"))]
        IRFunctionExpr::StringExpr(IRStringFunction::ConcatHorizontal {
            ignore_nulls: false,
            delimiter: _,
        }) => {
            let [e] = inputs else {
                return None;
            };

            rec_ae!(e.node())
        },

        _ => None,
    }
}
