use std::sync::Arc;

use polars_core::chunked_array::cast::CastOptions;
use polars_core::prelude::{FillNullStrategy, PlHashMap, PlHashSet};
use polars_core::schema::Schema;
use polars_core::series::IsSorted;
use polars_utils::arena::{Arena, Node};
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::unique_id::UniqueId;

#[cfg(all(feature = "strings", feature = "concat_str"))]
use crate::plans::IRStringFunction;
use crate::plans::{
    AExpr, ExprIR, FunctionIR, HintIR, IR, IRFunctionExpr, Sorted, ToFieldContext,
    constant_evaluate, into_column,
};

#[derive(Debug, Clone)]
pub struct IRSorted(pub Arc<[Sorted]>);

/// Are the keys together sorted in any way?
///
/// Returns the way in which the keys are sorted, if they are sorted.
pub fn are_keys_sorted_any(
    ir_sorted: Option<&IRSorted>,
    keys: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    input_schema: &Schema,
) -> Option<Vec<AExprSorted>> {
    let mut sortedness = Vec::with_capacity(keys.len());
    for (idx, key) in keys.iter().enumerate() {
        let s = aexpr_sortedness(
            expr_arena.get(key.node()),
            expr_arena,
            input_schema,
            Some(&ir_sorted?.0[idx..]),
        )?;
        sortedness.push(s);
    }
    Some(sortedness)
}

pub fn is_sorted(root: Node, ir_arena: &Arena<IR>, expr_arena: &Arena<AExpr>) -> Option<IRSorted> {
    let mut sortedness = PlHashMap::default();
    let mut cache_proxy = PlHashMap::default();
    let mut amort_passed_columns = PlHashSet::default();

    is_sorted_rec(
        root,
        ir_arena,
        expr_arena,
        &mut sortedness,
        &mut cache_proxy,
        &mut amort_passed_columns,
    )
}

#[recursive::recursive]
fn is_sorted_rec(
    root: Node,
    ir_arena: &Arena<IR>,
    expr_arena: &Arena<AExpr>,
    sortedness: &mut PlHashMap<Node, Option<IRSorted>>,
    cache_proxy: &mut PlHashMap<UniqueId, Option<IRSorted>>,
    amort_passed_columns: &mut PlHashSet<PlSmallStr>,
) -> Option<IRSorted> {
    if let Some(s) = sortedness.get(&root) {
        return s.clone();
    }

    macro_rules! rec {
        ($node:expr) => {{
            is_sorted_rec(
                $node,
                ir_arena,
                expr_arena,
                sortedness,
                cache_proxy,
                amort_passed_columns,
            )
        }};
    }

    sortedness.insert(root, None);

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
        IR::DataFrameScan { df, .. } => {
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
            (!sorted_cols.is_empty()).then(|| IRSorted(sorted_cols.into()))
        },
        IR::SimpleProjection { input, columns } => {
            let (input, columns) = (*input, columns.clone());
            match rec!(input) {
                None => None,
                Some(v) => {
                    let first_unsorted_key = v.0.iter().position(|v| !columns.contains(&v.column));
                    match first_unsorted_key {
                        None => Some(v),
                        Some(0) => None,
                        Some(i) => Some(IRSorted(v.0.iter().take(i).cloned().collect())),
                    }
                },
            }
        },
        IR::Select { input, expr, .. } => {
            let input = *input;
            let input_sorted = rec!(input);

            if let Some(input_sorted) = &input_sorted {
                // We can keep a sorted column if it was kept and not changed.

                amort_passed_columns.clear();
                amort_passed_columns.extend(expr.iter().filter_map(|e| {
                    let column = into_column(e.node(), expr_arena)?;
                    (column == e.output_name()).then(|| column.clone())
                }));

                let first_unkept_key = input_sorted
                    .0
                    .iter()
                    .position(|v| !amort_passed_columns.contains(&v.column));
                match first_unkept_key {
                    None => Some(input_sorted.clone()),
                    Some(0) => {
                        let input_schema = ir_arena.get(input).schema(ir_arena);
                        first_expr_ir_sorted(
                            expr,
                            expr_arena,
                            input_schema.as_ref(),
                            Some(&input_sorted.0),
                        )
                        .map(|s| IRSorted([s].into()))
                    },
                    Some(i) => Some(IRSorted(input_sorted.0.iter().take(i).cloned().collect())),
                }
            } else {
                let input_schema = ir_arena.get(input).schema(ir_arena);
                first_expr_ir_sorted(expr, expr_arena, input_schema.as_ref(), None)
                    .map(|s| IRSorted([s].into()))
            }
        },
        IR::HStack { input, exprs, .. } => {
            let input = *input;
            let input_sorted = rec!(input);

            if let Some(input_sorted) = &input_sorted {
                // We can keep a sorted column if it was not overwritten.

                amort_passed_columns.clear();
                amort_passed_columns.extend(exprs.iter().filter_map(|e| {
                    match into_column(e.node(), expr_arena) {
                        None => Some(e.output_name().clone()),
                        Some(c) if c == e.output_name() => None,
                        Some(_) => Some(e.output_name().clone()),
                    }
                }));

                let first_overwritten_key = input_sorted
                    .0
                    .iter()
                    .position(|v| amort_passed_columns.contains(&v.column));
                match first_overwritten_key {
                    None => Some(input_sorted.clone()),
                    Some(0) => {
                        let input_schema = ir_arena.get(input).schema(ir_arena);
                        first_expr_ir_sorted(
                            exprs,
                            expr_arena,
                            input_schema.as_ref(),
                            Some(&input_sorted.0),
                        )
                        .map(|s| IRSorted([s].into()))
                    },
                    Some(i) => Some(IRSorted(input_sorted.0.iter().take(i).cloned().collect())),
                }
            } else {
                let input_schema = ir_arena.get(input).schema(ir_arena);
                first_expr_ir_sorted(exprs, expr_arena, input_schema.as_ref(), None)
                    .map(|s| IRSorted([s].into()))
            }
        },
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
        IR::GroupBy {
            input,
            keys,
            options,
            maintain_order: true,
            ..
        } if !options.is_rolling() && !options.is_dynamic() => {
            let input = *input;
            let input_sorted = rec!(input)?;

            amort_passed_columns.clear();
            amort_passed_columns.extend(keys.iter().filter_map(|e| {
                let column = into_column(e.node(), expr_arena)?;
                (column == e.output_name()).then(|| column.clone())
            }));

            // We can keep a sorted key column if it was kept and not changed.

            let first_unkept_key = input_sorted
                .0
                .iter()
                .position(|v| !amort_passed_columns.contains(&v.column));
            match first_unkept_key {
                None => Some(input_sorted.clone()),
                Some(0) => {
                    let input_schema = ir_arena.get(input).schema(ir_arena);
                    first_expr_ir_sorted(keys, expr_arena, input_schema.as_ref(), None)
                        .map(|s| IRSorted([s].into()))
                },
                Some(i) => Some(IRSorted(input_sorted.0.iter().take(i).cloned().collect())),
            }
        },
        #[cfg(feature = "dynamic_group_by")]
        IR::GroupBy { options, .. } if options.is_rolling() => {
            let Some(rolling_options) = &options.rolling else {
                unreachable!()
            };
            Some(IRSorted(
                [Sorted {
                    column: rolling_options.index_column.clone(),
                    descending: None,
                    nulls_last: None,
                }]
                .into(),
            ))
        },
        #[cfg(feature = "dynamic_group_by")]
        IR::GroupBy { keys, options, .. } if options.is_dynamic() => {
            let Some(dynamic_options) = &options.dynamic else {
                unreachable!()
            };
            keys.is_empty().then(|| {
                IRSorted(
                    [Sorted {
                        column: dynamic_options.index_column.clone(),
                        descending: None,
                        nulls_last: None,
                    }]
                    .into(),
                )
            })
        },

        IR::GroupBy { .. } => None,
        IR::Join { .. } => None,
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
        IR::MergeSorted { key, .. } => Some(IRSorted(
            [Sorted {
                column: key.clone(),
                descending: None,
                nulls_last: None,
            }]
            .into(),
        )),
        IR::Distinct { input, options } => {
            if !options.maintain_order {
                return None;
            }

            let input = *input;
            rec!(input)
        },
        IR::Invalid => unreachable!(),
    };

    sortedness.insert(root, sorted.clone());
    sorted
}

#[derive(Debug, PartialEq)]
pub struct AExprSorted {
    pub descending: Option<bool>,
    pub nulls_last: Option<bool>,
}

fn first_expr_ir_sorted(
    exprs: &[ExprIR],
    arena: &Arena<AExpr>,
    schema: &Schema,
    input_sorted: Option<&[Sorted]>,
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
    input_sorted: Option<&[Sorted]>,
) -> Option<AExprSorted> {
    match aexpr {
        AExpr::Element => None,
        AExpr::Explode { .. } => None,
        AExpr::Column(col) => {
            let fst = input_sorted?.first()?;
            (fst.column == col).then_some(AExprSorted {
                descending: fst.descending,
                nulls_last: fst.nulls_last,
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
    input_sorted: Option<&[Sorted]>,
) -> Option<AExprSorted> {
    let nth_input =
        |n: usize| aexpr_sortedness(arena.get(inputs[n].node()), arena, schema, input_sorted);

    match function {
        #[cfg(feature = "rle")]
        IRFunctionExpr::RLEID => Some(AExprSorted {
            descending: Some(false),
            nulls_last: Some(false),
        }),
        IRFunctionExpr::SetSortedFlag(is_sorted) => match is_sorted {
            IsSorted::Ascending => Some(AExprSorted {
                descending: Some(false),
                nulls_last: None,
            }),
            IsSorted::Descending => Some(AExprSorted {
                descending: Some(true),
                nulls_last: None,
            }),
            IsSorted::Not => None,
        },

        IRFunctionExpr::Unique(true)
        | IRFunctionExpr::DropNulls
        | IRFunctionExpr::DropNans
        | IRFunctionExpr::FillNullWithStrategy(
            FillNullStrategy::Forward(None) | FillNullStrategy::Backward(None),
        ) => nth_input(0),
        #[cfg(feature = "mode")]
        IRFunctionExpr::Mode {
            maintain_order: true,
        } => nth_input(0),

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
            let mut sortedness = nth_input(0)?;
            if let Some(d) = &mut sortedness.descending {
                *d = !*d;
            }
            if let Some(n) = &mut sortedness.nulls_last {
                *n ^= !*n;
            }
            Some(sortedness)
        },

        #[cfg(all(feature = "strings", feature = "concat_str"))]
        IRFunctionExpr::StringExpr(IRStringFunction::ConcatHorizontal { ignore_nulls, .. }) => {
            // In cases like pl.concat_str(pl.lit("prefix"), pl.col("a"), pl.lit("suffix")),
            // we always want to return the sortedness of pl.col("a").
            let scalar_constants = inputs
                .iter()
                .map(|e| {
                    constant_evaluate(e.node(), arena, schema, 0)
                        .flatten()
                        .filter(|c| c.is_scalar())
                })
                .collect::<Vec<_>>();
            let scalar_constant_count = scalar_constants.iter().filter(|o| o.is_some()).count();
            for (idx, sc) in scalar_constants.iter().enumerate() {
                let Some(sortedness) = nth_input(idx) else {
                    continue;
                };
                if scalar_constant_count == inputs.len() - 1 && sc.is_none() {
                    return Some(sortedness);
                }
            }

            let sortedness = nth_input(0)?;
            if *ignore_nulls && sortedness.nulls_last? != sortedness.descending? {
                return None;
            }
            if (1..inputs.len()).all(|n| nth_input(n).as_ref() == Some(&sortedness)) {
                Some(sortedness)
            } else {
                None
            }
        },

        _ => None,
    }
}
