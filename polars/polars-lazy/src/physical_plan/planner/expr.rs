use polars_core::frame::groupby::GroupByMethod;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_core::utils::parallel_op_series;

use super::super::expressions as phys_expr;
use crate::prelude::*;

pub(crate) fn create_physical_expressions(
    exprs: &[Node],
    context: Context,
    expr_arena: &Arena<AExpr>,
    schema: Option<&SchemaRef>,
) -> PolarsResult<Vec<Arc<dyn PhysicalExpr>>> {
    exprs
        .iter()
        .map(|e| create_physical_expr(*e, context, expr_arena, schema))
        .collect()
}

pub(crate) fn create_physical_expr(
    expression: Node,
    ctxt: Context,
    expr_arena: &Arena<AExpr>,
    schema: Option<&SchemaRef>,
) -> PolarsResult<Arc<dyn PhysicalExpr>> {
    use AExpr::*;

    match expr_arena.get(expression).clone() {
        Count => Ok(Arc::new(phys_expr::CountExpr::new())),
        Window {
            mut function,
            partition_by,
            order_by: _,
            options,
        } => {
            // TODO! Order by
            let group_by =
                create_physical_expressions(&partition_by, Context::Default, expr_arena, schema)?;
            let phys_function =
                create_physical_expr(function, Context::Aggregation, expr_arena, schema)?;
            let mut out_name = None;
            let mut apply_columns = aexpr_to_leaf_names(function, expr_arena);
            // sort and then dedup removes consecutive duplicates == all duplicates
            apply_columns.sort();
            apply_columns.dedup();

            if apply_columns.is_empty() {
                if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Literal(_))) {
                    apply_columns.push(Arc::from("literal"))
                } else if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Count)) {
                    apply_columns.push(Arc::from("count"))
                } else {
                    let e = node_to_expr(function, expr_arena);
                    return Err(PolarsError::ComputeError(
                        format!(
                            "Cannot apply a window function, did not find a root column. \
                        This is likely due to a syntax error in this expression: {e:?}",
                        )
                        .into(),
                    ));
                }
            }

            if let Alias(expr, name) = expr_arena.get(function) {
                function = *expr;
                out_name = Some(name.clone());
            };
            let function = node_to_expr(function, expr_arena);

            Ok(Arc::new(WindowExpr {
                group_by,
                apply_columns,
                out_name,
                function,
                phys_function,
                options,
                expr: node_to_expr(expression, expr_arena),
            }))
        }
        Literal(value) => Ok(Arc::new(LiteralExpr::new(
            value,
            node_to_expr(expression, expr_arena),
        ))),
        BinaryExpr { left, op, right } => {
            let lhs = create_physical_expr(left, ctxt, expr_arena, schema)?;
            let rhs = create_physical_expr(right, ctxt, expr_arena, schema)?;
            Ok(Arc::new(phys_expr::BinaryExpr::new(
                lhs,
                op,
                rhs,
                node_to_expr(expression, expr_arena),
            )))
        }
        Column(column) => Ok(Arc::new(ColumnExpr::new(
            column,
            node_to_expr(expression, expr_arena),
            schema.cloned(),
        ))),
        Sort { expr, options } => {
            let phys_expr = create_physical_expr(expr, ctxt, expr_arena, schema)?;
            Ok(Arc::new(SortExpr::new(
                phys_expr,
                options,
                node_to_expr(expression, expr_arena),
            )))
        }
        Take { expr, idx } => {
            let phys_expr = create_physical_expr(expr, ctxt, expr_arena, schema)?;
            let phys_idx = create_physical_expr(idx, ctxt, expr_arena, schema)?;
            Ok(Arc::new(TakeExpr {
                phys_expr,
                idx: phys_idx,
                expr: node_to_expr(expression, expr_arena),
            }))
        }
        SortBy { expr, by, reverse } => {
            let phys_expr = create_physical_expr(expr, ctxt, expr_arena, schema)?;
            let phys_by = create_physical_expressions(&by, ctxt, expr_arena, schema)?;
            Ok(Arc::new(SortByExpr::new(
                phys_expr,
                phys_by,
                reverse,
                node_to_expr(expression, expr_arena),
            )))
        }
        Filter { input, by } => {
            let phys_input = create_physical_expr(input, ctxt, expr_arena, schema)?;
            let phys_by = create_physical_expr(by, ctxt, expr_arena, schema)?;
            Ok(Arc::new(FilterExpr::new(
                phys_input,
                phys_by,
                node_to_expr(expression, expr_arena),
            )))
        }
        Alias(expr, name) => {
            let phys_expr = create_physical_expr(expr, ctxt, expr_arena, schema)?;
            Ok(Arc::new(AliasExpr::new(
                phys_expr,
                name,
                node_to_expr(expression, expr_arena),
            )))
        }
        Agg(agg) => {
            match agg {
                AAggExpr::Min {
                    input: expr,
                    propagate_nans,
                } => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            if propagate_nans {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::NanMin)))
                            } else {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Min)))
                            }
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);

                                if propagate_nans && s.dtype().is_float() {
                                    #[cfg(feature = "propagate_nans")]
                                    {
                                        return parallel_op_series(
                                            |s| {
                                                Ok(polars_ops::prelude::nan_propagating_aggregate::nan_min_s(&s, s.name()))
                                            },
                                            s,
                                            None,
                                        );
                                    }
                                    #[cfg(not(feature = "propagate_nans"))]
                                    {
                                        panic!("activate 'propagate_nans' feature")
                                    }
                                }

                                match s.is_sorted_flag() {
                                    IsSorted::Ascending | IsSorted::Descending => {
                                        Ok(Some(s.min_as_series()))
                                    }
                                    IsSorted::Not => {
                                        parallel_op_series(|s| Ok(s.min_as_series()), s, None)
                                    }
                                }
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::Max {
                    input: expr,
                    propagate_nans,
                } => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            if propagate_nans {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::NanMax)))
                            } else {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Max)))
                            }
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);

                                if propagate_nans && s.dtype().is_float() {
                                    #[cfg(feature = "propagate_nans")]
                                    {
                                        return parallel_op_series(
                                            |s| {
                                                Ok(polars_ops::prelude::nan_propagating_aggregate::nan_max_s(&s, s.name()))
                                            },
                                            s,
                                            None,
                                        );
                                    }
                                    #[cfg(not(feature = "propagate_nans"))]
                                    {
                                        panic!("activate 'propagate_nans' feature")
                                    }
                                }

                                match s.is_sorted_flag() {
                                    IsSorted::Ascending | IsSorted::Descending => {
                                        Ok(Some(s.max_as_series()))
                                    }
                                    IsSorted::Not => {
                                        parallel_op_series(|s| Ok(s.max_as_series()), s, None)
                                    }
                                }
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::Sum(expr) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Sum)))
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                parallel_op_series(|s| Ok(s.sum_as_series()), s, None)
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::Std(expr, ddof) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => Ok(Arc::new(AggregationExpr::new(
                            input,
                            GroupByMethod::Std(ddof),
                        ))),
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                Ok(Some(s.std_as_series(ddof)))
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::Var(expr, ddof) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => Ok(Arc::new(AggregationExpr::new(
                            input,
                            GroupByMethod::Var(ddof),
                        ))),
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                Ok(Some(s.var_as_series(ddof)))
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::Mean(expr) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Mean)))
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                Ok(Some(s.mean_as_series()))
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::Median(expr) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Median)))
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                Ok(Some(s.median_as_series()))
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::First(expr) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::First)))
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                Ok(Some(s.head(Some(1))))
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::Last(expr) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Last)))
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                Ok(Some(s.tail(Some(1))))
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::List(expr) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::List)))
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = &s[0];
                                s.to_list().map(|ca| Some(ca.into_series()))
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::NUnique(expr) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => Ok(Arc::new(AggregationExpr::new(
                            input,
                            GroupByMethod::NUnique,
                        ))),
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                s.n_unique().map(|count| {
                                    Some(
                                        UInt32Chunked::from_slice(s.name(), &[count as u32])
                                            .into_series(),
                                    )
                                })
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
                AAggExpr::Quantile {
                    expr,
                    quantile,
                    interpol,
                } => {
                    // todo! add schema to get correct output type
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    let quantile = create_physical_expr(quantile, ctxt, expr_arena, schema)?;
                    Ok(Arc::new(AggQuantileExpr::new(input, quantile, interpol)))
                    //
                    // match ctxt {
                    //     Context::Aggregation => {
                    //
                    //         Ok(Arc::new(AggQuantileExpr::new(input, quantile, interpol)))
                    //     }
                    //     Context::Default => {
                    //         let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                    //             let s = std::mem::take(&mut s[0]);
                    //             s.quantile_as_series(quantile, interpol)
                    //         })
                    //             as Arc<dyn SeriesUdf>);
                    //         Ok(Arc::new(ApplyExpr::new_minimal(
                    //             vec![input],
                    //             function,
                    //             node_to_expr(expression, expr_arena),
                    //             ApplyOptions::ApplyFlat,
                    //         )))
                    //     }
                    // }
                }
                AAggExpr::AggGroups(expr) => {
                    if let Context::Default = ctxt {
                        panic!("agg groups expression only supported in aggregation context")
                    }
                    let phys_expr = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    Ok(Arc::new(AggregationExpr::new(
                        phys_expr,
                        GroupByMethod::Groups,
                    )))
                }
                AAggExpr::Count(expr) => {
                    let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
                    match ctxt {
                        Context::Aggregation => {
                            Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Count)))
                        }
                        Context::Default => {
                            let function = SpecialEq::new(Arc::new(move |s: &mut [Series]| {
                                let s = std::mem::take(&mut s[0]);
                                let count = s.len();
                                Ok(Some(
                                    UInt32Chunked::from_slice(s.name(), &[count as u32])
                                        .into_series(),
                                ))
                            })
                                as Arc<dyn SeriesUdf>);
                            Ok(Arc::new(ApplyExpr::new_minimal(
                                vec![input],
                                function,
                                node_to_expr(expression, expr_arena),
                                ApplyOptions::ApplyFlat,
                            )))
                        }
                    }
                }
            }
        }
        Cast {
            expr,
            data_type,
            strict,
        } => {
            let phys_expr = create_physical_expr(expr, ctxt, expr_arena, schema)?;
            Ok(Arc::new(CastExpr {
                input: phys_expr,
                data_type,
                expr: node_to_expr(expression, expr_arena),
                strict,
            }))
        }
        Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let predicate = create_physical_expr(predicate, ctxt, expr_arena, schema)?;
            let truthy = create_physical_expr(truthy, ctxt, expr_arena, schema)?;
            let falsy = create_physical_expr(falsy, ctxt, expr_arena, schema)?;
            Ok(Arc::new(TernaryExpr::new(
                predicate,
                truthy,
                falsy,
                node_to_expr(expression, expr_arena),
            )))
        }
        AnonymousFunction {
            input,
            function,
            output_type: _,
            options,
        } => {
            let input = create_physical_expressions(&input, ctxt, expr_arena, schema)?;

            Ok(Arc::new(ApplyExpr {
                inputs: input,
                function,
                expr: node_to_expr(expression, expr_arena),
                collect_groups: options.collect_groups,
                auto_explode: options.auto_explode,
                allow_rename: options.allow_rename,
                pass_name_to_apply: options.pass_name_to_apply,
                input_schema: schema.cloned(),
            }))
        }
        Function {
            input,
            function,
            options,
            ..
        } => {
            let input = create_physical_expressions(&input, ctxt, expr_arena, schema)?;

            Ok(Arc::new(ApplyExpr {
                inputs: input,
                function: function.into(),
                expr: node_to_expr(expression, expr_arena),
                collect_groups: options.collect_groups,
                auto_explode: options.auto_explode,
                allow_rename: options.allow_rename,
                pass_name_to_apply: options.pass_name_to_apply,
                input_schema: schema.cloned(),
            }))
        }
        Slice {
            input,
            offset,
            length,
        } => {
            let input = create_physical_expr(input, ctxt, expr_arena, schema)?;
            let offset = create_physical_expr(offset, ctxt, expr_arena, schema)?;
            let length = create_physical_expr(length, ctxt, expr_arena, schema)?;
            Ok(Arc::new(SliceExpr {
                input,
                offset,
                length,
                expr: node_to_expr(expression, expr_arena),
            }))
        }
        Explode(expr) => {
            let input = create_physical_expr(expr, ctxt, expr_arena, schema)?;
            let function =
                SpecialEq::new(Arc::new(move |s: &mut [Series]| s[0].explode().map(Some))
                    as Arc<dyn SeriesUdf>);
            Ok(Arc::new(ApplyExpr::new_minimal(
                vec![input],
                function,
                node_to_expr(expression, expr_arena),
                ApplyOptions::ApplyGroups,
            )))
        }
        Wildcard => panic!("should be no wildcard at this point"),
        Nth(_) => panic!("should be no nth at this point"),
    }
}
