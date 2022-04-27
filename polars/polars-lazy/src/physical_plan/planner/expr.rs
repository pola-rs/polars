use super::super::expressions as phys_expr;
use crate::prelude::*;
use polars_core::frame::groupby::GroupByMethod;
use polars_core::prelude::*;
use polars_core::utils::parallel_op_series;

impl DefaultPlanner {
    pub fn create_physical_expr(
        &self,
        expression: Node,
        ctxt: Context,
        expr_arena: &mut Arena<AExpr>,
    ) -> Result<Arc<dyn PhysicalExpr>> {
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
                    self.create_physical_expressions(&partition_by, Context::Default, expr_arena)?;
                let phys_function =
                    self.create_physical_expr(function, Context::Aggregation, expr_arena)?;
                let mut out_name = None;
                let mut apply_columns = aexpr_to_root_names(function, expr_arena);
                // sort and then dedup removes consecutive duplicates == all duplicates
                apply_columns.sort();
                apply_columns.dedup();

                if apply_columns.is_empty() {
                    if has_aexpr(function, expr_arena, |e| matches!(e, AExpr::Literal(_))) {
                        apply_columns.push(Arc::from("literal"))
                    } else {
                        let e = node_to_expr(function, expr_arena);
                        return Err(PolarsError::ComputeError(
                            format!(
                                "Cannot apply a window function, did not find a root column. \
                            This is likely due to a syntax error in this expression: {:?}",
                                e
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
                let lhs = self.create_physical_expr(left, ctxt, expr_arena)?;
                let rhs = self.create_physical_expr(right, ctxt, expr_arena)?;
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
            ))),
            Sort { expr, options } => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(SortExpr::new(
                    phys_expr,
                    options,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Take { expr, idx } => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let phys_idx = self.create_physical_expr(idx, ctxt, expr_arena)?;
                Ok(Arc::new(TakeExpr {
                    phys_expr,
                    idx: phys_idx,
                    expr: node_to_expr(expression, expr_arena),
                }))
            }
            SortBy { expr, by, reverse } => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let phys_by = self.create_physical_expressions(&by, ctxt, expr_arena)?;
                Ok(Arc::new(SortByExpr::new(
                    phys_expr,
                    phys_by,
                    reverse,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Filter { input, by } => {
                let phys_input = self.create_physical_expr(input, ctxt, expr_arena)?;
                let phys_by = self.create_physical_expr(by, ctxt, expr_arena)?;
                Ok(Arc::new(FilterExpr::new(
                    phys_input,
                    phys_by,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Not(expr) => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(NotExpr::new(
                    phys_expr,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Alias(expr, name) => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(AliasExpr::new(
                    phys_expr,
                    name,
                    node_to_expr(expression, expr_arena),
                )))
            }
            IsNull(expr) => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(IsNullExpr::new(
                    phys_expr,
                    node_to_expr(expression, expr_arena),
                )))
            }
            IsNotNull(expr) => {
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                Ok(Arc::new(IsNotNullExpr::new(
                    phys_expr,
                    node_to_expr(expression, expr_arena),
                )))
            }
            Agg(agg) => {
                match agg {
                    AAggExpr::Min(expr) => {
                        // todo! Output type is dependent on schema.
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Min)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    parallel_op_series(|s| Ok(s.min_as_series()), s, None)
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Max(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Max)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    parallel_op_series(|s| Ok(s.max_as_series()), s, None)
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Sum(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Sum)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    parallel_op_series(|s| Ok(s.sum_as_series()), s, None)
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Std(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Std)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.std_as_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Var(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Var)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.var_as_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Mean(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Mean)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.mean_as_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Median(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Median)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.median_as_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::First(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::First)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.head(Some(1)))
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Last(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Last)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    Ok(s.tail(Some(1)))
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::List(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::List)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = &s[0];
                                    s.to_list().map(|ca| ca.into_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::NUnique(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => Ok(Arc::new(AggregationExpr::new(
                                input,
                                GroupByMethod::NUnique,
                            ))),
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    s.n_unique().map(|count| {
                                        UInt32Chunked::from_slice(s.name(), &[count as u32])
                                            .into_series()
                                    })
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::Quantile {
                        expr,
                        quantile,
                        interpol,
                    } => {
                        // todo! add schema to get correct output type
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggQuantileExpr::new(input, quantile, interpol)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    s.quantile_as_series(quantile, interpol)
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
                            }
                        }
                    }
                    AAggExpr::AggGroups(expr) => {
                        if let Context::Default = ctxt {
                            panic!("agg groups expression only supported in aggregation context")
                        }
                        let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        Ok(Arc::new(AggregationExpr::new(
                            phys_expr,
                            GroupByMethod::Groups,
                        )))
                    }
                    AAggExpr::Count(expr) => {
                        let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                        match ctxt {
                            Context::Aggregation => {
                                Ok(Arc::new(AggregationExpr::new(input, GroupByMethod::Count)))
                            }
                            Context::Default => {
                                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                                    let s = std::mem::take(&mut s[0]);
                                    let count = s.len();
                                    Ok(UInt32Chunked::from_slice(s.name(), &[count as u32])
                                        .into_series())
                                })
                                    as Arc<dyn SeriesUdf>);
                                Ok(Arc::new(ApplyExpr {
                                    inputs: vec![input],
                                    function,
                                    expr: node_to_expr(expression, expr_arena),
                                    collect_groups: ApplyOptions::ApplyFlat,
                                    auto_explode: false,
                                }))
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
                let phys_expr = self.create_physical_expr(expr, ctxt, expr_arena)?;
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
                let predicate = self.create_physical_expr(predicate, ctxt, expr_arena)?;
                let truthy = self.create_physical_expr(truthy, ctxt, expr_arena)?;
                let falsy = self.create_physical_expr(falsy, ctxt, expr_arena)?;
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
                let input = self.create_physical_expressions(&input, ctxt, expr_arena)?;

                Ok(Arc::new(ApplyExpr {
                    inputs: input,
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: options.collect_groups,
                    auto_explode: options.auto_explode,
                }))
            }
            Function {
                input,
                function,
                options,
                ..
            } => {
                let input = self.create_physical_expressions(&input, ctxt, expr_arena)?;

                Ok(Arc::new(ApplyExpr {
                    inputs: input,
                    function: function.into(),
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: options.collect_groups,
                    auto_explode: options.auto_explode,
                }))
            }
            Shift { input, periods } => {
                let input = self.create_physical_expr(input, ctxt, expr_arena)?;
                Ok(Arc::new(phys_expr::ShiftExpr {
                    input,
                    periods,
                    expr: node_to_expr(expression, expr_arena),
                }))
            }
            Slice {
                input,
                offset,
                length,
            } => {
                let input = self.create_physical_expr(input, ctxt, expr_arena)?;
                let offset = self.create_physical_expr(offset, ctxt, expr_arena)?;
                let length = self.create_physical_expr(length, ctxt, expr_arena)?;
                Ok(Arc::new(SliceExpr {
                    input,
                    offset,
                    length,
                    expr: node_to_expr(expression, expr_arena),
                }))
            }
            Reverse(expr) => {
                let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                    let s = std::mem::take(&mut s[0]);
                    Ok(s.reverse())
                }) as Arc<dyn SeriesUdf>);
                Ok(Arc::new(ApplyExpr {
                    inputs: vec![input],
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: ApplyOptions::ApplyGroups,
                    auto_explode: false,
                }))
            }
            Duplicated(expr) => {
                let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                    let s = std::mem::take(&mut s[0]);
                    s.is_duplicated().map(|ca| ca.into_series())
                }) as Arc<dyn SeriesUdf>);
                Ok(Arc::new(ApplyExpr {
                    inputs: vec![input],
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: ApplyOptions::ApplyGroups,
                    auto_explode: false,
                }))
            }
            IsUnique(expr) => {
                let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                    let s = std::mem::take(&mut s[0]);
                    s.is_unique().map(|ca| ca.into_series())
                }) as Arc<dyn SeriesUdf>);
                Ok(Arc::new(ApplyExpr {
                    inputs: vec![input],
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: ApplyOptions::ApplyGroups,
                    auto_explode: false,
                }))
            }
            Explode(expr) => {
                let input = self.create_physical_expr(expr, ctxt, expr_arena)?;
                let function = NoEq::new(Arc::new(move |s: &mut [Series]| {
                    let s = std::mem::take(&mut s[0]);
                    s.explode()
                }) as Arc<dyn SeriesUdf>);
                Ok(Arc::new(ApplyExpr {
                    inputs: vec![input],
                    function,
                    expr: node_to_expr(expression, expr_arena),
                    collect_groups: ApplyOptions::ApplyFlat,
                    auto_explode: false,
                }))
            }
            Wildcard => panic!("should be no wildcard at this point"),
            Nth(_) => panic!("should be no nth at this point"),
        }
    }
}
