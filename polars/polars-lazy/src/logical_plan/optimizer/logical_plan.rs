use crate::prelude::*;

impl ALogicalPlan {
    /// Takes the expressions of an LP node and the inputs of that node and reconstruct
    pub fn from_exprs_and_input(&self, mut exprs: Vec<Node>, inputs: Vec<Node>) -> ALogicalPlan {
        use ALogicalPlan::*;

        match self {
            Melt {
                id_vars,
                value_vars,
                schema,
                ..
            } => Melt {
                input: inputs[0],
                id_vars: id_vars.clone(),
                value_vars: value_vars.clone(),
                schema: schema.clone(),
            },
            Slice { offset, len, .. } => Slice {
                input: inputs[0],
                offset: *offset,
                len: *len,
            },
            Selection { .. } => Selection {
                input: inputs[0],
                predicate: exprs[0],
            },
            LocalProjection { schema, .. } => LocalProjection {
                input: inputs[0],
                expr: exprs,
                schema: schema.clone(),
            },
            Projection { schema, .. } => Projection {
                input: inputs[0],
                expr: exprs,
                schema: schema.clone(),
            },
            Aggregate {
                keys,
                schema,
                apply,
                ..
            } => Aggregate {
                input: inputs[0],
                keys: exprs[..keys.len()].to_vec(),
                aggs: exprs[keys.len()..].to_vec(),
                schema: schema.clone(),
                apply: apply.clone(),
            },
            Join {
                schema,
                how,
                left_on,
                allow_par,
                force_par,
                ..
            } => Join {
                input_left: inputs[0],
                input_right: inputs[1],
                schema: schema.clone(),
                how: *how,
                left_on: exprs[..left_on.len()].to_vec(),
                right_on: exprs[left_on.len()..].to_vec(),
                allow_par: *allow_par,
                force_par: *force_par,
            },
            Sort {
                by_column, reverse, ..
            } => Sort {
                input: inputs[0],
                by_column: by_column.clone(),
                reverse: *reverse,
            },
            Explode { columns, .. } => Explode {
                input: inputs[0],
                columns: columns.clone(),
            },
            Cache { .. } => Cache { input: inputs[0] },
            Distinct {
                maintain_order,
                subset,
                ..
            } => Distinct {
                input: inputs[0],
                maintain_order: *maintain_order,
                subset: subset.clone(),
            },
            HStack { schema, .. } => HStack {
                input: inputs[0],
                exprs,
                schema: schema.clone(),
            },
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                with_columns,
                predicate,
                stop_after_n_rows,
                cache,
                ..
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }

                ParquetScan {
                    path: path.clone(),
                    schema: schema.clone(),
                    with_columns: with_columns.clone(),
                    predicate: new_predicate,
                    aggregate: exprs,
                    stop_after_n_rows: *stop_after_n_rows,
                    cache: *cache,
                }
            }
            CsvScan {
                path,
                schema,
                has_header,
                delimiter,
                ignore_errors,
                skip_rows,
                stop_after_n_rows,
                with_columns,
                predicate,
                cache,
                ..
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }
                CsvScan {
                    path: path.clone(),
                    schema: schema.clone(),
                    has_header: *has_header,
                    delimiter: *delimiter,
                    ignore_errors: *ignore_errors,
                    skip_rows: *skip_rows,
                    stop_after_n_rows: *stop_after_n_rows,
                    with_columns: with_columns.clone(),
                    predicate: new_predicate,
                    aggregate: exprs,
                    cache: *cache,
                }
            }
            DataFrameScan {
                df,
                schema,
                projection,
                selection,
            } => {
                let mut new_selection = None;
                if selection.is_some() {
                    new_selection = exprs.pop()
                }
                let mut new_projection = None;
                if projection.is_some() {
                    new_projection = Some(exprs)
                }

                DataFrameScan {
                    df: df.clone(),
                    schema: schema.clone(),
                    projection: new_projection,
                    selection: new_selection,
                }
            }
            Udf {
                function,
                predicate_pd,
                projection_pd,
                schema,
                ..
            } => Udf {
                input: inputs[0],
                function: function.clone(),
                predicate_pd: *predicate_pd,
                projection_pd: *projection_pd,
                schema: schema.clone(),
            },
        }
    }

    /// Get expressions in this node.
    pub fn get_exprs(&self) -> Vec<Node> {
        use ALogicalPlan::*;
        match self {
            Melt { .. }
            | Slice { .. }
            | Sort { .. }
            | Explode { .. }
            | Cache { .. }
            | Distinct { .. }
            | Udf { .. } => vec![],
            Selection { predicate, .. } => vec![*predicate],
            Projection { expr, .. } => expr.clone(),
            LocalProjection { expr, .. } => expr.clone(),
            Aggregate { keys, aggs, .. } => {
                keys.iter().copied().chain(aggs.iter().copied()).collect()
            }
            Join {
                left_on, right_on, ..
            } => left_on
                .iter()
                .copied()
                .chain(right_on.iter().copied())
                .collect(),
            HStack { exprs, .. } => exprs.clone(),
            #[cfg(feature = "parquet")]
            ParquetScan {
                predicate,
                aggregate,
                ..
            } => {
                let mut exprs = aggregate.clone();
                if let Some(node) = predicate {
                    exprs.push(*node)
                }
                exprs
            }
            CsvScan {
                predicate,
                aggregate,
                ..
            } => {
                let mut exprs = aggregate.clone();
                if let Some(node) = predicate {
                    exprs.push(*node)
                }
                exprs
            }
            DataFrameScan {
                projection,
                selection,
                ..
            } => {
                let mut exprs = vec![];
                if let Some(expr) = projection {
                    exprs.extend_from_slice(expr)
                }
                if let Some(expr) = selection {
                    exprs.push(*expr)
                }
                exprs
            }
        }
    }

    pub fn get_inputs(&self) -> Vec<Node> {
        use ALogicalPlan::*;
        let input = match self {
            Melt { input, .. } => *input,
            Slice { input, .. } => *input,
            Selection { input, .. } => *input,
            Projection { input, .. } => *input,
            LocalProjection { input, .. } => *input,
            Sort { input, .. } => *input,
            Explode { input, .. } => *input,
            Cache { input, .. } => *input,
            Aggregate { input, .. } => *input,
            Join {
                input_left,
                input_right,
                ..
            } => return vec![*input_left, *input_right],
            HStack { input, .. } => *input,
            Distinct { input, .. } => *input,
            Udf { input, .. } => *input,
            #[cfg(feature = "parquet")]
            ParquetScan { .. } => return vec![],
            CsvScan { .. } | DataFrameScan { .. } => return vec![],
        };
        vec![input]
    }
}
