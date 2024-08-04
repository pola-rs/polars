use super::*;

impl IR {
    /// Takes the expressions of an LP node and the inputs of that node and reconstruct
    pub fn with_exprs_and_input(&self, mut exprs: Vec<ExprIR>, mut inputs: Vec<Node>) -> IR {
        use IR::*;

        match self {
            #[cfg(feature = "python")]
            PythonScan { options } => PythonScan {
                options: options.clone(),
            },
            Union { options, .. } => Union {
                inputs,
                options: *options,
            },
            HConcat {
                schema, options, ..
            } => HConcat {
                inputs,
                schema: schema.clone(),
                options: *options,
            },
            Slice { offset, len, .. } => Slice {
                input: inputs[0],
                offset: *offset,
                len: *len,
            },
            Filter { .. } => Filter {
                input: inputs[0],
                predicate: exprs.pop().unwrap(),
            },
            Reduce { schema, .. } => Reduce {
                input: inputs[0],
                exprs,
                schema: schema.clone(),
            },
            Select {
                schema, options, ..
            } => Select {
                input: inputs[0],
                expr: exprs,
                schema: schema.clone(),
                options: *options,
            },
            GroupBy {
                keys,
                schema,
                apply,
                maintain_order,
                options: dynamic_options,
                ..
            } => GroupBy {
                input: inputs[0],
                keys: exprs[..keys.len()].to_vec(),
                aggs: exprs[keys.len()..].to_vec(),
                schema: schema.clone(),
                apply: apply.clone(),
                maintain_order: *maintain_order,
                options: dynamic_options.clone(),
            },
            Join {
                schema,
                left_on,
                options,
                ..
            } => Join {
                input_left: inputs[0],
                input_right: inputs[1],
                schema: schema.clone(),
                left_on: exprs[..left_on.len()].to_vec(),
                right_on: exprs[left_on.len()..].to_vec(),
                options: options.clone(),
            },
            Sort {
                by_column,
                slice,
                sort_options,
                ..
            } => Sort {
                input: inputs[0],
                by_column: by_column.clone(),
                slice: *slice,
                sort_options: sort_options.clone(),
            },
            Cache { id, cache_hits, .. } => Cache {
                input: inputs[0],
                id: *id,
                cache_hits: *cache_hits,
            },
            Distinct { options, .. } => Distinct {
                input: inputs[0],
                options: options.clone(),
            },
            HStack {
                schema, options, ..
            } => HStack {
                input: inputs[0],
                exprs,
                schema: schema.clone(),
                options: *options,
            },
            Scan {
                paths,
                file_info,
                hive_parts,
                output_schema,
                predicate,
                file_options: options,
                scan_type,
            } => {
                let mut new_predicate = None;
                if predicate.is_some() {
                    new_predicate = exprs.pop()
                }
                Scan {
                    paths: paths.clone(),
                    file_info: file_info.clone(),
                    hive_parts: hive_parts.clone(),
                    output_schema: output_schema.clone(),
                    file_options: options.clone(),
                    predicate: new_predicate,
                    scan_type: scan_type.clone(),
                }
            },
            DataFrameScan {
                df,
                schema,
                output_schema,
                filter: selection,
            } => {
                let mut new_selection = None;
                if selection.is_some() {
                    new_selection = exprs.pop()
                }

                DataFrameScan {
                    df: df.clone(),
                    schema: schema.clone(),
                    output_schema: output_schema.clone(),
                    filter: new_selection,
                }
            },
            MapFunction { function, .. } => MapFunction {
                input: inputs[0],
                function: function.clone(),
            },
            ExtContext { schema, .. } => ExtContext {
                input: inputs.pop().unwrap(),
                contexts: inputs,
                schema: schema.clone(),
            },
            Sink { payload, .. } => Sink {
                input: inputs.pop().unwrap(),
                payload: payload.clone(),
            },
            SimpleProjection { columns, .. } => SimpleProjection {
                input: inputs.pop().unwrap(),
                columns: columns.clone(),
            },
            Invalid => unreachable!(),
        }
    }

    /// Copy the exprs in this LP node to an existing container.
    pub fn copy_exprs(&self, container: &mut Vec<ExprIR>) {
        use IR::*;
        match self {
            Slice { .. } | Cache { .. } | Distinct { .. } | Union { .. } | MapFunction { .. } => {},
            Sort { by_column, .. } => container.extend_from_slice(by_column),
            Filter { predicate, .. } => container.push(predicate.clone()),
            Reduce { exprs, .. } => container.extend_from_slice(exprs),
            Select { expr, .. } => container.extend_from_slice(expr),
            GroupBy { keys, aggs, .. } => {
                let iter = keys.iter().cloned().chain(aggs.iter().cloned());
                container.extend(iter)
            },
            Join {
                left_on, right_on, ..
            } => {
                let iter = left_on.iter().cloned().chain(right_on.iter().cloned());
                container.extend(iter)
            },
            HStack { exprs, .. } => container.extend_from_slice(exprs),
            Scan { predicate, .. } => {
                if let Some(pred) = predicate {
                    container.push(pred.clone())
                }
            },
            DataFrameScan {
                filter: selection, ..
            } => {
                if let Some(expr) = selection {
                    container.push(expr.clone())
                }
            },
            #[cfg(feature = "python")]
            PythonScan { .. } => {},
            HConcat { .. } => {},
            ExtContext { .. } | Sink { .. } | SimpleProjection { .. } => {},
            Invalid => unreachable!(),
        }
    }

    /// Get expressions in this node.
    pub fn get_exprs(&self) -> Vec<ExprIR> {
        let mut exprs = Vec::new();
        self.copy_exprs(&mut exprs);
        exprs
    }

    /// Push inputs of the LP in of this node to an existing container.
    /// Most plans have typically one input. A join has two and a scan (CsvScan)
    /// or an in-memory DataFrame has none. A Union has multiple.
    pub fn copy_inputs<T>(&self, container: &mut T)
    where
        T: PushNode,
    {
        use IR::*;
        let input = match self {
            Union { inputs, .. } => {
                for node in inputs {
                    container.push_node(*node);
                }
                return;
            },
            HConcat { inputs, .. } => {
                for node in inputs {
                    container.push_node(*node);
                }
                return;
            },
            Slice { input, .. } => *input,
            Filter { input, .. } => *input,
            Select { input, .. } => *input,
            Reduce { input, .. } => *input,
            SimpleProjection { input, .. } => *input,
            Sort { input, .. } => *input,
            Cache { input, .. } => *input,
            GroupBy { input, .. } => *input,
            Join {
                input_left,
                input_right,
                ..
            } => {
                container.push_node(*input_left);
                container.push_node(*input_right);
                return;
            },
            HStack { input, .. } => *input,
            Distinct { input, .. } => *input,
            MapFunction { input, .. } => *input,
            Sink { input, .. } => *input,
            ExtContext {
                input, contexts, ..
            } => {
                for n in contexts {
                    container.push_node(*n)
                }
                *input
            },
            Scan { .. } => return,
            DataFrameScan { .. } => return,
            #[cfg(feature = "python")]
            PythonScan { .. } => return,
            Invalid => unreachable!(),
        };
        container.push_node(input)
    }

    pub fn get_inputs(&self) -> UnitVec<Node> {
        let mut inputs: UnitVec<Node> = unitvec!();
        self.copy_inputs(&mut inputs);
        inputs
    }

    pub fn get_inputs_vec(&self) -> Vec<Node> {
        let mut inputs = vec![];
        self.copy_inputs(&mut inputs);
        inputs
    }

    pub(crate) fn get_input(&self) -> Option<Node> {
        let mut inputs: UnitVec<Node> = unitvec!();
        self.copy_inputs(&mut inputs);
        inputs.first().copied()
    }
}
