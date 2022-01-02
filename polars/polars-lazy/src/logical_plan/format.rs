use crate::prelude::*;
use crate::utils::expr_to_root_column_names;
use std::fmt;
use std::fmt::Write;

impl fmt::Debug for LogicalPlan {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LogicalPlan::*;
        match self {
            Union { inputs, .. } => write!(f, "UNION {:?}", inputs),
            Cache { input } => write!(f, "CACHE {:?}", input),
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                predicate,
                options,
                ..
            } => {
                let total_columns = schema.fields().len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }
                write!(
                    f,
                    "PARQUET SCAN {}; PROJECT {}/{} COLUMNS; SELECTION: {:?}",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    predicate
                )
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                schema,
                options,
                predicate,
                ..
            } => {
                let total_columns = schema.fields().len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }
                write!(
                    f,
                    "IPC SCAN {}; PROJECT {}/{} COLUMNS; SELECTION: {:?}",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    predicate
                )
            }
            Selection { predicate, input } => {
                write!(f, "FILTER\n\t{:?}\nFROM\n\t{:?}", predicate, input)
            }
            Melt { input, .. } => {
                write!(f, "MELT\n\t{:?}", input)
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                options,
                schema,
                predicate,
                ..
            } => {
                let total_columns = schema.fields().len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }
                write!(
                    f,
                    "CSV SCAN {}; PROJECT {}/{} COLUMNS; SELECTION: {:?}",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    predicate
                )
            }
            DataFrameScan {
                schema,
                projection,
                selection,
                ..
            } => {
                let total_columns = schema.fields().len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = projection {
                    n_columns = format!("{}", columns.len());
                }

                write!(
                    f,
                    "TABLE: {:?}; PROJECT {}/{} COLUMNS; SELECTION: {:?}\\n
                    PROJECTION: {:?}",
                    schema
                        .fields()
                        .iter()
                        .map(|f| f.name())
                        .take(4)
                        .collect::<Vec<_>>(),
                    n_columns,
                    total_columns,
                    selection,
                    projection
                )
            }
            Projection { expr, input, .. } => {
                write!(
                    f,
                    "SELECT {:?} COLUMNS\n\
                 {:?}
                 \nFROM\n{:?}",
                    expr.len(),
                    expr,
                    input
                )
            }
            LocalProjection { expr, input, .. } => {
                write!(
                    f,
                    "LOCAL SELECT {:?} COLUMNS \nFROM\n{:?}",
                    expr.len(),
                    input
                )
            }
            Sort {
                input, by_column, ..
            } => write!(f, "SORT {:?} BY {:?}", input, by_column),
            Explode { input, columns, .. } => {
                write!(f, "EXPLODE COLUMN(S) {:?} OF {:?}", columns, input)
            }
            Aggregate {
                input, keys, aggs, ..
            } => write!(f, "Aggregate\n\t{:?} BY {:?} FROM {:?}", aggs, keys, input),
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                ..
            } => write!(
                f,
                "JOIN\n\t({:?})\nWITH\n\t({:?})\nON (left: {:?} right: {:?})",
                input_left, input_right, left_on, right_on
            ),
            HStack { input, exprs, .. } => {
                write!(f, "STACK [{:?}\n\tWITH COLUMN(S)\n{:?}\n]", input, exprs)
            }
            Distinct { input, .. } => write!(f, "DISTINCT {:?}", input),
            Slice { input, offset, len } => {
                write!(f, "SLICE {:?}, offset: {}, len: {}", input, offset, len)
            }
            Udf { input, .. } => write!(f, "UDF {:?}", input),
        }
    }
}

impl LogicalPlan {
    ///
    /// # Arguments
    /// `id` - (branch, id)
    ///     Used to make sure that the dot boxes are distinct.
    ///     branch is an id per join branch
    ///     id is incremented by the depth traversal of the tree.
    #[cfg(feature = "dot_diagram")]
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_diagram")))]
    pub(crate) fn dot(
        &self,
        acc_str: &mut String,
        id: (usize, usize),
        prev_node: &str,
    ) -> std::fmt::Result {
        use LogicalPlan::*;
        let (branch, id) = id;
        match self {
            Union { inputs, .. } => {
                for input in inputs {
                    let current_node = format!("UNION [{:?}]", (branch, id));
                    self.write_dot(acc_str, prev_node, &current_node, id)?;
                    input.dot(acc_str, (branch, id + 1), &current_node)?
                }
                Ok(())
            }
            Cache { input } => {
                let current_node = format!("CACHE [{:?}]", (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            Selection { predicate, input } => {
                let pred = fmt_predicate(Some(predicate));
                let current_node = format!("FILTER BY {} [{:?}]", pred, (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            #[cfg(feature = "csv-file")]
            CsvScan {
                path,
                options,
                schema,
                predicate,
                ..
            } => {
                let total_columns = schema.fields().len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }
                let pred = fmt_predicate(predicate.as_ref());

                let current_node = format!(
                    "CSV SCAN {};\nπ {}/{};\nσ {}\n[{:?}]",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    pred,
                    (branch, id)
                );
                if id == 0 {
                    self.write_dot(acc_str, prev_node, &current_node, id)?;
                    write!(acc_str, "\"{}\"", current_node)
                } else {
                    self.write_dot(acc_str, prev_node, &current_node, id)
                }
            }
            DataFrameScan {
                schema,
                projection,
                selection,
                ..
            } => {
                let total_columns = schema.fields().len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = projection {
                    n_columns = format!("{}", columns.len());
                }

                let pred = fmt_predicate(selection.as_ref());
                let current_node = format!(
                    "TABLE\nπ {}/{};\nσ {}\n[{:?}]",
                    n_columns,
                    total_columns,
                    pred,
                    (branch, id)
                );
                if id == 0 {
                    self.write_dot(acc_str, prev_node, &current_node, id)?;
                    write!(acc_str, "\"{}\"", current_node)
                } else {
                    self.write_dot(acc_str, prev_node, &current_node, id)
                }
            }
            Projection { expr, input, .. } => {
                let current_node = format!(
                    "π {}/{} [{:?}]",
                    expr.len(),
                    input.schema().fields().len(),
                    (branch, id)
                );
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            Sort {
                input, by_column, ..
            } => {
                let current_node = format!("SORT BY {:?} [{}]", by_column, id);
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            LocalProjection { expr, input, .. } => {
                let current_node = format!(
                    "LOCAL π {}/{} [{:?}]",
                    expr.len(),
                    input.schema().fields().len(),
                    (branch, id)
                );
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            Explode { input, columns, .. } => {
                let current_node = format!("EXPLODE {:?} [{:?}]", columns, (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            Melt { input, .. } => {
                let current_node = format!("MELT [{:?}]", (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            Aggregate {
                input, keys, aggs, ..
            } => {
                let mut s_keys = String::with_capacity(128);
                for key in keys.iter() {
                    s_keys.push_str(&format!("{:?}", key));
                }
                let current_node = format!("AGG {:?} BY {} [{:?}]", aggs, s_keys, (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            HStack { input, exprs, .. } => {
                let mut current_node = String::with_capacity(128);
                current_node.push_str("STACK");
                for e in exprs {
                    if let Expr::Alias(_, name) = e {
                        current_node.push_str(&format!(" {},", name));
                    } else {
                        for name in expr_to_root_column_names(e).iter().take(1) {
                            current_node.push_str(&format!(" {},", name));
                        }
                    }
                }
                current_node.push_str(&format!(" [{:?}]", (branch, id)));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            Slice { input, offset, len } => {
                let current_node = format!(
                    "SLICE offset: {}; len: {} [{:?}]",
                    offset,
                    len,
                    (branch, id)
                );
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            Distinct { input, subset, .. } => {
                let mut current_node = String::with_capacity(128);
                current_node.push_str("DISTINCT");
                if let Some(subset) = &**subset {
                    current_node.push_str(" BY ");
                    for name in subset.iter() {
                        current_node.push_str(&format!("{}, ", name));
                    }
                }
                current_node.push_str(&format!(" [{:?}]", (branch, id)));

                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            #[cfg(feature = "parquet")]
            ParquetScan {
                path,
                schema,
                predicate,
                options,
                ..
            } => {
                let total_columns = schema.fields().len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }

                let pred = fmt_predicate(predicate.as_ref());
                let current_node = format!(
                    "PARQUET SCAN {};\nπ {}/{};\nσ {} [{:?}]",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    pred,
                    (branch, id)
                );
                if id == 0 {
                    self.write_dot(acc_str, prev_node, &current_node, id)?;
                    write!(acc_str, "\"{}\"", current_node)
                } else {
                    self.write_dot(acc_str, prev_node, &current_node, id)
                }
            }
            #[cfg(feature = "ipc")]
            IpcScan {
                path,
                schema,
                options,
                predicate,
                ..
            } => {
                let total_columns = schema.fields().len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }

                let pred = fmt_predicate(predicate.as_ref());
                let current_node = format!(
                    "PARQUET SCAN {};\nπ {}/{};\nσ {} [{:?}]",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    pred,
                    (branch, id)
                );
                if id == 0 {
                    self.write_dot(acc_str, prev_node, &current_node, id)?;
                    write!(acc_str, "\"{}\"", current_node)
                } else {
                    self.write_dot(acc_str, prev_node, &current_node, id)
                }
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                ..
            } => {
                let current_node =
                    format!("JOIN left {:?}; right: {:?} [{}]", left_on, right_on, id);
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input_left.dot(acc_str, (branch + 10, id + 1), &current_node)?;
                input_right.dot(acc_str, (branch + 20, id + 1), &current_node)
            }
            Udf { input, .. } => {
                let current_node = format!("UDF [{:?}]", (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
        }
    }
}

fn fmt_predicate(predicate: Option<&Expr>) -> String {
    if let Some(predicate) = predicate {
        let n = 25;
        let mut pred_fmt = format!("{:?}", predicate);
        pred_fmt = pred_fmt.replace('[', "");
        pred_fmt = pred_fmt.replace(']', "");
        if pred_fmt.len() > n {
            pred_fmt.truncate(n);
            pred_fmt.push_str("...")
        }
        pred_fmt
    } else {
        "-".to_string()
    }
}
