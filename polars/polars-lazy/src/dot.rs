use crate::prelude::*;
use crate::utils::expr_to_root_column_names;
use polars_core::prelude::*;
use polars_utils::arena::Arena;
use std::fmt::Write;

impl LazyFrame {
    /// Get a dot language representation of the LogicalPlan.
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_diagram")))]
    pub fn to_dot(&self, optimized: bool) -> Result<String> {
        let mut s = String::with_capacity(512);

        let mut logical_plan = self.clone().get_plan_builder().build();
        if optimized {
            // initialize arena's
            let mut expr_arena = Arena::with_capacity(64);
            let mut lp_arena = Arena::with_capacity(32);

            let lp_top = self.clone().optimize(&mut lp_arena, &mut expr_arena)?;
            logical_plan = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);
        }

        logical_plan.dot(&mut s, (0, 0), "").expect("io error");
        s.push_str("\n}");
        Ok(s)
    }
}

impl LogicalPlan {
    fn write_dot(
        &self,
        acc_str: &mut String,
        prev_node: &str,
        current_node: &str,
        id: usize,
    ) -> std::fmt::Result {
        if id == 0 {
            writeln!(acc_str, "graph  polars_query {{")
        } else {
            writeln!(
                acc_str,
                "\"{}\" -- \"{}\"",
                prev_node.replace('"', r#"\""#),
                current_node.replace('"', r#"\""#)
            )
        }
    }

    ///
    /// # Arguments
    /// `id` - (branch, id)
    ///     Used to make sure that the dot boxes are distinct.
    ///     branch is an id per join/union branch
    ///     id is incremented by the depth traversal of the tree.
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_diagram")))]
    pub(crate) fn dot(
        &self,
        acc_str: &mut String,
        id: (usize, usize),
        prev_node: &str,
    ) -> std::fmt::Result {
        use LogicalPlan::*;
        let (mut branch, id) = id;
        match self {
            Union { inputs, .. } => {
                let current_node = format!("UNION [{:?}]", (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                for input in inputs {
                    input.dot(acc_str, (branch, id + 1), &current_node)?;
                    branch += 1;
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
                let total_columns = schema.len();
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
                let total_columns = schema.len();
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
                    input.schema().len(),
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
                    input.schema().len(),
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
                s_keys.push('[');
                for key in keys.iter() {
                    s_keys.push_str(&format!("{:?},", key));
                }
                s_keys.pop();
                s_keys.push(']');
                let current_node = format!("AGG {:?}\nBY\n{} [{:?}]", aggs, s_keys, (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            HStack { input, exprs, .. } => {
                let mut current_node = String::with_capacity(128);
                current_node.push_str("WITH COLUMNS [");
                for e in exprs {
                    if let Expr::Alias(_, name) = e {
                        current_node.push_str(&format!("\"{}\",", name));
                    } else {
                        for name in expr_to_root_column_names(e).iter().take(1) {
                            current_node.push_str(&format!("\"{}\",", name));
                        }
                    }
                }
                current_node.pop();
                current_node.push(']');
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
            Distinct { input, options, .. } => {
                let mut current_node = String::with_capacity(128);
                current_node.push_str("DISTINCT");
                if let Some(subset) = &options.subset {
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
                let total_columns = schema.len();
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
                let total_columns = schema.len();
                let mut n_columns = "*".to_string();
                if let Some(columns) = &options.with_columns {
                    n_columns = format!("{}", columns.len());
                }

                let pred = fmt_predicate(predicate.as_ref());
                let current_node = format!(
                    "IPC SCAN {};\nπ {}/{};\nσ {} [{:?}]",
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
                let current_node = format!(
                    r#"JOIN
                    left {:?};
                    right: {:?} [{}]"#,
                    left_on, right_on, id
                );
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input_left.dot(acc_str, (branch + 10, id + 1), &current_node)?;
                input_right.dot(acc_str, (branch + 20, id + 1), &current_node)
            }
            Udf { input, options, .. } => {
                let current_node = format!("{} [{:?}]", options.fmt_str, (branch, id));
                self.write_dot(acc_str, prev_node, &current_node, id)?;
                input.dot(acc_str, (branch, id + 1), &current_node)
            }
            Error { err, .. } => {
                let current_node = format!("{:?}", &**err);
                self.write_dot(acc_str, prev_node, &current_node, id)
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
