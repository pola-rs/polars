use std::fmt::Write;

use polars_core::prelude::*;
use polars_utils::arena::Arena;

use crate::prelude::*;
use crate::utils::expr_to_leaf_column_names;

impl Expr {
    /// Get a dot language representation of the Expression.
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_diagram")))]
    pub fn to_dot(&self) -> PolarsResult<String> {
        let mut s = String::with_capacity(512);
        self.dot_viz(&mut s, (0, 0), "").expect("io error");
        s.push_str("\n}");
        Ok(s)
    }

    fn write_dot(
        &self,
        acc_str: &mut String,
        prev_node: &str,
        current_node: &str,
        id: usize,
    ) -> std::fmt::Result {
        if id == 0 {
            writeln!(acc_str, "graph expr {{")
        } else {
            writeln!(
                acc_str,
                "\"{}\" -- \"{}\"",
                prev_node.replace('"', r#"\""#),
                current_node.replace('"', r#"\""#)
            )
        }
    }

    #[cfg_attr(docsrs, doc(cfg(feature = "dot_diagram")))]
    fn dot_viz(
        &self,
        acc_str: &mut String,
        id: (usize, usize),
        prev_node: &str,
    ) -> std::fmt::Result {
        let (mut branch, id) = id;

        match self {
            Expr::BinaryExpr { left, op, right } => {
                let current_node = format!(
                    r#"BINARY
                    left _;
                    op {:?},
                    right: _ [{},{}]"#,
                    op, branch, id
                );

                self.write_dot(acc_str, prev_node, &current_node, id)?;
                for input in [left, right] {
                    input.dot_viz(acc_str, (branch, id + 1), &current_node)?;
                    branch += 1;
                }
                Ok(())
            }
            _ => self.write_dot(acc_str, prev_node, &format!("{}{}", branch, id), id),
        }
    }
}

impl LazyFrame {
    /// Get a dot language representation of the LogicalPlan.
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_diagram")))]
    pub fn to_dot(&self, optimized: bool) -> PolarsResult<String> {
        let mut s = String::with_capacity(512);

        let mut logical_plan = self.clone().get_plan_builder().build();
        if optimized {
            // initialize arena's
            let mut expr_arena = Arena::with_capacity(64);
            let mut lp_arena = Arena::with_capacity(32);

            let lp_top = self.clone().optimize(&mut lp_arena, &mut expr_arena)?;
            logical_plan = node_to_lp(lp_top, &mut expr_arena, &mut lp_arena);
        }

        let prev_node = DotNode {
            branch: 0,
            id: 0,
            fmt: "",
        };

        // maps graphviz id to label
        // we use this to create this graph
        // first we create nodes including ids to make sure they are unique
        // A [id] -- B [id]
        // B [id] -- C [id]
        //
        // then later we hide the [id] by adding this to the graph
        // A [id] [label="A"]
        // B [id] [label="B"]
        // C [id] [label="C"]

        let mut id_map = PlHashMap::with_capacity(8);
        logical_plan
            .dot(&mut s, (0, 0), prev_node, &mut id_map)
            .expect("io error");
        s.push('\n');

        for (id, label) in id_map {
            // the label is wrapped in double quotes
            // the id already is wrapped in double quotes
            writeln!(s, "{}[label=\"{}\"]", id, label).unwrap();
        }
        s.push_str("\n}");
        Ok(s)
    }
}

#[derive(Copy, Clone)]
struct DotNode<'a> {
    branch: usize,
    id: usize,
    fmt: &'a str,
}

impl LogicalPlan {
    fn write_dot(
        &self,
        acc_str: &mut String,
        prev_node: DotNode,
        current_node: DotNode,
        id_map: &mut PlHashMap<String, String>,
    ) -> std::fmt::Result {
        if current_node.id == 0 && current_node.branch == 0 {
            writeln!(acc_str, "graph  polars_query {{")
        } else {
            let fmt_prev_node = prev_node.fmt.replace('"', r#"\""#);
            let fmt_current_node = current_node.fmt.replace('"', r#"\""#);

            let id_prev_node = format!(
                "\"{} [{:?}]\"",
                &fmt_prev_node,
                (prev_node.branch, prev_node.id)
            );
            let id_current_node = format!(
                "\"{} [{:?}]\"",
                &fmt_current_node,
                (current_node.branch, current_node.id)
            );

            writeln!(acc_str, "{} -- {}", &id_prev_node, &id_current_node)?;

            id_map.insert(id_current_node, fmt_current_node);
            id_map.insert(id_prev_node, fmt_prev_node);

            Ok(())
        }
    }

    ///
    /// # Arguments
    /// `id` - (branch, id)
    ///     Used to make sure that the dot boxes are distinct.
    ///     branch is an id per join/union branch
    ///     id is incremented by the depth traversal of the tree.
    #[cfg_attr(docsrs, doc(cfg(feature = "dot_diagram")))]
    fn dot(
        &self,
        acc_str: &mut String,
        id: (usize, usize),
        prev_node: DotNode,
        id_map: &mut PlHashMap<String, String>,
    ) -> std::fmt::Result {
        use LogicalPlan::*;
        let (mut branch, id) = id;
        match self {
            AnonymousScan { schema, .. } => {
                let total_columns = schema.len();

                let fmt = format!("ANONYMOUS SCAN;\nπ {}", total_columns);
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)
            }
            Union { inputs, .. } => {
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: "UNION",
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                for input in inputs {
                    input.dot(acc_str, (branch, id + 1), current_node, id_map)?;
                    branch += 1;
                }
                Ok(())
            }
            Cache {
                input,
                id: cache_id,
                count,
            } => {
                let fmt = if *count == usize::MAX {
                    "CACHE".to_string()
                } else {
                    format!("CACHE: {}times", *count)
                };
                let current_node = DotNode {
                    branch: *cache_id,
                    id: *cache_id,
                    fmt: &fmt,
                };
                // here we take the cache id, to ensure the same cached subplans get the same ids
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (*cache_id, cache_id + 1), current_node, id_map)
            }
            Selection { predicate, input } => {
                let pred = fmt_predicate(Some(predicate));
                let fmt = format!("FILTER BY {}", pred);

                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };

                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            #[cfg(feature = "python")]
            PythonScan { options } => {
                let schema = &options.schema;
                let total_columns = schema.len();
                let n_columns = if let Some(columns) = &options.with_columns {
                    format!("{}", columns.len())
                } else {
                    "*".to_string()
                };

                let fmt = format!("PYTHON SCAN;\nπ {}/{};", n_columns, total_columns,);
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)
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

                let fmt = format!(
                    "CSV SCAN {};\nπ {}/{};\nσ {};",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    pred,
                );
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)
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
                let fmt = format!("TABLE\nπ {}/{};\nσ {};", n_columns, total_columns, pred,);
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)
            }
            Projection { expr, input, .. } => {
                let schema = input.schema().map_err(|_| {
                    eprintln!("could not determine schema");
                    std::fmt::Error
                })?;

                let fmt = format!("π {}/{}", expr.len(), schema.len());

                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            Sort {
                input, by_column, ..
            } => {
                let fmt = format!("SORT BY {:?}", by_column);
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            LocalProjection { expr, input, .. } => {
                let schema = input.schema().map_err(|_| {
                    eprintln!("could not determine schema");
                    std::fmt::Error
                })?;

                let fmt = format!("LOCAL π {}/{}", expr.len(), schema.len(),);
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            Explode { input, columns, .. } => {
                let fmt = format!("EXPLODE {:?}", columns);

                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            Melt { input, .. } => {
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: "MELT",
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            Aggregate {
                input, keys, aggs, ..
            } => {
                let mut s_keys = String::with_capacity(128);
                s_keys.push('[');
                for key in keys.iter() {
                    write!(s_keys, "{:?},", key)?
                }
                s_keys.pop();
                s_keys.push(']');
                let fmt = format!("AGG {:?}\nBY\n{} [{:?}]", aggs, s_keys, (branch, id));
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            HStack { input, exprs, .. } => {
                let mut fmt = String::with_capacity(128);
                fmt.push_str("WITH COLUMNS [");
                for e in exprs {
                    if let Expr::Alias(_, name) = e {
                        write!(fmt, "\"{}\",", name)?
                    } else {
                        for name in expr_to_leaf_column_names(e).iter().take(1) {
                            write!(fmt, "\"{}\",", name)?
                        }
                    }
                }
                fmt.pop();
                fmt.push(']');
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            Slice { input, offset, len } => {
                let fmt = format!("SLICE offset: {}; len: {}", offset, len,);
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            Distinct { input, options, .. } => {
                let mut fmt = String::with_capacity(128);
                fmt.push_str("DISTINCT");
                if let Some(subset) = &options.subset {
                    fmt.push_str(" BY ");
                    for name in subset.iter() {
                        write!(fmt, "{}", name)?
                    }
                }
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };

                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
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
                let fmt = format!(
                    "PARQUET SCAN {};\nπ {}/{};\nσ {}",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    pred,
                );
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)
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
                let fmt = format!(
                    "IPC SCAN {};\nπ {}/{};\nσ {}",
                    path.to_string_lossy(),
                    n_columns,
                    total_columns,
                    pred,
                );
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)
            }
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                ..
            } => {
                let fmt = format!(
                    r#"JOIN
                    left {:?};
                    right: {:?}"#,
                    left_on, right_on
                );
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input_left.dot(acc_str, (branch + 100, id + 1), current_node, id_map)?;
                input_right.dot(acc_str, (branch + 200, id + 1), current_node, id_map)
            }
            MapFunction {
                input, function, ..
            } => {
                let fmt = format!("{}", function);
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            ExtContext { input, .. } => {
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: "EXTERNAL_CONTEXT",
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)?;
                input.dot(acc_str, (branch, id + 1), current_node, id_map)
            }
            Error { err, .. } => {
                let fmt = format!("{:?}", &**err);
                let current_node = DotNode {
                    branch,
                    id,
                    fmt: &fmt,
                };
                self.write_dot(acc_str, prev_node, current_node, id_map)
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
