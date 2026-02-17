use std::fmt::{self, Display, Formatter, Write};

use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_io::RowIndex;
use polars_utils::format_list_truncated;
use polars_utils::slice_enum::Slice;
use recursive::recursive;

use self::ir::dot::ScanSourcesDisplay;
use crate::dsl::deletion::DeletionFilesList;
use crate::prelude::*;

const INDENT_INCREMENT: usize = 2;

pub struct IRDisplay<'a> {
    lp: IRPlanRef<'a>,
}

#[derive(Clone, Copy)]
pub struct ExprIRDisplay<'a> {
    pub(crate) node: Node,
    pub(crate) output_name: &'a OutputName,
    pub(crate) expr_arena: &'a Arena<AExpr>,
}

impl<'a> ExprIRDisplay<'a> {
    pub fn display_node(node: Node, expr_arena: &'a Arena<AExpr>) -> Self {
        Self {
            node,
            output_name: &OutputName::None,
            expr_arena,
        }
    }
}

/// Utility structure to display several [`ExprIR`]'s in a nice way
pub(crate) struct ExprIRSliceDisplay<'a, T: AsExpr> {
    pub(crate) exprs: &'a [T],
    pub(crate) expr_arena: &'a Arena<AExpr>,
}

pub(crate) trait AsExpr {
    fn node(&self) -> Node;
    fn output_name(&self) -> &OutputName;
}

impl AsExpr for Node {
    fn node(&self) -> Node {
        *self
    }
    fn output_name(&self) -> &OutputName {
        &OutputName::None
    }
}

impl AsExpr for ExprIR {
    fn node(&self) -> Node {
        self.node()
    }
    fn output_name(&self) -> &OutputName {
        self.output_name_inner()
    }
}

#[allow(clippy::too_many_arguments)]
fn write_scan(
    f: &mut dyn fmt::Write,
    name: &str,
    sources: &ScanSources,
    indent: usize,
    n_columns: i64,
    total_columns: usize,
    row_estimation: Option<usize>,
    predicate: &Option<ExprIRDisplay<'_>>,
    pre_slice: Option<Slice>,
    row_index: Option<&RowIndex>,
    deletion_files: Option<&DeletionFilesList>,
) -> fmt::Result {
    write!(
        f,
        "{:indent$}{name} SCAN {}",
        "",
        ScanSourcesDisplay(sources),
    )?;

    let total_columns = total_columns - usize::from(row_index.is_some());
    if n_columns > 0 {
        write!(
            f,
            "\n{:indent$}PROJECT {n_columns}/{total_columns} COLUMNS",
            "",
        )?;
    } else {
        write!(f, "\n{:indent$}PROJECT */{total_columns} COLUMNS", "")?;
    }
    if let Some(predicate) = predicate {
        write!(f, "\n{:indent$}SELECTION: {predicate}", "")?;
    }
    if let Some(pre_slice) = pre_slice {
        write!(f, "\n{:indent$}SLICE: {pre_slice:?}", "")?;
    }
    if let Some(row_index) = row_index {
        write!(f, "\n{:indent$}ROW_INDEX: {}", "", row_index.name)?;
        if row_index.offset != 0 {
            write!(f, " (offset: {})", row_index.offset)?;
        }
    }
    if let Some(deletion_files) = deletion_files {
        write!(f, "\n{deletion_files}")?;
    }
    if let Some(row_estimation) = row_estimation {
        write!(f, "\n{:indent$}ESTIMATED ROWS: {row_estimation}", "")?;
    }
    Ok(())
}

impl<'a> IRDisplay<'a> {
    pub fn new(lp: IRPlanRef<'a>) -> Self {
        Self { lp }
    }

    fn root(&self) -> &IR {
        self.lp.root()
    }

    fn with_root(&self, root: Node) -> Self {
        Self {
            lp: self.lp.with_root(root),
        }
    }

    fn display_expr(&self, root: &'a ExprIR) -> ExprIRDisplay<'a> {
        ExprIRDisplay {
            node: root.node(),
            output_name: root.output_name_inner(),
            expr_arena: self.lp.expr_arena,
        }
    }

    fn display_expr_slice(&self, exprs: &'a [ExprIR]) -> ExprIRSliceDisplay<'a, ExprIR> {
        ExprIRSliceDisplay {
            exprs,
            expr_arena: self.lp.expr_arena,
        }
    }

    #[recursive]
    fn _format(&self, f: &mut Formatter, indent: usize) -> fmt::Result {
        if indent != 0 {
            writeln!(f)?;
        }

        let sub_indent = indent + INDENT_INCREMENT;
        use IR::*;

        let ir_node = self.root();
        let output_schema = ir_node.schema(self.lp.lp_arena);
        let output_schema = output_schema.as_ref();
        match ir_node {
            Union { inputs, options } => {
                write_ir_non_recursive(f, ir_node, self.lp.expr_arena, output_schema, indent)?;
                let name = if let Some(slice) = options.slice {
                    format!("SLICED UNION: {slice:?}")
                } else {
                    "UNION".to_string()
                };

                // 3 levels of indentation
                // - 0 => UNION ... END UNION
                // - 1 => PLAN 0, PLAN 1, ... PLAN N
                // - 2 => actual formatting of plans
                let sub_sub_indent = sub_indent + INDENT_INCREMENT;
                for (i, plan) in inputs.iter().enumerate() {
                    write!(f, "\n{:sub_indent$}PLAN {i}:", "")?;
                    self.with_root(*plan)._format(f, sub_sub_indent)?;
                }
                write!(f, "\n{:indent$}END {name}", "")
            },
            HConcat { inputs, .. } => {
                let sub_sub_indent = sub_indent + INDENT_INCREMENT;
                write_ir_non_recursive(f, ir_node, self.lp.expr_arena, output_schema, indent)?;
                for (i, plan) in inputs.iter().enumerate() {
                    write!(f, "\n{:sub_indent$}PLAN {i}:", "")?;
                    self.with_root(*plan)._format(f, sub_sub_indent)?;
                }
                write!(f, "\n{:indent$}END HCONCAT", "")
            },
            GroupBy { input, .. } => {
                write_ir_non_recursive(f, ir_node, self.lp.expr_arena, output_schema, indent)?;
                write!(f, "\n{:sub_indent$}FROM", "")?;
                self.with_root(*input)._format(f, sub_indent)?;
                Ok(())
            },
            Join {
                input_left,
                input_right,
                left_on,
                right_on,
                options,
                ..
            } => {
                let left_on = self.display_expr_slice(left_on);
                let right_on = self.display_expr_slice(right_on);

                // Fused cross + filter (show as nested loop join)
                if let Some(JoinTypeOptionsIR::CrossAndFilter { predicate }) = &options.options {
                    let predicate = self.display_expr(predicate);
                    let name = "NESTED LOOP";
                    write!(f, "{:indent$}{name} JOIN ON {predicate}:", "")?;
                    write!(f, "\n{:indent$}LEFT PLAN:", "")?;
                    self.with_root(*input_left)._format(f, sub_indent)?;
                    write!(f, "\n{:indent$}RIGHT PLAN:", "")?;
                    self.with_root(*input_right)._format(f, sub_indent)?;
                    write!(f, "\n{:indent$}END {name} JOIN", "")
                } else {
                    let how = &options.args.how;
                    write!(f, "{:indent$}{how} JOIN:", "")?;
                    write!(f, "\n{:indent$}LEFT PLAN ON: {left_on}", "")?;
                    self.with_root(*input_left)._format(f, sub_indent)?;
                    write!(f, "\n{:indent$}RIGHT PLAN ON: {right_on}", "")?;
                    self.with_root(*input_right)._format(f, sub_indent)?;
                    write!(f, "\n{:indent$}END {how} JOIN", "")
                }
            },
            MapFunction { input, .. } => {
                write_ir_non_recursive(f, ir_node, self.lp.expr_arena, output_schema, indent)?;
                self.with_root(*input)._format(f, sub_indent)
            },
            SinkMultiple { inputs } => {
                write_ir_non_recursive(f, ir_node, self.lp.expr_arena, output_schema, indent)?;

                // 3 levels of indentation
                // - 0 => SINK_MULTIPLE ... END SINK_MULTIPLE
                // - 1 => PLAN 0, PLAN 1, ... PLAN N
                // - 2 => actual formatting of plans
                let sub_sub_indent = sub_indent + 2;
                for (i, plan) in inputs.iter().enumerate() {
                    write!(f, "\n{:sub_indent$}PLAN {i}:", "")?;
                    self.with_root(*plan)._format(f, sub_sub_indent)?;
                }
                write!(f, "\n{:indent$}END SINK_MULTIPLE", "")
            },
            #[cfg(feature = "merge_sorted")]
            MergeSorted {
                input_left,
                input_right,
                key: _,
            } => {
                write_ir_non_recursive(f, ir_node, self.lp.expr_arena, output_schema, indent)?;
                write!(f, ":")?;

                write!(f, "\n{:indent$}LEFT PLAN:", "")?;
                self.with_root(*input_left)._format(f, sub_indent)?;
                write!(f, "\n{:indent$}RIGHT PLAN:", "")?;
                self.with_root(*input_right)._format(f, sub_indent)?;
                write!(f, "\n{:indent$}END MERGE_SORTED", "")
            },
            ir_node => {
                write_ir_non_recursive(f, ir_node, self.lp.expr_arena, output_schema, indent)?;
                for input in ir_node.inputs() {
                    self.with_root(input)._format(f, sub_indent)?;
                }
                Ok(())
            },
        }
    }
}

impl<'a> ExprIRDisplay<'a> {
    fn with_slice<T: AsExpr>(&self, exprs: &'a [T]) -> ExprIRSliceDisplay<'a, T> {
        ExprIRSliceDisplay {
            exprs,
            expr_arena: self.expr_arena,
        }
    }

    fn with_root<T: AsExpr>(&self, root: &'a T) -> Self {
        Self {
            node: root.node(),
            output_name: root.output_name(),
            expr_arena: self.expr_arena,
        }
    }
}

impl Display for IRDisplay<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self._format(f, 0)
    }
}

impl fmt::Debug for IRDisplay<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(&self, f)
    }
}

impl<T: AsExpr> Display for ExprIRSliceDisplay<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        // Display items in slice delimited by a comma

        use std::fmt::Write;

        let mut iter = self.exprs.iter();

        f.write_char('[')?;
        if let Some(fst) = iter.next() {
            let fst = ExprIRDisplay {
                node: fst.node(),
                output_name: fst.output_name(),
                expr_arena: self.expr_arena,
            };
            write!(f, "{fst}")?;
        }

        for expr in iter {
            let expr = ExprIRDisplay {
                node: expr.node(),
                output_name: expr.output_name(),
                expr_arena: self.expr_arena,
            };
            write!(f, ", {expr}")?;
        }

        f.write_char(']')?;

        Ok(())
    }
}

impl<T: AsExpr> fmt::Debug for ExprIRSliceDisplay<'_, T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl Display for ExprIRDisplay<'_> {
    #[recursive]
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let root = self.expr_arena.get(self.node);

        use AExpr::*;
        match root {
            Element => f.write_str("element()"),
            #[cfg(feature = "dynamic_group_by")]
            Rolling {
                function,
                index_column,
                period,
                offset,
                closed_window: _,
            } => {
                let function = self.with_root(function);
                let index_column = self.with_root(index_column);
                write!(
                    f,
                    "{function}.rolling(by='{index_column}', offset={offset}, period={period})",
                )
            },
            Over {
                function,
                partition_by,
                order_by,
                mapping: _,
            } => {
                let function = self.with_root(function);
                let partition_by = self.with_slice(partition_by);
                if let Some((order_by, _)) = order_by {
                    let order_by = self.with_root(order_by);
                    write!(
                        f,
                        "{function}.over(partition_by: {partition_by}, order_by: {order_by})"
                    )
                } else {
                    write!(f, "{function}.over({partition_by})")
                }
            },
            Len => write!(f, "len()"),
            Explode { expr, options } => {
                let expr = self.with_root(expr);
                write!(f, "{expr}.explode(")?;
                match (options.empty_as_null, options.keep_nulls) {
                    (true, true) => {},
                    (true, false) => f.write_str("keep_nulls=false")?,
                    (false, true) => f.write_str("empty_as_null=false")?,
                    (false, false) => f.write_str("empty_as_null=false, keep_nulls=false")?,
                }
                f.write_char(')')
            },
            Column(name) => write!(f, "col(\"{name}\")"),
            #[cfg(feature = "dtype-struct")]
            StructField(name) => write!(f, "field(\"{name}\")"),
            Literal(v) => write!(f, "{v:?}"),
            BinaryExpr { left, op, right } => {
                let left = self.with_root(left);
                let right = self.with_root(right);
                write!(f, "[({left}) {op:?} ({right})]")
            },
            Sort { expr, options } => {
                let expr = self.with_root(expr);
                if options.descending {
                    write!(f, "{expr}.sort(desc)")
                } else {
                    write!(f, "{expr}.sort(asc)")
                }
            },
            SortBy {
                expr,
                by,
                sort_options,
            } => {
                let expr = self.with_root(expr);
                let by = self.with_slice(by);
                write!(f, "{expr}.sort_by(by={by}, sort_option={sort_options:?})",)
            },
            Filter { input, by } => {
                let input = self.with_root(input);
                let by = self.with_root(by);

                write!(f, "{input}.filter({by})")
            },
            Gather {
                expr,
                idx,
                returns_scalar,
                null_on_oob: _,
            } => {
                let expr = self.with_root(expr);
                let idx = self.with_root(idx);
                expr.fmt(f)?;

                if *returns_scalar {
                    write!(f, ".get({idx})")
                } else {
                    write!(f, ".gather({idx})")
                }
            },
            Agg(agg) => {
                use IRAggExpr::*;
                match agg {
                    Min {
                        input,
                        propagate_nans,
                    } => {
                        self.with_root(input).fmt(f)?;
                        if *propagate_nans {
                            write!(f, ".nan_min()")
                        } else {
                            write!(f, ".min()")
                        }
                    },
                    Max {
                        input,
                        propagate_nans,
                    } => {
                        self.with_root(input).fmt(f)?;
                        if *propagate_nans {
                            write!(f, ".nan_max()")
                        } else {
                            write!(f, ".max()")
                        }
                    },
                    Median(expr) => write!(f, "{}.median()", self.with_root(expr)),
                    Mean(expr) => write!(f, "{}.mean()", self.with_root(expr)),
                    First(expr) => write!(f, "{}.first()", self.with_root(expr)),
                    FirstNonNull(expr) => write!(f, "{}.first_non_null()", self.with_root(expr)),
                    Last(expr) => write!(f, "{}.last()", self.with_root(expr)),
                    LastNonNull(expr) => write!(f, "{}.last_non_null()", self.with_root(expr)),
                    Item { input, allow_empty } => {
                        self.with_root(input).fmt(f)?;
                        if *allow_empty {
                            write!(f, ".item(allow_empty=true)")
                        } else {
                            write!(f, ".item()")
                        }
                    },
                    Implode(expr) => write!(f, "{}.implode()", self.with_root(expr)),
                    NUnique(expr) => write!(f, "{}.n_unique()", self.with_root(expr)),
                    Sum(expr) => write!(f, "{}.sum()", self.with_root(expr)),
                    AggGroups(expr) => write!(f, "{}.groups()", self.with_root(expr)),
                    Count {
                        input,
                        include_nulls: false,
                    } => write!(f, "{}.count()", self.with_root(input)),
                    Count {
                        input,
                        include_nulls: true,
                    } => write!(f, "{}.len()", self.with_root(input)),
                    Var(expr, _) => write!(f, "{}.var()", self.with_root(expr)),
                    Std(expr, _) => write!(f, "{}.std()", self.with_root(expr)),
                    Quantile {
                        expr,
                        quantile,
                        method,
                    } => write!(
                        f,
                        "{}.quantile({}, interpolation='{}')",
                        self.with_root(expr),
                        self.with_root(quantile),
                        <&'static str>::from(method),
                    ),
                }
            },
            Cast {
                expr,
                dtype,
                options,
            } => {
                self.with_root(expr).fmt(f)?;
                if options.is_strict() {
                    write!(f, ".strict_cast({dtype:?})")
                } else {
                    write!(f, ".cast({dtype:?})")
                }
            },
            Ternary {
                predicate,
                truthy,
                falsy,
            } => {
                let predicate = self.with_root(predicate);
                let truthy = self.with_root(truthy);
                let falsy = self.with_root(falsy);
                write!(f, "when({predicate}).then({truthy}).otherwise({falsy})",)
            },
            Function {
                input, function, ..
            } => {
                let fst = self.with_root(&input[0]);
                fst.fmt(f)?;
                if input.len() >= 2 {
                    write!(f, ".{function}({})", self.with_slice(&input[1..]))
                } else {
                    write!(f, ".{function}()")
                }
            },
            AnonymousFunction { input, fmt_str, .. } | AnonymousAgg { input, fmt_str, .. } => {
                let fst = self.with_root(&input[0]);
                fst.fmt(f)?;
                if input.len() >= 2 {
                    write!(f, ".{fmt_str}({})", self.with_slice(&input[1..]))
                } else {
                    write!(f, ".{fmt_str}()")
                }
            },
            Eval {
                expr,
                evaluation,
                variant,
            } => {
                let expr = self.with_root(expr);
                let evaluation = self.with_root(evaluation);
                match variant {
                    EvalVariant::List => write!(f, "{expr}.list.eval({evaluation})"),
                    EvalVariant::ListAgg => write!(f, "{expr}.list.agg({evaluation})"),
                    EvalVariant::Array { as_list: false } => {
                        write!(f, "{expr}.array.eval({evaluation})")
                    },
                    EvalVariant::Array { as_list: true } => {
                        write!(f, "{expr}.array.eval({evaluation}, as_list=true)")
                    },
                    EvalVariant::ArrayAgg => write!(f, "{expr}.array.agg({evaluation})"),
                    EvalVariant::Cumulative { min_samples } => write!(
                        f,
                        "{expr}.cumulative_eval({evaluation}, min_samples={min_samples})"
                    ),
                }
            },
            #[cfg(feature = "dtype-struct")]
            StructEval { expr, evaluation } => {
                let expr = self.with_root(expr);
                let evaluation = self.with_slice(evaluation);
                write!(f, "{expr}.struct.with_fields({evaluation})")
            },
            Slice {
                input,
                offset,
                length,
            } => {
                let input = self.with_root(input);
                let offset = self.with_root(offset);
                let length = self.with_root(length);

                write!(f, "{input}.slice(offset={offset}, length={length})")
            },
        }?;

        match self.output_name {
            OutputName::None => {},
            OutputName::LiteralLhs(_) => {},
            OutputName::ColumnLhs(_) => {},
            #[cfg(feature = "dtype-struct")]
            OutputName::Field(_) => {},
            OutputName::Alias(name) => {
                if root.to_name(self.expr_arena) != name {
                    write!(f, r#".alias("{name}")"#)?;
                }
            },
        }

        Ok(())
    }
}

impl fmt::Debug for ExprIRDisplay<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

pub(crate) struct ColumnsDisplay<'a>(pub(crate) &'a Schema);

impl fmt::Display for ColumnsDisplay<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let len = self.0.len();
        let mut iter_names = self.0.iter_names().enumerate();

        const MAX_LEN: usize = 32;
        const ADD_PER_ITEM: usize = 4;

        let mut current_len = 0;

        if let Some((_, fst)) = iter_names.next() {
            write!(f, "\"{fst}\"")?;

            current_len += fst.len() + ADD_PER_ITEM;
        }

        for (i, col) in iter_names {
            current_len += col.len() + ADD_PER_ITEM;

            if current_len > MAX_LEN {
                write!(f, ", ... {} other ", len - i)?;
                if len - i == 1 {
                    f.write_str("column")?;
                } else {
                    f.write_str("columns")?;
                }

                break;
            }

            write!(f, ", \"{col}\"")?;
        }

        Ok(())
    }
}

impl fmt::Debug for Operator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

impl fmt::Debug for LiteralValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use LiteralValue::*;

        match self {
            Self::Scalar(sc) => write!(f, "{}", sc.value()),
            Self::Series(s) => {
                let name = s.name();
                if name.is_empty() {
                    write!(f, "Series")
                } else {
                    write!(f, "Series[{name}]")
                }
            },
            Range(range) => fmt::Debug::fmt(range, f),
            Dyn(d) => fmt::Debug::fmt(d, f),
        }
    }
}

impl fmt::Debug for DynLiteralValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        match self {
            Self::Int(v) => write!(f, "dyn int: {v}"),
            Self::Float(v) => write!(f, "dyn float: {v}"),
            Self::Str(v) => write!(f, "dyn str: {v}"),
            Self::List(_) => todo!(),
        }
    }
}

impl fmt::Debug for RangeLiteralValue {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "range({}, {})", self.low, self.high)
    }
}

pub fn write_ir_non_recursive(
    f: &mut dyn fmt::Write,
    ir: &IR,
    expr_arena: &Arena<AExpr>,
    output_schema: &Schema,
    indent: usize,
) -> fmt::Result {
    match ir {
        #[cfg(feature = "python")]
        IR::PythonScan { options } => {
            let total_columns = options.schema.len();
            let n_columns = options
                .with_columns
                .as_ref()
                .map(|s| s.len() as i64)
                .unwrap_or(-1);

            let predicate = match &options.predicate {
                PythonPredicate::Polars(e) => Some(e.display(expr_arena)),
                PythonPredicate::PyArrow(_) => None,
                PythonPredicate::None => None,
            };

            write_scan(
                f,
                "PYTHON",
                &ScanSources::default(),
                indent,
                n_columns,
                total_columns,
                None,
                &predicate,
                options
                    .n_rows
                    .map(|len| polars_utils::slice_enum::Slice::Positive { offset: 0, len }),
                None,
                None,
            )
        },
        IR::Slice {
            input: _,
            offset,
            len,
        } => {
            write!(f, "{:indent$}SLICE[offset: {offset}, len: {len}]", "")
        },
        IR::Filter {
            input: _,
            predicate,
        } => {
            let predicate = predicate.display(expr_arena);
            // this one is writeln because we don't increase indent (which inserts a line)
            write!(f, "{:indent$}FILTER {predicate}", "")?;
            write!(f, "\n{:indent$}FROM", "")
        },
        IR::Scan {
            sources,
            file_info,
            predicate,
            predicate_file_skip_applied: _,
            scan_type,
            unified_scan_args,
            hive_parts: _,
            output_schema: _,
        } => {
            let n_columns = unified_scan_args
                .projection
                .as_ref()
                .map(|columns| columns.len() as i64)
                .unwrap_or(-1);

            let row_estimation = if file_info.row_estimation.1 != usize::MAX {
                Some(file_info.row_estimation.1)
            } else {
                None
            };

            let predicate = predicate.as_ref().map(|p| p.display(expr_arena));

            write_scan(
                f,
                (&**scan_type).into(),
                sources,
                indent,
                n_columns,
                file_info.schema.len(),
                row_estimation,
                &predicate,
                unified_scan_args.pre_slice.clone(),
                unified_scan_args.row_index.as_ref(),
                unified_scan_args.deletion_files.as_ref(),
            )
        },
        IR::DataFrameScan {
            df: _,
            schema,
            output_schema,
        } => {
            let total_columns = schema.len();
            let (n_columns, projected) = if let Some(schema) = output_schema {
                (
                    format!("{}", schema.len()),
                    format_list_truncated!(schema.iter_names(), 4, '"'),
                )
            } else {
                ("*".to_string(), "".to_string())
            };
            write!(
                f,
                "{:indent$}DF {}; PROJECT{} {}/{} COLUMNS",
                "",
                format_list_truncated!(schema.iter_names(), 4, '"'),
                projected,
                n_columns,
                total_columns,
            )
        },
        IR::SimpleProjection { input: _, columns } => {
            let num_columns = columns.as_ref().len();
            let total_columns = output_schema.len();

            let columns = ColumnsDisplay(columns.as_ref());
            write!(
                f,
                "{:indent$}simple π {num_columns}/{total_columns} [{columns}]",
                ""
            )
        },
        IR::Select {
            input: _,
            expr,
            schema: _,
            options: _,
        } => {
            // @NOTE: Maybe there should be a clear delimiter here?
            let exprs = ExprIRSliceDisplay {
                exprs: expr,
                expr_arena,
            };
            write!(f, "{:indent$}SELECT {exprs}", "")?;
            Ok(())
        },
        IR::Sort {
            input: _,
            by_column,
            slice,
            sort_options,
        } => {
            write!(f, "{:indent$}", "")?;

            f.write_str("SORT BY ")?;

            if slice.is_some()
                || sort_options.maintain_order
                || sort_options.descending.iter().any(|v| *v)
                || sort_options.nulls_last.iter().any(|v| *v)
            {
                f.write_char('[')?;

                let mut comma = false;
                if let Some((o, l, dyn_pred)) = slice {
                    if let Some(dyn_pred) = &dyn_pred {
                        write!(f, "slice: ({o}, {l}, {dyn_pred:?})")?;
                    } else {
                        write!(f, "slice: ({o}, {l})")?;
                    }
                    comma = true;
                }
                if sort_options.maintain_order {
                    if comma {
                        f.write_str(", ")?;
                    }
                    f.write_str("maintain_order: true")?;
                    comma = true;
                }
                if sort_options.descending.iter().any(|v| *v) {
                    if comma {
                        f.write_str(", ")?;
                    }
                    write!(f, "descending: {:?}", sort_options.descending.as_slice())?;
                    comma = true;
                }
                if sort_options.nulls_last.iter().any(|v| *v) {
                    if comma {
                        f.write_str(", ")?;
                    }
                    write!(f, "nulls_last: {:?}", sort_options.nulls_last.as_slice())?;
                }

                f.write_str("] ")?;
            }

            write!(
                f,
                "{}",
                ExprIRSliceDisplay {
                    exprs: by_column,
                    expr_arena,
                }
            )
        },
        IR::Cache { input: _, id } => write!(f, "{:indent$}CACHE[id: {id}]", ""),
        IR::GroupBy {
            input: _,
            keys,
            aggs,
            schema: _,
            maintain_order,
            options: _,
            apply,
        } => write_group_by(
            f,
            indent,
            expr_arena,
            keys,
            aggs,
            apply.as_ref(),
            *maintain_order,
        ),
        IR::Join {
            input_left: _,
            input_right: _,
            schema: _,
            left_on,
            right_on,
            options,
        } => {
            let left_on = ExprIRSliceDisplay {
                exprs: left_on,
                expr_arena,
            };
            let right_on = ExprIRSliceDisplay {
                exprs: right_on,
                expr_arena,
            };

            // Fused cross + filter (show as nested loop join)
            if let Some(JoinTypeOptionsIR::CrossAndFilter { predicate }) = &options.options {
                let predicate = predicate.display(expr_arena);
                write!(f, "{:indent$}NESTED_LOOP JOIN ON {predicate}", "")?;
            } else {
                let how = &options.args.how;
                write!(f, "{:indent$}{how} JOIN", "")?;
                write!(f, "\n{:indent$}LEFT PLAN ON: {left_on}", "")?;
                write!(f, "\n{:indent$}RIGHT PLAN ON: {right_on}", "")?;
            }

            Ok(())
        },
        IR::HStack {
            input: _,
            exprs,
            schema: _,
            options: _,
        } => {
            // @NOTE: Maybe there should be a clear delimiter here?
            let exprs = ExprIRSliceDisplay { exprs, expr_arena };

            write!(f, "{:indent$} WITH_COLUMNS:", "",)?;
            write!(f, "\n{:indent$} {exprs} ", "")
        },
        IR::Distinct { input: _, options } => {
            write!(
                f,
                "{:indent$}UNIQUE[maintain_order: {:?}, keep_strategy: {:?}] BY {:?}",
                "", options.maintain_order, options.keep_strategy, options.subset
            )
        },
        IR::MapFunction { input: _, function } => write!(f, "{:indent$}{function}", ""),
        IR::Union { inputs: _, options } => {
            let name = if let Some(slice) = options.slice {
                format!("SLICED UNION: {slice:?}")
            } else {
                "UNION".to_string()
            };
            write!(f, "{:indent$}{name}", "")
        },
        IR::HConcat {
            inputs: _,
            schema: _,
            options: _,
        } => write!(f, "{:indent$}HCONCAT", ""),
        IR::ExtContext {
            input: _,
            contexts: _,
            schema: _,
        } => write!(f, "{:indent$}EXTERNAL_CONTEXT", ""),
        IR::Sink { input: _, payload } => {
            let name = match payload {
                SinkTypeIR::Memory => "SINK (memory)",
                SinkTypeIR::Callback { .. } => "SINK (callback)",
                SinkTypeIR::File { .. } => "SINK (file)",
                SinkTypeIR::Partitioned { .. } => "SINK (partition)",
            };
            write!(f, "{:indent$}{name}", "")
        },
        IR::SinkMultiple { inputs: _ } => write!(f, "{:indent$}SINK_MULTIPLE", ""),
        #[cfg(feature = "merge_sorted")]
        IR::MergeSorted {
            input_left: _,
            input_right: _,
            key,
        } => write!(f, "{:indent$}MERGE SORTED ON '{key}'", ""),
        IR::Invalid => write!(f, "{:indent$}INVALID", ""),
    }
}

pub fn write_group_by(
    f: &mut dyn fmt::Write,
    indent: usize,
    expr_arena: &Arena<AExpr>,
    keys: &[ExprIR],
    aggs: &[ExprIR],
    apply: Option<&PlanCallback<DataFrame, DataFrame>>,
    maintain_order: bool,
) -> fmt::Result {
    let sub_indent = indent + INDENT_INCREMENT;
    let keys = ExprIRSliceDisplay {
        exprs: keys,
        expr_arena,
    };
    write!(
        f,
        "{:indent$}AGGREGATE[maintain_order: {}]",
        "", maintain_order
    )?;
    if apply.is_some() {
        write!(f, "\n{:sub_indent$}MAP_GROUPS BY {keys}", "")?;
        write!(f, "\n{:sub_indent$}FROM", "")?;
    } else {
        let aggs = ExprIRSliceDisplay {
            exprs: aggs,
            expr_arena,
        };
        write!(f, "\n{:sub_indent$}{aggs} BY {keys}", "")?;
    }

    Ok(())
}
