use std::fmt::Write;

use polars_plan::dsl::PartitionStrategyIR;
use polars_plan::plans::expr_ir::ExprIR;
use polars_plan::plans::{AExpr, EscapeLabel};
use polars_plan::prelude::FileWriteFormat;
use polars_time::ClosedWindow;
#[cfg(feature = "dynamic_group_by")]
use polars_time::DynamicGroupOptions;
use polars_utils::arena::Arena;
use polars_utils::slice_enum::Slice;
use slotmap::{Key, SecondaryMap, SlotMap};

use super::{PhysNode, PhysNodeKey, PhysNodeKind};
use crate::physical_plan::ZipBehavior;

/// A style of a graph node.
pub enum NodeStyle {
    InMemoryFallback,
    MemoryIntensive,
    Generic,
}

impl NodeStyle {
    const COLOR_IN_MEM_FALLBACK: &str = "0.0 0.3 1.0"; // Pastel red
    const COLOR_MEM_INTENSIVE: &str = "0.16 0.3 1.0"; // Pastel yellow

    /// Returns a style for a node kind.
    pub fn for_node_kind(kind: &PhysNodeKind) -> Self {
        use PhysNodeKind as K;
        match kind {
            K::InMemoryMap { .. } | K::InMemoryJoin { .. } => Self::InMemoryFallback,
            K::InMemorySource { .. }
            | K::InputIndependentSelect { .. }
            | K::NegativeSlice { .. }
            | K::InMemorySink { .. }
            | K::Sort { .. }
            | K::GroupBy { .. }
            | K::EquiJoin { .. }
            | K::SemiAntiJoin { .. }
            | K::Multiplexer { .. } => Self::MemoryIntensive,
            #[cfg(feature = "merge_sorted")]
            K::MergeSorted { .. } => Self::MemoryIntensive,
            _ => Self::Generic,
        }
    }

    /// Returns extra styling attributes (if any) for the graph node.
    pub fn node_attrs(&self) -> Option<String> {
        match self {
            Self::InMemoryFallback => Some(format!(
                "style=filled,fillcolor=\"{}\"",
                Self::COLOR_IN_MEM_FALLBACK
            )),
            Self::MemoryIntensive => Some(format!(
                "style=filled,fillcolor=\"{}\"",
                Self::COLOR_MEM_INTENSIVE
            )),
            Self::Generic => None,
        }
    }

    /// Returns a legend explaining the node style meaning.
    pub fn legend() -> String {
        format!(
            "fontname=\"Helvetica\"\nfontsize=\"10\"\nlabelloc=\"b\"\nlabel=<<BR/><BR/><B>Legend</B><BR/><BR/>◯ streaming engine node <FONT COLOR=\"{}\">⬤</FONT> potentially memory-intensive node <FONT COLOR=\"{}\">⬤</FONT> in-memory engine fallback>",
            Self::COLOR_MEM_INTENSIVE,
            Self::COLOR_IN_MEM_FALLBACK,
        )
    }
}

fn escape_graphviz(s: &str) -> String {
    s.replace('\\', "\\\\")
        .replace('\n', "\\n")
        .replace('"', "\\\"")
}

fn fmt_expr(f: &mut dyn Write, expr: &ExprIR, expr_arena: &Arena<AExpr>) -> std::fmt::Result {
    // Remove the alias to make the display better
    let without_alias = ExprIR::from_node(expr.node(), expr_arena);
    write!(
        f,
        "{} = {}",
        expr.output_name(),
        without_alias.display(expr_arena)
    )
}

pub enum FormatExprStyle {
    Select,
    NoAliases,
}

pub fn fmt_exprs_to_label(
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    style: FormatExprStyle,
) -> String {
    let mut buffer = String::new();
    let mut f = EscapeLabel(&mut buffer);
    fmt_exprs(&mut f, exprs, expr_arena, style);
    buffer
}

pub fn fmt_exprs(
    f: &mut dyn Write,
    exprs: &[ExprIR],
    expr_arena: &Arena<AExpr>,
    style: FormatExprStyle,
) {
    if matches!(style, FormatExprStyle::Select) {
        let mut formatted = Vec::new();

        let mut max_name_width = 0;
        let mut max_expr_width = 0;

        for e in exprs {
            let mut name = String::new();
            let mut expr = String::new();

            // Remove the alias to make the display better
            let without_alias = ExprIR::from_node(e.node(), expr_arena);

            write!(name, "{}", e.output_name()).unwrap();
            write!(expr, "{}", without_alias.display(expr_arena)).unwrap();

            max_name_width = max_name_width.max(name.chars().count());
            max_expr_width = max_expr_width.max(expr.chars().count());

            formatted.push((name, expr));
        }

        for (name, expr) in formatted {
            writeln!(f, "{name:>max_name_width$} = {expr:<max_expr_width$}").unwrap();
        }
    } else {
        let Some(e) = exprs.first() else {
            return;
        };

        fmt_expr(f, e, expr_arena).unwrap();

        for e in &exprs[1..] {
            f.write_str("\n").unwrap();
            fmt_expr(f, e, expr_arena).unwrap();
        }
    }
}

#[recursive::recursive]
fn visualize_plan_rec(
    node_key: PhysNodeKey,
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
    visited: &mut SecondaryMap<PhysNodeKey, ()>,
    out: &mut Vec<String>,
) {
    if visited.contains_key(node_key) {
        return;
    }
    visited.insert(node_key, ());

    let kind = &phys_sm[node_key].kind;

    use std::slice::from_ref;
    let (label, inputs) = match kind {
        PhysNodeKind::InMemorySource {
            df,
            disable_morsel_split: _,
        } => (
            format!(
                "in-memory-source\\ncols: {}",
                df.get_column_names_owned().join(", ")
            ),
            &[][..],
        ),
        #[cfg(feature = "python")]
        PhysNodeKind::PythonScan { .. } => ("python-scan".to_string(), &[][..]),
        PhysNodeKind::SinkMultiple { sinks } => {
            for sink in sinks {
                visualize_plan_rec(*sink, phys_sm, expr_arena, visited, out);
            }
            return;
        },
        PhysNodeKind::Select {
            input,
            selectors,
            extend_original,
        } => {
            let label = if *extend_original {
                "with-columns"
            } else {
                "select"
            };
            (
                format!(
                    "{label}\\n{}",
                    fmt_exprs_to_label(selectors, expr_arena, FormatExprStyle::Select)
                ),
                from_ref(input),
            )
        },
        PhysNodeKind::WithRowIndex {
            input,
            name,
            offset,
        } => (
            format!("with-row-index\\nname: {name}\\noffset: {offset:?}"),
            from_ref(input),
        ),
        PhysNodeKind::InputIndependentSelect { selectors } => (
            format!(
                "input-independent-select\\n{}",
                fmt_exprs_to_label(selectors, expr_arena, FormatExprStyle::Select)
            ),
            &[][..],
        ),
        PhysNodeKind::Reduce { input, exprs } => (
            format!(
                "reduce\\n{}",
                fmt_exprs_to_label(exprs, expr_arena, FormatExprStyle::Select)
            ),
            from_ref(input),
        ),
        PhysNodeKind::StreamingSlice {
            input,
            offset,
            length,
        } => (
            format!("slice\\noffset: {offset}, length: {length}"),
            from_ref(input),
        ),
        PhysNodeKind::NegativeSlice {
            input,
            offset,
            length,
        } => (
            format!("slice\\noffset: {offset}, length: {length}"),
            from_ref(input),
        ),
        PhysNodeKind::DynamicSlice {
            input,
            offset,
            length,
        } => ("slice".to_owned(), &[*input, *offset, *length][..]),
        PhysNodeKind::Shift {
            input,
            offset,
            fill: Some(fill),
        } => ("shift".to_owned(), &[*input, *offset, *fill][..]),
        PhysNodeKind::Shift {
            input,
            offset,
            fill: None,
        } => ("shift".to_owned(), &[*input, *offset][..]),
        PhysNodeKind::Filter { input, predicate } => (
            format!(
                "filter\\n{}",
                fmt_exprs_to_label(from_ref(predicate), expr_arena, FormatExprStyle::Select)
            ),
            from_ref(input),
        ),
        PhysNodeKind::SimpleProjection { input, columns } => (
            format!("select\\ncols: {}", columns.join(", ")),
            from_ref(input),
        ),
        PhysNodeKind::InMemorySink { input } => ("in-memory-sink".to_string(), from_ref(input)),
        PhysNodeKind::CallbackSink { input, .. } => ("callback-sink".to_string(), from_ref(input)),
        PhysNodeKind::FileSink { input, options } => match options.file_format {
            #[cfg(feature = "parquet")]
            FileWriteFormat::Parquet(_) => ("parquet-sink".to_string(), from_ref(input)),
            #[cfg(feature = "ipc")]
            FileWriteFormat::Ipc(_) => ("ipc-sink".to_string(), from_ref(input)),
            #[cfg(feature = "csv")]
            FileWriteFormat::Csv(_) => ("csv-sink".to_string(), from_ref(input)),
            #[cfg(feature = "json")]
            FileWriteFormat::NDJson(_) => ("ndjson-sink".to_string(), from_ref(input)),
            #[allow(unreachable_patterns)]
            _ => todo!(),
        },
        PhysNodeKind::PartitionedSink { input, options } => {
            let variant = match options.partition_strategy {
                PartitionStrategyIR::Keyed { .. } => "partition-keyed",
                PartitionStrategyIR::FileSize => "partition-file-size",
            };

            match options.file_format {
                #[cfg(feature = "parquet")]
                FileWriteFormat::Parquet(_) => (format!("{variant}[parquet]"), from_ref(input)),
                #[cfg(feature = "ipc")]
                FileWriteFormat::Ipc(_) => (format!("{variant}[ipc]"), from_ref(input)),
                #[cfg(feature = "csv")]
                FileWriteFormat::Csv(_) => (format!("{variant}[csv]"), from_ref(input)),
                #[cfg(feature = "json")]
                FileWriteFormat::NDJson(_) => (format!("{variant}[ndjson]"), from_ref(input)),
                #[allow(unreachable_patterns)]
                _ => todo!(),
            }
        },
        PhysNodeKind::InMemoryMap {
            input,
            map: _,
            format_str,
        } => {
            let mut label = String::new();
            label.push_str("in-memory-map");
            if let Some(format_str) = format_str {
                label.write_str("\\n").unwrap();

                let mut f = EscapeLabel(&mut label);
                f.write_str(format_str).unwrap();
            }
            (label, from_ref(input))
        },
        PhysNodeKind::Map {
            input,
            map: _,
            format_str,
        } => {
            let mut label = String::new();
            label.push_str("map");
            if let Some(format_str) = format_str {
                label.push_str("\\n");

                let mut f = EscapeLabel(&mut label);
                f.write_str(format_str).unwrap();
            }
            (label, from_ref(input))
        },
        PhysNodeKind::SortedGroupBy {
            input,
            key,
            aggs,
            slice,
        } => {
            let mut s = String::new();
            s.push_str("sorted-group-by\\n");
            let f = &mut s;
            write!(f, "key: {key}\\n").unwrap();
            if let Some((offset, length)) = slice {
                write!(f, "slice: {offset}, {length}\\n").unwrap();
            }
            write!(
                f,
                "aggs:\\n{}",
                fmt_exprs_to_label(aggs, expr_arena, FormatExprStyle::Select)
            )
            .unwrap();

            (s, from_ref(input))
        },
        PhysNodeKind::Sort {
            input,
            by_column,
            slice: _,
            sort_options: _,
        } => (
            format!(
                "sort\\n{}",
                fmt_exprs_to_label(by_column, expr_arena, FormatExprStyle::NoAliases)
            ),
            from_ref(input),
        ),
        PhysNodeKind::TopK {
            input,
            k,
            by_column,
            reverse,
            nulls_last: _,
        } => {
            let name = if reverse.iter().all(|r| *r) {
                "bottom-k"
            } else {
                "top-k"
            };
            (
                format!(
                    "{name}\\n{}",
                    fmt_exprs_to_label(by_column, expr_arena, FormatExprStyle::NoAliases)
                ),
                &[*input, *k][..],
            )
        },
        PhysNodeKind::Repeat { value, repeats } => ("repeat".to_owned(), &[*value, *repeats][..]),
        #[cfg(feature = "cum_agg")]
        PhysNodeKind::CumAgg { input, kind } => {
            use crate::nodes::cum_agg::CumAggKind;

            (
                format!(
                    "cum_{}",
                    match kind {
                        CumAggKind::Min => "min",
                        CumAggKind::Max => "max",
                        CumAggKind::Sum => "sum",
                        CumAggKind::Count => "count",
                        CumAggKind::Prod => "prod",
                        CumAggKind::Mean => "mean",
                    }
                ),
                &[*input][..],
            )
        },
        PhysNodeKind::GatherEvery { input, n, offset } => (
            format!("gather_every\\nn: {n}, offset: {offset}"),
            &[*input][..],
        ),
        PhysNodeKind::Rle(input) => ("rle".to_owned(), &[*input][..]),
        PhysNodeKind::RleId(input) => ("rle_id".to_owned(), &[*input][..]),
        PhysNodeKind::PeakMinMax { input, is_peak_max } => (
            if *is_peak_max { "peak_max" } else { "peak_min" }.to_owned(),
            &[*input][..],
        ),
        PhysNodeKind::OrderedUnion { inputs } => ("ordered-union".to_string(), inputs.as_slice()),
        PhysNodeKind::UnorderedUnion { inputs } => {
            ("unordered-union".to_string(), inputs.as_slice())
        },
        PhysNodeKind::Zip {
            inputs,
            zip_behavior,
        } => {
            let label = match zip_behavior {
                ZipBehavior::NullExtend => "zip-null-extend",
                ZipBehavior::Broadcast => "zip-broadcast",
                ZipBehavior::Strict => "zip-strict",
            };
            (label.to_string(), inputs.as_slice())
        },
        PhysNodeKind::Multiplexer { input } => ("multiplexer".to_string(), from_ref(input)),
        PhysNodeKind::MultiScan {
            scan_sources,
            file_reader_builder,
            cloud_options: _,
            file_projection_builder,
            output_schema,
            row_index,
            pre_slice,
            predicate,
            predicate_file_skip_applied: _,
            hive_parts,
            include_file_paths,
            cast_columns_policy: _,
            missing_columns_policy: _,
            forbid_extra_columns: _,
            deletion_files,
            table_statistics: _,
            file_schema: _,
            disable_morsel_split: _,
        } => {
            let mut out = format!("multi-scan[{}]", file_reader_builder.reader_name());
            let mut f = EscapeLabel(&mut out);

            write!(f, "\n{} source", scan_sources.len()).unwrap();

            if scan_sources.len() != 1 {
                write!(f, "s").unwrap();
            }

            write!(
                f,
                "\nproject: {} total, {} from file",
                output_schema.len(),
                file_projection_builder.num_projections(),
            )
            .unwrap();

            if let Some(ri) = row_index {
                write!(f, "\nrow index: name: {}, offset: {:?}", ri.name, ri.offset).unwrap();
            }

            if let Some(col_name) = include_file_paths {
                write!(f, "\nfile path column: {col_name}").unwrap();
            }

            if let Some(pre_slice) = pre_slice {
                write!(f, "\nslice: offset: ").unwrap();

                match pre_slice {
                    Slice::Positive { offset, len: _ } => write!(f, "{}", *offset),
                    Slice::Negative {
                        offset_from_end,
                        len: _,
                    } => write!(f, "-{}", *offset_from_end),
                }
                .unwrap();

                write!(f, ", len: {}", pre_slice.len()).unwrap()
            }

            if let Some(predicate) = predicate {
                write!(f, "\nfilter: {}", predicate.display(expr_arena)).unwrap();
            }

            if let Some(v) = hive_parts.as_ref().map(|h| h.df().width()) {
                write!(f, "\nhive: {v} column").unwrap();

                if v != 1 {
                    write!(f, "s").unwrap();
                }
            }

            if let Some(deletion_files) = deletion_files {
                write!(f, "\n{deletion_files}").unwrap();
            }

            (out, &[][..])
        },
        PhysNodeKind::GroupBy {
            inputs,
            key_per_input,
            aggs_per_input,
        } => {
            let mut out = String::from("group-by");
            for (key, aggs) in key_per_input.iter().zip(aggs_per_input) {
                write!(
                    &mut out,
                    "\\nkey:\\n{}\\naggs:\\n{}",
                    fmt_exprs_to_label(key, expr_arena, FormatExprStyle::Select),
                    fmt_exprs_to_label(aggs, expr_arena, FormatExprStyle::Select)
                )
                .ok();
            }
            (out, inputs.as_slice())
        },
        #[cfg(feature = "dynamic_group_by")]
        PhysNodeKind::DynamicGroupBy {
            input,
            options,
            aggs,
            slice,
        } => {
            use polars_time::prelude::{Label, StartBy};

            let DynamicGroupOptions {
                index_column,
                every,
                period,
                offset,
                label,
                include_boundaries,
                closed_window,
                start_by,
            } = options;
            let mut s = String::new();
            let f = &mut s;
            f.write_str("dynamic-group-by\\n").unwrap();
            write!(f, "index column: {index_column}\\n").unwrap();
            write!(f, "every: {every}").unwrap();
            if every != period {
                write!(f, ", period: {period}").unwrap();
            }
            if !offset.is_zero() {
                write!(f, ", offset: {offset}").unwrap();
            }
            f.write_str("\\n").unwrap();
            if *label != Label::Left {
                write!(f, "label: {}\\n", <&'static str>::from(label)).unwrap();
            }
            if *include_boundaries {
                write!(f, "include_boundaries: true\\n").unwrap();
            }
            if *start_by != StartBy::WindowBound {
                write!(f, "start_by: {}\\n", <&'static str>::from(start_by)).unwrap();
            }
            if *closed_window != ClosedWindow::Left {
                write!(
                    f,
                    "closed_window: {}\\n",
                    <&'static str>::from(closed_window)
                )
                .unwrap();
            }
            if let Some((offset, length)) = slice {
                write!(f, "slice: {offset}, {length}\\n").unwrap();
            }
            write!(
                f,
                "aggs:\\n{}",
                fmt_exprs_to_label(aggs, expr_arena, FormatExprStyle::Select)
            )
            .unwrap();

            (s, from_ref(input))
        },
        #[cfg(feature = "dynamic_group_by")]
        PhysNodeKind::RollingGroupBy {
            input,
            index_column,
            period,
            offset,
            closed,
            slice,
            aggs,
        } => {
            let mut s = String::new();
            let f = &mut s;
            f.write_str("rolling-group-by\\n").unwrap();
            write!(f, "index column: {index_column}\\n").unwrap();
            write!(f, "period: {period}, offset: {offset}\\n").unwrap();
            write!(f, "closed: {}\\n", <&'static str>::from(*closed)).unwrap();
            if let Some((offset, length)) = slice {
                write!(f, "slice: {offset}, {length}\\n").unwrap();
            }
            write!(
                f,
                "aggs:\\n{}",
                fmt_exprs_to_label(aggs, expr_arena, FormatExprStyle::Select)
            )
            .unwrap();

            (s, from_ref(input))
        },
        PhysNodeKind::MergeJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            ..
        } => {
            let mut label = "merge-join".to_string();
            let how: &'static str = (&args.how).into();
            write!(
                label,
                r"\nleft_on:\n{}",
                left_on
                    .iter()
                    .map(|s| escape_graphviz(&s[..]))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .unwrap();
            write!(
                label,
                r"\nright_on:\n{}",
                right_on
                    .iter()
                    .map(|s| escape_graphviz(&s[..]))
                    .collect::<Vec<_>>()
                    .join("\n"),
            )
            .unwrap();
            write!(label, r"\nhow: {}", escape_graphviz(how)).unwrap();
            if args.nulls_equal {
                write!(label, r"\njoin-nulls").unwrap();
            }
            (label, &[*input_left, *input_right][..])
        },
        PhysNodeKind::InMemoryJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            ..
        }
        | PhysNodeKind::EquiJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
        }
        | PhysNodeKind::SemiAntiJoin {
            input_left,
            input_right,
            left_on,
            right_on,
            args,
            output_bool: _,
        } => {
            let label = match phys_sm[node_key].kind {
                PhysNodeKind::MergeJoin { .. } => "merge-join",
                PhysNodeKind::EquiJoin { .. } => "equi-join",
                PhysNodeKind::InMemoryJoin { .. } => "in-memory-join",
                PhysNodeKind::CrossJoin { .. } => "cross-join",
                PhysNodeKind::SemiAntiJoin {
                    output_bool: false, ..
                } if args.how.is_semi() => "semi-join",
                PhysNodeKind::SemiAntiJoin {
                    output_bool: false, ..
                } if args.how.is_anti() => "anti-join",
                PhysNodeKind::SemiAntiJoin {
                    output_bool: true, ..
                } if args.how.is_semi() => "is-in",
                PhysNodeKind::SemiAntiJoin {
                    output_bool: true, ..
                } if args.how.is_anti() => "is-not-in",
                _ => unreachable!(),
            };
            let mut label = label.to_string();
            write!(
                label,
                r"\nleft_on:\n{}",
                fmt_exprs_to_label(left_on, expr_arena, FormatExprStyle::NoAliases)
            )
            .unwrap();
            write!(
                label,
                r"\nright_on:\n{}",
                fmt_exprs_to_label(right_on, expr_arena, FormatExprStyle::NoAliases)
            )
            .unwrap();
            if args.how.is_equi() {
                write!(
                    label,
                    r"\nhow: {}",
                    escape_graphviz(&format!("{:?}", args.how))
                )
                .unwrap();
            }
            if args.nulls_equal {
                write!(label, r"\njoin-nulls").unwrap();
            }
            (label, &[*input_left, *input_right][..])
        },
        PhysNodeKind::CrossJoin {
            input_left,
            input_right,
            args: _,
        } => ("cross-join".to_string(), &[*input_left, *input_right][..]),
        PhysNodeKind::AsOfJoin {
            input_left,
            input_right,
            ..
        } => ("asof_join".to_string(), &[*input_left, *input_right][..]),
        #[cfg(feature = "merge_sorted")]
        PhysNodeKind::MergeSorted {
            input_left,
            input_right,
        } => ("merge-sorted".to_string(), &[*input_left, *input_right][..]),
        #[cfg(feature = "ewma")]
        PhysNodeKind::EwmMean { input, options: _ } => ("ewm-mean".to_string(), &[*input][..]),
        #[cfg(feature = "ewma")]
        PhysNodeKind::EwmVar { input, options: _ } => ("ewm-var".to_string(), &[*input][..]),
        #[cfg(feature = "ewma")]
        PhysNodeKind::EwmStd { input, options: _ } => ("ewm-std".to_string(), &[*input][..]),
    };

    let node_id = node_key.data().as_ffi();
    let style = NodeStyle::for_node_kind(kind);

    if let Some(attrs) = style.node_attrs() {
        out.push(format!("{node_id} [label=\"{label}\",{attrs}];"));
    } else {
        out.push(format!("{node_id} [label=\"{label}\"];"));
    }
    for input in inputs {
        visualize_plan_rec(input.node, phys_sm, expr_arena, visited, out);
        out.push(format!(
            "{} -> {};",
            input.node.data().as_ffi(),
            node_key.data().as_ffi()
        ));
    }
}

pub fn visualize_plan(
    root: PhysNodeKey,
    phys_sm: &SlotMap<PhysNodeKey, PhysNode>,
    expr_arena: &Arena<AExpr>,
) -> String {
    let mut visited: SecondaryMap<PhysNodeKey, ()> = SecondaryMap::new();
    let mut out = Vec::with_capacity(phys_sm.len() + 3);
    out.push("digraph polars {\nrankdir=\"BT\"\nnode [fontname=\"Monospace\"]".to_string());
    out.push(NodeStyle::legend());
    visualize_plan_rec(root, phys_sm, expr_arena, &mut visited, &mut out);
    out.push("}".to_string());
    out.join("\n")
}
