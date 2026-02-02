mod count;
mod dsl;
mod hint;
#[cfg(feature = "python")]
mod python_udf;
mod schema;

use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::Arc;

pub use dsl::*;
pub use hint::*;
use polars_core::error::feature_gated;
use polars_core::prelude::*;
use polars_core::series::IsSorted;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

#[cfg(feature = "python")]
use crate::dsl::python_dsl::PythonFunction;
use crate::plans::ir::ScanSourcesDisplay;
use crate::prelude::*;

#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[derive(Clone, IntoStaticStr)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionIR {
    RowIndex {
        name: PlSmallStr,
        offset: Option<IdxSize>,
        // Might be cached.
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    #[cfg(feature = "python")]
    OpaquePython(OpaquePythonUdf),

    FastCount {
        sources: ScanSources,
        scan_type: Box<FileScanIR>,
        alias: Option<PlSmallStr>,
    },

    Unnest {
        columns: Arc<[PlSmallStr]>,
        separator: Option<PlSmallStr>,
    },
    Rechunk,
    Explode {
        columns: Arc<[PlSmallStr]>,
        options: ExplodeOptions,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    #[cfg(feature = "pivot")]
    Unpivot {
        args: Arc<UnpivotArgsIR>,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    Opaque {
        function: Arc<dyn DataFrameUdf>,
        schema: Option<Arc<dyn UdfSchema>>,
        ///  allow predicate pushdown optimizations
        predicate_pd: bool,
        ///  allow projection pushdown optimizations
        projection_pd: bool,
        streamable: bool,
        // used for formatting
        fmt_str: PlSmallStr,
    },
    Hint(HintIR),
}

impl Hash for FunctionIR {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            #[cfg(feature = "python")]
            FunctionIR::OpaquePython { .. } => {},
            FunctionIR::Opaque { fmt_str, .. } => fmt_str.hash(state),
            FunctionIR::FastCount {
                sources,
                scan_type,
                alias,
            } => {
                sources.hash(state);
                scan_type.hash(state);
                alias.hash(state);
            },
            FunctionIR::Unnest { columns, separator } => {
                columns.hash(state);
                separator.hash(state);
            },
            FunctionIR::Rechunk => {},
            FunctionIR::Explode {
                columns,
                options,
                schema: _,
            } => {
                columns.hash(state);
                options.hash(state);
            },
            #[cfg(feature = "pivot")]
            FunctionIR::Unpivot { args, schema: _ } => args.hash(state),
            FunctionIR::RowIndex {
                name,
                schema: _,
                offset,
            } => {
                name.hash(state);
                offset.hash(state);
            },
            FunctionIR::Hint(hint) => hint.hash(state),
        }
    }
}

impl FunctionIR {
    /// Whether this function can run on batches of data at a time.
    pub fn is_streamable(&self) -> bool {
        use FunctionIR::*;
        match self {
            Rechunk => false,
            FastCount { .. } | Unnest { .. } | Explode { .. } => true,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            Opaque { streamable, .. } => *streamable,
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { streamable, .. }) => *streamable,
            RowIndex { .. } => false,
            Hint(_) => true,
        }
    }

    /// Whether this function will increase the number of rows
    pub fn expands_rows(&self) -> bool {
        use FunctionIR::*;
        match self {
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            Explode { .. } => true,
            _ => false,
        }
    }

    pub(crate) fn allow_predicate_pd(&self) -> bool {
        use FunctionIR::*;
        match self {
            Opaque { predicate_pd, .. } => *predicate_pd,
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { predicate_pd, .. }) => *predicate_pd,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            Rechunk | Unnest { .. } | Explode { .. } | Hint(_) => true,
            RowIndex { .. } | FastCount { .. } => false,
        }
    }

    pub(crate) fn allow_projection_pd(&self) -> bool {
        use FunctionIR::*;
        match self {
            Opaque { projection_pd, .. } => *projection_pd,
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { projection_pd, .. }) => *projection_pd,
            Rechunk | FastCount { .. } | Unnest { .. } | Explode { .. } | Hint(_) => true,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            RowIndex { .. } => true,
        }
    }

    pub(crate) fn additional_projection_pd_columns(&self) -> Cow<'_, [PlSmallStr]> {
        use FunctionIR::*;
        match self {
            Unnest { columns, .. } => Cow::Borrowed(columns.as_ref()),
            Explode { columns, .. } => Cow::Borrowed(columns.as_ref()),
            _ => Cow::Borrowed(&[]),
        }
    }

    pub fn evaluate(&self, mut df: DataFrame) -> PolarsResult<DataFrame> {
        use FunctionIR::*;
        match self {
            Opaque { function, .. } => function.call_udf(df),
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf {
                function,
                validate_output,
                schema,
                ..
            }) => python_udf::call_python_udf(function, df, *validate_output, schema.clone()),
            FastCount {
                sources,
                scan_type,
                alias,
            } => count::count_rows(sources, scan_type, alias.clone()),
            Rechunk => {
                df.rechunk_mut_par();
                Ok(df)
            },
            Unnest { columns, separator } => {
                feature_gated!(
                    "dtype-struct",
                    df.unnest(columns.iter().cloned(), separator.as_deref())
                )
            },
            Explode {
                columns, options, ..
            } => df.explode(columns.iter().cloned(), *options),
            #[cfg(feature = "pivot")]
            Unpivot { args, .. } => {
                use polars_ops::unpivot::UnpivotDF;
                let args = (**args).clone();
                df.unpivot2(args)
            },
            RowIndex { name, offset, .. } => df.with_row_index(name.clone(), *offset),
            Hint(hint) => {
                #[expect(irrefutable_let_patterns)]
                if let HintIR::Sorted(s) = &hint
                    && let Some(s) = s.first()
                {
                    let idx = df.try_get_column_index(&s.column)?;
                    let col = &mut unsafe { df.columns_mut_retain_schema() }[idx];
                    if let Some(d) = s.descending {
                        let flag = if d {
                            IsSorted::Descending
                        } else {
                            IsSorted::Ascending
                        };
                        col.set_sorted_flag(flag);
                    }
                }

                Ok(df)
            },
        }
    }

    pub fn is_order_producing(&self, is_input_ordered: bool) -> bool {
        match self {
            FunctionIR::RowIndex { .. } => true,
            FunctionIR::FastCount { .. } => false,
            FunctionIR::Unnest { .. } => is_input_ordered,
            FunctionIR::Rechunk => is_input_ordered,
            #[cfg(feature = "python")]
            FunctionIR::OpaquePython(..) => true,
            FunctionIR::Explode { .. } => true,
            #[cfg(feature = "pivot")]
            FunctionIR::Unpivot { .. } => true,
            FunctionIR::Opaque { .. } => true,
            FunctionIR::Hint(_) => is_input_ordered,
        }
    }

    pub fn is_elementwise(&self) -> bool {
        match self {
            Self::Unnest { .. } | Self::Hint(_) => true,
            #[cfg(feature = "python")]
            Self::OpaquePython(..) => false,
            #[cfg(feature = "pivot")]
            Self::Unpivot { .. } => false,
            Self::RowIndex { .. }
            | Self::FastCount { .. }
            | Self::Rechunk
            | Self::Explode { .. }
            | Self::Opaque { .. } => false,
        }
    }

    pub fn observes_input_order(&self) -> bool {
        true
    }

    /// Is the input ordering always the same as the output ordering.
    pub fn has_equal_order(&self) -> bool {
        match self {
            Self::Unnest { .. } | Self::Rechunk | Self::Hint(_) => true,
            #[cfg(feature = "python")]
            Self::OpaquePython(..) => false,
            #[cfg(feature = "pivot")]
            Self::Unpivot { .. } => false,
            Self::RowIndex { .. }
            | Self::FastCount { .. }
            | Self::Explode { .. }
            | Self::Opaque { .. } => false,
        }
    }
}

impl Debug for FunctionIR {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self}")
    }
}

impl Display for FunctionIR {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use FunctionIR::*;
        match self {
            Hint(hint) => {
                write!(f, "hint.{hint}")
            },
            Opaque { fmt_str, .. } => write!(f, "{fmt_str}"),
            Unnest { columns, separator } => {
                write!(f, "UNNEST by:")?;
                let columns = columns.as_ref();
                fmt_column_delimited(f, columns, "[", "]")?;
                if let Some(separator) = separator {
                    write!(f, ", separator: {separator}")?;
                }
                Ok(())
            },
            FastCount {
                sources,
                scan_type,
                alias,
            } => {
                let scan_type: &str = (&(**scan_type)).into();
                let default_column_name = PlSmallStr::from_static(crate::constants::LEN);
                let alias = alias.as_ref().unwrap_or(&default_column_name);

                write!(
                    f,
                    "FAST COUNT ({scan_type}) {} as \"{alias}\"",
                    ScanSourcesDisplay(sources)
                )
            },
            RowIndex {
                name,
                offset,
                schema: _,
            } => {
                write!(f, "ROW INDEX name: {name}")?;
                if let Some(offset) = offset {
                    write!(f, ", offset: {offset}")?;
                }

                Ok(())
            },
            Explode {
                columns,
                options,
                schema: _,
            } => {
                f.write_str("EXPLODE ")?;
                fmt_column_delimited(f, columns, "[", "]")?;
                if !options.empty_as_null {
                    f.write_str(", empty_as_null: false")?;
                }
                if !options.keep_nulls {
                    f.write_str(", keep_nulls: false")?;
                }
                Ok(())
            },
            #[cfg(feature = "pivot")]
            Unpivot { args, schema: _ } => {
                let UnpivotArgsIR {
                    on,
                    index,
                    variable_name,
                    value_name,
                } = args.as_ref();

                f.write_str("UNPIVOT on: ")?;
                fmt_column_delimited(f, on, "[", "]")?;
                fmt_column_delimited(f, index, "[", "]")?;
                write!(f, ", variable_name: {variable_name}")?;
                write!(f, ", value_name: {value_name}")?;
                Ok(())
            },
            #[cfg(feature = "python")]
            OpaquePython(_) => f.write_str(<&'static str>::from(self)),
            Rechunk => f.write_str(<&'static str>::from(self)),
        }
    }
}
