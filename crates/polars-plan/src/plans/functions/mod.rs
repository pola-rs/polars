mod count;
mod dsl;
#[cfg(feature = "merge_sorted")]
mod merge_sorted;
#[cfg(feature = "python")]
mod python_udf;
mod rename;
mod schema;

use std::borrow::Cow;
use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex};

pub use dsl::*;
use polars_core::error::feature_gated;
use polars_core::prelude::*;
use polars_utils::pl_str::PlSmallStr;
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
use strum_macros::IntoStaticStr;

use self::visitor::AexprNode;
#[cfg(feature = "python")]
use crate::dsl::python_udf::PythonFunction;
#[cfg(feature = "merge_sorted")]
use crate::plans::functions::merge_sorted::merge_sorted;
use crate::plans::ir::ScanSourcesDisplay;
use crate::prelude::*;

#[cfg_attr(feature = "ir_serde", derive(Serialize, Deserialize))]
#[derive(Clone, IntoStaticStr)]
#[strum(serialize_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionIR {
    #[cfg(feature = "python")]
    OpaquePython(OpaquePythonUdf),
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
    FastCount {
        sources: ScanSources,
        scan_type: FileScan,
        alias: Option<PlSmallStr>,
    },
    /// Streaming engine pipeline
    #[cfg_attr(feature = "ir_serde", serde(skip))]
    Pipeline {
        function: Arc<Mutex<dyn DataFrameUdfMut>>,
        schema: SchemaRef,
        original: Option<Arc<IRPlan>>,
    },
    Unnest {
        columns: Arc<[PlSmallStr]>,
    },
    Rechunk,
    // The two DataFrames are temporary concatenated
    // this indicates until which chunk the data is from the left df
    // this trick allows us to reuse the `Union` architecture to get map over
    // two DataFrames
    #[cfg(feature = "merge_sorted")]
    MergeSorted {
        // sorted column that serves as the key
        column: PlSmallStr,
    },
    Rename {
        existing: Arc<[PlSmallStr]>,
        new: Arc<[PlSmallStr]>,
        // A column name gets swapped with an existing column
        swapping: bool,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    Explode {
        columns: Arc<[PlSmallStr]>,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    #[cfg(feature = "pivot")]
    Unpivot {
        args: Arc<UnpivotArgsIR>,
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
    },
    RowIndex {
        name: PlSmallStr,
        // Might be cached.
        #[cfg_attr(feature = "ir_serde", serde(skip))]
        schema: CachedSchema,
        offset: Option<IdxSize>,
    },
    Assert {
        name: Option<PlSmallStr>,
        predicate: AexprNode,
        flags: AssertFlags,
        expr_format: Arc<str>,
    },
}

pub struct HashableEqFunctionIR<'a, 'b> {
    function: &'a FunctionIR,
    expr_arena: &'b Arena<AExpr>,
}

impl FunctionIR {
    pub fn hashable_and_eq<'a, 'b>(
        &'a self,
        expr_arena: &'b Arena<AExpr>,
    ) -> HashableEqFunctionIR<'a, 'b> {
        HashableEqFunctionIR {
            function: self,
            expr_arena,
        }
    }
}

impl Eq for HashableEqFunctionIR<'_, '_> {}

impl PartialEq for HashableEqFunctionIR<'_, '_> {
    fn eq(&self, other: &Self) -> bool {
        use FunctionIR::*;
        match (self.function, other.function) {
            (Rechunk, Rechunk) => true,
            (
                FastCount {
                    sources: srcs_l, ..
                },
                FastCount {
                    sources: srcs_r, ..
                },
            ) => srcs_l == srcs_r,
            (
                Rename {
                    existing: existing_l,
                    new: new_l,
                    ..
                },
                Rename {
                    existing: existing_r,
                    new: new_r,
                    ..
                },
            ) => existing_l == existing_r && new_l == new_r,
            (Explode { columns: l, .. }, Explode { columns: r, .. }) => l == r,
            #[cfg(feature = "pivot")]
            (Unpivot { args: l, .. }, Unpivot { args: r, .. }) => l == r,
            (RowIndex { name: l, .. }, RowIndex { name: r, .. }) => l == r,
            #[cfg(feature = "merge_sorted")]
            (MergeSorted { column: l }, MergeSorted { column: r }) => l == r,
            (
                Assert {
                    name: l_name,
                    predicate: l_pred,
                    flags: l_flags,
                    expr_format: _,
                },
                Assert {
                    name: r_name,
                    predicate: r_pred,
                    flags: r_flags,
                    expr_format: _,
                },
            ) => {
                l_name == r_name
                    && l_flags == r_flags
                    && l_pred.hashable_and_cmp(self.expr_arena)
                        == r_pred.hashable_and_cmp(other.expr_arena)
            },
            _ => false,
        }
    }
}

impl Hash for HashableEqFunctionIR<'_, '_> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self.function).hash(state);
        match self.function {
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
            FunctionIR::Pipeline { .. } => {},
            FunctionIR::Unnest { columns } => columns.hash(state),
            FunctionIR::Rechunk => {},
            #[cfg(feature = "merge_sorted")]
            FunctionIR::MergeSorted { column } => column.hash(state),
            FunctionIR::Rename {
                existing,
                new,
                swapping: _,
                ..
            } => {
                existing.hash(state);
                new.hash(state);
            },
            FunctionIR::Explode { columns, schema: _ } => columns.hash(state),
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
            FunctionIR::Assert {
                name,
                predicate,
                flags,
                expr_format: _,
            } => {
                name.hash(state);
                predicate
                    .to_expr_ir()
                    .traverse_and_hash(self.expr_arena, state);
                flags.hash(state);
            },
        }
    }
}

impl FunctionIR {
    /// Whether this function can run on batches of data at a time.
    pub fn is_streamable(&self, expr_arena: &Arena<AExpr>) -> bool {
        use FunctionIR::*;
        match self {
            Rechunk | Pipeline { .. } => false,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => false,
            FastCount { .. } | Unnest { .. } | Rename { .. } | Explode { .. } => true,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            Opaque { streamable, .. } => *streamable,
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { streamable, .. }) => *streamable,
            RowIndex { .. } => false,
            Assert { predicate, .. } => is_streamable(
                predicate.node(),
                expr_arena,
                IsStreamableContext::new(Context::Default).with_allow_cast_categorical(false),
            ),
        }
    }

    /// Whether this function will increase the number of rows
    pub fn expands_rows(&self) -> bool {
        use FunctionIR::*;
        match self {
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
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
            Rechunk | Unnest { .. } | Rename { .. } | Explode { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            RowIndex { .. } | FastCount { .. } => false,
            Assert { flags, .. } => flags.contains(AssertFlags::ALLOW_PREDICATE_PD),
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn allow_projection_pd(&self) -> bool {
        use FunctionIR::*;
        match self {
            Opaque { projection_pd, .. } => *projection_pd,
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf { projection_pd, .. }) => *projection_pd,
            Rechunk
            | FastCount { .. }
            | Unnest { .. }
            | Rename { .. }
            | Explode { .. }
            | Assert { .. } => true,
            #[cfg(feature = "pivot")]
            Unpivot { .. } => true,
            #[cfg(feature = "merge_sorted")]
            MergeSorted { .. } => true,
            RowIndex { .. } => true,
            Pipeline { .. } => unimplemented!(),
        }
    }

    pub(crate) fn additional_projection_pd_columns(
        &self,
        expr_arena: &Arena<AExpr>,
    ) -> Cow<[PlSmallStr]> {
        use FunctionIR::*;
        match self {
            Unnest { columns } => Cow::Borrowed(columns.as_ref()),
            Explode { columns, .. } => Cow::Borrowed(columns.as_ref()),
            Assert { predicate, .. } => {
                aexpr_to_leaf_names_iter(predicate.node(), expr_arena).collect()
            },
            #[cfg(feature = "merge_sorted")]
            MergeSorted { column, .. } => Cow::Owned(vec![column.clone()]),
            _ => Cow::Borrowed(&[]),
        }
    }

    pub fn copy_exprs(&self, container: &mut Vec<ExprIR>) {
        match self {
            FunctionIR::Assert { predicate, .. } => container.push(predicate.to_expr_ir()),
            _ => {},
        }
    }

    pub fn get_exprs(&self) -> Vec<ExprIR> {
        let mut exprs = Vec::new();
        self.copy_exprs(&mut exprs);
        exprs
    }

    pub fn evaluate(&self, mut df: DataFrame, exprs: &[Column]) -> PolarsResult<DataFrame> {
        use FunctionIR::*;
        match self {
            Opaque { function, .. } => function.call_udf(df),
            #[cfg(feature = "python")]
            OpaquePython(OpaquePythonUdf {
                function,
                validate_output,
                schema,
                ..
            }) => python_udf::call_python_udf(function, df, *validate_output, schema.as_deref()),
            FastCount {
                sources,
                scan_type,
                alias,
            } => count::count_rows(sources, scan_type, alias.clone()),
            Rechunk => {
                df.as_single_chunk_par();
                Ok(df)
            },
            #[cfg(feature = "merge_sorted")]
            MergeSorted { column } => merge_sorted(&df, column.as_ref()),
            Unnest { columns: _columns } => {
                feature_gated!("dtype-struct", df.unnest(_columns.iter().cloned()))
            },
            Pipeline { function, .. } => {
                // we use a global string cache here as streaming chunks all have different rev maps
                #[cfg(feature = "dtype-categorical")]
                {
                    let _sc = StringCacheHolder::hold();
                    function.lock().unwrap().call_udf(df)
                }

                #[cfg(not(feature = "dtype-categorical"))]
                {
                    function.lock().unwrap().call_udf(df)
                }
            },
            Rename { existing, new, .. } => rename::rename_impl(df, existing, new),
            Explode { columns, .. } => df.explode(columns.iter().cloned()),
            #[cfg(feature = "pivot")]
            Unpivot { args, .. } => {
                use polars_ops::pivot::UnpivotDF;
                let args = (**args).clone();
                df.unpivot2(args)
            },
            RowIndex { name, offset, .. } => df.with_row_index(name.clone(), *offset),
            Assert {
                name,
                predicate: _,
                flags,
                expr_format,
            } => {
                if std::env::var("POLARS_SKIP_ASSERTS").as_deref() == Ok("1") {
                    return Ok(df);
                }

                assert_eq!(exprs.len(), 1);
                match exprs[0].and_reduce()?.value() {
                    AnyValue::Null | AnyValue::Boolean(false) => {
                        if flags.contains(AssertFlags::WARN_ON_FAIL) {
                            if std::env::var("POLARS_SILENCE_ASSERT_WARN").as_deref() == Ok("1") {
                                return Ok(df);
                            }

                            match &name {
                                None => eprintln!(
                                    "WARN: Assertion with predicate '{expr_format}' failed."
                                ),
                                Some(name) => {
                                    eprintln!("WARN: Assertion '{name}' with predicate '{expr_format}' failed.")
                                },
                            }

                            Ok(df)
                        } else {
                            Err(polars_err!(AssertionFailed: match &name {
                                None => format!("Assertion with predicate '{expr_format}' failed."),
                                Some(name) => format!("Assertion '{name}' with predicate '{expr_format}' failed."),
                            }))
                        }
                    },
                    AnyValue::Boolean(true) => Ok(df),
                    _ => polars_bail!(InvalidOperation: "Assertion produced a non-boolean"),
                }
            },
        }
    }

    pub fn to_streaming_lp(&self) -> Option<IRPlanRef> {
        let Self::Pipeline {
            function: _,
            schema: _,
            original,
        } = self
        else {
            return None;
        };

        Some(original.as_ref()?.as_ref().as_ref())
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
            Opaque { fmt_str, .. } => write!(f, "{fmt_str}"),
            Unnest { columns } => {
                write!(f, "UNNEST by:")?;
                let columns = columns.as_ref();
                fmt_column_delimited(f, columns, "[", "]")
            },
            Pipeline { original, .. } => {
                if let Some(original) = original {
                    let ir_display = original.as_ref().display();

                    writeln!(f, "--- STREAMING")?;
                    write!(f, "{ir_display}")?;
                    let indent = 2;
                    write!(f, "{:indent$}--- END STREAMING", "")
                } else {
                    write!(f, "STREAMING")
                }
            },
            Assert {
                name,
                flags,
                expr_format,
                predicate: _,
            } => {
                let on_fail_str = if flags.contains(AssertFlags::WARN_ON_FAIL) {
                    "warn"
                } else {
                    "error"
                };
                match name {
                    None => write!(f, "ASSERT[{on_fail_str}] {expr_format}"),
                    Some(name) => write!(f, "ASSERT[{on_fail_str}] {name} = {expr_format}"),
                }
            },
            FastCount {
                sources,
                scan_type,
                alias,
            } => {
                let scan_type: &str = scan_type.into();
                let default_column_name = PlSmallStr::from_static(crate::constants::LEN);
                let alias = alias.as_ref().unwrap_or(&default_column_name);

                write!(
                    f,
                    "FAST COUNT ({scan_type}) {} as \"{alias}\"",
                    ScanSourcesDisplay(sources)
                )
            },
            v => {
                let s: &str = v.into();
                write!(f, "{s}")
            },
        }
    }
}
