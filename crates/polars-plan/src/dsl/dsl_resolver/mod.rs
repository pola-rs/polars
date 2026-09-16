use std::pin::Pin;
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_utils::aliases::PlIndexSet;
use polars_utils::arena::Arena;
use polars_utils::pl_str::PlSmallStr;

use crate::dsl::{DslPlan, Expr};
use crate::plans::{AExpr, ExprIR};

#[cfg(feature = "python")]
pub mod python;

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct DslResolver {
    variant: DslResolverVariant,
}

impl std::hash::Hash for DslResolver {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        std::mem::discriminant(&self.variant).hash(state);

        match &self.variant {
            #[cfg(feature = "python")]
            DslResolverVariant::Python(x) => x.hash(state),
            DslResolverVariant::Rust(()) => unimplemented!(),
        }
    }
}

impl PartialEq for DslResolver {
    fn eq(&self, other: &Self) -> bool {
        match (&self.variant, &other.variant) {
            #[cfg(feature = "python")]
            (DslResolverVariant::Python(l), DslResolverVariant::Python(r)) => {
                l.cse_eq(r).ok() == Some(true)
            },
            (DslResolverVariant::Rust(()), DslResolverVariant::Rust(())) => unimplemented!(),
            _ => false,
        }
    }
}

impl Eq for DslResolver {}

#[derive(Debug, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
enum DslResolverVariant {
    #[cfg(feature = "python")]
    Python(python::PythonDslResolver),
    Rust(()), // Unimplemented. Placeholder to avoid empty enum without python feature.
}

pub trait DslResolverTrait {
    fn name(&self) -> PolarsResult<PlSmallStr>;

    fn schema(
        &self,
        #[cfg(feature = "python")] py_node_resolve_threadpool: Arc<
            polars_utils::python_thread_pool::PyThreadPool,
        >,
    ) -> PolarsResult<Pin<Box<dyn Future<Output = PolarsResult<SchemaRef>> + Send>>>;

    fn resolve_dsl(
        &self,
        args: ResolveDslArgs,
        filters_eir: Buffer<ExprIR>,
        existing_resolved_version_key: Option<PlSmallStr>,
        expr_arena: &Arena<AExpr>,
        #[cfg(feature = "python")] py_node_resolve_threadpool: Arc<
            polars_utils::python_thread_pool::PyThreadPool,
        >,
    ) -> PolarsResult<Pin<Box<dyn Future<Output = PolarsResult<ResolvedDsl>> + Send>>>;

    fn cse_eq(&self, other: &Self) -> PolarsResult<bool>;
}

impl DslResolverTrait for DslResolver {
    fn name(&self) -> PolarsResult<PlSmallStr> {
        match &self.variant {
            #[cfg(feature = "python")]
            DslResolverVariant::Python(py_resolver) => py_resolver.name(),
            DslResolverVariant::Rust(()) => unimplemented!(),
        }
    }

    fn schema(
        &self,
        #[cfg(feature = "python")] py_node_resolve_threadpool: Arc<
            polars_utils::python_thread_pool::PyThreadPool,
        >,
    ) -> PolarsResult<Pin<Box<dyn Future<Output = PolarsResult<SchemaRef>> + Send>>> {
        match &self.variant {
            #[cfg(feature = "python")]
            DslResolverVariant::Python(py_resolver) => {
                py_resolver.schema(py_node_resolve_threadpool)
            },
            DslResolverVariant::Rust(()) => unimplemented!(),
        }
    }

    fn resolve_dsl(
        &self,
        args: ResolveDslArgs,
        filters_eir: Buffer<ExprIR>,
        existing_resolved_version_key: Option<PlSmallStr>,
        expr_arena: &Arena<AExpr>,
        #[cfg(feature = "python")] py_node_resolve_threadpool: Arc<
            polars_utils::python_thread_pool::PyThreadPool,
        >,
    ) -> PolarsResult<Pin<Box<dyn Future<Output = PolarsResult<ResolvedDsl>> + Send>>> {
        match &self.variant {
            #[cfg(feature = "python")]
            DslResolverVariant::Python(py_resolver) => py_resolver.resolve_dsl(
                args,
                filters_eir,
                existing_resolved_version_key,
                expr_arena,
                py_node_resolve_threadpool,
            ),
            DslResolverVariant::Rust(()) => unimplemented!(),
        }
    }

    fn cse_eq(&self, other: &Self) -> PolarsResult<bool> {
        match (&self.variant, &other.variant) {
            #[cfg(feature = "python")]
            (DslResolverVariant::Python(l), DslResolverVariant::Python(r)) => l.cse_eq(r),
            (DslResolverVariant::Rust(()), DslResolverVariant::Rust(())) => unimplemented!(),
            _ => Ok(false),
        }
    }
}

#[derive(Debug, Default, Clone, Hash, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolveDslArgs {
    pub projection: Option<Buffer<PlSmallStr>>,
    pub slice: Option<(i64, u64)>,
    pub filters: Buffer<Expr>,
    pub filter_columns: Buffer<PlSmallStr>,
    pub filter_drop_columns_idx: Option<usize>,
}

#[derive(Debug, Default, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct ResolvedDsl {
    pub dsl: Option<DslPlan>,
    pub version_key: Option<PlSmallStr>,
    pub applied_filters: PlIndexSet<usize>,
    pub slice_offset_applied: bool,
}

pub struct ResolverExplainHeadingDisplay<'a> {
    pub indent: usize,
    pub resolver: &'a Arc<DslResolver>,
    pub resolved_dsl:
        &'a Arc<std::sync::Mutex<polars_utils::aliases::PlIndexMap<ResolveDslArgs, ResolvedDsl>>>,
}

impl std::fmt::Display for ResolverExplainHeadingDisplay<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let indent = self.indent;
        let name = match self.resolver.name() {
            Ok(x) => x,
            Err(e) => PlSmallStr::from_string(format!("(error: {e})")),
        };
        let n_cached_resolves = { self.resolved_dsl.lock().unwrap().len() };
        write!(
            f,
            "{:indent$}RESOLVER[name: {name}, cached_resolves: {n_cached_resolves}]",
            ""
        )
    }
}
