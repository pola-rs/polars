use std::pin::Pin;
#[cfg(feature = "python")]
use std::sync::Arc;
use std::sync::OnceLock;

use polars_buffer::Buffer;
use polars_core::runtime::ASYNC;
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_utils::arena::Arena;
use polars_utils::async_utils::tokio_handle_ext::AbortOnDropHandle;
use polars_utils::itertools::Itertools;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::python_function::PythonObject;
use polars_utils::python_interns;
use pyo3::pybacked::PyBackedStr;
use pyo3::sync::PyOnceLock;
use pyo3::types::{PyAnyMethods, PyDict, PyList, PyListMethods};
use pyo3::{Bound, IntoPyObjectExt, Py, PyAny, Python, intern};

use crate::dsl::Expr;
#[cfg(feature = "python")]
use crate::dsl::dsl_resolver::DslResolverVariant;
use crate::dsl::dsl_resolver::{DslResolver, DslResolverTrait, ResolveDslArgs, ResolvedDsl};
use crate::plans::pyarrow::{aexpr_to_pyarrow, predicate_to_pa};
use crate::plans::{AExpr, ExprIR};

pub static PY_DSL_RESOLVER_VTABLE: OnceLock<PyDslResolverVTable> = OnceLock::new();

fn py_dsl_resolver_vtable() -> &'static PyDslResolverVTable {
    PY_DSL_RESOLVER_VTABLE
        .get()
        .unwrap_or_else(|| panic!("PY_DSL_RESOLVER_VTABLE not initialized"))
}

pub struct PyDslResolverVTable {
    pub extract_schema: fn(py: Python<'_>, schema: Py<PyAny>) -> PolarsResult<SchemaRef>,
    pub to_py_plexpr: fn(py: Python<'_>, expr: Expr) -> Py<PyAny>,
    pub extract_py_resolved_dsl:
        fn(py: Python<'_>, py_resolved_node: Py<PyAny>) -> PolarsResult<ResolvedDsl>,
}

#[derive(Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct PythonDslResolver {
    resolver: PythonObject,
}

impl std::hash::Hash for PythonDslResolver {
    fn hash<H: std::hash::Hasher>(&self, _state: &mut H) {}
}

impl DslResolver {
    pub fn new_python(dsl_resolver: PythonObject) -> Self {
        Self {
            variant: DslResolverVariant::Python(PythonDslResolver {
                resolver: dsl_resolver,
            }),
        }
    }
}

fn to_py_resolve_dsl_args<'py>(
    py: Python<'py>,
    resolve_dsl_args: &ResolveDslArgs,
    existing_resolved_version_key: Option<&str>,
    filters_eir: &[ExprIR],
    expr_arena: &Arena<AExpr>,
) -> PolarsResult<Bound<'py, PyDict>> {
    let ResolveDslArgs {
        projection,
        slice,
        filters,
        filter_columns,
        filter_drop_columns_idx,
    } = resolve_dsl_args;

    let py_resolve_dsl_kwargs: Bound<'py, PyDict> = PyDict::new(py);

    py_resolve_dsl_kwargs.set_item(
        intern!(py, "existing_resolved_version_key"),
        existing_resolved_version_key,
    )?;
    py_resolve_dsl_kwargs.set_item(intern!(py, "projection"), projection.as_deref())?;

    let limit = slice
        .filter(|(offset, _)| *offset >= 0)
        .map(|(offset, len)| offset as u128 + len as u128);

    py_resolve_dsl_kwargs.set_item(intern!(py, "limit"), limit)?;

    let py_dsl_filters = PyList::empty(py);
    let pyarrow_compute = py.import("pyarrow.compute");

    for (expr, eir) in filters.iter().zip_eq(filters_eir.iter()) {
        let node = eir.node();

        let kwargs = PyDict::new(py);
        kwargs.set_item(
            intern!(py, "expr"),
            (py_dsl_resolver_vtable().to_py_plexpr)(py, expr.clone()),
        )?;
        kwargs.set_item(
            intern!(py, "_pyarrow_expr"),
            match pyarrow_compute.as_ref() {
                Ok(pc) => aexpr_to_pyarrow(py, pc, node, expr_arena).into_py_any(py)?,
                Err(e) => e.into_py_any(py)?,
            },
        )?;
        kwargs.set_item(
            intern!(py, "pyarrow_str"),
            predicate_to_pa(node, expr_arena),
        )?;

        py_dsl_filters.append(py_filter_dataclass(py).call(py, (), Some(&kwargs))?)?;
    }

    py_resolve_dsl_kwargs.set_item(intern!(py, "filters"), PyList::new(py, py_dsl_filters)?)?;
    py_resolve_dsl_kwargs.set_item(intern!(py, "filter_columns"), filter_columns.as_slice())?;
    py_resolve_dsl_kwargs.set_item(
        intern!(py, "filter_drop_columns_idx"),
        filter_drop_columns_idx,
    )?;

    return Ok(py_resolve_dsl_kwargs);

    fn py_filter_dataclass(py: Python<'_>) -> &'static Py<PyAny> {
        static CLS: PyOnceLock<Py<PyAny>> = PyOnceLock::new();

        CLS.get_or_init(py, || {
            py.import("polars.lazyframe.resolver")
                .unwrap()
                .getattr("FilterExpr")
                .unwrap()
                .unbind()
        })
    }
}

impl DslResolverTrait for PythonDslResolver {
    fn name(&self) -> PolarsResult<PlSmallStr> {
        Python::attach(|py| {
            Ok(PlSmallStr::from_str(
                &self
                    .resolver
                    .getattr(py, python_interns::DUNDER_CLASS.get(py))?
                    .getattr(py, python_interns::DUNDER_NAME.get(py))?
                    .extract::<PyBackedStr>(py)?,
            ))
        })
    }

    fn schema(
        &self,
        #[cfg(feature = "python")] py_dsl_resolve_threadpool: Arc<
            polars_utils::python_thread_pool::PyThreadPool,
        >,
    ) -> PolarsResult<Pin<Box<dyn Future<Output = PolarsResult<SchemaRef>> + Send>>> {
        let schema_fn = Python::attach(|py| self.resolver.getattr(py, "schema"))?;

        let fut = AbortOnDropHandle(ASYNC.spawn_blocking(move || {
            Python::attach(|py| {
                (py_dsl_resolver_vtable().extract_schema)(
                    py,
                    py_dsl_resolve_threadpool.spawn_call(py, &schema_fn, (), None)?,
                )
            })
        }));

        Ok(Box::pin(async move { fut.await.unwrap() }))
    }

    fn resolve_dsl(
        &self,
        args: ResolveDslArgs,
        filters_eir: Buffer<ExprIR>,
        existing_resolved_version_key: Option<PlSmallStr>,
        expr_arena: &Arena<AExpr>,
        #[cfg(feature = "python")] py_dsl_resolve_threadpool: Arc<
            polars_utils::python_thread_pool::PyThreadPool,
        >,
    ) -> PolarsResult<Pin<Box<dyn Future<Output = PolarsResult<ResolvedDsl>> + Send>>> {
        let (resolve_dsl_fn, kwargs) = Python::attach(|py| {
            PolarsResult::Ok((
                self.resolver
                    .getattr(py, intern!(py, "resolve_lazyframe"))?,
                to_py_resolve_dsl_args(
                    py,
                    &args,
                    existing_resolved_version_key.as_deref(),
                    &filters_eir,
                    expr_arena,
                )?
                .unbind(),
            ))
        })?;

        let fut = AbortOnDropHandle(ASYNC.spawn_blocking(move || {
            Python::attach(|py| {
                let py_resolved_dsl = py_dsl_resolve_threadpool.spawn_call(
                    py,
                    &resolve_dsl_fn,
                    (),
                    Some(kwargs.bind(py)),
                )?;

                (py_dsl_resolver_vtable().extract_py_resolved_dsl)(py, py_resolved_dsl)
            })
        }));

        Ok(Box::pin(async move { fut.await.unwrap() }))
    }

    fn cse_eq(&self, other: &Self) -> PolarsResult<bool> {
        Python::attach(|py| {
            PolarsResult::Ok(
                self.resolver
                    .getattr(py, intern!(py, "cse_eq"))?
                    .call1(py, (&other.resolver,))?
                    .extract::<bool>(py)?,
            )
        })
    }
}
