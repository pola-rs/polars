use parking_lot::RwLock;
use polars::prelude::{PlHashMap, PlSmallStr, Scalar};
use polars::sql::{
    CallArg, CatalogAlter, CatalogRefKind, CatalogRefMode, CatalogStatement, QueryParams,
    SQLContext, VersionSpec, extract_table_identifiers,
};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};

use crate::conversion::Wrap;
use crate::conversion::any_value::py_object_to_any_value;
use crate::error::PyPolarsErr;
use crate::utils::EnterPolarsExt;
use crate::{PyExpr, PyLazyFrame};

/// A catalog table that a query reads (see `PySQLContext::catalog_tables`).
#[pyclass(frozen)]
pub struct PyCatalogRead {
    /// The name the catalog was attached under.
    #[pyo3(get)]
    catalog: String,
    /// Namespace parts, then the table name.
    #[pyo3(get)]
    identifier: Vec<String>,
    /// None, ("snapshot", id), ("ref", name) or ("timestamp", microseconds since the epoch).
    #[pyo3(get)]
    version: Option<Py<PyTuple>>,
    /// The name to pass the scan of this read back under.
    #[pyo3(get)]
    key: Vec<String>,
}

fn param_value(ob: &Bound<'_, PyAny>) -> PyResult<Scalar> {
    let value = py_object_to_any_value(ob, true, false, None)?;
    Ok(Scalar::new(value.dtype(), value))
}

impl<'a, 'py> FromPyObject<'a, 'py> for Wrap<QueryParams> {
    type Error = PyErr;

    fn extract(ob: Borrowed<'a, 'py, PyAny>) -> PyResult<Self> {
        if let Ok(dict) = ob.cast::<PyDict>() {
            let mut values = PlHashMap::with_capacity_and_hasher(dict.len(), Default::default());
            for (key, value) in dict.iter() {
                let key: String = key.extract().map_err(|_| {
                    PyTypeError::new_err("the names of query parameters must be strings")
                })?;
                values.insert(PlSmallStr::from_string(key), param_value(&value)?);
            }
            return Ok(Wrap(QueryParams::Named(values)));
        }
        if ob.is_instance_of::<PyList>() || ob.is_instance_of::<PyTuple>() {
            let values = ob
                .try_iter()?
                .map(|value| param_value(&value?))
                .collect::<PyResult<_>>()?;
            return Ok(Wrap(QueryParams::Positional(values)));
        }
        Err(PyTypeError::new_err(
            "query parameters must be a list, tuple or dict",
        ))
    }
}

#[pyclass(frozen)]
#[repr(transparent)]
pub struct PySQLContext {
    pub context: RwLock<SQLContext>,
}

impl Clone for PySQLContext {
    fn clone(&self) -> Self {
        Self {
            context: RwLock::new(self.context.read().clone()),
        }
    }
}

#[pymethods]
#[allow(
    clippy::wrong_self_convention,
    clippy::should_implement_trait,
    clippy::len_without_is_empty
)]
impl PySQLContext {
    #[staticmethod]
    #[allow(clippy::new_without_default)]
    pub fn new(allow_file_reads: bool, allow_file_writes: bool) -> PySQLContext {
        let context = SQLContext::new()
            .with_file_reads(allow_file_reads)
            .with_file_writes(allow_file_writes);
        PySQLContext {
            context: RwLock::new(context),
        }
    }

    /// Execute a SQL query in the current SQLContext.
    #[pyo3(signature = (query, params=None))]
    pub fn execute(
        &self,
        py: Python<'_>,
        query: &str,
        params: Option<Wrap<QueryParams>>,
    ) -> PyResult<PyLazyFrame> {
        let params = params.map(|p| p.0).unwrap_or_default();
        py.enter_polars(|| self.context.write().execute_with_params(query, &params))
            .map(Into::into)
    }

    /// Get a list of table names registered in the current SQLContext.
    pub fn get_tables(&self) -> PyResult<Vec<String>> {
        Ok(self.context.read().get_tables())
    }

    /// Register a table in the current SQLContext.
    pub fn register(&self, name: &str, lf: PyLazyFrame) {
        self.context.write().register(name, lf.ldf.into_inner())
    }

    /// Unregister a table from the current SQLContext.
    pub fn unregister(&self, name: &str) {
        self.context.write().unregister(name)
    }

    /// Attach a catalog: table names resolve against it (see `SQLContext::register_catalog`).
    #[pyo3(signature = (name, default_namespace, default))]
    pub fn register_catalog(
        &self,
        name: &str,
        default_namespace: Option<Vec<String>>,
        default: bool,
    ) -> PyResult<()> {
        self.context
            .write()
            .register_catalog(name, default_namespace, default)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    /// The name of the default catalog, if one is set.
    pub fn default_catalog(&self) -> Option<String> {
        self.context.read().default_catalog().map(str::to_string)
    }

    /// Make an attached catalog the default.
    pub fn set_default_catalog(&self, name: &str) -> PyResult<()> {
        self.context
            .write()
            .set_default_catalog(name)
            .map_err(PyPolarsErr::from)?;
        Ok(())
    }

    /// Detach a catalog, returning the name it was attached under.
    pub fn unregister_catalog(&self, name: &str) -> Option<String> {
        self.context.write().unregister_catalog(name)
    }

    /// The catalog tables a query reads, once per table and version.
    pub fn catalog_tables(&self, py: Python<'_>, query: &str) -> PyResult<Vec<PyCatalogRead>> {
        let reads = self
            .context
            .read()
            .catalog_tables(query)
            .map_err(PyPolarsErr::from)?;
        reads
            .into_iter()
            .map(|read| {
                let version = match read.version {
                    None => None,
                    Some(VersionSpec::Snapshot(id)) => Some(("snapshot", id).into_pyobject(py)?),
                    Some(VersionSpec::Ref(name)) => Some(("ref", name).into_pyobject(py)?),
                    Some(VersionSpec::Timestamp(us)) => Some(("timestamp", us).into_pyobject(py)?),
                };
                Ok(PyCatalogRead {
                    catalog: read.table.catalog,
                    identifier: read.table.identifier,
                    version: version.map(Bound::unbind),
                    key: read.key,
                })
            })
            .collect()
    }

    /// The catalog statement of a query, as a dict keyed by `kind`, or None.
    pub fn catalog_statement<'py>(
        &self,
        py: Python<'py>,
        query: &str,
    ) -> PyResult<Option<Bound<'py, PyDict>>> {
        let stmt = self
            .context
            .read()
            .catalog_statement(query)
            .map_err(PyPolarsErr::from)?;
        let Some(stmt) = stmt else {
            return Ok(None);
        };
        let dict = PyDict::new(py);
        match stmt {
            CatalogStatement::DropTable {
                name,
                table,
                if_exists,
                purge,
            } => {
                dict.set_item("kind", "drop_table")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                dict.set_item("if_exists", if_exists)?;
                dict.set_item("purge", purge)?;
            },
            CatalogStatement::CreateTable {
                name,
                table,
                if_not_exists,
                schema,
                query,
                required,
                location,
                properties,
                partition_by,
            } => {
                dict.set_item("kind", "create_table")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                dict.set_item("if_not_exists", if_not_exists)?;
                dict.set_item("schema", Wrap(schema))?;
                dict.set_item("query", query)?;
                dict.set_item("required", required)?;
                dict.set_item("location", location)?;
                let props = PyDict::new(py);
                for (key, value) in properties {
                    props.set_item(key, value)?;
                }
                dict.set_item("properties", props)?;
                let partition_by: Vec<(Vec<String>, String)> = partition_by
                    .into_iter()
                    .map(|field| (field.path, field.transform))
                    .collect();
                dict.set_item("partition_by", partition_by)?;
            },
            CatalogStatement::CreateSchema {
                name,
                namespace,
                if_not_exists,
                properties,
            } => {
                dict.set_item("kind", "create_schema")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", namespace.catalog)?;
                dict.set_item("namespace", namespace.namespace)?;
                dict.set_item("if_not_exists", if_not_exists)?;
                let props = PyDict::new(py);
                for (key, value) in properties {
                    props.set_item(key, value)?;
                }
                dict.set_item("properties", props)?;
            },
            CatalogStatement::DropSchema {
                name,
                namespace,
                if_exists,
            } => {
                dict.set_item("kind", "drop_schema")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", namespace.catalog)?;
                dict.set_item("namespace", namespace.namespace)?;
                dict.set_item("if_exists", if_exists)?;
            },
            CatalogStatement::Insert {
                name,
                table,
                columns,
                query,
                overwrite,
            } => {
                dict.set_item("kind", "insert")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                dict.set_item("columns", columns)?;
                dict.set_item("query", query)?;
                dict.set_item("overwrite", overwrite)?;
            },
            CatalogStatement::Delete {
                name,
                table,
                condition,
                subquery,
                hint_unsupported_part,
            } => {
                dict.set_item("kind", "delete")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                dict.set_item("condition", condition)?;
                dict.set_item("subquery", subquery)?;
                dict.set_item("hint_unsupported_part", hint_unsupported_part)?;
            },
            CatalogStatement::Update {
                name,
                table,
                columns,
                condition,
                subquery,
                hint_unsupported_part,
            } => {
                dict.set_item("kind", "update")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                dict.set_item("columns", columns)?;
                dict.set_item("condition", condition)?;
                dict.set_item("subquery", subquery)?;
                dict.set_item("hint_unsupported_part", hint_unsupported_part)?;
            },
            CatalogStatement::Merge { name, table } => {
                dict.set_item("kind", "merge")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
            },
            CatalogStatement::Truncate {
                name,
                table,
                if_exists,
            } => {
                dict.set_item("kind", "truncate")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                dict.set_item("if_exists", if_exists)?;
            },
            CatalogStatement::AlterTable {
                name,
                table,
                if_exists,
                operation,
            } => {
                dict.set_item("kind", "alter_table")?;
                dict.set_item("name", name)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                dict.set_item("if_exists", if_exists)?;
                let op = PyDict::new(py);
                match operation {
                    CatalogAlter::AddColumn {
                        name,
                        dtype,
                        if_not_exists,
                    } => {
                        op.set_item("op", "add_column")?;
                        op.set_item("column", name)?;
                        op.set_item("dtype", &Wrap(dtype))?;
                        op.set_item("if_not_exists", if_not_exists)?;
                    },
                    CatalogAlter::DropColumn { name, if_exists } => {
                        op.set_item("op", "drop_column")?;
                        op.set_item("column", name)?;
                        op.set_item("if_exists", if_exists)?;
                    },
                    CatalogAlter::RenameColumn { old, new } => {
                        op.set_item("op", "rename_column")?;
                        op.set_item("column", old)?;
                        op.set_item("new_name", new)?;
                    },
                    CatalogAlter::SetDataType { name, dtype } => {
                        op.set_item("op", "set_data_type")?;
                        op.set_item("column", name)?;
                        op.set_item("dtype", &Wrap(dtype))?;
                    },
                    CatalogAlter::DropNotNull { name } => {
                        op.set_item("op", "drop_not_null")?;
                        op.set_item("column", name)?;
                    },
                    CatalogAlter::SetProperties { properties } => {
                        op.set_item("op", "set_properties")?;
                        let props = PyDict::new(py);
                        for (key, value) in properties {
                            props.set_item(key, value)?;
                        }
                        op.set_item("properties", props)?;
                    },
                    CatalogAlter::SetPartitioning { fields } => {
                        op.set_item("op", "set_partitioning")?;
                        let fields: Vec<(Vec<String>, String)> = fields
                            .into_iter()
                            .map(|field| (field.path, field.transform))
                            .collect();
                        op.set_item("fields", fields)?;
                    },
                    CatalogAlter::SetSortOrder { fields } => {
                        op.set_item("op", "set_sort_order")?;
                        let fields: Vec<(Vec<String>, String, bool, bool)> = fields
                            .into_iter()
                            .map(|f| (f.path, f.transform, f.descending, f.nulls_first))
                            .collect();
                        op.set_item("fields", fields)?;
                    },
                    CatalogAlter::SetRef {
                        kind,
                        name,
                        mode,
                        snapshot_id,
                        max_ref_age_ms,
                        min_snapshots_to_keep,
                        max_snapshot_age_ms,
                    } => {
                        op.set_item("op", "set_ref")?;
                        op.set_item("ref_kind", ref_kind_name(kind))?;
                        op.set_item("ref", name)?;
                        op.set_item(
                            "mode",
                            match mode {
                                CatalogRefMode::Create => "create",
                                CatalogRefMode::CreateIfNotExists => "create_if_not_exists",
                                CatalogRefMode::Replace => "replace",
                                CatalogRefMode::CreateOrReplace => "create_or_replace",
                            },
                        )?;
                        op.set_item("snapshot_id", snapshot_id)?;
                        op.set_item("max_ref_age_ms", max_ref_age_ms)?;
                        op.set_item("min_snapshots_to_keep", min_snapshots_to_keep)?;
                        op.set_item("max_snapshot_age_ms", max_snapshot_age_ms)?;
                    },
                    CatalogAlter::DropRef {
                        kind,
                        name,
                        if_exists,
                    } => {
                        op.set_item("op", "drop_ref")?;
                        op.set_item("ref_kind", ref_kind_name(kind))?;
                        op.set_item("ref", name)?;
                        op.set_item("if_exists", if_exists)?;
                    },
                }
                dict.set_item("operation", op)?;
            },
            CatalogStatement::Call {
                name,
                procedure,
                table,
                args,
            } => {
                dict.set_item("kind", "call")?;
                dict.set_item("name", name)?;
                dict.set_item("procedure", procedure)?;
                dict.set_item("catalog", table.catalog)?;
                dict.set_item("identifier", table.identifier)?;
                let py_args = PyDict::new(py);
                for (arg, value) in args {
                    match value {
                        CallArg::String(s) => py_args.set_item(arg, s)?,
                        CallArg::Int(i) => py_args.set_item(arg, i)?,
                        CallArg::Bool(b) => py_args.set_item(arg, b)?,
                        CallArg::Array(values) => py_args.set_item(arg, values)?,
                        CallArg::Map(pairs) => {
                            let map = PyDict::new(py);
                            for (k, v) in pairs {
                                map.set_item(k, v)?;
                            }
                            py_args.set_item(arg, map)?
                        },
                    }
                }
                dict.set_item("args", py_args)?;
            },
        }
        Ok(Some(dict))
    }

    /// Execute a query with the scans of the catalog tables it reads, by full name.
    #[pyo3(signature = (query, tables, params=None))]
    pub fn execute_with_catalog_tables(
        &self,
        py: Python<'_>,
        query: &str,
        tables: Vec<(Vec<String>, PyLazyFrame)>,
        params: Option<Wrap<QueryParams>>,
    ) -> PyResult<PyLazyFrame> {
        let tables = tables
            .into_iter()
            .map(|(full_name, lf)| (full_name, lf.ldf.into_inner()))
            .collect();
        let params = params.map(|p| p.0).unwrap_or_default();
        py.enter_polars(|| {
            self.context
                .write()
                .execute_with_catalog_tables(query, tables, &params)
        })
        .map(Into::into)
    }

    /// The predicate of the rows a catalog `DELETE` removes, planned against the scan of its
    /// table, by full name.
    pub fn catalog_delete_predicate(
        &self,
        py: Python<'_>,
        query: &str,
        tables: Vec<(Vec<String>, PyLazyFrame)>,
    ) -> PyResult<PyExpr> {
        let tables = tables
            .into_iter()
            .map(|(full_name, lf)| (full_name, lf.ldf.into_inner()))
            .collect();
        py.enter_polars(|| self.context.write().catalog_delete_predicate(query, tables))
            .map(Into::into)
    }

    /// The predicate of the `where` argument of a `CALL`, planned against the scan of its table.
    pub fn catalog_call_predicate(
        &self,
        py: Python<'_>,
        condition: &str,
        lf: PyLazyFrame,
    ) -> PyResult<PyExpr> {
        let lf = lf.ldf.into_inner();
        py.enter_polars(|| self.context.write().catalog_call_predicate(condition, lf))
            .map(Into::into)
    }

    /// The predicate of the rows a catalog `UPDATE` changes and the value of each `SET` column,
    /// planned against the scan of its table, by full name.
    pub fn catalog_update_plan(
        &self,
        py: Python<'_>,
        query: &str,
        tables: Vec<(Vec<String>, PyLazyFrame)>,
    ) -> PyResult<(PyExpr, Vec<(String, PyExpr)>)> {
        let tables = tables
            .into_iter()
            .map(|(full_name, lf)| (full_name, lf.ldf.into_inner()))
            .collect();
        let (predicate, values) =
            py.enter_polars(|| self.context.write().catalog_update_plan(query, tables))?;
        Ok((
            predicate.into(),
            values.into_iter().map(|(c, e)| (c, e.into())).collect(),
        ))
    }

    /// The source of a catalog `MERGE` (the columns it reads) and the keys of its `ON`, planned
    /// against the scans of the catalog tables it reads, by full name.
    pub fn catalog_merge_source(
        &self,
        py: Python<'_>,
        query: &str,
        tables: Vec<(Vec<String>, PyLazyFrame)>,
    ) -> PyResult<(PyLazyFrame, Vec<(String, PyExpr)>)> {
        let tables = tables
            .into_iter()
            .map(|(full_name, lf)| (full_name, lf.ldf.into_inner()))
            .collect();
        let plan = py.enter_polars(|| self.context.write().catalog_merge_source(query, tables))?;
        Ok((
            plan.source.into(),
            plan.keys.into_iter().map(|(c, e)| (c, e.into())).collect(),
        ))
    }

    /// A catalog `MERGE` of `source` into the scan of its target: the frame of matching pairs,
    /// the frame of source rows to insert, and per `WHEN MATCHED` and `WHEN NOT MATCHED` clause
    /// its kind and, per column written, the frame column holding the value.
    #[allow(clippy::type_complexity)]
    pub fn catalog_merge_plan(
        &self,
        py: Python<'_>,
        query: &str,
        tables: Vec<(Vec<String>, PyLazyFrame)>,
        source: PyLazyFrame,
    ) -> PyResult<(
        PyLazyFrame,
        PyLazyFrame,
        Vec<(String, usize, Vec<(String, PyExpr)>)>,
        Vec<(String, usize, Vec<(String, PyExpr)>)>,
    )> {
        let tables = tables
            .into_iter()
            .map(|(full_name, lf)| (full_name, lf.ldf.into_inner()))
            .collect();
        let source = source.ldf.into_inner();
        let plan = py.enter_polars(|| {
            self.context
                .write()
                .catalog_merge_plan(query, tables, source)
        })?;
        let clauses = |clauses: Vec<polars::sql::MergeClausePlan>| {
            clauses
                .into_iter()
                .map(|c| {
                    let columns = c.columns.into_iter().map(|(n, e)| (n, e.into())).collect();
                    (c.kind.to_string(), c.number, columns)
                })
                .collect()
        };
        Ok((
            plan.pairs.into(),
            plan.inserts.into(),
            clauses(plan.matched),
            clauses(plan.not_matched),
        ))
    }

    /// Extract table identifiers from a SQL query string.
    #[staticmethod]
    #[pyo3(signature = (query, include_schema=true, unique=false))]
    pub fn table_identifiers(
        query: &str,
        include_schema: bool,
        unique: bool,
    ) -> PyResult<Vec<String>> {
        extract_table_identifiers(query, include_schema, unique)
            .map_err(PyPolarsErr::from)
            .map_err(Into::into)
    }
}

fn ref_kind_name(kind: CatalogRefKind) -> &'static str {
    match kind {
        CatalogRefKind::Tag => "tag",
        CatalogRefKind::Branch => "branch",
    }
}
