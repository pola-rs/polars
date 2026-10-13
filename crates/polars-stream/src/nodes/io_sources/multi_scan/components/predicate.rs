use polars_io::predicates::ScanIOPredicate;

#[derive(Debug, Clone)]
pub struct Predicate {
    pub scan_io_predicate: ScanIOPredicate,
    #[cfg(feature = "python")]
    pub py_filter_exprs: Option<std::sync::Arc<pyo3::Py<pyo3::PyAny>>>,
}
