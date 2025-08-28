mod samplers;

use polars::prelude::*;
use pyo3::prelude::*;
use pyo3_polars::error::PyPolarsErr;
use pyo3_polars::{PyDataFrame, PyExpr, PySchema};

use crate::samplers::PySampler;

#[pyclass]
pub struct RandomSource {
    columns: Vec<PySampler>,
    size_hint: usize,
    n_rows: usize,
    predicate: Option<Expr>,
    with_columns: Option<Vec<usize>>,
}

#[pymethods]
impl RandomSource {
    #[new]
    #[pyo3(signature = (columns, size_hint, n_rows))]
    fn new_source(
        columns: Vec<PySampler>,
        size_hint: Option<usize>,
        n_rows: Option<usize>,
    ) -> Self {
        let n_rows = n_rows.unwrap_or(usize::MAX);
        let size_hint = size_hint.unwrap_or(10_000);

        Self {
            columns,
            size_hint,
            n_rows,
            predicate: None,
            with_columns: None,
        }
    }

    fn schema(&self) -> PySchema {
        let schema = self
            .columns
            .iter()
            .map(|s| {
                let s = s.0.lock().unwrap();
                Field::new(s.name().into(), s.dtype())
            })
            .collect::<Schema>();
        PySchema(Arc::new(schema))
    }

    fn try_set_predicate(&mut self, predicate: PyExpr) {
        self.predicate = Some(predicate.0);
    }

    fn set_with_columns(&mut self, columns: Vec<String>) {
        let schema = self.schema().0;

        let indexes = columns
            .iter()
            .map(|name| {
                schema
                    .index_of(name.as_ref())
                    .expect("schema should be correct")
            })
            .collect();

        self.with_columns = Some(indexes)
    }

    fn next(&mut self) -> PyResult<Option<PyDataFrame>> {
        if self.n_rows > 0 {
            // Apply projection pushdown.
            // This prevents unneeded sampling.
            let s_iter = if let Some(idx) = &self.with_columns {
                Box::new(idx.iter().copied().map(|i| &self.columns[i]))
                    as Box<dyn Iterator<Item = _>>
            } else {
                Box::new(self.columns.iter())
            };

            let columns = s_iter
                .map(|s| {
                    let mut s = s.0.lock().unwrap();

                    // Apply slice pushdown.
                    // This prevents unneeded sampling.
                    s.next_n(std::cmp::min(self.size_hint, self.n_rows))
                        .into_column()
                })
                .collect::<Vec<_>>();

            let mut df = DataFrame::new(columns).map_err(PyPolarsErr::from)?;
            self.n_rows = self.n_rows.saturating_sub(self.size_hint);

            // Apply predicate pushdown.
            // This is done after the fact, but there could be sources where this could be applied
            // lower.
            if let Some(predicate) = &self.predicate {
                df = df
                    .lazy()
                    .filter(predicate.clone())
                    ._with_eager(true)
                    .collect()
                    .map_err(PyPolarsErr::from)?;
            }

            Ok(Some(PyDataFrame(df)))
        } else {
            Ok(None)
        }
    }
}

#[pymodule]
fn io_plugin(m: &Bound<PyModule>) -> PyResult<()> {
    m.add_class::<RandomSource>().unwrap();
    m.add_class::<PySampler>().unwrap();
    m.add_wrapped(wrap_pyfunction!(samplers::new_bernoulli))
        .unwrap();
    m.add_wrapped(wrap_pyfunction!(samplers::new_uniform))
        .unwrap();

    Ok(())
}
