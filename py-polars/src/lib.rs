use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod dataframe;
pub mod error;
pub mod npy;
pub mod series;

use crate::{dataframe::PyDataFrame, series::PySeries};

macro_rules! create_aligned_buffer {
    ($name:ident, $type:ty) => {
        #[pyfunction]
        pub fn $name(size: usize) -> (Py<PyArray1<$type>>, usize) {
            let (buf, ptr) = npy::aligned_array::<$type>(size);
            (buf, ptr as usize)
        }
    };
}

create_aligned_buffer!(aligned_array_f32, f32);
create_aligned_buffer!(aligned_array_f64, f64);
create_aligned_buffer!(aligned_array_i32, i32);
create_aligned_buffer!(aligned_array_i64, i64);

#[pymodule]
fn polars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_wrapped(wrap_pyfunction!(aligned_array_f32)).unwrap();
    m.add_wrapped(wrap_pyfunction!(aligned_array_f64)).unwrap();
    m.add_wrapped(wrap_pyfunction!(aligned_array_i32)).unwrap();
    m.add_wrapped(wrap_pyfunction!(aligned_array_i64)).unwrap();
    Ok(())
}
