use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

pub mod dataframe;
pub mod error;
pub mod npy;
pub mod series;

use crate::{dataframe::PyDataFrame, series::PySeries};
use polars::chunked_array::builder::AlignedVec;
use polars::prelude::*;

macro_rules! create_aligned_buffer {
    ($name:ident, $type:ty) => {
        #[pyfunction]
        pub fn $name(size: usize) -> (Py<PyArray1<$type>>, usize) {
            let (buf, ptr) = npy::aligned_array::<$type>(size);
            (buf, ptr as usize)
        }
    };
}

create_aligned_buffer!(aligned_array_f16, f16);
create_aligned_buffer!(aligned_array_f32, f32);
create_aligned_buffer!(aligned_array_f64, f64);
create_aligned_buffer!(aligned_array_i8, i8);
create_aligned_buffer!(aligned_array_i16, i16);
create_aligned_buffer!(aligned_array_i32, i32);
create_aligned_buffer!(aligned_array_i64, i64);
create_aligned_buffer!(aligned_array_u8, u8);
create_aligned_buffer!(aligned_array_u16, u16);
create_aligned_buffer!(aligned_array_u32, u32);
create_aligned_buffer!(aligned_array_u64, u64);

#[pyfunction]
pub fn series_from_ptr_f64(name: &str, ptr: usize, len: usize) -> PySeries {
    let v: Vec<f64> = unsafe { npy::vec_from_ptr(ptr, len) };
    let av = AlignedVec::new(v).unwrap();
    let ca = ChunkedArray::new_from_aligned_vec(name, av);
    PySeries::new(Series::Float64(ca))
}

#[pymodule]
fn pypolars(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<PySeries>().unwrap();
    m.add_class::<PyDataFrame>().unwrap();
    m.add_wrapped(wrap_pyfunction!(aligned_array_f32)).unwrap();
    m.add_wrapped(wrap_pyfunction!(aligned_array_f64)).unwrap();
    m.add_wrapped(wrap_pyfunction!(aligned_array_i32)).unwrap();
    m.add_wrapped(wrap_pyfunction!(aligned_array_i64)).unwrap();
    m.add_wrapped(wrap_pyfunction!(series_from_ptr_f64))
        .unwrap();
    Ok(())
}
