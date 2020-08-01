use numpy::{Element, PyArray1};
use polars::prelude::*;
use pyo3::prelude::*;

/// Create an empty numpy array arrows 64 byte alignment
pub fn aligned_array<T: Element>(size: usize) -> Py<PyArray1<T>> {
    let mut buf: Vec<T> = Vec::with_capacity_aligned(size);
    unsafe { buf.set_len(size) }
    let gil = Python::acquire_gil();
    let python = gil.python();
    PyArray1::from_vec(python, buf).to_owned()
}
