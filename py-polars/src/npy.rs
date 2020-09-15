use ndarray::IntoDimension;
use numpy::{
    npyffi::{self, flags, types::npy_intp},
    ToNpyDims, PY_ARRAY_API,
};
use numpy::{Element, PyArray1};
use polars::prelude::*;
use pyo3::methods::PyMethods;
use pyo3::prelude::*;
use pyo3::proto_methods::PyProtoMethods;
use pyo3::pyclass::{PyClassAlloc, PyClassSend, ThreadCheckerStub};
use pyo3::pyclass_slots::PyClassDummySlot;
use pyo3::{type_object, PyClass};
use std::{mem, ptr};

pub(crate) struct SliceBox<T> {
    pub(crate) data: *mut [T],
}

impl<T> SliceBox<T> {
    pub(crate) fn new(value: Box<[T]>) -> Self {
        SliceBox {
            data: Box::into_raw(value),
        }
    }
}
impl<T> PyClassAlloc for SliceBox<T> {}
impl<T> PyClass for SliceBox<T> {
    type Dict = PyClassDummySlot;
    type WeakRef = PyClassDummySlot;
    type BaseNativeType = PyAny;
}

unsafe impl<T> type_object::PyTypeInfo for SliceBox<T> {
    type Type = ();
    type BaseType = PyAny;
    type BaseLayout = pyo3::pycell::PyCellBase<PyAny>;
    type Layout = PyCell<Self>;
    type Initializer = PyClassInitializer<Self>;
    type AsRefTarget = PyCell<Self>;
    const NAME: &'static str = "SliceBox";
    const MODULE: Option<&'static str> = Some("_rust_numpy");
    const DESCRIPTION: &'static str = "Memory store for PyArray using rust's Box<[T]> \0";
    const FLAGS: usize = 0;

    #[inline]
    fn type_object_raw(py: pyo3::Python) -> *mut pyo3::ffi::PyTypeObject {
        use pyo3::type_object::LazyStaticType;
        static TYPE_OBJECT: LazyStaticType = LazyStaticType::new();
        TYPE_OBJECT.get_or_init::<Self>(py)
    }
}

// Some stubs to use PyClass
impl<T> PyMethods for SliceBox<T> {}
impl<T> PyProtoMethods for SliceBox<T> {}
unsafe impl<T> Send for SliceBox<T> {}
impl<T> PyClassSend for SliceBox<T> {
    type ThreadChecker = ThreadCheckerStub<Self>;
}

impl<T> Drop for SliceBox<T> {
    fn drop(&mut self) {
        let _boxed_slice = unsafe { Box::from_raw(self.data) };
    }
}
/// Create an empty numpy array arrows 64 byte alignment
pub fn aligned_array<T: Element>(size: usize) -> (Py<PyArray1<T>>, *mut T) {
    let mut buf: Vec<T> = Vec::with_capacity_aligned(size);
    unsafe { buf.set_len(size) }
    let gil = Python::acquire_gil();
    let py = gil.python();
    // PyArray1::from_vec(python, buf).to_owned()

    // modified from
    // numpy-0.10.0/src/array.rs:375

    let len = buf.len();
    let buffer_ptr = buf.as_mut_ptr();
    let slice = buf.into_boxed_slice();

    let dims = [len].into_dimension();
    let strides = [mem::size_of::<T>() as npy_intp];
    let container = SliceBox::new(slice);
    let data_ptr = container.data;
    unsafe {
        let ptr = PY_ARRAY_API.PyArray_New(
            PY_ARRAY_API.get_type_object(npyffi::ArrayType::PyArray_Type),
            dims.ndim_cint(),
            dims.as_dims_ptr(),
            T::ffi_dtype() as i32,
            strides.as_ptr() as *mut _, // strides
            data_ptr as _,              // data
            mem::size_of::<T>() as i32, // itemsize
            flags::NPY_ARRAY_OUT_ARRAY, // flag // set to read only to prevent segfault
            ptr::null_mut(),            //obj
        );
        // I believe this make Python owner of the memory

        // let cell = pyo3::PyClassInitializer::from(container)
        //     .create_cell(py)
        //     .expect("Object creation failed.");

        // if we uncomment this, do we leak memory?
        // PY_ARRAY_API.PyArray_SetBaseObject(ptr as *mut npyffi::PyArrayObject, cell as _);
        mem::forget(container);
        (PyArray1::from_owned_ptr(py, ptr).to_owned(), buffer_ptr)
    }
}

pub unsafe fn vec_from_ptr<T>(ptr: usize, len: usize) -> Vec<T> {
    let ptr = ptr as *mut T;
    Vec::from_raw_parts(ptr, len, len)
}
