use std::fs::File;
use std::io;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom, Write};
use std::path::PathBuf;

use polars::io::mmap::MmapBytesReader;
use polars_error::polars_warn;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::error::PyPolarsErr;
use crate::prelude::resolve_homedir;

#[derive(Clone)]
pub struct PyFileLikeObject {
    inner: PyObject,
}

/// Wraps a `PyObject`, and implements read, seek, and write for it.
impl PyFileLikeObject {
    /// Creates an instance of a `PyFileLikeObject` from a `PyObject`.
    /// To assert the object has the required methods methods,
    /// instantiate it with `PyFileLikeObject::require`
    pub fn new(object: PyObject) -> Self {
        PyFileLikeObject { inner: object }
    }

    pub fn as_buffer(&self) -> std::io::Cursor<Vec<u8>> {
        let data = self.as_file_buffer().into_inner();
        std::io::Cursor::new(data)
    }

    pub fn as_file_buffer(&self) -> Cursor<Vec<u8>> {
        let buf = Python::with_gil(|py| {
            let bytes = self
                .inner
                .call_method_bound(py, "read", (), None)
                .expect("no read method found");

            let bytes: &Bound<'_, PyBytes> = bytes
                .downcast_bound(py)
                .expect("Expecting to be able to downcast into bytes from read result.");

            bytes.as_bytes().to_vec()
        });

        Cursor::new(buf)
    }

    /// Same as `PyFileLikeObject::new`, but validates that the underlying
    /// python object has a `read`, `write`, and `seek` methods in respect to parameters.
    /// Will return a `TypeError` if object does not have `read`, `seek`, and `write` methods.
    pub fn with_requirements(
        object: PyObject,
        read: bool,
        write: bool,
        seek: bool,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            if read && object.getattr(py, "read").is_err() {
                return Err(PyErr::new::<PyTypeError, _>(
                    "Object does not have a .read() method.",
                ));
            }

            if seek && object.getattr(py, "seek").is_err() {
                return Err(PyErr::new::<PyTypeError, _>(
                    "Object does not have a .seek() method.",
                ));
            }

            if write && object.getattr(py, "write").is_err() {
                return Err(PyErr::new::<PyTypeError, _>(
                    "Object does not have a .write() method.",
                ));
            }

            Ok(PyFileLikeObject::new(object))
        })
    }
}

/// Extracts a string repr from, and returns an IO error to send back to rust.
fn pyerr_to_io_err(e: PyErr) -> io::Error {
    Python::with_gil(|py| {
        let e_as_object: PyObject = e.into_py(py);

        match e_as_object.call_method_bound(py, "__str__", (), None) {
            Ok(repr) => match repr.extract::<String>(py) {
                Ok(s) => io::Error::new(io::ErrorKind::Other, s),
                Err(_e) => io::Error::new(io::ErrorKind::Other, "An unknown error has occurred"),
            },
            Err(_) => io::Error::new(io::ErrorKind::Other, "Err doesn't have __str__"),
        }
    })
}

impl Read for PyFileLikeObject {
    fn read(&mut self, mut buf: &mut [u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let bytes = self
                .inner
                .call_method_bound(py, "read", (buf.len(),), None)
                .map_err(pyerr_to_io_err)?;

            let bytes: &Bound<'_, PyBytes> = bytes
                .downcast_bound(py)
                .expect("Expecting to be able to downcast into bytes from read result.");

            buf.write_all(bytes.as_bytes())?;

            bytes.len().map_err(pyerr_to_io_err)
        })
    }
}

impl Write for PyFileLikeObject {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let pybytes = PyBytes::new_bound(py, buf);

            let number_bytes_written = self
                .inner
                .call_method_bound(py, "write", (pybytes,), None)
                .map_err(pyerr_to_io_err)?;

            number_bytes_written.extract(py).map_err(pyerr_to_io_err)
        })
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        Python::with_gil(|py| {
            self.inner
                .call_method_bound(py, "flush", (), None)
                .map_err(pyerr_to_io_err)?;

            Ok(())
        })
    }
}

impl Seek for PyFileLikeObject {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, io::Error> {
        Python::with_gil(|py| {
            let (whence, offset) = match pos {
                SeekFrom::Start(i) => (0, i as i64),
                SeekFrom::Current(i) => (1, i),
                SeekFrom::End(i) => (2, i),
            };

            let new_position = self
                .inner
                .call_method_bound(py, "seek", (offset, whence), None)
                .map_err(pyerr_to_io_err)?;

            new_position.extract(py).map_err(pyerr_to_io_err)
        })
    }
}

pub trait FileLike: Read + Write + Seek + Sync + Send {}

impl FileLike for File {}
impl FileLike for PyFileLikeObject {}
impl MmapBytesReader for PyFileLikeObject {}

pub enum EitherRustPythonFile {
    Py(PyFileLikeObject),
    Rust(BufReader<File>),
}

///
/// # Arguments
/// * `truncate` - open or create a new file.
pub fn get_either_file(py_f: PyObject, truncate: bool) -> PyResult<EitherRustPythonFile> {
    Python::with_gil(|py| {
        if let Ok(pstring) = py_f.downcast_bound::<PyString>(py) {
            let s = pstring.to_cow()?;
            let file_path = std::path::Path::new(&*s);
            let file_path = resolve_homedir(file_path);
            let f = if truncate {
                File::create(file_path)?
            } else {
                polars_utils::open_file(&file_path).map_err(PyPolarsErr::from)?
            };
            let reader = BufReader::new(f);
            Ok(EitherRustPythonFile::Rust(reader))
        } else {
            let f = PyFileLikeObject::with_requirements(py_f, !truncate, truncate, !truncate)?;
            Ok(EitherRustPythonFile::Py(f))
        }
    })
}

pub fn get_file_like(f: PyObject, truncate: bool) -> PyResult<Box<dyn FileLike>> {
    use EitherRustPythonFile::*;
    match get_either_file(f, truncate)? {
        Py(f) => Ok(Box::new(f)),
        Rust(f) => Ok(Box::new(f.into_inner())),
    }
}

/// If the give file-like is a BytesIO, read its contents.
pub fn read_if_bytesio(py_f: Bound<PyAny>) -> Bound<PyAny> {
    if py_f.getattr("read").is_ok() {
        let Ok(bytes) = py_f.call_method0("getvalue") else {
            return py_f;
        };
        if bytes.downcast::<PyBytes>().is_ok() {
            return bytes.clone();
        }
    }
    py_f
}

/// Create reader from PyBytes or a file-like object. To get BytesIO to have
/// better performance, use read_if_bytesio() before calling this.
pub fn get_mmap_bytes_reader<'a>(
    py_f: &'a Bound<'a, PyAny>,
) -> PyResult<Box<dyn MmapBytesReader + 'a>> {
    get_mmap_bytes_reader_and_path(py_f).map(|t| t.0)
}

pub fn get_mmap_bytes_reader_and_path<'a>(
    py_f: &'a Bound<'a, PyAny>,
) -> PyResult<(Box<dyn MmapBytesReader + 'a>, Option<PathBuf>)> {
    // bytes object
    if let Ok(bytes) = py_f.downcast::<PyBytes>() {
        Ok((Box::new(Cursor::new(bytes.as_bytes())), None))
    }
    // string so read file
    else if let Ok(pstring) = py_f.downcast::<PyString>() {
        let s = pstring.to_cow()?;
        let p = std::path::Path::new(&*s);
        let p_resolved = resolve_homedir(p);
        let f = polars_utils::open_file(p_resolved).map_err(PyPolarsErr::from)?;
        Ok((Box::new(f), Some(p.to_path_buf())))
    }
    // hopefully a normal python file: with open(...) as f:.
    else {
        // we can still get a file name, inform the user of possibly wrong API usage.
        if py_f.getattr("read").is_ok() && py_f.getattr("name").is_ok() {
            polars_warn!("Polars found a filename. \
            Ensure you pass a path to the file instead of a python file object when possible for best \
            performance.")
        }
        // don't really know what we got here, just read.
        let f = Python::with_gil(|py| {
            PyFileLikeObject::with_requirements(py_f.to_object(py), true, false, true)
        })?;
        Ok((Box::new(f), None))
    }
}
