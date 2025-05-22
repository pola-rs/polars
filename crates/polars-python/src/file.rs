use std::borrow::Cow;
#[cfg(target_family = "unix")]
use std::fs;
use std::fs::File;
use std::io;
use std::io::{Cursor, ErrorKind, Read, Seek, SeekFrom, Write};
#[cfg(target_family = "unix")]
use std::os::fd::{FromRawFd, RawFd};
use std::path::PathBuf;
use std::sync::Arc;

use polars::io::mmap::MmapBytesReader;
use polars::prelude::file::DynWriteable;
use polars::prelude::sync_on_close::SyncOnCloseType;
use polars_error::polars_err;
use polars_utils::create_file;
use polars_utils::file::{ClosableFile, WriteClose};
use polars_utils::mmap::MemSlice;
use pyo3::IntoPyObjectExt;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString, PyStringMethods};

use crate::error::PyPolarsErr;
use crate::prelude::resolve_homedir;

pub(crate) struct PyFileLikeObject {
    inner: PyObject,
    /// The object expects a string instead of a bytes for `write`.
    expects_str: bool,
    /// The object has a flush method.
    has_flush: bool,
}

impl WriteClose for PyFileLikeObject {}
impl DynWriteable for PyFileLikeObject {
    fn as_dyn_write(&self) -> &(dyn io::Write + Send + 'static) {
        self as _
    }
    fn as_mut_dyn_write(&mut self) -> &mut (dyn io::Write + Send + 'static) {
        self as _
    }
    fn close(self: Box<Self>) -> io::Result<()> {
        Ok(())
    }
    fn sync_on_close(&mut self, _sync_on_close: SyncOnCloseType) -> io::Result<()> {
        Ok(())
    }
}

impl Clone for PyFileLikeObject {
    fn clone(&self) -> Self {
        Python::with_gil(|py| Self {
            inner: self.inner.clone_ref(py),
            expects_str: self.expects_str,
            has_flush: self.has_flush,
        })
    }
}

/// Wraps a `PyObject`, and implements read, seek, and write for it.
impl PyFileLikeObject {
    /// Creates an instance of a `PyFileLikeObject` from a `PyObject`.
    /// To assert the object has the required methods,
    /// instantiate it with `PyFileLikeObject::require`
    pub(crate) fn new(object: PyObject, expects_str: bool, has_flush: bool) -> Self {
        PyFileLikeObject {
            inner: object,
            expects_str,
            has_flush,
        }
    }

    pub(crate) fn to_memslice(&self) -> MemSlice {
        Python::with_gil(|py| {
            let bytes = self
                .inner
                .call_method(py, "read", (), None)
                .expect("no read method found");

            if let Ok(b) = bytes.downcast_bound::<PyBytes>(py) {
                return MemSlice::from_arc(b.as_bytes(), Arc::new(bytes.clone_ref(py)));
            }

            if let Ok(b) = bytes.downcast_bound::<PyString>(py) {
                return match b.to_cow().expect("PyString is not valid UTF-8") {
                    Cow::Borrowed(v) => {
                        MemSlice::from_arc(v.as_bytes(), Arc::new(bytes.clone_ref(py)))
                    },
                    Cow::Owned(v) => MemSlice::from_vec(v.into_bytes()),
                };
            }

            panic!("Expecting to be able to downcast into bytes from read result.");
        })
    }

    /// Validates that the underlying
    /// python object has a `read`, `write`, and `seek` methods in respect to parameters.
    /// Will return a `TypeError` if object does not have `read`, `seek`, and `write` methods.
    pub(crate) fn ensure_requirements(
        object: &Bound<PyAny>,
        read: bool,
        write: bool,
        seek: bool,
    ) -> PyResult<()> {
        if read && object.getattr("read").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .read() method.",
            ));
        }

        if seek && object.getattr("seek").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .seek() method.",
            ));
        }

        if write && object.getattr("write").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .write() method.",
            ));
        }

        Ok(())
    }
}

/// Extracts a string repr from, and returns an IO error to send back to rust.
fn pyerr_to_io_err(e: PyErr) -> io::Error {
    Python::with_gil(|py| {
        let e_as_object: PyObject = e.into_py_any(py).unwrap();

        match e_as_object.call_method(py, "__str__", (), None) {
            Ok(repr) => match repr.extract::<String>(py) {
                Ok(s) => io::Error::other(s),
                Err(_e) => io::Error::other("An unknown error has occurred"),
            },
            Err(_) => io::Error::other("Err doesn't have __str__"),
        }
    })
}

impl Read for PyFileLikeObject {
    fn read(&mut self, mut buf: &mut [u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let bytes = self
                .inner
                .call_method(py, "read", (buf.len(),), None)
                .map_err(pyerr_to_io_err)?;

            let opt_bytes = bytes.downcast_bound::<PyBytes>(py);

            if let Ok(bytes) = opt_bytes {
                buf.write_all(bytes.as_bytes())?;

                bytes.len().map_err(pyerr_to_io_err)
            } else if let Ok(s) = bytes.downcast_bound::<PyString>(py) {
                let s = s.to_cow().map_err(pyerr_to_io_err)?;
                buf.write_all(s.as_bytes())?;
                Ok(s.len())
            } else {
                Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    polars_err!(InvalidOperation: "could not read from input"),
                ))
            }
        })
    }
}

impl Write for PyFileLikeObject {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let number_bytes_written = if self.expects_str {
                self.inner.call_method(
                    py,
                    "write",
                    (PyString::new(
                        py,
                        std::str::from_utf8(buf).map_err(io::Error::other)?,
                    ),),
                    None,
                )
            } else {
                self.inner
                    .call_method(py, "write", (PyBytes::new(py, buf),), None)
            }
            .map_err(pyerr_to_io_err)?;

            let n = number_bytes_written.extract(py).map_err(pyerr_to_io_err)?;

            Ok(n)
        })
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        if self.has_flush {
            Python::with_gil(|py| {
                self.inner
                    .call_method(py, "flush", (), None)
                    .map_err(pyerr_to_io_err)
            })?;
        }

        Ok(())
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
                .call_method(py, "seek", (offset, whence), None)
                .map_err(pyerr_to_io_err)?;

            new_position.extract(py).map_err(pyerr_to_io_err)
        })
    }
}

pub(crate) trait FileLike: Read + Write + Seek + Sync + Send {}

impl FileLike for File {}
impl FileLike for ClosableFile {}
impl FileLike for PyFileLikeObject {}
impl MmapBytesReader for PyFileLikeObject {}

pub(crate) enum EitherRustPythonFile {
    Py(PyFileLikeObject),
    Rust(ClosableFile),
}

impl EitherRustPythonFile {
    pub(crate) fn into_dyn(self) -> Box<dyn FileLike> {
        match self {
            EitherRustPythonFile::Py(f) => Box::new(f),
            EitherRustPythonFile::Rust(f) => Box::new(f),
        }
    }

    fn into_scan_source_input(self) -> PythonScanSourceInput {
        match self {
            EitherRustPythonFile::Py(f) => PythonScanSourceInput::Buffer(f.to_memslice()),
            EitherRustPythonFile::Rust(f) => PythonScanSourceInput::File(f),
        }
    }

    pub(crate) fn into_writeable(self) -> Box<dyn DynWriteable> {
        match self {
            Self::Py(f) => Box::new(f),
            Self::Rust(f) => Box::new(f),
        }
    }
}

pub(crate) enum PythonScanSourceInput {
    Buffer(MemSlice),
    Path(PathBuf),
    File(ClosableFile),
}

pub(crate) fn try_get_pyfile(
    py: Python<'_>,
    py_f: Bound<'_, PyAny>,
    write: bool,
) -> PyResult<(EitherRustPythonFile, Option<PathBuf>)> {
    let io = py.import("io")?;
    let is_utf8_encoding = |py_f: &Bound<PyAny>| -> PyResult<bool> {
        let encoding = py_f.getattr("encoding")?;
        let encoding = encoding.extract::<Cow<str>>()?;
        Ok(encoding.eq_ignore_ascii_case("utf-8") || encoding.eq_ignore_ascii_case("utf8"))
    };

    #[cfg(target_family = "unix")]
    if let Some(fd) = (py_f.is_exact_instance(&io.getattr("FileIO").unwrap())
        || (py_f.is_exact_instance(&io.getattr("BufferedReader").unwrap())
            || py_f.is_exact_instance(&io.getattr("BufferedWriter").unwrap())
            || py_f.is_exact_instance(&io.getattr("BufferedRandom").unwrap())
            || py_f.is_exact_instance(&io.getattr("BufferedRWPair").unwrap())
            || (py_f.is_exact_instance(&io.getattr("TextIOWrapper").unwrap())
                && is_utf8_encoding(&py_f)?))
            && if write {
                // invalidate read buffer
                py_f.call_method0("flush").is_ok()
            } else {
                // flush write buffer
                py_f.call_method1("seek", (0, 1)).is_ok()
            })
    .then(|| {
        py_f.getattr("fileno")
            .and_then(|fileno| fileno.call0())
            .and_then(|fileno| fileno.extract::<libc::c_int>())
            .ok()
    })
    .flatten()
    .map(|fileno| unsafe {
        // `File::from_raw_fd()` takes the ownership of the file descriptor.
        // When the File is dropped, it closes the file descriptor.
        // This is undesired - the Python file object will become invalid.
        // Therefore, we duplicate the file descriptor here.
        // Closing the duplicated file descriptor will not close
        // the original file descriptor;
        // and the status, e.g. stream position, is still shared with
        // the original file descriptor.
        // We use `F_DUPFD_CLOEXEC` here instead of `dup()`
        // because it also sets the `O_CLOEXEC` flag on the duplicated file descriptor,
        // which `dup()` clears.
        // `open()` in both Rust and Python automatically set `O_CLOEXEC` flag;
        // it prevents leaking file descriptors across processes,
        // and we want to be consistent with them.
        // `F_DUPFD_CLOEXEC` is defined in POSIX.1-2008
        // and is present on all alive UNIX(-like) systems.
        libc::fcntl(fileno, libc::F_DUPFD_CLOEXEC, 0)
    })
    .filter(|fileno| *fileno != -1)
    .map(|fileno| fileno as RawFd)
    {
        return Ok((
            EitherRustPythonFile::Rust(unsafe { File::from_raw_fd(fd).into() }),
            // This works on Linux and BSD with procfs mounted,
            // otherwise it fails silently.
            fs::canonicalize(format!("/proc/self/fd/{fd}")).ok(),
        ));
    }

    // Unwrap TextIOWrapper
    // Allow subclasses to allow things like pytest.capture.CaptureIO
    let py_f = if py_f
        .is_instance(&io.getattr("TextIOWrapper").unwrap())
        .unwrap_or_default()
    {
        if !is_utf8_encoding(&py_f)? {
            return Err(PyPolarsErr::from(
                polars_err!(InvalidOperation: "file encoding is not UTF-8"),
            )
            .into());
        }
        // XXX: we have to clear buffer here.
        // Is there a better solution?
        if write {
            py_f.call_method0("flush")?;
        } else {
            py_f.call_method1("seek", (0, 1))?;
        }
        py_f.getattr("buffer")?
    } else {
        py_f
    };
    PyFileLikeObject::ensure_requirements(&py_f, !write, write, !write)?;
    let expects_str = py_f.is_instance(&io.getattr("TextIOBase").unwrap())?;
    let has_flush = py_f
        .getattr_opt("flush")?
        .is_some_and(|flush| flush.is_callable());
    let f = PyFileLikeObject::new(py_f.unbind(), expects_str, has_flush);
    Ok((EitherRustPythonFile::Py(f), None))
}

pub(crate) fn get_python_scan_source_input(
    py_f: PyObject,
    write: bool,
) -> PyResult<PythonScanSourceInput> {
    Python::with_gil(|py| {
        let py_f = py_f.into_bound(py);

        // CPython has some internal tricks that means much of the time
        // BytesIO.getvalue() involves no memory copying, unlike
        // BytesIO.read(). So we want to handle BytesIO specially in order
        // to save memory.
        let py_f = read_if_bytesio(py_f);

        // If the pyobject is a `bytes` class
        if let Ok(b) = py_f.downcast::<PyBytes>() {
            return Ok(PythonScanSourceInput::Buffer(MemSlice::from_arc(
                b.as_bytes(),
                // We want to specifically keep alive the PyBytes object.
                Arc::new(b.clone().unbind()),
            )));
        }

        if let Ok(s) = py_f.extract::<Cow<str>>() {
            let file_path = resolve_homedir(&&*s);
            Ok(PythonScanSourceInput::Path(file_path))
        } else {
            Ok(try_get_pyfile(py, py_f, write)?.0.into_scan_source_input())
        }
    })
}

fn get_either_buffer_or_path(
    py_f: PyObject,
    write: bool,
) -> PyResult<(EitherRustPythonFile, Option<PathBuf>)> {
    Python::with_gil(|py| {
        let py_f = py_f.into_bound(py);
        if let Ok(s) = py_f.extract::<Cow<str>>() {
            let file_path = resolve_homedir(&&*s);
            let f = if write {
                create_file(&file_path).map_err(PyPolarsErr::from)?
            } else {
                polars_utils::open_file(&file_path).map_err(PyPolarsErr::from)?
            };
            Ok((EitherRustPythonFile::Rust(f.into()), Some(file_path)))
        } else {
            try_get_pyfile(py, py_f, write)
        }
    })
}

///
/// # Arguments
/// * `write` - open for writing; will truncate existing file and create new file if not.
pub(crate) fn get_either_file(py_f: PyObject, write: bool) -> PyResult<EitherRustPythonFile> {
    Ok(get_either_buffer_or_path(py_f, write)?.0)
}

pub(crate) fn get_file_like(f: PyObject, truncate: bool) -> PyResult<Box<dyn FileLike>> {
    Ok(get_either_file(f, truncate)?.into_dyn())
}

/// If the give file-like is a BytesIO, read its contents in a memory-efficient
/// way.
fn read_if_bytesio(py_f: Bound<PyAny>) -> Bound<PyAny> {
    let bytes_io = py_f.py().import("io").unwrap().getattr("BytesIO").unwrap();
    if py_f.is_instance(&bytes_io).unwrap() {
        // Note that BytesIO has some memory optimizations ensuring that much of
        // the time getvalue() doesn't need to copy the underlying data:
        let Ok(bytes) = py_f.call_method0("getvalue") else {
            return py_f;
        };
        return bytes;
    }
    py_f
}

/// Create reader from PyBytes or a file-like object.
pub(crate) fn get_mmap_bytes_reader(py_f: &Bound<PyAny>) -> PyResult<Box<dyn MmapBytesReader>> {
    get_mmap_bytes_reader_and_path(py_f).map(|t| t.0)
}

pub(crate) fn get_mmap_bytes_reader_and_path(
    py_f: &Bound<PyAny>,
) -> PyResult<(Box<dyn MmapBytesReader>, Option<PathBuf>)> {
    let py_f = read_if_bytesio(py_f.clone());

    // bytes object
    if let Ok(bytes) = py_f.downcast::<PyBytes>() {
        Ok((
            Box::new(Cursor::new(MemSlice::from_arc(
                bytes.as_bytes(),
                Arc::new(py_f.clone().unbind()),
            ))),
            None,
        ))
    }
    // string so read file
    else {
        match get_either_buffer_or_path(py_f.to_owned().unbind(), false)? {
            (EitherRustPythonFile::Rust(f), path) => Ok((Box::new(f), path)),
            (EitherRustPythonFile::Py(f), path) => {
                Ok((Box::new(Cursor::new(f.to_memslice())), path))
            },
        }
    }
}
