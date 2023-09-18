use std::ffi::{CStr, CString};
use std::ops::DerefMut;

use crate::{array::Array, datatypes::Field, error::Error};

use super::{export_array_to_c, export_field_to_c, import_array_from_c, import_field_from_c};
use super::{ArrowArray, ArrowArrayStream, ArrowSchema};

impl Drop for ArrowArrayStream {
    fn drop(&mut self) {
        match self.release {
            None => (),
            Some(release) => unsafe { release(self) },
        };
    }
}

impl ArrowArrayStream {
    /// Creates an empty [`ArrowArrayStream`] used to import from a producer.
    pub fn empty() -> Self {
        Self {
            get_schema: None,
            get_next: None,
            get_last_error: None,
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }
}

unsafe fn handle_error(iter: &mut ArrowArrayStream) -> Error {
    let error = unsafe { (iter.get_last_error.unwrap())(&mut *iter) };

    if error.is_null() {
        return Error::External(
            "C stream".to_string(),
            Box::new(Error::ExternalFormat("an unspecified error".to_string())),
        );
    }

    let error = unsafe { CStr::from_ptr(error) };
    Error::External(
        "C stream".to_string(),
        Box::new(Error::ExternalFormat(error.to_str().unwrap().to_string())),
    )
}

/// Implements an iterator of [`Array`] consumed from the [C stream interface](https://arrow.apache.org/docs/format/CStreamInterface.html).
pub struct ArrowArrayStreamReader<Iter: DerefMut<Target = ArrowArrayStream>> {
    iter: Iter,
    field: Field,
}

impl<Iter: DerefMut<Target = ArrowArrayStream>> ArrowArrayStreamReader<Iter> {
    /// Returns a new [`ArrowArrayStreamReader`]
    /// # Error
    /// Errors iff the [`ArrowArrayStream`] is out of specification,
    /// or was already released prior to calling this function.
    /// # Safety
    /// This method is intrinsically `unsafe` since it assumes that the `ArrowArrayStream`
    /// contains a valid Arrow C stream interface.
    /// In particular:
    /// * The `ArrowArrayStream` fulfills the invariants of the C stream interface
    /// * The schema `get_schema` produces fulfills the C data interface
    pub unsafe fn try_new(mut iter: Iter) -> Result<Self, Error> {
        if iter.release.is_none() {
            return Err(Error::InvalidArgumentError(
                "The C stream was already released".to_string(),
            ));
        };

        if iter.get_next.is_none() {
            return Err(Error::OutOfSpec(
                "The C stream MUST contain a non-null get_next".to_string(),
            ));
        };

        if iter.get_last_error.is_none() {
            return Err(Error::OutOfSpec(
                "The C stream MUST contain a non-null get_last_error".to_string(),
            ));
        };

        let mut field = ArrowSchema::empty();
        let status = if let Some(f) = iter.get_schema {
            unsafe { (f)(&mut *iter, &mut field) }
        } else {
            return Err(Error::OutOfSpec(
                "The C stream MUST contain a non-null get_schema".to_string(),
            ));
        };

        if status != 0 {
            return Err(unsafe { handle_error(&mut iter) });
        }

        let field = unsafe { import_field_from_c(&field)? };

        Ok(Self { iter, field })
    }

    /// Returns the field provided by the stream
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Advances this iterator by one array
    /// # Error
    /// Errors iff:
    /// * The C stream interface returns an error
    /// * The C stream interface returns an invalid array (that we can identify, see Safety below)
    /// # Safety
    /// Calling this iterator's `next` assumes that the [`ArrowArrayStream`] produces arrow arrays
    /// that fulfill the C data interface
    pub unsafe fn next(&mut self) -> Option<Result<Box<dyn Array>, Error>> {
        let mut array = ArrowArray::empty();
        let status = unsafe { (self.iter.get_next.unwrap())(&mut *self.iter, &mut array) };

        if status != 0 {
            return Some(Err(unsafe { handle_error(&mut self.iter) }));
        }

        // last paragraph of https://arrow.apache.org/docs/format/CStreamInterface.html#c.ArrowArrayStream.get_next
        array.release?;

        // Safety: assumed from the C stream interface
        unsafe { import_array_from_c(array, self.field.data_type.clone()) }
            .map(Some)
            .transpose()
    }
}

struct PrivateData {
    iter: Box<dyn Iterator<Item = Result<Box<dyn Array>, Error>>>,
    field: Field,
    error: Option<CString>,
}

unsafe extern "C" fn get_next(iter: *mut ArrowArrayStream, array: *mut ArrowArray) -> i32 {
    if iter.is_null() {
        return 2001;
    }
    let private = &mut *((*iter).private_data as *mut PrivateData);

    match private.iter.next() {
        Some(Ok(item)) => {
            // check that the array has the same data_type as field
            let item_dt = item.data_type();
            let expected_dt = private.field.data_type();
            if item_dt != expected_dt {
                private.error = Some(CString::new(format!("The iterator produced an item of data type {item_dt:?} but the producer expects data type {expected_dt:?}").as_bytes().to_vec()).unwrap());
                return 2001; // custom application specific error (since this is never a result of this interface)
            }

            std::ptr::write(array, export_array_to_c(item));

            private.error = None;
            0
        }
        Some(Err(err)) => {
            private.error = Some(CString::new(err.to_string().as_bytes().to_vec()).unwrap());
            2001 // custom application specific error (since this is never a result of this interface)
        }
        None => {
            let a = ArrowArray::empty();
            std::ptr::write_unaligned(array, a);
            private.error = None;
            0
        }
    }
}

unsafe extern "C" fn get_schema(iter: *mut ArrowArrayStream, schema: *mut ArrowSchema) -> i32 {
    if iter.is_null() {
        return 2001;
    }
    let private = &mut *((*iter).private_data as *mut PrivateData);

    std::ptr::write(schema, export_field_to_c(&private.field));
    0
}

unsafe extern "C" fn get_last_error(iter: *mut ArrowArrayStream) -> *const ::std::os::raw::c_char {
    if iter.is_null() {
        return std::ptr::null();
    }
    let private = &mut *((*iter).private_data as *mut PrivateData);

    private
        .error
        .as_ref()
        .map(|x| x.as_ptr())
        .unwrap_or(std::ptr::null())
}

unsafe extern "C" fn release(iter: *mut ArrowArrayStream) {
    if iter.is_null() {
        return;
    }
    let _ = Box::from_raw((*iter).private_data as *mut PrivateData);
    (*iter).release = None;
    // private drops automatically
}

/// Exports an iterator to the [C stream interface](https://arrow.apache.org/docs/format/CStreamInterface.html)
pub fn export_iterator(
    iter: Box<dyn Iterator<Item = Result<Box<dyn Array>, Error>>>,
    field: Field,
) -> ArrowArrayStream {
    let private_data = Box::new(PrivateData {
        iter,
        field,
        error: None,
    });

    ArrowArrayStream {
        get_schema: Some(get_schema),
        get_next: Some(get_next),
        get_last_error: Some(get_last_error),
        release: Some(release),
        private_data: Box::into_raw(private_data) as *mut ::std::os::raw::c_void,
    }
}
