use std::ffi::{CStr, CString};
use std::sync::RwLock;

use arrow::ffi::{import_field_from_c, ArrowSchema};
use libloading::Library;
use once_cell::sync::Lazy;
use polars_ffi::*;

use super::*;

static LOADED: Lazy<RwLock<PlHashMap<String, Library>>> = Lazy::new(Default::default);

fn get_lib(lib: &str) -> PolarsResult<&'static Library> {
    let lib_map = LOADED.read().unwrap();
    if let Some(library) = lib_map.get(lib) {
        // lifetime is static as we never remove libraries.
        Ok(unsafe { std::mem::transmute::<&Library, &'static Library>(library) })
    } else {
        drop(lib_map);
        let library = unsafe {
            Library::new(lib).map_err(|e| {
                PolarsError::ComputeError(format!("error loading dynamic library: {e}").into())
            })?
        };

        let mut lib_map = LOADED.write().unwrap();
        lib_map.insert(lib.to_string(), library);
        drop(lib_map);

        get_lib(lib)
    }
}

unsafe fn retrieve_error_msg(lib: &Library) -> CString {
    let symbol: libloading::Symbol<
        unsafe extern "C" fn() -> *mut std::os::raw::c_char,
    > = lib.get(b"get_last_error_message\0").unwrap();
    let msg_ptr = symbol();
    CString::from_raw(msg_ptr)
}

pub(super) unsafe fn call_plugin(
    s: &[Series],
    lib: &str,
    symbol: &str,
    kwargs: &CStr,
) -> PolarsResult<Series> {
    let lib = get_lib(lib)?;

    let symbol: libloading::Symbol<
        unsafe extern "C" fn(
            *const SeriesExport,
            usize,
            *const std::os::raw::c_char,
        ) -> *mut SeriesExport,
    > = lib.get(symbol.as_bytes()).unwrap();

    let n_args = s.len();

    let input = s.iter().map(export_series).collect::<Vec<_>>();
    let slice_ptr = input.as_ptr();
    let out = symbol(slice_ptr, n_args, kwargs.as_ptr());

    // The inputs get dropped when the ffi side calls the drop callback.
    for e in input {
        std::mem::forget(e);
    }

    if !out.is_null() {
        let out = Box::from_raw(out);
        import_series(*out)
    } else {
        let msg = retrieve_error_msg(lib);
        let msg = msg.to_string_lossy();
        polars_bail!(ComputeError: "the plugin failed with message: {}", msg)
    }

}

pub(super) unsafe fn plugin_field(
    fields: &[Field],
    lib: &str,
    symbol: &str,
) -> PolarsResult<Field> {
    let lib = get_lib(lib)?;

    let symbol: libloading::Symbol<unsafe extern "C" fn(*const ArrowSchema, usize, *mut ArrowSchema)> =
        lib.get(symbol.as_bytes()).unwrap();

    // we deallocate the fields buffer
    let fields = fields
        .iter()
        .map(|field| arrow::ffi::export_field_to_c(&field.to_arrow()))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let n_args = fields.len();
    let slice_ptr = fields.as_ptr();

    let mut return_value = ArrowSchema::empty();
    let return_value_ptr = &mut return_value as *mut ArrowSchema;
    symbol(slice_ptr, n_args, return_value_ptr);

    if return_value.is_null() {
        let msg = retrieve_error_msg(lib);
        let msg = msg.to_string_lossy();
        polars_bail!(ComputeError: "the plugin failed with message: {}", msg)
    } else {
        let arrow_field = import_field_from_c(&return_value)?;
        let out = Field::from(&arrow_field);
        Ok(out)
    }
}
