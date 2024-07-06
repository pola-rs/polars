use std::ffi::CStr;
use std::sync::RwLock;

use arrow::ffi::{import_field_from_c, ArrowSchema};
use libloading::Library;
use once_cell::sync::Lazy;

use super::*;

type PluginAndVersion = (Library, u16, u16);
static LOADED: Lazy<RwLock<PlHashMap<String, PluginAndVersion>>> = Lazy::new(Default::default);

fn get_lib(lib: &str) -> PolarsResult<&'static PluginAndVersion> {
    let lib_map = LOADED.read().unwrap();
    if let Some(library) = lib_map.get(lib) {
        // lifetime is static as we never remove libraries.
        Ok(unsafe { std::mem::transmute::<&PluginAndVersion, &'static PluginAndVersion>(library) })
    } else {
        drop(lib_map);
        let library = unsafe {
            Library::new(lib).map_err(|e| {
                PolarsError::ComputeError(format!("error loading dynamic library: {e}").into())
            })?
        };
        let version_function: libloading::Symbol<unsafe extern "C" fn() -> u32> = unsafe {
            library
                .get("_polars_plugin_get_version".as_bytes())
                .unwrap()
        };

        let version = unsafe { version_function() };
        let major = (version >> 16) as u16;
        let minor = version as u16;

        let mut lib_map = LOADED.write().unwrap();
        lib_map.insert(lib.to_string(), (library, major, minor));
        drop(lib_map);

        get_lib(lib)
    }
}

unsafe fn retrieve_error_msg(lib: &Library) -> &CStr {
    let symbol: libloading::Symbol<unsafe extern "C" fn() -> *mut std::os::raw::c_char> =
        lib.get(b"_polars_plugin_get_last_error_message\0").unwrap();
    let msg_ptr = symbol();
    CStr::from_ptr(msg_ptr)
}

pub(super) unsafe fn call_plugin(
    s: &[Series],
    lib: &str,
    symbol: &str,
    kwargs: &[u8],
) -> PolarsResult<Series> {
    let plugin = get_lib(lib)?;
    let lib = &plugin.0;
    let major = plugin.1;

    if major == 0 {
        use polars_ffi::version_0::*;
        // *const SeriesExport: pointer to Box<SeriesExport>
        // * usize: length of that pointer
        // *const u8: pointer to &[u8]
        // usize: length of the u8 slice
        // *mut SeriesExport: pointer where return value should be written.
        // *const CallerContext
        let symbol: libloading::Symbol<
            unsafe extern "C" fn(
                *const SeriesExport,
                usize,
                *const u8,
                usize,
                *mut SeriesExport,
                *const CallerContext,
            ),
        > = lib
            .get(format!("_polars_plugin_{}", symbol).as_bytes())
            .unwrap();

        let input = s.iter().map(export_series).collect::<Vec<_>>();
        let input_len = s.len();
        let slice_ptr = input.as_ptr();

        let kwargs_ptr = kwargs.as_ptr();
        let kwargs_len = kwargs.len();

        let mut return_value = SeriesExport::empty();
        let return_value_ptr = &mut return_value as *mut SeriesExport;
        let context = CallerContext::default();
        let context_ptr = &context as *const CallerContext;
        symbol(
            slice_ptr,
            input_len,
            kwargs_ptr,
            kwargs_len,
            return_value_ptr,
            context_ptr,
        );

        // The inputs get dropped when the ffi side calls the drop callback.
        for e in input {
            std::mem::forget(e);
        }

        if !return_value.is_null() {
            import_series(return_value)
        } else {
            let msg = retrieve_error_msg(lib);
            let msg = msg.to_string_lossy();
            check_panic(msg.as_ref())?;
            polars_bail!(ComputeError: "the plugin failed with message: {}", msg)
        }
    } else {
        polars_bail!(ComputeError: "this polars engine doesn't support plugin version: {}", major)
    }
}

pub(super) unsafe fn plugin_field(
    fields: &[Field],
    lib: &str,
    symbol: &str,
    kwargs: &[u8],
) -> PolarsResult<Field> {
    let plugin = get_lib(lib)?;
    let lib = &plugin.0;
    let major = plugin.1;
    let minor = plugin.2;

    // we deallocate the fields buffer
    let ffi_fields = fields
        .iter()
        .map(|field| arrow::ffi::export_field_to_c(&field.to_arrow(CompatLevel::newest())))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let n_args = ffi_fields.len();
    let slice_ptr = ffi_fields.as_ptr();

    let mut return_value = ArrowSchema::empty();
    let return_value_ptr = &mut return_value as *mut ArrowSchema;

    if major == 0 {
        match minor {
            0 => {
                let views = fields.iter().any(|field| field.dtype.contains_views());
                polars_ensure!(!views, ComputeError: "cannot call plugin\n\nThis Polars' version has a different 'binary/string' layout. Please compile with latest 'pyo3-polars'");

                // *const ArrowSchema: pointer to heap Box<ArrowSchema>
                // usize: length of the boxed slice
                // *mut ArrowSchema: pointer where the return value can be written
                let symbol: libloading::Symbol<
                    unsafe extern "C" fn(*const ArrowSchema, usize, *mut ArrowSchema),
                > = lib
                    .get((format!("_polars_plugin_field_{}", symbol)).as_bytes())
                    .unwrap();
                symbol(slice_ptr, n_args, return_value_ptr);
            },
            1 => {
                // *const ArrowSchema: pointer to heap Box<ArrowSchema>
                // usize: length of the boxed slice
                // *mut ArrowSchema: pointer where the return value can be written
                // *const u8: pointer to &[u8] (kwargs)
                // usize: length of the u8 slice
                let symbol: libloading::Symbol<
                    unsafe extern "C" fn(
                        *const ArrowSchema,
                        usize,
                        *mut ArrowSchema,
                        *const u8,
                        usize,
                    ),
                > = lib
                    .get((format!("_polars_plugin_field_{}", symbol)).as_bytes())
                    .unwrap();

                let kwargs_ptr = kwargs.as_ptr();
                let kwargs_len = kwargs.len();

                symbol(slice_ptr, n_args, return_value_ptr, kwargs_ptr, kwargs_len);
            },
            _ => {
                polars_bail!(ComputeError: "this Polars engine doesn't support plugin version: {}-{}", major, minor)
            },
        }
        if !return_value.is_null() {
            let arrow_field = import_field_from_c(&return_value)?;
            let out = Field::from(&arrow_field);
            Ok(out)
        } else {
            let msg = retrieve_error_msg(lib);
            let msg = msg.to_string_lossy();
            check_panic(msg.as_ref())?;
            polars_bail!(ComputeError: "the plugin failed with message: {}", msg)
        }
    } else {
        polars_bail!(ComputeError: "this Polars engine doesn't support plugin version: {}", major)
    }
}

fn check_panic(msg: &str) -> PolarsResult<()> {
    polars_ensure!(msg != "PANIC", ComputeError: "the plugin panicked\n\nThe message is suppressed. Set POLARS_VERBOSE=1 to send the panic message to stderr.");
    Ok(())
}
