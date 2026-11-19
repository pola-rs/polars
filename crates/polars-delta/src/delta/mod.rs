use polars_io_ext_ffi::common::FfiResult;
use polars_io_ext_ffi::delta_v1;
mod actions;
mod checkpoint;
mod error;
mod host;
mod log;
mod output;
mod runtime;
mod schema;
mod values;
pub static PLUGIN_V1: delta_v1::Plugin = delta_v1::Plugin {
    plan,
    decode_inline: crate::ffi::decode_inline,
    decode_frame: crate::ffi::decode_frame,
    check_file_version: crate::ffi::check_file_version,
};
unsafe extern "C" fn plan(
    host: *const delta_v1::Host,
    request: *const delta_v1::FfiRequest,
) -> FfiResult<delta_v1::Output> {
    crate::catch_panic("the planner", || {
        if request.is_null() {
            return Err(error::invalid("null request"));
        }
        let request = delta_v1::Request::from_ffi(unsafe { &*request })
            .map_err(|e| error::invalid(e.to_string()))?;
        if request.version.is_some() && request.timestamp_ms.is_some() {
            return Err(error::input("version and timestamp are mutually exclusive"));
        }
        if request.version.is_some_and(|v| v < 0) {
            return Err(error::input("negative Delta version"));
        }
        let host = unsafe { host::Host::from_raw(host, request.verbose)? };
        runtime::block_on(request.max_threads, output::resolve(host, &request))
    })
    .into()
}
#[cfg(test)]
mod tests;
