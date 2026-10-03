//! Native Iceberg scan planning, loaded by Polars as an I/O plugin.
//!
//! Implements the contract `polars.io_plugin.iceberg.v1` ([`PLUGIN_V1`]). Polars calls it from a
//! blocking thread without the GIL. Planning runs on the plugin's own runtime ([`runtime`]); all
//! storage IO goes through the host, which executes it on the Polars IO runtime.
//!
//! Planning is implemented here (no dependency on another Iceberg library): [`spec`] parses table
//! metadata, [`avro`] and [`manifest`] decode manifest lists and manifests, [`planner`] selects
//! snapshots and assigns delete files, and [`resolve`] builds the output.
use polars_io_ext_ffi::common::FfiResult;
use polars_io_ext_ffi::iceberg_v1;

use crate::catch_panic;
use crate::iceberg::host::Host;

mod arrow_types;
mod avro;
mod error;
mod expr;
mod host;
mod manifest;
mod output;
mod planner;
mod prune;
mod request;
mod resolve;
mod runtime;
mod spec;
mod values;

pub static PLUGIN_V1: iceberg_v1::Plugin = iceberg_v1::Plugin { plan: plan_v1 };

unsafe extern "C" fn plan_v1(
    host: *const iceberg_v1::Host,
    request: *const iceberg_v1::FfiRequest,
) -> FfiResult<iceberg_v1::Output> {
    catch_panic(|| {
        let request = unsafe { request::parse(request)? };
        let host = unsafe { Host::from_raw(host, request.verbose)? };
        runtime::block_on(request.max_threads, resolve::resolve(host, &request))?.into_ffi()
    })
    .into()
}
