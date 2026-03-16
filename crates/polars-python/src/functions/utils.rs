use polars::prelude::_set_check_length;
use pyo3::prelude::*;

#[pyfunction]
pub fn check_length(check: bool) {
    unsafe { _set_check_length(check) }
}

#[pyfunction(name = "get_engine_affinity")]
pub fn py_get_engine_affinity() -> PyResult<String> {
    Ok(polars_config::config()
        .engine_affinity()
        .as_static_str()
        .to_string())
}

#[pyfunction]
pub fn config_reload_env_vars() {
    polars_config::config().reload_env_vars();
}

#[pyfunction]
pub fn config_reload_env_var(var: &str) {
    polars_config::config().reload_env_var(var);
}
