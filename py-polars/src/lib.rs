const fn _const_str_equality(l: &str, r: &str) -> bool {
    if l.len() != r.len() {
        return false;
    }

    let mut i = 0;
    while i < l.len() {
        if l.as_bytes()[i] != r.as_bytes()[i] {
            return false;
        }
        i += 1;
    }

    true
}
const _CORRECT_VERSION_SET: () = const {
    let module_version = polars_python::c_api::PYPOLARS_VERSION;
    let package_version = env!("CARGO_PKG_VERSION");

    // You probably need to update the PYPOLARS_VERSION, to match the package version.
    assert!(_const_str_equality(module_version, package_version));
};
pub use polars_python::c_api::polars;
