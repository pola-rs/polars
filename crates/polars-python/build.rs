fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=POLARS_BUILD_COMMIT");
    let channel = version_check::Channel::read().unwrap();
    if channel.is_nightly() {
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }
}
