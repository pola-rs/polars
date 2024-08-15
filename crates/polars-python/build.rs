fn main() {
    println!("cargo::rustc-check-cfg=cfg(allocator, values(\"default\", \"mimalloc\"))");
    println!("cargo:rerun-if-changed=build.rs");
    let channel = version_check::Channel::read().unwrap();
    if channel.is_nightly() {
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }
}
