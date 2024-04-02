fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let channel = version_check::Channel::read().unwrap();
    if channel.is_nightly() {
        println!("cargo:rustc-cfg=feature=\"nightly\"");
    }
    #[cfg(all(test, feature = "io_capnp"))]
    {
        capnpc::CompilerCommand::new()
        .src_prefix("src/io/capnp/read/tests/schema/")
        .file("src/io/capnp/read/tests/schema/test_all_types.capnp")
        .run()
        .expect("capnp schema compiler command failed");
    }
}
