fn main() {
    println!("cargo::rustc-check-cfg=cfg(allocator, values(\"default\", \"mimalloc\"))");
    println!(
        "cargo:rustc-env=TARGET={}",
        std::env::var("TARGET").unwrap()
    );
}
