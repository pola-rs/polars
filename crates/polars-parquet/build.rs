use std::process::Command;

fn main() {
    let output = Command::new("git")
        .args(&["rev-parse", "HEAD"])
        .output()
        .expect("`git` should be available");
    let git_hash = String::from_utf8(output.stdout)
        .expect("couldn't parse `git rev-parse HEAD` output");
    println!("cargo:rustc-env=POLARS_GIT_HASH={}", git_hash);
}