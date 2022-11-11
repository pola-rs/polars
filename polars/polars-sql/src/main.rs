#![allow(unused)]

#[cfg(target_os = "linux")]
use jemallocator::Jemalloc;

#[cfg(feature = "cli")]
mod cli;

#[global_allocator]
#[cfg(target_os = "linux")]
static ALLOC: Jemalloc = Jemalloc;

fn main() -> std::io::Result<()> {
    #[cfg(feature = "cli")]
    return cli::run();

    Ok(())
}
