#![allow(unused)]

#[cfg(feature = "cli")]
mod cli;

fn main() -> std::io::Result<()> {
    #[cfg(feature = "cli")]
    return cli::run();

    Ok(())
}
