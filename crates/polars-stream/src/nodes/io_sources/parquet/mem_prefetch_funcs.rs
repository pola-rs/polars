pub(super) use polars_utils::mem::{
    force_populate_read, madvise_populate_read, madvise_sequential, madvise_willneed, prefetch_l2,
};
pub(super) fn no_prefetch(_: &[u8]) {}

pub(super) fn get_memory_prefetch_func(verbose: bool) -> fn(&[u8]) -> () {
    let memory_prefetch_func = match std::env::var("POLARS_MEMORY_PREFETCH").ok().as_deref() {
        None => {
            // madvise_willneed performed the best on both MacOS on Apple Silicon and Ubuntu on x86-64,
            // using PDS-H query 3 SF=10 after clearing file cache as a benchmark.
            #[cfg(target_family = "unix")]
            {
                madvise_willneed
            }
            #[cfg(not(target_family = "unix"))]
            {
                no_prefetch
            }
        },
        Some("no_prefetch") => no_prefetch,
        Some("prefetch_l2") => prefetch_l2,
        Some("madvise_sequential") => {
            #[cfg(target_family = "unix")]
            {
                madvise_sequential
            }
            #[cfg(not(target_family = "unix"))]
            {
                panic!("POLARS_MEMORY_PREFETCH=madvise_sequential is not supported by this system");
            }
        },
        Some("madvise_willneed") => {
            #[cfg(target_family = "unix")]
            {
                madvise_willneed
            }
            #[cfg(not(target_family = "unix"))]
            {
                panic!("POLARS_MEMORY_PREFETCH=madvise_willneed is not supported by this system");
            }
        },
        Some("madvise_populate_read") => {
            #[cfg(target_os = "linux")]
            {
                madvise_populate_read
            }
            #[cfg(not(target_os = "linux"))]
            {
                panic!(
                    "POLARS_MEMORY_PREFETCH=madvise_populate_read is not supported by this system"
                );
            }
        },
        Some("force_populate_read") => force_populate_read,
        Some(v) => panic!("invalid value for POLARS_MEMORY_PREFETCH: {}", v),
    };

    if verbose {
        let func_name = match memory_prefetch_func as usize {
            v if v == no_prefetch as usize => "no_prefetch",
            v if v == prefetch_l2 as usize => "prefetch_l2",
            v if v == madvise_sequential as usize => "madvise_sequential",
            v if v == madvise_willneed as usize => "madvise_willneed",
            v if v == madvise_populate_read as usize => "madvise_populate_read",
            v if v == force_populate_read as usize => "force_populate_read",
            _ => unreachable!(),
        };

        eprintln!("[ParquetSource] Memory prefetch function: {}", func_name);
    }

    memory_prefetch_func
}
