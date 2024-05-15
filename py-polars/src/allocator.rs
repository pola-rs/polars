#[cfg(all(
    target_family = "unix",
    not(feature = "default_allocator"),
    not(feature = "use_mimalloc"),
))]
use jemallocator::Jemalloc;
#[cfg(all(
    not(debug_assertions),
    not(feature = "default_allocator"),
    any(not(target_family = "unix"), feature = "use_mimalloc"),
))]
use mimalloc::MiMalloc;

#[cfg(all(
    debug_assertions,
    target_family = "unix",
    not(feature = "default_allocator"),
    not(feature = "use_mimalloc"),
))]
use crate::memory::TracemallocAllocator;

#[global_allocator]
#[cfg(all(
    not(debug_assertions),
    not(feature = "use_mimalloc"),
    not(feature = "default_allocator"),
    target_family = "unix",
))]
static ALLOC: Jemalloc = Jemalloc;

#[global_allocator]
#[cfg(all(
    not(debug_assertions),
    not(feature = "default_allocator"),
    any(not(target_family = "unix"), feature = "use_mimalloc"),
))]
static ALLOC: MiMalloc = MiMalloc;

// On Windows tracemalloc does work. However, we build abi3 wheels, and the
// relevant C APIs are not part of the limited stable CPython API. As a result,
// linking breaks on Windows if we use tracemalloc C APIs. So we only use this
// on Unix for now.
#[global_allocator]
#[cfg(all(
    debug_assertions,
    not(feature = "default_allocator"),
    target_family = "unix",
))]
static ALLOC: TracemallocAllocator<Jemalloc> = TracemallocAllocator::new(Jemalloc);
