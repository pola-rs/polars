//! Out-of-core memory manager for the Polars streaming engine.
//!
//! Streaming pipeline nodes produce and consume DataFrames in small chunks
//! (morsels). Pipeline-breaker nodes (sort, group-by, join build-side) must
//! buffer all incoming data before they can produce output. This crate provides
//! a central [`MemoryManager`] that holds those buffered DataFrames, returning
//! lightweight [`Token`] handles (16 bytes, `Copy`) so nodes never hold large
//! allocations directly.
//!
//! Each thread checks its local memory usage against a 64 MB per-thread
//! budget. When a thread exceeds this limit, it calls the coordinator which
//! checks the overall used memory and triggers spilling if needed. Reloading
//! from disk happens on demand (not yet implemented). The per-thread budget
//! avoids frequent atomic contention on the global counter. The 64 MB
//! threshold needs future benchmarking to be adjusted for different workloads
//! and hardware.
mod memory_manager;
mod spiller;
mod token;

pub use memory_manager::{MemoryManager, SpillPolicy, mm};
pub use token::Token;
