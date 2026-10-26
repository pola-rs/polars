use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
use std::sync::{LazyLock, Once};
use std::time::Duration;

mod engine;
mod file_advice;
mod parse;
mod resolve_mode;
mod spill_format;
pub mod spill_path;

pub use engine::Engine;
pub use file_advice::FileAdvice;
use polars_error::polars_warn;
pub use resolve_mode::ResolveMode;
pub use spill_format::SpillFormat;

// Public.
const VERBOSE: &str = "POLARS_VERBOSE";
const DEFAULT_VERBOSE: bool = false;

const WARN_UNKNOWN_CONFIG: &str = "POLARS_WARN_UNKNOWN_CONFIG";
const DEFAULT_WARN_UNKNOWN_CONFIG: bool = false;
// TODO: Turn DEFAULT_WARN_UNKNOWN_CONFIG on once we support all stable config options.

const WARN_UNSTABLE: &str = "POLARS_WARN_UNSTABLE";
const DEFAULT_WARN_UNSTABLE: bool = true;

const MAX_THREADS: &str = "POLARS_MAX_THREADS";
fn default_max_threads() -> u64 {
    std::thread::available_parallelism()
        .unwrap_or(std::num::NonZeroUsize::new(4).unwrap())
        .get() as u64
}

const IDEAL_MORSEL_SIZE: &str = "POLARS_IDEAL_MORSEL_SIZE";
const DEFAULT_IDEAL_MORSEL_SIZE: u64 = 100_000;

const ENGINE_AFFINITY: &str = "POLARS_ENGINE_AFFINITY";
const DEFAULT_ENGINE_AFFINITY: Engine = Engine::Auto;

const PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH: &str =
    "POLARS_PARQUET_BINARY_STATISTICS_TRUNCATE_LEN";
const DEFAULT_PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH: u64 = 64;

const PRUNE_PARQUET_METADATA: &str = "POLARS_PRUNE_PARQUET_METADATA";
const DEFAULT_PRUNE_PARQUET_METADATA: bool = false;

const RESOLVE_METADATA_LEVEL: &str = "POLARS_RESOLVE_METADATA_LEVEL";

const RESOLVE_SAMPLE_LIMIT: &str = "POLARS_RESOLVE_SAMPLE_LIMIT";
// 0 = auto (see `resolve_sample_limit()`).
const DEFAULT_RESOLVE_SAMPLE_LIMIT: u64 = 0;

// Private.
const VERBOSE_SENSITIVE: &str = "POLARS_VERBOSE_SENSITIVE";
const DEFAULT_VERBOSE_SENSITIVE: bool = false;

const FORCE_ASYNC: &str = "POLARS_FORCE_ASYNC";
const DEFAULT_FORCE_ASYNC: bool = false;

const SCAN_CHUNK_STATISTICS: &str = "POLARS_SCAN_CHUNK_STATISTICS";
const DEFAULT_SCAN_CHUNK_STATISTICS: bool = false;

const IMPORT_INTERVAL_AS_STRUCT: &str = "POLARS_IMPORT_INTERVAL_AS_STRUCT";
const DEFAULT_IMPORT_INTERVAL_AS_STRUCT: bool = false;

const OOC_DRIFT_THRESHOLD: &str = "POLARS_OOC_DRIFT_THRESHOLD";
const DEFAULT_OOC_DRIFT_THRESHOLD: u64 = 4 * 1024 * 1024;

// Unused at the moment, always IPC.
const OOC_SPILL_FORMAT: &str = "POLARS_OOC_SPILL_FORMAT";
const DEFAULT_OOC_SPILL_FORMAT: SpillFormat = SpillFormat::Ipc;

const OOC_SPILL_COMPRESSION_LEVEL: &str = "POLARS_OOC_SPILL_COMPRESSION_LEVEL";
const DEFAULT_OOC_SPILL_COMPRESSION_LEVEL: u64 = 1;

const OOC_MEMORY_BUDGET_FRACTION: &str = "POLARS_OOC_MEMORY_BUDGET_FRACTION";
const DEFAULT_OOC_MEMORY_BUDGET_FRACTION: f64 = 0.8;

const OOC_MEMORY_BUDGET_MB: &str = "POLARS_OOC_MEMORY_BUDGET_MB";
const DEFAULT_OOC_MEMORY_BUDGET_MB: u64 = u64::MAX;

const OOC_MEMORY_PREFETCH_FRACTION: &str = "POLARS_OOC_MEMORY_PREFETCH_FRACTION";
const DEFAULT_OOC_MEMORY_PREFETCH_FRACTION: f64 = 0.5;

const OOC_DISK_BUDGET_MB: &str = "POLARS_OOC_DISK_BUDGET_MB";
const DEFAULT_OOC_DISK_BUDGET_MB: u64 = 64 * 1000; // 64 GB

const OOC_SPILL_MIN_BYTES: &str = "POLARS_OOC_SPILL_MIN_BYTES";
const DEFAULT_OOC_SPILL_MIN_BYTES: u64 = 64 * 1024; // 64 KiB

const OOC_MAX_PARALLEL_SPILL_TASKS: &str = "POLARS_OOC_MAX_PARALLEL_SPILL_TASKS";
const DEFAULT_OOC_MAX_PARALLEL_SPILL_TASKS: u64 = 64;

const OOC_MAX_PARALLEL_PREFETCH_TASKS: &str = "POLARS_OOC_MAX_PARALLEL_PREFETCH_TASKS";
const DEFAULT_OOC_MAX_PARALLEL_PREFETCH_TASKS: u64 = 64;

const OOC_LOG_METRICS: &str = "POLARS_OOC_LOG_METRICS";
const DEFAULT_OOC_LOG_METRICS: bool = false;

const OOC_PRESSURE_DECAY: &str = "POLARS_OOC_PRESSURE_DECAY";
const DEFAULT_OOC_PRESSURE_DECAY: bool = true;

const OOC_PRESSURE_DECAY_GRACE_MS: &str = "POLARS_OOC_PRESSURE_DECAY_GRACE_MS";
const DEFAULT_OOC_PRESSURE_DECAY_GRACE_MS: u64 = 1000;

const OOMKILL_THRESHOLD_FRACTION: &str = "POLARS_OOMKILL_THRESHOLD_FRACTION";
const DEFAULT_OOMKILL_THRESHOLD_FRACTION: f64 = f64::INFINITY;

const OOMKILL_THRESHOLD_MB: &str = "POLARS_OOMKILL_THRESHOLD_MB";
const DEFAULT_OOMKILL_THRESHOLD_MB: u64 = u64::MAX;

const JOIN_SAMPLE_LIMIT: &str = "POLARS_JOIN_SAMPLE_LIMIT";
const DEFAULT_JOIN_SAMPLE_LIMIT: u64 = 10_000_000;

/// Whether hash joins publish key ranges to the parquet scans below them.
const JOIN_RUNTIME_FILTERS: &str = "POLARS_JOIN_RUNTIME_FILTERS";
const DEFAULT_JOIN_RUNTIME_FILTERS: bool = true;

/// Allows pruning of strict hconcat inputs in projection pushdown. This can reduce data loading
/// but may discard shape errors.
const PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS: &str =
    "POLARS_PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS";
const DEFAULT_PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS: bool = false;

const ALLOW_NESTED_CSPE: &str = "POLARS_ALLOW_NESTED_CSPE";
const DEFAULT_ALLOW_NESTED_CSPE: bool = false;

const DNS_LOG_THRESHOLD_MS: &str = "POLARS_DNS_LOG_THRESHOLD_MS";
/// Sentinel meaning "env var not set / logging disabled".
const DNS_LOG_THRESHOLD_DISABLED: u64 = u64::MAX;

const NUMA_AWARE: &str = "POLARS_NUMA_AWARE";
const DEFAULT_NUMA_AWARE: bool = false;

const NUMA_MOCK_REGIONS: &str = "POLARS_NUMA_MOCK_REGIONS";
const DEFAULT_NUMA_MOCK_REGIONS: u64 = 0;

const DISABLE_HTTP_RATE_LIMIT: &str = "POLARS_DISABLE_HTTP_RATE_LIMIT";
const DEFAULT_DISABLE_HTTP_RATE_LIMIT: bool = false;

const HTTP_SKIP_SYSTEM_CERTIFICATES: &str = "POLARS_HTTP_SKIP_SYSTEM_CERTIFICATES";
const DEFAULT_HTTP_SKIP_SYSTEM_CERTIFICATES: bool = false;

/// Max number of cached per-origin HTTP object stores; 0 disables the cache.
const HTTP_STORE_CACHE_SIZE: &str = "POLARS_HTTP_STORE_CACHE_SIZE";
const DEFAULT_HTTP_STORE_CACHE_SIZE: u64 = 256;

/// Use direct I/O (Linux: `O_DIRECT`), bypassing the page cache.
const DIRECT_IO: &str = "POLARS_DIRECT_IO";
const DEFAULT_DIRECT_IO: bool = false;

const FILE_READ_CONCURRENCY: &str = "POLARS_FILE_READ_CONCURRENCY";
const DEFAULT_FILE_READ_CONCURRENCY: u64 = 32;

/// Access pattern hint for local file reads (Linux: `posix_fadvise`).
const FILE_POSIX_FADV: &str = "POLARS_FILE_POSIX_FADV";
const DEFAULT_FILE_POSIX_FADV: FileAdvice = FileAdvice::Normal;

/// Read the column chunks of row groups that are in the page cache in the tasks that decode them,
/// instead of prefetching them (Linux 6.5+, parquet).
const FILE_DEFER_CACHED_READS: &str = "POLARS_FILE_DEFER_CACHED_READS";
const DEFAULT_FILE_DEFER_CACHED_READS: bool = true;

/// Initial number of slots of each hot table in the streaming group-by.
const HOT_TABLE_SIZE: &str = "POLARS_HOT_TABLE_SIZE";
const DEFAULT_HOT_TABLE_SIZE: u64 = if cfg!(debug_assertions) { 4 } else { 4096 };

/// Number of slots up to which the hot tables in the streaming group-by may grow.
const MAX_HOT_TABLE_SIZE: &str = "POLARS_MAX_HOT_TABLE_SIZE";
const DEFAULT_MAX_HOT_TABLE_SIZE: u64 = if cfg!(debug_assertions) { 16 } else { 1 << 17 };

static KNOWN_OPTIONS: &[&str] = &[
    // Public.
    VERBOSE,
    WARN_UNKNOWN_CONFIG,
    WARN_UNSTABLE,
    MAX_THREADS,
    IDEAL_MORSEL_SIZE,
    ENGINE_AFFINITY,
    PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH,
    PRUNE_PARQUET_METADATA,
    ALLOW_NESTED_CSPE,
    RESOLVE_METADATA_LEVEL,
    RESOLVE_SAMPLE_LIMIT,
    /*
    Not yet supported public options:

        "POLARS_FMT_STR_LEN"
        "POLARS_FMT_MAX_COLS"
        "POLARS_FMT_TABLE_FORMATTING"
        "POLARS_FMT_TABLE_INLINE_COLUMN_DATA_TYPE"
        "POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW"
        "POLARS_FMT_TABLE_FORMATTING"
        "POLARS_FMT_TABLE_ROUNDED_CORNERS"
        "POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES"
        "POLARS_FMT_TABLE_HIDE_COLUMN_NAMES"
        "POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR"
        "POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION"
        "POLARS_FMT_TABLE_CELL_ALIGNMENT"
        "POLARS_FMT_TABLE_CELL_NUMERIC_ALIGNMENT"
        "POLARS_FMT_TABLE_CELL_LIST_LEN"
        "POLARS_FMT_MAX_ROWS"
        "POLARS_TABLE_WIDTH"
        "POLARS_MAX_EXPR_DEPTH"
    */
    // Private.
    VERBOSE_SENSITIVE,
    FORCE_ASYNC,
    SCAN_CHUNK_STATISTICS,
    IMPORT_INTERVAL_AS_STRUCT,
    OOC_DRIFT_THRESHOLD,
    OOC_SPILL_FORMAT,
    OOC_SPILL_COMPRESSION_LEVEL,
    OOC_MEMORY_BUDGET_FRACTION,
    OOC_MEMORY_BUDGET_MB,
    OOC_MEMORY_PREFETCH_FRACTION,
    OOC_DISK_BUDGET_MB,
    OOC_SPILL_MIN_BYTES,
    OOC_MAX_PARALLEL_SPILL_TASKS,
    OOC_MAX_PARALLEL_PREFETCH_TASKS,
    OOC_LOG_METRICS,
    OOC_PRESSURE_DECAY,
    OOC_PRESSURE_DECAY_GRACE_MS,
    OOMKILL_THRESHOLD_FRACTION,
    OOMKILL_THRESHOLD_MB,
    JOIN_SAMPLE_LIMIT,
    JOIN_RUNTIME_FILTERS,
    PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS,
    DNS_LOG_THRESHOLD_MS,
    NUMA_AWARE,
    NUMA_MOCK_REGIONS,
    DISABLE_HTTP_RATE_LIMIT,
    HTTP_SKIP_SYSTEM_CERTIFICATES,
    HTTP_STORE_CACHE_SIZE,
    DIRECT_IO,
    FILE_READ_CONCURRENCY,
    FILE_POSIX_FADV,
    FILE_DEFER_CACHED_READS,
    HOT_TABLE_SIZE,
    MAX_HOT_TABLE_SIZE,
];

pub struct Config {
    // Public.
    verbose: AtomicBool,
    warn_unknown_config: AtomicBool,
    warn_unstable: AtomicBool,
    max_threads: AtomicU64,
    ideal_morsel_size: AtomicU64,
    engine_affinity: AtomicU8,
    parquet_binary_statistics_truncate_length: AtomicU64,
    prune_parquet_metadata: AtomicBool,
    allow_nested_cspe: AtomicBool,
    resolve_metadata_level: AtomicU8,
    resolve_sample_limit: AtomicU64,

    // Private.
    verbose_sensitive: AtomicBool,
    force_async: AtomicBool,
    scan_chunk_statistics: AtomicBool,
    import_interval_as_struct: AtomicBool,
    ooc_spill_format: AtomicU8,
    ooc_spill_compression_level: AtomicU64,
    ooc_memory_budget_fraction: AtomicU64,
    ooc_memory_budget_bytes: AtomicU64,
    ooc_memory_prefetch_fraction: AtomicU64,
    ooc_disk_budget_bytes: AtomicU64,
    ooc_spill_min_bytes: AtomicU64,
    ooc_max_parallel_spill_tasks: AtomicU64,
    ooc_max_parallel_prefetch_tasks: AtomicU64,
    ooc_log_metrics: AtomicBool,
    ooc_pressure_decay: AtomicBool,
    ooc_pressure_decay_grace_ms: AtomicU64,
    oomkill_threshold_fraction: AtomicU64,
    oomkill_threshold_bytes: AtomicU64,
    join_sample_limit: AtomicU64,
    join_runtime_filters: AtomicBool,
    projection_pushdown_prune_strict_hconcat_inputs: AtomicBool,
    dns_log_threshold_ms: AtomicU64,
    numa_aware: AtomicBool,
    numa_mock_regions: AtomicU64,
    disable_http_rate_limit: AtomicBool,
    http_skip_system_certificates: AtomicBool,
    http_store_cache_size: AtomicU64,
    direct_io: AtomicBool,
    file_read_concurrency: AtomicU64,
    file_posix_fadv: AtomicU8,
    file_defer_cached_reads: AtomicBool,
    hot_table_size: AtomicU64,
    max_hot_table_size: AtomicU64,

    // Derived from others.
    ooc_effective_memory_budget_bytes: AtomicU64,
    ooc_memory_prefetch_bytes: AtomicU64,
}

impl Config {
    fn new() -> Self {
        let cfg = Self {
            // Public.
            verbose: AtomicBool::new(DEFAULT_VERBOSE),
            warn_unknown_config: AtomicBool::new(DEFAULT_WARN_UNKNOWN_CONFIG),
            warn_unstable: AtomicBool::new(DEFAULT_WARN_UNSTABLE),
            max_threads: AtomicU64::new(default_max_threads()),
            ideal_morsel_size: AtomicU64::new(DEFAULT_IDEAL_MORSEL_SIZE),
            engine_affinity: AtomicU8::new(DEFAULT_ENGINE_AFFINITY as u8),
            parquet_binary_statistics_truncate_length: AtomicU64::new(
                DEFAULT_PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH,
            ),
            prune_parquet_metadata: AtomicBool::new(DEFAULT_PRUNE_PARQUET_METADATA),
            resolve_metadata_level: AtomicU8::new(ResolveMode::default() as u8),
            resolve_sample_limit: AtomicU64::new(DEFAULT_RESOLVE_SAMPLE_LIMIT),

            // Private.
            verbose_sensitive: AtomicBool::new(DEFAULT_VERBOSE_SENSITIVE),
            force_async: AtomicBool::new(DEFAULT_FORCE_ASYNC),
            scan_chunk_statistics: AtomicBool::new(DEFAULT_SCAN_CHUNK_STATISTICS),
            import_interval_as_struct: AtomicBool::new(DEFAULT_IMPORT_INTERVAL_AS_STRUCT),
            ooc_spill_format: AtomicU8::new(DEFAULT_OOC_SPILL_FORMAT as u8),
            ooc_spill_compression_level: AtomicU64::new(DEFAULT_OOC_SPILL_COMPRESSION_LEVEL),
            ooc_memory_budget_fraction: AtomicU64::new(
                DEFAULT_OOC_MEMORY_BUDGET_FRACTION.to_bits(),
            ),
            ooc_memory_budget_bytes: AtomicU64::new(
                DEFAULT_OOC_MEMORY_BUDGET_MB.saturating_mul(1_000_000),
            ),
            ooc_memory_prefetch_fraction: AtomicU64::new(
                DEFAULT_OOC_MEMORY_PREFETCH_FRACTION.to_bits(),
            ),
            ooc_disk_budget_bytes: AtomicU64::new(
                DEFAULT_OOC_DISK_BUDGET_MB.saturating_mul(1_000_000),
            ),
            ooc_spill_min_bytes: AtomicU64::new(DEFAULT_OOC_SPILL_MIN_BYTES),
            ooc_max_parallel_spill_tasks: AtomicU64::new(DEFAULT_OOC_MAX_PARALLEL_SPILL_TASKS),
            ooc_max_parallel_prefetch_tasks: AtomicU64::new(
                DEFAULT_OOC_MAX_PARALLEL_PREFETCH_TASKS,
            ),
            ooc_log_metrics: AtomicBool::new(false),
            ooc_pressure_decay: AtomicBool::new(DEFAULT_OOC_PRESSURE_DECAY),
            ooc_pressure_decay_grace_ms: AtomicU64::new(DEFAULT_OOC_PRESSURE_DECAY_GRACE_MS),
            oomkill_threshold_fraction: AtomicU64::new(
                DEFAULT_OOMKILL_THRESHOLD_FRACTION.to_bits(),
            ),
            oomkill_threshold_bytes: AtomicU64::new(
                DEFAULT_OOMKILL_THRESHOLD_MB.saturating_mul(1_000_000),
            ),
            join_sample_limit: AtomicU64::new(DEFAULT_JOIN_SAMPLE_LIMIT),
            join_runtime_filters: AtomicBool::new(DEFAULT_JOIN_RUNTIME_FILTERS),
            projection_pushdown_prune_strict_hconcat_inputs: AtomicBool::new(
                DEFAULT_PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS,
            ),
            allow_nested_cspe: AtomicBool::new(DEFAULT_ALLOW_NESTED_CSPE),
            dns_log_threshold_ms: AtomicU64::new(DNS_LOG_THRESHOLD_DISABLED),
            numa_aware: AtomicBool::new(DEFAULT_NUMA_AWARE),
            numa_mock_regions: AtomicU64::new(DEFAULT_NUMA_MOCK_REGIONS),
            disable_http_rate_limit: AtomicBool::new(DEFAULT_DISABLE_HTTP_RATE_LIMIT),
            http_skip_system_certificates: AtomicBool::new(DEFAULT_HTTP_SKIP_SYSTEM_CERTIFICATES),
            http_store_cache_size: AtomicU64::new(DEFAULT_HTTP_STORE_CACHE_SIZE),
            direct_io: AtomicBool::new(DEFAULT_DIRECT_IO),
            file_read_concurrency: AtomicU64::new(DEFAULT_FILE_READ_CONCURRENCY),
            file_posix_fadv: AtomicU8::new(DEFAULT_FILE_POSIX_FADV as u8),
            file_defer_cached_reads: AtomicBool::new(DEFAULT_FILE_DEFER_CACHED_READS),
            hot_table_size: AtomicU64::new(DEFAULT_HOT_TABLE_SIZE),
            max_hot_table_size: AtomicU64::new(DEFAULT_MAX_HOT_TABLE_SIZE),
            ooc_effective_memory_budget_bytes: AtomicU64::new(0),
            ooc_memory_prefetch_bytes: AtomicU64::new(0),
        };
        cfg.reload_env_vars();
        cfg
    }

    /// Reload the config from all environment variables.
    pub fn reload_env_vars(&self) {
        // Reload the warning config first to ensure we respect it.
        self.reload_env_var("POLARS_WARN_UNKNOWN_CONFIG");

        for var in KNOWN_OPTIONS {
            self.reload_env_var(var);
        }

        self.recompute_derived();
    }

    /// Reload a specific environment variable.
    pub fn reload_env_var(&self, var: &str) {
        self.apply_env_var(var, std::env::var(var).ok().as_deref());
        self.recompute_derived();
    }

    fn recompute_derived(&self) {
        static LOGGED_TOTAL_MEMORY: Once = Once::new();
        if self.verbose() {
            LOGGED_TOTAL_MEMORY.call_once(|| {
                let v = total_memory();
                let gib = (v as f64) / (1024.0 * 1024.0 * 1024.0);
                eprintln!("total memory: {gib:.3} GiB ({v} bytes)")
            });
        }

        let budget_frac = f64::from_bits(self.ooc_memory_budget_fraction.load(Ordering::Relaxed));
        let bytes = u64::min(
            self.ooc_memory_budget_bytes.load(Ordering::Relaxed),
            (total_memory() as f64 * budget_frac) as u64,
        );
        self.ooc_effective_memory_budget_bytes
            .store(bytes, Ordering::Relaxed);
        let frac = f64::from_bits(self.ooc_memory_prefetch_fraction.load(Ordering::Relaxed));
        self.ooc_memory_prefetch_bytes
            .store((bytes as f64 * frac) as u64, Ordering::Relaxed);

        let oomkill_frac = f64::from_bits(self.oomkill_threshold_fraction.load(Ordering::Relaxed));
        let oomkill_frac_bytes = if oomkill_frac.is_finite() {
            (total_memory() as f64 * oomkill_frac) as u64
        } else {
            u64::MAX
        };
        let oomkill_mb_bytes = self.oomkill_threshold_bytes.load(Ordering::Relaxed);
        OOMKILL_THRESHOLD_BYTES_ATOMIC.store(
            u64::min(oomkill_mb_bytes, oomkill_frac_bytes),
            Ordering::Relaxed,
        );
    }

    fn apply_env_var(&self, var: &str, val: Option<&str>) {
        match var {
            // Documented / public flags.
            WARN_UNKNOWN_CONFIG => self.warn_unknown_config.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_WARN_UNKNOWN_CONFIG),
                Ordering::Relaxed,
            ),
            WARN_UNSTABLE => self.warn_unstable.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_WARN_UNSTABLE),
                Ordering::Relaxed,
            ),
            VERBOSE => self.verbose.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_VERBOSE),
                Ordering::Relaxed,
            ),
            MAX_THREADS => self.max_threads.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(default_max_threads()),
                Ordering::Relaxed,
            ),
            IDEAL_MORSEL_SIZE => self.ideal_morsel_size.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_IDEAL_MORSEL_SIZE),
                Ordering::Relaxed,
            ),
            ENGINE_AFFINITY => self.engine_affinity.store(
                val.and_then(|x| parse::parse_engine(var, x))
                    .unwrap_or(DEFAULT_ENGINE_AFFINITY) as u8,
                Ordering::Relaxed,
            ),
            PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH => {
                self.parquet_binary_statistics_truncate_length.store(
                    val.and_then(|x| parse::parse_u64(var, x))
                        .unwrap_or(DEFAULT_PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH),
                    Ordering::Relaxed,
                )
            },
            PRUNE_PARQUET_METADATA => self.prune_parquet_metadata.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_PRUNE_PARQUET_METADATA),
                Ordering::Relaxed,
            ),
            ALLOW_NESTED_CSPE => self.allow_nested_cspe.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_ALLOW_NESTED_CSPE),
                Ordering::Relaxed,
            ),
            RESOLVE_METADATA_LEVEL => self.resolve_metadata_level.store(
                val.and_then(|x| parse::parse_resolve_mode(var, x))
                    .unwrap_or_default() as u8,
                Ordering::Relaxed,
            ),
            RESOLVE_SAMPLE_LIMIT => self.resolve_sample_limit.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_RESOLVE_SAMPLE_LIMIT),
                Ordering::Relaxed,
            ),

            // Private flags.
            VERBOSE_SENSITIVE => self.verbose_sensitive.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_VERBOSE_SENSITIVE),
                Ordering::Relaxed,
            ),
            FORCE_ASYNC => self.force_async.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_FORCE_ASYNC),
                Ordering::Relaxed,
            ),
            SCAN_CHUNK_STATISTICS => self.scan_chunk_statistics.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_SCAN_CHUNK_STATISTICS),
                Ordering::Relaxed,
            ),
            IMPORT_INTERVAL_AS_STRUCT => self.import_interval_as_struct.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_IMPORT_INTERVAL_AS_STRUCT),
                Ordering::Relaxed,
            ),
            OOC_DRIFT_THRESHOLD => OOC_DRIFT_THRESHOLD_ATOMIC.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_DRIFT_THRESHOLD),
                Ordering::Relaxed,
            ),
            OOC_SPILL_FORMAT => self.ooc_spill_format.store(
                val.and_then(|x| parse::parse_spill_format(var, x))
                    .unwrap_or(DEFAULT_OOC_SPILL_FORMAT) as u8,
                Ordering::Relaxed,
            ),
            OOC_SPILL_COMPRESSION_LEVEL => self.ooc_spill_compression_level.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_SPILL_COMPRESSION_LEVEL),
                Ordering::Relaxed,
            ),
            OOC_MEMORY_BUDGET_FRACTION => self.ooc_memory_budget_fraction.store(
                val.and_then(|x| parse::parse_f64(var, x))
                    .unwrap_or(DEFAULT_OOC_MEMORY_BUDGET_FRACTION)
                    .to_bits(),
                Ordering::Relaxed,
            ),
            OOC_MEMORY_BUDGET_MB => self.ooc_memory_budget_bytes.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_MEMORY_BUDGET_MB)
                    .saturating_mul(1_000_000),
                Ordering::Relaxed,
            ),
            OOC_MEMORY_PREFETCH_FRACTION => self.ooc_memory_prefetch_fraction.store(
                val.and_then(|x| parse::parse_f64_with_limits(var, x, 0.0, 0.99))
                    .unwrap_or(DEFAULT_OOC_MEMORY_PREFETCH_FRACTION)
                    .to_bits(),
                Ordering::Relaxed,
            ),
            OOC_DISK_BUDGET_MB => self.ooc_disk_budget_bytes.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_DISK_BUDGET_MB)
                    .saturating_mul(1_000_000),
                Ordering::Relaxed,
            ),
            OOC_SPILL_MIN_BYTES => self.ooc_spill_min_bytes.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_SPILL_MIN_BYTES),
                Ordering::Relaxed,
            ),
            OOC_MAX_PARALLEL_SPILL_TASKS => self.ooc_max_parallel_spill_tasks.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_MAX_PARALLEL_SPILL_TASKS)
                    .max(1), // A semaphore with zero permits would deadlock.
                Ordering::Relaxed,
            ),
            OOC_MAX_PARALLEL_PREFETCH_TASKS => self.ooc_max_parallel_prefetch_tasks.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_MAX_PARALLEL_PREFETCH_TASKS)
                    .max(1),
                Ordering::Relaxed,
            ),
            OOC_LOG_METRICS => self.ooc_log_metrics.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_OOC_LOG_METRICS),
                Ordering::Relaxed,
            ),
            OOC_PRESSURE_DECAY => self.ooc_pressure_decay.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_OOC_PRESSURE_DECAY),
                Ordering::Relaxed,
            ),
            OOC_PRESSURE_DECAY_GRACE_MS => self.ooc_pressure_decay_grace_ms.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_PRESSURE_DECAY_GRACE_MS),
                Ordering::Relaxed,
            ),
            OOMKILL_THRESHOLD_FRACTION => self.oomkill_threshold_fraction.store(
                val.and_then(|x| parse::parse_f64(var, x))
                    .unwrap_or(DEFAULT_OOMKILL_THRESHOLD_FRACTION)
                    .to_bits(),
                Ordering::Relaxed,
            ),
            OOMKILL_THRESHOLD_MB => self.oomkill_threshold_bytes.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOMKILL_THRESHOLD_MB)
                    .saturating_mul(1_000_000),
                Ordering::Relaxed,
            ),
            JOIN_SAMPLE_LIMIT => self.join_sample_limit.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_JOIN_SAMPLE_LIMIT),
                Ordering::Relaxed,
            ),
            JOIN_RUNTIME_FILTERS => self.join_runtime_filters.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_JOIN_RUNTIME_FILTERS),
                Ordering::Relaxed,
            ),
            PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS => {
                self.projection_pushdown_prune_strict_hconcat_inputs.store(
                    val.and_then(|x| parse::parse_bool(var, x))
                        .unwrap_or(DEFAULT_PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS),
                    Ordering::Relaxed,
                )
            },
            DNS_LOG_THRESHOLD_MS => self.dns_log_threshold_ms.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DNS_LOG_THRESHOLD_DISABLED),
                Ordering::Relaxed,
            ),
            NUMA_AWARE => self.numa_aware.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_NUMA_AWARE),
                Ordering::Relaxed,
            ),
            NUMA_MOCK_REGIONS => self.numa_mock_regions.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_NUMA_MOCK_REGIONS),
                Ordering::Relaxed,
            ),
            DISABLE_HTTP_RATE_LIMIT => self.disable_http_rate_limit.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_DISABLE_HTTP_RATE_LIMIT),
                Ordering::Relaxed,
            ),
            HTTP_SKIP_SYSTEM_CERTIFICATES => self.http_skip_system_certificates.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_HTTP_SKIP_SYSTEM_CERTIFICATES),
                Ordering::Relaxed,
            ),
            HTTP_STORE_CACHE_SIZE => self.http_store_cache_size.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_HTTP_STORE_CACHE_SIZE),
                Ordering::Relaxed,
            ),
            DIRECT_IO => self.direct_io.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_DIRECT_IO),
                Ordering::Relaxed,
            ),
            FILE_READ_CONCURRENCY => self.file_read_concurrency.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_FILE_READ_CONCURRENCY),
                Ordering::Relaxed,
            ),
            FILE_POSIX_FADV => self.file_posix_fadv.store(
                val.and_then(|x| parse::parse_file_advice(var, x))
                    .unwrap_or(DEFAULT_FILE_POSIX_FADV) as u8,
                Ordering::Relaxed,
            ),
            FILE_DEFER_CACHED_READS => self.file_defer_cached_reads.store(
                val.and_then(|x| parse::parse_bool(var, x))
                    .unwrap_or(DEFAULT_FILE_DEFER_CACHED_READS),
                Ordering::Relaxed,
            ),
            HOT_TABLE_SIZE => self.hot_table_size.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_HOT_TABLE_SIZE),
                Ordering::Relaxed,
            ),
            MAX_HOT_TABLE_SIZE => self.max_hot_table_size.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_MAX_HOT_TABLE_SIZE),
                Ordering::Relaxed,
            ),
            _ => {
                if var.starts_with("POLARS_") {
                    if self.warn_unknown_config.load(Ordering::Relaxed) {
                        polars_warn!(
                            "unknown config option '{var}' found in environment variables.\n\nYou can silence this warning by specifying POLARS_WARN_UNKNOWN_CONFIG=0."
                        )
                    }
                }
            },
        }
    }

    /// Whether we should do verbose printing.
    #[inline(always)]
    pub fn verbose(&self) -> bool {
        self.verbose.load(Ordering::Relaxed)
    }

    /// Whether we should warn when unstable features are used.
    #[inline(always)]
    pub fn warn_unstable(&self) -> bool {
        self.warn_unstable.load(Ordering::Relaxed)
    }

    /// The number of threads Polars should ideally use for CPU-intensive work.
    #[inline(always)]
    pub fn max_threads(&self) -> usize {
        self.max_threads.load(Ordering::Relaxed).try_into().unwrap()
    }

    /// The ideal size of a morsel, in rows.
    #[inline(always)]
    pub fn ideal_morsel_size(&self) -> u64 {
        self.ideal_morsel_size.load(Ordering::Relaxed)
    }

    /// Which engine to use by default.
    #[inline(always)]
    pub fn engine_affinity(&self) -> Engine {
        Engine::from_discriminant(self.engine_affinity.load(Ordering::Relaxed))
    }

    /// Target byte length to truncate statistics to for binary/string columns in parquet.
    #[inline(always)]
    pub fn parquet_binary_statistics_truncate_length(&self) -> u64 {
        self.parquet_binary_statistics_truncate_length
            .load(Ordering::Relaxed)
    }

    /// Whether the optimizer should prune parquet metadata to projected/predicate columns
    /// before serializing the IR plan. See `parquet_metadata_prune` in `polars-plan`.
    #[inline(always)]
    pub fn prune_parquet_metadata(&self) -> bool {
        self.prune_parquet_metadata.load(Ordering::Relaxed)
    }

    /// Nested common subplan elimination.
    #[inline(always)]
    pub fn allow_nested_cspe(&self) -> bool {
        self.allow_nested_cspe.load(Ordering::Relaxed)
    }

    /// How much per-file metadata `parquet_file_info` resolves at planning
    /// time. See [`ResolveMode`] for the variants and their cost / IR-shape
    /// trade-offs.
    #[inline(always)]
    pub fn resolve_metadata_level(&self) -> ResolveMode {
        ResolveMode::from_discriminant(self.resolve_metadata_level.load(Ordering::Relaxed))
    }

    /// Caps how many footers a `Sampled` metadata resolve reads. `None` (the
    /// default) leaves the cap to the resolver; an explicit value is
    /// authoritative, even a low one.
    #[inline(always)]
    pub fn resolve_sample_limit(&self) -> Option<u64> {
        match self.resolve_sample_limit.load(Ordering::Relaxed) {
            0 => None,
            n => Some(n),
        }
    }

    /// Whether we should do verbose printing on sensitive information.
    #[inline(always)]
    pub fn verbose_sensitive(&self) -> bool {
        self.verbose_sensitive.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn force_async(&self) -> bool {
        self.force_async.load(Ordering::Relaxed)
    }

    /// Whether a parquet scan, besides its per-column totals, keeps the min, max, null count and
    /// sizes that each sampled row group records. Planners use these to estimate how a column's
    /// values are distributed across the files, e.g. how many rows a filter keeps.
    #[inline(always)]
    pub fn scan_chunk_statistics(&self) -> bool {
        self.scan_chunk_statistics.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn import_interval_as_struct(&self) -> bool {
        self.import_interval_as_struct.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn ooc_drift_threshold(&self) -> u64 {
        get_ooc_drift_threshold()
    }

    #[inline(always)]
    pub fn ooc_spill_format(&self) -> SpillFormat {
        SpillFormat::from_discriminant(self.ooc_spill_format.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn ooc_spill_compression_level(&self) -> u64 {
        self.ooc_spill_compression_level.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn ooc_memory_budget_fraction(&self) -> f64 {
        f64::from_bits(self.ooc_memory_budget_fraction.load(Ordering::Relaxed))
    }

    /// The stricter of `POLARS_OOC_MEMORY_BUDGET_FRACTION` times the total memory and
    /// `POLARS_OOC_MEMORY_BUDGET_MB`.
    #[inline(always)]
    pub fn ooc_memory_budget_bytes(&self) -> u64 {
        self.ooc_effective_memory_budget_bytes
            .load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn ooc_memory_prefetch_fraction(&self) -> f64 {
        f64::from_bits(self.ooc_memory_prefetch_fraction.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn ooc_memory_prefetch_bytes(&self) -> u64 {
        self.ooc_memory_prefetch_bytes.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn ooc_disk_budget_bytes(&self) -> u64 {
        self.ooc_disk_budget_bytes.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn ooc_spill_min_bytes(&self) -> u64 {
        self.ooc_spill_min_bytes.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn ooc_max_parallel_spill_tasks(&self) -> usize {
        self.ooc_max_parallel_spill_tasks.load(Ordering::Relaxed) as usize
    }

    #[inline(always)]
    pub fn ooc_max_parallel_prefetch_tasks(&self) -> usize {
        self.ooc_max_parallel_prefetch_tasks.load(Ordering::Relaxed) as usize
    }

    #[inline(always)]
    pub fn ooc_log_metrics(&self) -> bool {
        self.ooc_log_metrics.load(Ordering::Relaxed)
    }

    /// Whether the allocator returns freed memory right away while out-of-core
    /// data is being spilled or is spilled.
    #[inline(always)]
    pub fn ooc_pressure_decay(&self) -> bool {
        self.ooc_pressure_decay.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn ooc_pressure_decay_grace_ms(&self) -> u64 {
        self.ooc_pressure_decay_grace_ms.load(Ordering::Relaxed)
    }

    pub fn ooc_spill_dir(&self) -> std::path::PathBuf {
        if let Ok(dir) = std::env::var("POLARS_OOC_SPILL_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            spill_path::default_ooc_spill_dir()
        }
    }

    #[inline(always)]
    pub fn oomkill_threshold_bytes(&self) -> u64 {
        get_oomkill_threshold_bytes()
    }

    #[inline(always)]
    pub fn join_sample_limit(&self) -> u64 {
        self.join_sample_limit.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn join_runtime_filters(&self) -> bool {
        self.join_runtime_filters.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn projection_pushdown_prune_strict_hconcat_inputs(&self) -> bool {
        self.projection_pushdown_prune_strict_hconcat_inputs
            .load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn dns_log_threshold(&self) -> Option<Duration> {
        match self.dns_log_threshold_ms.load(Ordering::Relaxed) {
            DNS_LOG_THRESHOLD_DISABLED => None,
            ms => Some(Duration::from_millis(ms)),
        }
    }

    #[inline(always)]
    pub fn numa_aware(&self) -> bool {
        self.numa_aware.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn numa_mock_regions(&self) -> u64 {
        self.numa_mock_regions.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn disable_http_rate_limit(&self) -> bool {
        self.disable_http_rate_limit.load(Ordering::Relaxed)
    }

    /// Whether cleartext `http` clients skip the system trust store. Such a client fails the
    /// handshake on a redirect to `https`.
    #[inline(always)]
    pub fn http_skip_system_certificates(&self) -> bool {
        self.http_skip_system_certificates.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn http_store_cache_size(&self) -> u64 {
        self.http_store_cache_size.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn direct_io(&self) -> bool {
        self.direct_io.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn file_read_concurrency(&self) -> u64 {
        self.file_read_concurrency.load(Ordering::Relaxed)
    }

    #[inline(always)]
    pub fn file_posix_fadv(&self) -> FileAdvice {
        FileAdvice::from_discriminant(self.file_posix_fadv.load(Ordering::Relaxed))
    }

    #[inline(always)]
    pub fn file_defer_cached_reads(&self) -> bool {
        self.file_defer_cached_reads.load(Ordering::Relaxed)
    }

    /// Initial number of slots of each hot table in the streaming group-by.
    #[inline(always)]
    pub fn hot_table_size(&self) -> u64 {
        self.hot_table_size.load(Ordering::Relaxed)
    }

    /// Number of slots up to which the hot tables in the streaming group-by may grow.
    #[inline(always)]
    pub fn max_hot_table_size(&self) -> u64 {
        self.max_hot_table_size.load(Ordering::Relaxed)
    }
}

pub fn config() -> &'static Config {
    static CONFIG: LazyLock<Config> = LazyLock::new(Config::new);
    &CONFIG
}

/// Return the total system memory in bytes, respecting cgroup limits and
/// `POLARS_OVERRIDE_TOTAL_MEMORY_MB`.
pub fn total_memory() -> u64 {
    static TOTAL_MEMORY: LazyLock<u64> = LazyLock::new(|| {
        if let Ok(s) = std::env::var("POLARS_OVERRIDE_TOTAL_MEMORY_MB") {
            return s
                .parse::<u64>()
                .unwrap_or_else(|_| {
                    panic!("invalid value for POLARS_OVERRIDE_TOTAL_MEMORY_MB: {s}")
                })
                .saturating_mul(1_000_000);
        }

        let mut sys = sysinfo::System::new();
        sys.refresh_memory_specifics(sysinfo::MemoryRefreshKind::nothing().with_ram());
        let system_limit = sys.cgroup_limits().map(|limits| limits.total_memory);
        [
            Some(sys.total_memory()),
            system_limit,
            process_cgroup_memory_limit(),
        ]
        .into_iter()
        .flatten()
        .min()
        .unwrap()
    });
    *TOTAL_MEMORY
}

/// The memory limit of the cgroup of this process, including the limits of
/// its parent cgroups.
///
/// The root cgroup files only hold this limit if the process has its own cgroup
/// namespace. We don't use sysinfo's process cgroup limits, as refreshing a
/// process there raises the soft open file limit of this process.
#[cfg(target_os = "linux")]
fn process_cgroup_memory_limit() -> Option<u64> {
    let cgroups = std::fs::read_to_string("/proc/self/cgroup").ok()?;
    let mountinfo = std::fs::read_to_string("/proc/self/mountinfo").ok()?;
    cgroup_memory_limit(&cgroups, &mountinfo)
}

/// The memory limit for the cgroups in `cgroups` (the format of
/// /proc/self/cgroup), with the hierarchies mounted as in `mountinfo`.
#[cfg(target_os = "linux")]
fn cgroup_memory_limit(cgroups: &str, mountinfo: &str) -> Option<u64> {
    use std::path::{Path, PathBuf};

    let read_u64 = |path: &Path| {
        std::fs::read_to_string(path)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    };

    // The cgroup mounts as (is cgroup v2, mount root, mount point). A container
    // may mount only its own part of the hierarchy, so cgroup paths are
    // resolved against the mount root.
    let mounts: Vec<(bool, PathBuf, PathBuf)> = mountinfo
        .lines()
        .filter_map(|line| {
            let (mount, fs) = line.split_once(" - ")?;
            let mut mount = mount.split(' ');
            let mount_root = mount.nth(3)?;
            let mount_point = mount.next()?;
            let mut fs = fs.split(' ');
            let fs_type = fs.next()?;
            let super_options = fs.nth(1)?;
            let is_v2 = match fs_type {
                "cgroup2" => true,
                "cgroup" if super_options.split(',').any(|o| o == "memory") => false,
                _ => return None,
            };
            Some((
                is_v2,
                unescape_mount_path(mount_root),
                unescape_mount_path(mount_point),
            ))
        })
        .collect();

    let cgroup_limit = |is_v2: bool, path: &str| {
        mounts
            .iter()
            .filter(|(mount_is_v2, _, _)| *mount_is_v2 == is_v2)
            .filter_map(|(_, mount_root, mount_point)| {
                let dir = mount_point.join(Path::new(path).strip_prefix(mount_root).ok()?);
                if is_v2 {
                    // The limit of every parent applies. "max" fails to parse
                    // and means no limit.
                    dir.ancestors()
                        .take_while(|dir| dir.starts_with(mount_point))
                        .filter_map(|dir| read_u64(&dir.join("memory.max")))
                        .min()
                } else {
                    // The kernel reports the limit that applies, taking
                    // memory.use_hierarchy into account.
                    std::fs::read_to_string(dir.join("memory.stat"))
                        .ok()?
                        .lines()
                        .find_map(|l| l.strip_prefix("hierarchical_memory_limit "))?
                        .trim()
                        .parse::<u64>()
                        .ok()
                }
            })
            .min()
    };

    cgroups
        .lines()
        .filter_map(|line| {
            let mut fields = line.splitn(3, ':');
            let hierarchy_id = fields.next()?;
            let controllers = fields.next()?;
            let path = fields.next()?;

            if hierarchy_id == "0" && controllers.is_empty() {
                cgroup_limit(true, path)
            } else if controllers.split(',').any(|c| c == "memory") {
                cgroup_limit(false, path)
            } else {
                None
            }
        })
        .min()
}

/// Decodes the octal escapes (such as `\040` for a space) of a path in
/// /proc/self/mountinfo.
#[cfg(target_os = "linux")]
fn unescape_mount_path(path: &str) -> std::path::PathBuf {
    use std::os::unix::ffi::OsStringExt;

    let bytes = path.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] == b'\\'
            && let Some(code) = bytes
                .get(i + 1..i + 4)
                .and_then(|d| u8::from_str_radix(std::str::from_utf8(d).ok()?, 8).ok())
        {
            out.push(code);
            i += 4;
        } else {
            out.push(bytes[i]);
            i += 1;
        }
    }
    std::ffi::OsString::from_vec(out).into()
}

#[cfg(not(target_os = "linux"))]
fn process_cgroup_memory_limit() -> Option<u64> {
    None
}

#[cfg(all(test, target_os = "linux"))]
mod tests {
    use std::fs;
    use std::path::{Path, PathBuf};

    use super::cgroup_memory_limit;

    fn fake_mount(name: &str, files: &[(&str, &str)]) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("polars-cgroup-{}-{name}", std::process::id()));
        let _ = fs::remove_dir_all(&dir);
        for (path, content) in files {
            let path = dir.join(path);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, content).unwrap();
        }
        dir
    }

    fn mountinfo_line(root: &str, mount_point: &Path, fs: &str) -> String {
        format!(
            "36 27 0:30 {root} {} rw,relatime shared:9 - {fs}",
            mount_point.display()
        )
    }

    #[test]
    fn test_cgroup_v2_parent_limit() {
        let mount = fake_mount(
            "v2",
            &[
                ("memory.max", "max\n"),
                ("a/memory.max", "2147483648\n"),
                ("a/b/memory.max", "5368709120\n"),
            ],
        );
        let mountinfo = mountinfo_line("/", &mount, "cgroup2 cgroup2 rw,nsdelegate");
        assert_eq!(cgroup_memory_limit("0::/a/b\n", &mountinfo), Some(2 << 30));
        assert_eq!(cgroup_memory_limit("0::/\n", &mountinfo), None);
        fs::remove_dir_all(mount).unwrap();
    }

    #[test]
    fn test_cgroup_v2_subtree_mount() {
        // Only /docker/abc is mounted, with the process in its child "job".
        let mount = fake_mount(
            "v2-subtree",
            &[
                ("memory.max", "5368709120\n"),
                ("job/memory.max", "2147483648\n"),
            ],
        );
        let mountinfo = mountinfo_line("/docker/abc", &mount, "cgroup2 cgroup2 rw");
        assert_eq!(
            cgroup_memory_limit("0::/docker/abc/job\n", &mountinfo),
            Some(2 << 30)
        );
        fs::remove_dir_all(mount).unwrap();
    }

    #[test]
    fn test_cgroup_v2_skips_unrelated_mount() {
        let unrelated = fake_mount("v2-unrelated", &[("memory.max", "1073741824\n")]);
        let mount = fake_mount("v2-full", &[("a/job/memory.max", "2147483648\n")]);
        let mountinfo = [
            mountinfo_line("/b", &unrelated, "cgroup2 cgroup2 rw"),
            mountinfo_line("/", &mount, "cgroup2 cgroup2 rw"),
        ]
        .join("\n");
        assert_eq!(
            cgroup_memory_limit("0::/a/job\n", &mountinfo),
            Some(2 << 30)
        );
        fs::remove_dir_all(unrelated).unwrap();
        fs::remove_dir_all(mount).unwrap();
    }

    #[test]
    fn test_cgroup_escaped_mount_paths() {
        let mount = fake_mount("v2 escaped", &[("job/memory.max", "2147483648\n")]);
        let escaped_mount = mount.display().to_string().replace(' ', "\\040");
        let mountinfo = mountinfo_line(
            "/batch\\040jobs",
            Path::new(&escaped_mount),
            "cgroup2 cgroup2 rw",
        );
        assert_eq!(
            cgroup_memory_limit("0::/batch jobs/job\n", &mountinfo),
            Some(2 << 30)
        );
        fs::remove_dir_all(mount).unwrap();
    }

    #[test]
    fn test_cgroup_v1_subtree_mount() {
        let mount = fake_mount(
            "v1-subtree",
            &[(
                "job/memory.stat",
                "cache 0\nhierarchical_memory_limit 2147483648\n",
            )],
        );
        let mountinfo = [
            mountinfo_line(
                "/docker/abc",
                Path::new("/nonexistent/cpu"),
                "cgroup cgroup rw,cpu",
            ),
            mountinfo_line("/docker/abc", &mount, "cgroup cgroup rw,memory"),
        ]
        .join("\n");
        let cgroups = "4:cpu:/docker/abc/job\n7:memory:/docker/abc/job\n";
        assert_eq!(cgroup_memory_limit(cgroups, &mountinfo), Some(2 << 30));
        fs::remove_dir_all(mount).unwrap();
    }
}

// These have to be standalone because LazyLock may not be called from allocator.
// Plus, it's faster this way.
static OOC_DRIFT_THRESHOLD_ATOMIC: AtomicU64 = AtomicU64::new(DEFAULT_OOC_DRIFT_THRESHOLD);
static OOMKILL_THRESHOLD_BYTES_ATOMIC: AtomicU64 =
    AtomicU64::new(DEFAULT_OOMKILL_THRESHOLD_MB.saturating_mul(1_000_000));

#[inline(always)]
pub fn get_ooc_drift_threshold() -> u64 {
    OOC_DRIFT_THRESHOLD_ATOMIC.load(Ordering::Relaxed)
}

#[inline(always)]
pub fn get_oomkill_threshold_bytes() -> u64 {
    OOMKILL_THRESHOLD_BYTES_ATOMIC.load(Ordering::Relaxed)
}
