use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

mod engine;
mod parse;
mod spill_format;
pub mod spill_path;
mod spill_policy;

pub use engine::Engine;
use polars_error::polars_warn;
pub use spill_format::SpillFormat;
pub use spill_policy::SpillPolicy;

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
const STREAMING_CHUNK_SIZE: &str = "POLARS_STREAMING_CHUNK_SIZE"; // Backwards compatibility.
const DEFAULT_IDEAL_MORSEL_SIZE: u64 = 100_000;

const ENGINE_AFFINITY: &str = "POLARS_ENGINE_AFFINITY";
const DEFAULT_ENGINE_AFFINITY: Engine = Engine::Auto;

const PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH: &str =
    "POLARS_PARQUET_BINARY_STATISTICS_TRUNCATE_LEN";
const DEFAULT_PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH: u64 = 64;

const PRUNE_PARQUET_METADATA: &str = "POLARS_PRUNE_PARQUET_METADATA";
const DEFAULT_PRUNE_PARQUET_METADATA: bool = false;

// Private.
const VERBOSE_SENSITIVE: &str = "POLARS_VERBOSE_SENSITIVE";
const DEFAULT_VERBOSE_SENSITIVE: bool = false;

const FORCE_ASYNC: &str = "POLARS_FORCE_ASYNC";
const DEFAULT_FORCE_ASYNC: bool = false;

const IMPORT_INTERVAL_AS_STRUCT: &str = "POLARS_IMPORT_INTERVAL_AS_STRUCT";
const DEFAULT_IMPORT_INTERVAL_AS_STRUCT: bool = false;

const OOC_DRIFT_THRESHOLD: &str = "POLARS_OOC_DRIFT_THRESHOLD";
const DEFAULT_OOC_DRIFT_THRESHOLD: u64 = 4 * 1024 * 1024;

const OOC_SPILL_POLICY: &str = "POLARS_OOC_SPILL_POLICY";
const DEFAULT_OOC_SPILL_POLICY: SpillPolicy = SpillPolicy::NoSpill;

const OOC_SPILL_FORMAT: &str = "POLARS_OOC_SPILL_FORMAT";
const DEFAULT_OOC_SPILL_FORMAT: SpillFormat = SpillFormat::Ipc;

const OOC_MEMORY_BUDGET_FRACTION: &str = "POLARS_OOC_MEMORY_BUDGET_FRACTION";
const DEFAULT_OOC_MEMORY_BUDGET_FRACTION: f64 = 0.8;

const OOC_SPILL_MIN_BYTES: &str = "POLARS_OOC_SPILL_MIN_BYTES";
const DEFAULT_OOC_SPILL_MIN_BYTES: u64 = 100 * 1024; // 100 KB

const JOIN_SAMPLE_LIMIT: &str = "POLARS_JOIN_SAMPLE_LIMIT";
const DEFAULT_JOIN_SAMPLE_LIMIT: u64 = 10_000_000;

/// Allows pruning of strict hconcat inputs in projection pushdown. This can reduce data loading
/// but may discard shape errors.
const PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS: &str =
    "POLARS_PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS";
const DEFAULT_PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS: bool = false;

const ALLOW_NESTED_CSPE: &str = "POLARS_ALLOW_NESTED_CSPE";
const DEFAULT_ALLOW_NESTED_CSPE: bool = false;

static KNOWN_OPTIONS: &[&str] = &[
    // Public.
    VERBOSE,
    WARN_UNKNOWN_CONFIG,
    WARN_UNSTABLE,
    MAX_THREADS,
    IDEAL_MORSEL_SIZE,
    STREAMING_CHUNK_SIZE,
    ENGINE_AFFINITY,
    PARQUET_BINARY_STATISTICS_TRUNCATE_LENGTH,
    PRUNE_PARQUET_METADATA,
    ALLOW_NESTED_CSPE,
    /*
    Not yet supported public options:

        "POLARS_AUTO_STRUCTIFY"
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
    IMPORT_INTERVAL_AS_STRUCT,
    OOC_DRIFT_THRESHOLD,
    OOC_SPILL_POLICY,
    OOC_SPILL_FORMAT,
    OOC_MEMORY_BUDGET_FRACTION,
    OOC_SPILL_MIN_BYTES,
    JOIN_SAMPLE_LIMIT,
    PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS,
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

    // Private.
    verbose_sensitive: AtomicBool,
    force_async: AtomicBool,
    import_interval_as_struct: AtomicBool,
    ooc_spill_policy: AtomicU8,
    ooc_spill_format: AtomicU8,
    ooc_memory_budget_fraction: AtomicU64,
    ooc_spill_min_bytes: AtomicU64,
    join_sample_limit: AtomicU64,
    projection_pushdown_prune_strict_hconcat_inputs: AtomicBool,
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

            // Private.
            verbose_sensitive: AtomicBool::new(DEFAULT_VERBOSE_SENSITIVE),
            force_async: AtomicBool::new(DEFAULT_FORCE_ASYNC),
            import_interval_as_struct: AtomicBool::new(DEFAULT_IMPORT_INTERVAL_AS_STRUCT),
            ooc_spill_policy: AtomicU8::new(DEFAULT_OOC_SPILL_POLICY as u8),
            ooc_spill_format: AtomicU8::new(DEFAULT_OOC_SPILL_FORMAT as u8),
            ooc_memory_budget_fraction: AtomicU64::new(
                DEFAULT_OOC_MEMORY_BUDGET_FRACTION.to_bits(),
            ),
            ooc_spill_min_bytes: AtomicU64::new(DEFAULT_OOC_SPILL_MIN_BYTES),
            join_sample_limit: AtomicU64::new(DEFAULT_JOIN_SAMPLE_LIMIT),
            projection_pushdown_prune_strict_hconcat_inputs: AtomicBool::new(
                DEFAULT_PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS,
            ),
            allow_nested_cspe: AtomicBool::new(DEFAULT_ALLOW_NESTED_CSPE),
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
    }

    /// Reload a specific environment variable.
    pub fn reload_env_var(&self, var: &str) {
        self.apply_env_var(var, std::env::var(var).ok().as_deref());
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
            IDEAL_MORSEL_SIZE | STREAMING_CHUNK_SIZE => self.ideal_morsel_size.store(
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
            OOC_SPILL_POLICY => self.ooc_spill_policy.store(
                val.and_then(|x| parse::parse_spill_policy(var, x))
                    .unwrap_or(DEFAULT_OOC_SPILL_POLICY) as u8,
                Ordering::Relaxed,
            ),
            OOC_SPILL_FORMAT => self.ooc_spill_format.store(
                val.and_then(|x| parse::parse_spill_format(var, x))
                    .unwrap_or(DEFAULT_OOC_SPILL_FORMAT) as u8,
                Ordering::Relaxed,
            ),
            OOC_MEMORY_BUDGET_FRACTION => self.ooc_memory_budget_fraction.store(
                val.and_then(|x| parse::parse_f64(var, x))
                    .unwrap_or(DEFAULT_OOC_MEMORY_BUDGET_FRACTION)
                    .to_bits(),
                Ordering::Relaxed,
            ),
            OOC_SPILL_MIN_BYTES => self.ooc_spill_min_bytes.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_OOC_SPILL_MIN_BYTES),
                Ordering::Relaxed,
            ),
            JOIN_SAMPLE_LIMIT => self.join_sample_limit.store(
                val.and_then(|x| parse::parse_u64(var, x))
                    .unwrap_or(DEFAULT_JOIN_SAMPLE_LIMIT),
                Ordering::Relaxed,
            ),
            PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS => {
                self.projection_pushdown_prune_strict_hconcat_inputs.store(
                    val.and_then(|x| parse::parse_bool(var, x))
                        .unwrap_or(DEFAULT_PROJECTION_PUSHDOWN_PRUNE_STRICT_HCONCAT_INPUTS),
                    Ordering::Relaxed,
                )
            },
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
    pub fn verbose(&self) -> bool {
        self.verbose.load(Ordering::Relaxed)
    }

    /// Whether we should warn when unstable features are used.
    pub fn warn_unstable(&self) -> bool {
        self.warn_unstable.load(Ordering::Relaxed)
    }

    /// The number of threads Polars should ideally use for CPU-intensive work.
    pub fn max_threads(&self) -> usize {
        self.max_threads.load(Ordering::Relaxed).try_into().unwrap()
    }

    /// The ideal size of a morsel, in rows.
    pub fn ideal_morsel_size(&self) -> u64 {
        self.ideal_morsel_size.load(Ordering::Relaxed)
    }

    /// Which engine to use by default.
    pub fn engine_affinity(&self) -> Engine {
        Engine::from_discriminant(self.engine_affinity.load(Ordering::Relaxed))
    }

    /// Target byte length to truncate statistics to for binary/string columns in parquet.
    pub fn parquet_binary_statistics_truncate_length(&self) -> u64 {
        self.parquet_binary_statistics_truncate_length
            .load(Ordering::Relaxed)
    }

    /// Whether the optimizer should prune parquet metadata to projected/predicate columns
    /// before serializing the IR plan. See `parquet_metadata_prune` in `polars-plan`.
    pub fn prune_parquet_metadata(&self) -> bool {
        self.prune_parquet_metadata.load(Ordering::Relaxed)
    }

    /// Nested common subplan elimination.
    pub fn allow_nested_cspe(&self) -> bool {
        self.allow_nested_cspe.load(Ordering::Relaxed)
    }

    /// Whether we should do verbose printing on sensitive information.
    pub fn verbose_sensitive(&self) -> bool {
        self.verbose_sensitive.load(Ordering::Relaxed)
    }

    pub fn force_async(&self) -> bool {
        self.force_async.load(Ordering::Relaxed)
    }

    pub fn import_interval_as_struct(&self) -> bool {
        self.import_interval_as_struct.load(Ordering::Relaxed)
    }

    pub fn ooc_drift_threshold(&self) -> u64 {
        get_ooc_drift_threshold()
    }

    pub fn ooc_spill_policy(&self) -> SpillPolicy {
        SpillPolicy::from_discriminant(self.ooc_spill_policy.load(Ordering::Relaxed))
    }

    pub fn ooc_spill_format(&self) -> SpillFormat {
        SpillFormat::from_discriminant(self.ooc_spill_format.load(Ordering::Relaxed))
    }

    pub fn ooc_memory_budget_fraction(&self) -> f64 {
        f64::from_bits(self.ooc_memory_budget_fraction.load(Ordering::Relaxed))
    }

    pub fn ooc_spill_min_bytes(&self) -> u64 {
        self.ooc_spill_min_bytes.load(Ordering::Relaxed)
    }

    pub fn ooc_spill_dir(&self) -> std::path::PathBuf {
        if let Ok(dir) = std::env::var("POLARS_OOC_SPILL_DIR") {
            std::path::PathBuf::from(dir)
        } else {
            spill_path::default_ooc_spill_dir()
        }
    }

    pub fn join_sample_limit(&self) -> u64 {
        self.join_sample_limit.load(Ordering::Relaxed)
    }

    pub fn projection_pushdown_prune_strict_hconcat_inputs(&self) -> bool {
        self.projection_pushdown_prune_strict_hconcat_inputs
            .load(Ordering::Relaxed)
    }
}

pub fn config() -> &'static Config {
    static CONFIG: LazyLock<Config> = LazyLock::new(Config::new);
    &CONFIG
}

// Has to be a standalone because LazyLock may not be called from allocator.
// Plus, it's faster this way.
static OOC_DRIFT_THRESHOLD_ATOMIC: AtomicU64 = AtomicU64::new(DEFAULT_OOC_DRIFT_THRESHOLD);

#[inline(always)]
pub fn get_ooc_drift_threshold() -> u64 {
    OOC_DRIFT_THRESHOLD_ATOMIC.load(Ordering::Relaxed)
}
