use std::sync::LazyLock;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};

mod engine;
mod parse;

pub use engine::Engine;
use polars_error::polars_warn;

// Public.
const VERBOSE: &'static str = "POLARS_VERBOSE";
const DEFAULT_VERBOSE: bool = false;

const WARN_UNKNOWN_CONFIG: &'static str = "POLARS_WARN_UNKNOWN_CONFIG";
const DEFAULT_WARN_UNKNOWN_CONFIG: bool = true;

const WARN_UNSTABLE: &'static str = "POLARS_WARN_UNSTABLE";
const DEFAULT_WARN_UNSTABLE: bool = true;

const IDEAL_MORSEL_SIZE: &'static str = "POLARS_IDEAL_MORSEL_SIZE";
const STREAMING_CHUNK_SIZE: &'static str = "POLARS_STREAMING_CHUNK_SIZE"; // Backwards compatibility.
const DEFAULT_IDEAL_MORSEL_SIZE: u64 = 100_000;

const ENGINE_AFFINITY: &'static str = "POLARS_ENGINE_AFFINITY";
const DEFAULT_ENGINE_AFFINITY: Engine = Engine::Auto;

// Private.
const FORCE_ASYNC: &'static str = "POLARS_FORCE_ASYNC";
const DEFAULT_FORCE_ASYNC: bool = false;

const IMPORT_INTERVAL_AS_STRUCT: &'static str = "POLARS_IMPORT_INTERVAL_AS_STRUCT";
const DEFAULT_IMPORT_INTERVAL_AS_STRUCT: bool = false;

pub struct Config {
    // Public.
    verbose: AtomicBool,
    warn_unknown_config: AtomicBool,
    warn_unstable: AtomicBool,
    ideal_morsel_size: AtomicU64,
    engine_affinity: AtomicU8,

    // Private.
    force_async: AtomicBool,
    import_interval_as_struct: AtomicBool,
}

impl Config {
    fn new() -> Self {
        let cfg = Self {
            // Public.
            verbose: AtomicBool::new(DEFAULT_VERBOSE),
            warn_unknown_config: AtomicBool::new(DEFAULT_WARN_UNKNOWN_CONFIG),
            warn_unstable: AtomicBool::new(DEFAULT_WARN_UNSTABLE),
            ideal_morsel_size: AtomicU64::new(DEFAULT_IDEAL_MORSEL_SIZE),
            engine_affinity: AtomicU8::new(DEFAULT_ENGINE_AFFINITY as u8),

            // Private.
            force_async: AtomicBool::new(DEFAULT_FORCE_ASYNC),
            import_interval_as_struct: AtomicBool::new(DEFAULT_IMPORT_INTERVAL_AS_STRUCT),
        };
        cfg.reload_env_vars();
        cfg
    }

    /// Reload the config from all environment variables.
    pub fn reload_env_vars(&self) {
        // Reload the warning config first to ensure we respect it.
        self.reload_env_var("POLARS_WARN_UNKNOWN_CONFIG");
        for (var, val) in std::env::vars() {
            self.apply_env_var(var.as_str(), Some(val.as_str()));
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

            // Private flags.
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

    /// The ideal size of a morsel, in rows.
    pub fn ideal_morsel_size(&self) -> u64 {
        self.ideal_morsel_size.load(Ordering::Relaxed)
    }

    /// Which engine to use by default.
    pub fn engine_affinity(&self) -> Engine {
        Engine::from_discriminant(self.engine_affinity.load(Ordering::Relaxed))
    }

    pub fn force_async(&self) -> bool {
        self.force_async.load(Ordering::Relaxed)
    }
    
    pub fn import_interval_as_struct(&self) -> bool {
        self.import_interval_as_struct.load(Ordering::Relaxed)
    }
}

pub fn config() -> &'static Config {
    static CONFIG: LazyLock<Config> = LazyLock::new(Config::new);
    &*CONFIG
}

/*

plr.config_reload_env_var("POLARS_AUTO_STRUCTIFY")
plr.config_reload_env_var("POLARS_FMT_STR_LEN")
plr.config_reload_env_var("POLARS_STREAMING_CHUNK_SIZE")
plr.config_reload_env_var("POLARS_FMT_MAX_COLS")
plr.config_reload_env_var("POLARS_FMT_TABLE_FORMATTING")
plr.config_reload_env_var("POLARS_FMT_TABLE_INLINE_COLUMN_DATA_TYPE")
plr.config_reload_env_var("POLARS_FMT_TABLE_DATAFRAME_SHAPE_BELOW")
plr.config_reload_env_var("POLARS_FMT_TABLE_FORMATTING")
plr.config_reload_env_var("POLARS_FMT_TABLE_ROUNDED_CORNERS")
plr.config_reload_env_var("POLARS_FMT_TABLE_HIDE_COLUMN_DATA_TYPES")
plr.config_reload_env_var("POLARS_FMT_TABLE_HIDE_COLUMN_NAMES")
plr.config_reload_env_var("POLARS_FMT_TABLE_HIDE_COLUMN_SEPARATOR")
plr.config_reload_env_var("POLARS_FMT_TABLE_HIDE_DATAFRAME_SHAPE_INFORMATION")
plr.config_reload_env_var("POLARS_FMT_TABLE_CELL_ALIGNMENT")
plr.config_reload_env_var("POLARS_FMT_TABLE_CELL_NUMERIC_ALIGNMENT")
plr.config_reload_env_var("POLARS_FMT_TABLE_CELL_LIST_LEN")
plr.config_reload_env_var("POLARS_FMT_MAX_ROWS")
plr.config_reload_env_var("POLARS_TABLE_WIDTH")
plr.config_reload_env_var("POLARS_MAX_EXPR_DEPTH")

*/
