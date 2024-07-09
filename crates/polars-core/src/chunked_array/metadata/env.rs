#[derive(Debug, Clone, Copy)]
pub struct MetadataEnv(u32);

impl MetadataEnv {
    const ENV_VAR: &'static str = "POLARS_METADATA_USE";

    const ENABLED: u32 = 0x1;
    const EXPERIMENTAL: u32 = 0x2;
    const LOG: u32 = 0x4;

    #[inline]
    fn get_cached() -> Self {
        if cfg!(debug_assertions) {
            let Ok(env) = std::env::var(Self::ENV_VAR) else {
                return Self(Self::ENABLED);
            };

            // @NOTE
            // We use a RwLock here so that we can mutate it for specific runs or sections of runs
            // when we perform A/B tests.
            static CACHED: std::sync::RwLock<Option<(String, MetadataEnv)>> =
                std::sync::RwLock::new(None);

            if let Some((cached_str, cached_value)) = CACHED.read().unwrap().as_ref() {
                if cached_str == &env[..] {
                    return *cached_value;
                }
            }

            let v = Self::get();
            *CACHED.write().unwrap() = Some((env.to_string(), v));
            v
        } else {
            static CACHED: std::sync::OnceLock<MetadataEnv> = std::sync::OnceLock::new();
            *CACHED.get_or_init(Self::get)
        }
    }

    #[inline(never)]
    fn get() -> Self {
        let Ok(env) = std::env::var(Self::ENV_VAR) else {
            return Self(Self::ENABLED);
        };

        match &env[..] {
            "0" => Self(0),
            "1" => Self(Self::ENABLED),
            "experimental" => Self(Self::ENABLED | Self::EXPERIMENTAL),
            "experimental,log" => Self(Self::ENABLED | Self::EXPERIMENTAL | Self::LOG),
            "log" => Self(Self::ENABLED | Self::LOG),
            _ => {
                eprintln!("Invalid `{}` environment variable", Self::ENV_VAR);
                eprintln!("Possible values:");
                eprintln!("  - 0                = Turn off all usage of metadata");
                eprintln!("  - 1                = Turn on usage of metadata (default)");
                eprintln!(
                    "  - experimental     = Turn on normal and experimental usage of metadata"
                );
                eprintln!("  - experimental,log = Turn on normal, experimental usage and logging of metadata usage");
                eprintln!("  - log              = Turn on normal and logging of metadata usage");
                eprintln!();
                panic!("Invalid environment variable")
            },
        }
    }

    #[inline(always)]
    pub fn disabled() -> bool {
        !Self::enabled()
    }

    #[inline(always)]
    pub fn enabled() -> bool {
        if cfg!(debug_assertions) {
            Self::get_cached().0 & Self::ENABLED != 0
        } else {
            true
        }
    }

    #[inline(always)]
    pub fn log() -> bool {
        if cfg!(debug_assertions) {
            Self::get_cached().0 & Self::LOG != 0
        } else {
            false
        }
    }

    #[inline(always)]
    pub fn experimental_enabled() -> bool {
        Self::get_cached().0 & Self::EXPERIMENTAL != 0
    }

    #[cfg(debug_assertions)]
    pub fn logfile() -> &'static std::sync::Mutex<std::fs::File> {
        static CACHED: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> =
            std::sync::OnceLock::new();
        CACHED.get_or_init(|| {
            std::sync::Mutex::new(std::fs::File::create(".polars-metadata.log").unwrap())
        })
    }
}

macro_rules! mdlog {
    ($s:literal$(, $arg:expr)* $(,)?) => {
        #[cfg(debug_assertions)]
        {
            use std::io::Write;
            let file = MetadataEnv::logfile();
            writeln!(file.lock().unwrap(), $s$(, $arg)*).unwrap();
        }

        #[cfg(not(debug_assertions))]
        {
            _ = $s;
            $(
            _ = $arg;
            )*
        }
    };
}
