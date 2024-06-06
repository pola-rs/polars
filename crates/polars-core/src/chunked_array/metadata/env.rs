#[derive(Debug, Clone, Copy)]
pub struct MetadataEnv(#[cfg(debug_assertions)] u32);

#[cfg(debug_assertions)]
impl MetadataEnv {
    pub const ENABLED: u32 = 0x1;
    pub const EXTENSIVE_USE: u32 = 0x2;
    pub const LOG: u32 = 0x4;

    #[inline(always)]
    fn get_cached() -> Self {
        static CACHED: std::sync::OnceLock<MetadataEnv> = std::sync::OnceLock::new();
        *CACHED.get_or_init(Self::get)
    }

    #[inline(always)]
    fn get() -> Self {
        let Ok(env) = std::env::var("POLARS_METADATA_FLAGS") else {
            return Self(Self::ENABLED);
        };

        if env == "0" {
            return Self(0);
        }

        // @NOTE
        // We use a RwLock here so that we can mutate it for specific runs or sections of runs when
        // we perform A/B tests.
        static CACHED: std::sync::RwLock<Option<(String, MetadataEnv)>> =
            std::sync::RwLock::new(None);

        if let Some((cached_str, cached_value)) = CACHED.read().unwrap().as_ref() {
            if cached_str == &env {
                return *cached_value;
            }
        };

        let mut mdenv = Self(Self::ENABLED);
        for arg in env.split(',') {
            match &arg.trim().to_lowercase()[..] {
                "extensive" => mdenv.0 |= Self::EXTENSIVE_USE,
                "log" => mdenv.0 |= Self::LOG,
                _ => panic!("Invalid `POLARS_METADATA_FLAGS` environment variable"),
            }
        }

        mdenv
    }

    #[inline(always)]
    pub fn disabled() -> bool {
        !Self::enabled()
    }

    #[inline(always)]
    pub fn enabled() -> bool {
        Self::get().0 & Self::ENABLED != 0
    }

    #[inline(always)]
    pub fn log() -> bool {
        Self::get_cached().0 & Self::LOG != 0
    }

    #[inline(always)]
    pub fn extensive_use() -> bool {
        Self::get().0 & Self::EXTENSIVE_USE != 0
    }

    pub fn logfile() -> &'static std::sync::Mutex<std::fs::File> {
        static CACHED: std::sync::OnceLock<std::sync::Mutex<std::fs::File>> =
            std::sync::OnceLock::new();
        CACHED.get_or_init(|| {
            std::sync::Mutex::new(std::fs::File::create(".polars-metadata.log").unwrap())
        })
    }
}

#[cfg(not(debug_assertions))]
impl MetadataEnv {
    #[inline(always)]
    pub const fn disabled() -> bool {
        false
    }

    #[inline(always)]
    pub const fn enabled() -> bool {
        true
    }

    #[inline(always)]
    pub const fn log() -> bool {
        false
    }

    #[inline(always)]
    pub const fn extensive_use() -> bool {
        false
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
