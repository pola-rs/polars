use std::cell::RefCell;

use regex::{Regex, RegexBuilder};

use crate::cache::LruCache;

fn get_size_limit() -> Option<usize> {
    Some(
        std::env::var("POLARS_REGEX_SIZE_LIMIT")
            .ok()
            .filter(|l| !l.is_empty())?
            .parse()
            .expect("invalid POLARS_REGEX_SIZE_LIMIT"),
    )
}

// Regex compilation is really heavy, and the resulting regexes can be large as
// well, so we should have a good caching scheme.
//
// TODO: add larger global cache which has time-based flush.

/// A cache for compiled regular expressions.
pub struct RegexCache {
    cache: LruCache<String, Regex>,
    size_limit: Option<usize>,
}

impl RegexCache {
    fn new() -> Self {
        Self {
            cache: LruCache::with_capacity(32),
            size_limit: get_size_limit(),
        }
    }

    pub fn compile(&mut self, re: &str) -> Result<&Regex, regex::Error> {
        let r = self.cache.try_get_or_insert_with(re, |re| {
            // We do this little loop to only check POLARS_REGEX_SIZE_LIMIT when
            // a regex fails to compile due to the size limit.
            loop {
                let mut builder = RegexBuilder::new(re);
                if let Some(bytes) = self.size_limit {
                    builder.size_limit(bytes);
                }
                match builder.build() {
                    err @ Err(regex::Error::CompiledTooBig(_)) => {
                        let new_size_limit = get_size_limit();
                        if new_size_limit != self.size_limit {
                            self.size_limit = new_size_limit;
                            continue; // Try to compile again.
                        }
                        break err;
                    },
                    r => break r,
                };
            }
        });
        Ok(&*r?)
    }
}

thread_local! {
    static LOCAL_REGEX_CACHE: RefCell<RegexCache> = RefCell::new(RegexCache::new());
}

pub fn compile_regex(re: &str) -> Result<Regex, regex::Error> {
    LOCAL_REGEX_CACHE.with_borrow_mut(|cache| cache.compile(re).cloned())
}

pub fn with_regex_cache<R, F: FnOnce(&mut RegexCache) -> R>(f: F) -> R {
    LOCAL_REGEX_CACHE.with_borrow_mut(f)
}

#[macro_export]
macro_rules! cached_regex {
    () => {};

    ($vis:vis static $name:ident = $regex:expr; $($rest:tt)*) => {
        #[allow(clippy::disallowed_methods)]
        $vis static $name: std::sync::LazyLock<regex::Regex> = std::sync::LazyLock::new(|| regex::Regex::new($regex).unwrap());
        $crate::regex_cache::cached_regex!($($rest)*);
    };
}
pub use cached_regex;
