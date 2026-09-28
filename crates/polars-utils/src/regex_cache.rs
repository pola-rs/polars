use std::cell::RefCell;
use std::sync::Arc;

use regex::bytes::Regex as BytesRegex;
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
    bytes_cache: LruCache<usize, (Arc<BytesRegex>, BytesRegex)>,
    size_limit: Option<usize>,
}

impl RegexCache {
    fn new() -> Self {
        Self {
            cache: LruCache::with_capacity(32),
            bytes_cache: LruCache::with_capacity(32),
            size_limit: get_size_limit(),
        }
    }

    pub fn compile(&mut self, re: &str) -> Result<&Regex, regex::Error> {
        let size_limit = &mut self.size_limit;
        let r = self.cache.try_get_or_insert_with(re, |re| {
            build_within_size_limit(size_limit, |limit| {
                let mut builder = RegexBuilder::new(re);
                if let Some(bytes) = limit {
                    builder.size_limit(bytes);
                }
                builder.build()
            })
        });
        Ok(&*r?)
    }

    /// Borrows a cached clone of the supplied regex, keyed by its Arc pointer.
    pub fn get_or_insert_bytes(&mut self, re: &Arc<BytesRegex>) -> &BytesRegex {
        // Retain the original Arc so its address cannot be reused while cached.
        &self
            .bytes_cache
            .get_or_insert_with(&(Arc::as_ptr(re) as usize), |_| {
                (re.clone(), re.as_ref().clone())
            })
            .1
    }
}

// We do this little loop to only check POLARS_REGEX_SIZE_LIMIT when a regex
// fails to compile due to the size limit.
fn build_within_size_limit<R>(
    size_limit: &mut Option<usize>,
    build: impl Fn(Option<usize>) -> Result<R, regex::Error>,
) -> Result<R, regex::Error> {
    loop {
        match build(*size_limit) {
            err @ Err(regex::Error::CompiledTooBig(_)) => {
                let new_size_limit = get_size_limit();
                if new_size_limit != *size_limit {
                    *size_limit = new_size_limit;
                    continue; // Try to compile again.
                }
                break err;
            },
            r => break r,
        }
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

#[cfg(test)]
mod tests {
    use regex::bytes::RegexBuilder as BytesRegexBuilder;

    use super::*;

    #[test]
    fn bytes_cache_preserves_builder_options() {
        let mut cache = RegexCache::new();
        let sensitive = Arc::new(BytesRegexBuilder::new("abc").build().unwrap());
        let insensitive = Arc::new(
            BytesRegexBuilder::new("abc")
                .case_insensitive(true)
                .build()
                .unwrap(),
        );

        for _ in 0..2 {
            assert!(!cache.get_or_insert_bytes(&sensitive).is_match(b"ABC"));
            assert!(cache.get_or_insert_bytes(&insensitive).is_match(b"ABC"));
        }
    }

    #[test]
    fn bytes_cache_reuses_cloned_arc() {
        let mut cache = RegexCache::new();
        let re = Arc::new(BytesRegexBuilder::new("abc").build().unwrap());
        let cloned = re.clone();
        let cached = std::ptr::from_ref(cache.get_or_insert_bytes(&re));

        assert_eq!(
            cached,
            std::ptr::from_ref(cache.get_or_insert_bytes(&cloned))
        );
        assert_ne!(cached, Arc::as_ptr(&re));
    }

    #[test]
    fn bytes_cache_retains_arc_until_eviction() {
        let mut cache = RegexCache::new();
        let re = Arc::new(BytesRegexBuilder::new("abc").build().unwrap());
        let weak = Arc::downgrade(&re);
        cache.get_or_insert_bytes(&re);
        drop(re);
        assert!(weak.upgrade().is_some());

        for _ in 0..32 {
            let re = Arc::new(BytesRegexBuilder::new("abc").build().unwrap());
            cache.get_or_insert_bytes(&re);
        }
        assert!(weak.upgrade().is_none());
    }
}
