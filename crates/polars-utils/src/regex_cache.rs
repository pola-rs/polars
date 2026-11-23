use std::cell::RefCell;
use std::sync::Arc;

use regex::bytes::Regex as BytesRegex;
use regex::{Regex, RegexBuilder};
use regex_syntax::ast::{self, Ast, RepetitionKind, RepetitionRange};
use regex_syntax::hir::translate::Translator;

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
    is_match_cache: LruCache<String, Regex>,
    bytes_cache: LruCache<usize, (Arc<BytesRegex>, BytesRegex)>,
    size_limit: Option<usize>,
}

impl RegexCache {
    fn new() -> Self {
        Self {
            cache: LruCache::with_capacity(32),
            is_match_cache: LruCache::with_capacity(32),
            bytes_cache: LruCache::with_capacity(32),
            size_limit: get_size_limit(),
        }
    }

    pub fn compile(&mut self, re: &str) -> Result<&Regex, regex::Error> {
        compile_cached(&mut self.cache, &mut self.size_limit, re)
    }

    /// Like [`Self::compile`], for `is_match` only, see [`is_match_pattern`].
    pub fn compile_is_match(&mut self, re: &str) -> Result<&Regex, regex::Error> {
        let Self {
            cache,
            is_match_cache,
            size_limit,
            ..
        } = self;
        let r = is_match_cache.try_get_or_insert_with(re, |re| match is_match_pattern(re) {
            Some(trimmed) => build_regex(size_limit, &trimmed),
            None => compile_cached(cache, size_limit, re).cloned(),
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

fn compile_cached<'a>(
    cache: &'a mut LruCache<String, Regex>,
    size_limit: &mut Option<usize>,
    re: &str,
) -> Result<&'a Regex, regex::Error> {
    let r = cache.try_get_or_insert_with(re, |re| build_regex(size_limit, re));
    Ok(&*r?)
}

fn build_regex(size_limit: &mut Option<usize>, re: &str) -> Result<Regex, regex::Error> {
    build_within_size_limit(size_limit, |limit| {
        let mut builder = RegexBuilder::new(re);
        if let Some(bytes) = limit {
            builder.size_limit(bytes);
        }
        builder.build()
    })
}

/// `pat` without the repetitions at either end that can match nothing (`.*foo.*` -> `foo`): same
/// `is_match`, but since regex 1.13.1 a leading `.*` disables its reverse suffix/inner search.
pub fn is_match_pattern(pat: &str) -> Option<String> {
    let ast = ast::parse::Parser::new().parse(pat).ok()?;
    let Ast::Concat(concat) = &ast else {
        return None;
    };
    let items = &concat.asts;
    let is_flags = |a: &Ast| matches!(a, Ast::Flags(_));
    let can_match_empty = |a: &Ast| match a {
        Ast::Repetition(rep) => match rep.op.kind {
            RepetitionKind::ZeroOrOne | RepetitionKind::ZeroOrMore => true,
            RepetitionKind::OneOrMore => false,
            RepetitionKind::Range(
                RepetitionRange::Exactly(min)
                | RepetitionRange::AtLeast(min)
                | RepetitionRange::Bounded(min, _),
            ) => min == 0,
        },
        _ => false,
    };
    let kept = |a: &Ast| !is_flags(a) && !can_match_empty(a);
    let first = items.iter().position(kept)?;
    let last = items.iter().rposition(kept)?;
    if items[..first]
        .iter()
        .chain(&items[last + 1..])
        .all(is_flags)
    {
        return None;
    }
    // Only rewrite patterns that compile, so errors stay the same.
    Translator::new().translate(pat, &ast).ok()?;
    // Slice `pat` rather than print a `Hir`, which doesn't round-trip nested repetitions.
    let span = |a: &Ast| a.span().start.offset..a.span().end.offset;
    let mut out = String::new();
    // Flags apply to what follows them.
    for a in items[..first].iter().filter(|a| is_flags(a)) {
        out.push_str(&pat[span(a)]);
    }
    out.push_str(&pat[span(&items[first]).start..span(&items[last]).end]);
    Some(out)
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

/// Like [`compile_regex`], for `is_match` only, see [`is_match_pattern`].
pub fn compile_is_match_regex(re: &str) -> Result<Regex, regex::Error> {
    LOCAL_REGEX_CACHE.with_borrow_mut(|cache| cache.compile_is_match(re).cloned())
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
        let regex = BytesRegexBuilder::new("abc").build().unwrap();
        let re = Arc::new(regex.clone());
        let weak = Arc::downgrade(&re);
        cache.get_or_insert_bytes(&re);
        drop(re);
        assert!(weak.upgrade().is_some());

        for _ in 0..32 {
            let re = Arc::new(regex.clone());
            cache.get_or_insert_bytes(&re);
        }
        assert!(weak.upgrade().is_none());
    }
}
