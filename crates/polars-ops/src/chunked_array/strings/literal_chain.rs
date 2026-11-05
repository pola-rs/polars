use std::cell::RefCell;
use std::sync::Arc;

use memchr::memmem::Finder;
use polars_utils::cache::LruCache;
use regex_syntax::hir::{Class, Hir, HirKind, Look};

/// A regex of plain literals joined by `(?s).*`, e.g. `(?s)foo.*bar` from SQL
/// `LIKE '%foo%bar%'`, matched with substring searches in order.
pub(super) struct LiteralChain {
    prefix: Option<Box<[u8]>>,
    middle: Vec<Finder<'static>>,
    suffix: Option<Box<[u8]>>,
}

enum Token {
    Start,
    End,
    Any,
    Literal(Box<[u8]>),
}

fn is_any_repetition(hir: &Hir) -> bool {
    let HirKind::Repetition(rep) = hir.kind() else {
        return false;
    };
    if rep.min != 0 || rep.max.is_some() {
        return false;
    }
    match rep.sub.kind() {
        HirKind::Class(Class::Unicode(cls)) => {
            cls.ranges().len() == 1
                && cls.ranges()[0].start() == '\0'
                && cls.ranges()[0].end() == char::MAX
        },
        _ => false,
    }
}

fn to_token(hir: &Hir) -> Option<Token> {
    match hir.kind() {
        HirKind::Look(Look::Start) => Some(Token::Start),
        HirKind::Look(Look::End) => Some(Token::End),
        HirKind::Literal(lit) => Some(Token::Literal(lit.0.clone())),
        _ if is_any_repetition(hir) => Some(Token::Any),
        _ => None,
    }
}

thread_local! {
    static LOCAL_CHAIN_CACHE: RefCell<LruCache<String, Option<Arc<LiteralChain>>>> =
        RefCell::new(LruCache::with_capacity(32));
}

impl LiteralChain {
    /// Like `parse`, but cached per thread, including patterns that are not a chain.
    pub(super) fn cached(pat: &str) -> Option<Arc<Self>> {
        LOCAL_CHAIN_CACHE.with_borrow_mut(|cache| {
            cache
                .get_or_insert_with(pat, |pat| Self::parse(pat).map(Arc::new))
                .clone()
        })
    }

    fn parse(pat: &str) -> Option<Self> {
        let hir = regex_syntax::parse(pat).ok()?;
        let HirKind::Concat(items) = hir.kind() else {
            return None;
        };
        let tokens = items.iter().map(to_token).collect::<Option<Vec<_>>>()?;

        let (prefix, rest) = match tokens.as_slice() {
            [Token::Start, Token::Literal(lit), rest @ ..] => (Some(lit.clone()), rest),
            [Token::Start, rest @ ..] => (None, rest),
            rest => (None, rest),
        };
        let (suffix, rest) = match rest {
            [rest @ .., Token::Literal(lit), Token::End] => (Some(lit.clone()), rest),
            [rest @ .., Token::End] => (None, rest),
            rest => (None, rest),
        };

        // Without any `.*` this is a plain literal or exact match, which the regex engine handles.
        let mut seen_any = false;
        let mut middle = Vec::new();
        for token in rest {
            match token {
                Token::Any => seen_any = true,
                Token::Literal(lit) => middle.push(Finder::new(lit).into_owned()),
                _ => return None,
            }
        }
        if !seen_any {
            return None;
        }

        Some(Self {
            prefix,
            middle,
            suffix,
        })
    }

    pub(super) fn is_match(&self, s: &[u8]) -> bool {
        let mut start = 0;
        let mut end = s.len();
        if let Some(prefix) = &self.prefix {
            if !s.starts_with(prefix) {
                return false;
            }
            start = prefix.len();
        }
        if let Some(suffix) = &self.suffix {
            if end < start + suffix.len() || !s.ends_with(suffix) {
                return false;
            }
            end -= suffix.len();
        }
        for finder in &self.middle {
            match finder.find(&s[start..end]) {
                Some(i) => start += i + finder.needle().len(),
                None => return false,
            }
        }
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn check(pat: &str, haystacks: &[&str]) {
        let chain = LiteralChain::parse(pat).unwrap();
        let re = polars_utils::regex_cache::compile_regex(pat).unwrap();
        for s in haystacks {
            assert_eq!(
                chain.is_match(s.as_bytes()),
                re.is_match(s),
                "{pat} on {s:?}"
            );
        }
    }

    #[test]
    fn test_literal_chain_matches_regex() {
        let haystacks = [
            "", "a", "ab", "aab", "abab", "a\nb", "xaybz", "aXb", "abXab", "é€b", "bXa",
        ];
        for pat in [
            "^(?s).*a.*b.*$",
            "(?s)a.*b",
            "^(?s)a.*b$",
            "^(?s)a.*$",
            "^(?s).*b$",
            "^(?s)ab.*ab$",
            "^(?s)a.*a$",
            "^(?s).*$",
            "(?s)€.*b",
            "^(?s)a.*b.*a.*$",
        ] {
            check(pat, &haystacks);
        }
    }

    #[test]
    fn test_literal_chain_rejects() {
        for pat in [
            "a.*b",
            "(?s)ab",
            "^ab$",
            "(?is)a.*b",
            "(?s)a.+b",
            "(?s)a.b",
            "(?s)a|b.*c",
            "(?m)^(?s)a.*b$",
            "[",
        ] {
            assert!(LiteralChain::parse(pat).is_none(), "{pat}");
        }
    }
}
