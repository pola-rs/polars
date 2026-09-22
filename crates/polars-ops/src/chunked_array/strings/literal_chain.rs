use memchr::memmem::Finder;
use regex_syntax::hir::{Class, Hir, HirKind, Look};

/// A regex of plain literals joined by `(?s).*`, e.g. `^(?s).*foo.*bar.*$` (which is what SQL
/// `LIKE '%foo%bar%'` becomes). Matching it with substring searches in order is several times
/// faster than running the regex engine.
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

impl LiteralChain {
    pub(super) fn parse(pat: &str) -> Option<Self> {
        let hir = regex_syntax::parse(pat).ok()?;
        let HirKind::Concat(items) = hir.kind() else {
            return None;
        };
        let mut tokens = items.iter().map(to_token).collect::<Option<Vec<_>>>()?;

        let mut prefix = None;
        if matches!(tokens.first(), Some(Token::Start)) {
            tokens.remove(0);
            match tokens.first() {
                Some(Token::Literal(_)) => {
                    let Token::Literal(lit) = tokens.remove(0) else {
                        unreachable!()
                    };
                    prefix = Some(lit);
                },
                Some(Token::Any) => {},
                _ => return None,
            }
        }
        let mut suffix = None;
        if matches!(tokens.last(), Some(Token::End)) {
            tokens.pop();
            match tokens.last() {
                Some(Token::Literal(_)) => {
                    let Some(Token::Literal(lit)) = tokens.pop() else {
                        unreachable!()
                    };
                    suffix = Some(lit);
                },
                Some(Token::Any) => {},
                _ => return None,
            }
        }

        // What is left must alternate between `.*` and literals. Without any `.*` the regex
        // engine is already fast (plain literal or exact match), so leave those to it.
        let mut seen_any = false;
        let mut middle = Vec::new();
        let mut prev_is_literal = prefix.is_some();
        for token in tokens {
            match token {
                Token::Any => {
                    seen_any = true;
                    prev_is_literal = false;
                },
                Token::Literal(lit) if !prev_is_literal => {
                    middle.push(Finder::new(&lit).into_owned());
                    prev_is_literal = true;
                },
                _ => return None,
            }
        }
        if !seen_any || (prev_is_literal && suffix.is_some()) {
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
        let re = regex::Regex::new(pat).unwrap();
        for s in haystacks {
            assert_eq!(chain.is_match(s.as_bytes()), re.is_match(s), "{pat} on {s:?}");
        }
    }

    #[test]
    fn test_literal_chain_matches_regex() {
        let haystacks = [
            "",
            "a",
            "ab",
            "aab",
            "ba",
            "abab",
            "a\nb",
            "xaybz",
            "aXb",
            "abXab",
            "é€b",
            "bXa",
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
