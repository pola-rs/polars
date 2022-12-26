use memchr::memmem::find;

use super::*;

pub trait BinaryNameSpaceImpl: AsBinary {
    /// Check if binary contains given literal
    fn contains(&self, lit: &[u8]) -> PolarsResult<BooleanChunked> {
        let ca = self.as_binary();
        let f = |s: &[u8]| find(s, lit).is_some();
        let mut out: BooleanChunked = if !ca.has_validity() {
            ca.into_no_null_iter().map(f).collect()
        } else {
            ca.into_iter().map(|opt_s| opt_s.map(f)).collect()
        };
        out.rename(ca.name());
        Ok(out)
    }

    /// Check if strings contain a given literal
    fn contains_literal(&self, lit: &[u8]) -> PolarsResult<BooleanChunked> {
        self.contains(lit)
    }

    /// Check if strings ends with a substring
    fn ends_with(&self, sub: &[u8]) -> BooleanChunked {
        let ca = self.as_binary();
        let f = |s: &[u8]| s.ends_with(sub);
        let mut out: BooleanChunked = ca.into_iter().map(|opt_s| opt_s.map(f)).collect();
        out.rename(ca.name());
        out
    }

    /// Check if strings starts with a substring
    fn starts_with(&self, sub: &[u8]) -> BooleanChunked {
        let ca = self.as_binary();
        let f = |s: &[u8]| s.starts_with(sub);
        let mut out: BooleanChunked = ca.into_iter().map(|opt_s| opt_s.map(f)).collect();
        out.rename(ca.name());
        out
    }
}

impl BinaryNameSpaceImpl for BinaryChunked {}
