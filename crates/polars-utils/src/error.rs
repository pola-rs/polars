use std::borrow::Cow;
use std::fmt::{Display, Formatter};

use crate::config::verbose;
use crate::format_pl_smallstr;

type ErrString = Cow<'static, str>;

#[derive(Debug)]
pub enum PolarsUtilsError {
    ComputeError(ErrString),
}

impl Display for PolarsUtilsError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match self {
            PolarsUtilsError::ComputeError(s) => {
                let s = s.as_ref();
                write!(f, "{s}")
            },
        }
    }
}

pub type Result<T> = std::result::Result<T, PolarsUtilsError>;

/// Utility whose Display impl truncates the string unless POLARS_VERBOSE is set.
pub struct TruncateErrorDetail<'a>(pub &'a str);

impl std::fmt::Display for TruncateErrorDetail<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let maybe_truncated = if verbose() {
            self.0
        } else {
            // Clamp the output on non-verbose
            &self.0[..self.0.len().min(4096)]
        };

        f.write_str(maybe_truncated)?;

        if maybe_truncated.len() != self.0.len() {
            let n_more = self.0.len() - maybe_truncated.len();
            f.write_str(" ...(set POLARS_VERBOSE=1 to see full response (")?;
            f.write_str(&format_pl_smallstr!("{}", n_more))?;
            f.write_str(" more characters))")?;
        };

        Ok(())
    }
}

/// Sanitize a string for safe display in error messages.
///
/// This removes ASCII control characters (0x00-0x1F and 0x7F) that could
/// interfere with terminal output (e.g., escape sequences that clear the screen).
/// Non-printable characters are replaced with the Unicode replacement character.
pub fn sanitize_for_error_display(s: &str) -> Cow<'_, str> {
    // Check if sanitization is needed
    if s.bytes().all(|b| b >= 0x20 && b != 0x7F) {
        return Cow::Borrowed(s);
    }

    // Build sanitized string
    let sanitized: String = s
        .chars()
        .map(|c| {
            if c.is_control() {
                '\u{FFFD}' // Unicode replacement character
            } else {
                c
            }
        })
        .collect();

    Cow::Owned(sanitized)
}

/// Utility whose Display impl sanitizes and truncates strings for error messages.
///
/// Control characters (ASCII 0x00-0x1F and 0x7F) are replaced with the Unicode
/// replacement character to prevent terminal manipulation via escape sequences.
/// Long strings are truncated unless POLARS_VERBOSE is set.
pub struct SanitizeErrorDetail<'a>(pub &'a str);

impl std::fmt::Display for SanitizeErrorDetail<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let sanitized = sanitize_for_error_display(self.0);
        let maybe_truncated = if verbose() {
            sanitized.as_ref()
        } else {
            // Clamp the output on non-verbose
            let max_len = sanitized.len().min(256);
            // Find a valid UTF-8 boundary
            let truncate_at = sanitized
                .char_indices()
                .take_while(|(i, _)| *i < max_len)
                .last()
                .map(|(i, c)| i + c.len_utf8())
                .unwrap_or(0);
            &sanitized[..truncate_at]
        };

        f.write_str(maybe_truncated)?;

        if maybe_truncated.len() != sanitized.len() {
            let n_more = sanitized.len() - maybe_truncated.len();
            f.write_str(" ...(set POLARS_VERBOSE=1 to see full value (")?;
            f.write_str(&format_pl_smallstr!("{}", n_more))?;
            f.write_str(" more characters))")?;
        };

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sanitize_for_error_display_clean_string() {
        let s = "hello world";
        let result = sanitize_for_error_display(s);
        assert!(matches!(result, Cow::Borrowed(_)));
        assert_eq!(result, "hello world");
    }

    #[test]
    fn test_sanitize_for_error_display_with_control_chars() {
        // Test with escape sequence that clears terminal (\x1b[2J) and other control chars
        let s = "prefix\x1b[2Jsuffix\x00end";
        let result = sanitize_for_error_display(s);
        assert!(matches!(result, Cow::Owned(_)));
        // Control characters should be replaced with Unicode replacement character
        assert!(!result.contains('\x1b'));
        assert!(!result.contains('\x00'));
        assert!(result.contains('\u{FFFD}'));
        assert!(result.contains("prefix"));
        assert!(result.contains("suffix"));
        assert!(result.contains("end"));
    }

    #[test]
    fn test_sanitize_for_error_display_binary_data() {
        // Simulate binary parquet data with control characters
        let s = "PAR1\x00\x00\x00\x1b\x1bsome\tdata\r\n";
        let result = sanitize_for_error_display(s);
        // Should contain replacement characters but not original control chars
        assert!(!result.chars().any(|c| c.is_control() && c != '\u{FFFD}'));
    }

    #[test]
    fn test_sanitize_error_detail_display() {
        let s = "test\x00value";
        let detail = SanitizeErrorDetail(s);
        let displayed = format!("{}", detail);
        // Should not contain null byte
        assert!(!displayed.contains('\x00'));
        assert!(displayed.contains("test"));
        assert!(displayed.contains("value"));
    }
}
