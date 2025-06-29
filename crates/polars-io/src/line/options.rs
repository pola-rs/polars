use std::path::PathBuf;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub struct LineReadOptions {
    pub path: Option<PathBuf>,
    // Line-wise options
    pub n_lines: Option<usize>,
    pub skip_lines: usize,
    // Performance related options
    pub n_threads: Option<usize>,
    pub low_memory: bool,
    pub chunk_size: usize,
    // Parse options
    pub eol_char: u8,
    pub encoding: TextEncoding,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Hash)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum TextEncoding {
    /// Utf8 encoding.
    #[default]
    Utf8,
    /// Utf8 encoding and unknown bytes are replaced with �.
    LossyUtf8,
}

impl Default for LineReadOptions {
    fn default() -> Self {
        Self {
            path: None,
            n_lines: None,
            skip_lines: 0,
            n_threads: None,
            low_memory: false,
            chunk_size: 1 << 18,
            eol_char: b'\n',
            encoding: Default::default(),
        }
    }
}

impl LineReadOptions {
    pub fn with_path<P: Into<PathBuf>>(mut self, path: Option<P>) -> Self {
        self.path = path.map(|p| p.into());
        self
    }

    /// Limits the number of lines to read.
    pub fn with_n_lines(mut self, n_lines: Option<usize>) -> Self {
        self.n_lines = n_lines;
        self
    }

    /// Number of threads to use for reading. Defaults to the size of the polars
    /// thread pool.
    pub fn with_n_threads(mut self, n_threads: Option<usize>) -> Self {
        self.n_threads = n_threads;
        self
    }

    /// Start reading after `skip_lines` lines. The header will be parsed at this
    /// offset.
    pub fn with_skip_lines(mut self, skip_lines: usize) -> Self {
        self.skip_lines = skip_lines;
        self
    }

    /// Set the character used to indicate an end-of-line (eol).
    pub fn with_eol_char(mut self, eol_char: u8) -> Self {
        self.eol_char = eol_char;
        self
    }

    /// Set the encoding used by the file.
    pub fn with_encoding(mut self, encoding: TextEncoding) -> Self {
        self.encoding = encoding;
        self
    }

    /// Reduce memory consumption at the expense of performance.
    pub fn low_memory(mut self, toggle: bool) -> Self {
        self.low_memory = toggle;
        self
    }

    /// Sets the chunk size used by the parser. This influences performance.
    pub fn with_chunk_size(mut self, chunk_size: usize) -> Self {
        self.chunk_size = chunk_size;
        self
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;
    use std::str::FromStr;

    use super::{LineReadOptions, TextEncoding};

    #[test]
    fn lines_read_options_default() {
        let options = LineReadOptions::default();
        assert_eq!(
            options,
            LineReadOptions {
                path: None,
                n_threads: None,
                n_lines: None,
                skip_lines: 0,
                low_memory: false,
                chunk_size: 1 << 18,
                eol_char: b'\n',
                encoding: TextEncoding::Utf8,
            }
        );
    }

    #[test]
    fn lines_read_options_builder() {
        let options = LineReadOptions::default()
            .with_path(Some("/test/path"))
            .with_encoding(TextEncoding::LossyUtf8)
            .with_eol_char(b'\t')
            .with_n_lines(Some(5))
            .with_n_threads(Some(2))
            .with_skip_lines(3)
            .low_memory(true)
            .with_chunk_size(1024);
        assert_eq!(
            options,
            LineReadOptions {
                path: Some(PathBuf::from_str("/test/path").unwrap()),
                n_lines: Some(5),
                n_threads: Some(2),
                skip_lines: 3,
                low_memory: true,
                chunk_size: 1024,
                eol_char: b'\t',
                encoding: TextEncoding::LossyUtf8,
            }
        );
    }
}
