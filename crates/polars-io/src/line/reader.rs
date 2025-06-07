use std::fs::File;
use std::path::PathBuf;

use polars_core::prelude::*;

use super::options::LineReadOptions;
use super::read_impl::CoreReader;
use crate::mmap::MmapBytesReader;
use crate::path_utils::resolve_homedir;
use crate::predicates::PhysicalIoExpr;
use crate::shared::SerReader;
use crate::utils::get_reader_bytes;

/// Create a new DataFrame by reading a text file.
///
/// # Example
///
/// ```
/// use polars_core::prelude::*;
/// use polars_io::prelude::*;
/// use std::fs::File;
///
/// fn example() -> PolarsResult<DataFrame> {
///     LineReadOptions::default()
///             .try_into_reader_with_file_path(Some("app.log".into()))?
///             .finish()
/// }
/// ```
#[must_use]
pub struct LineReader<R>
where
    R: MmapBytesReader,
{
    /// File or Stream object.
    reader: R,
    /// Options for the line reader.
    options: LineReadOptions,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
}

impl<R: MmapBytesReader> LineReader<R> {
    fn core_reader(&mut self) -> PolarsResult<CoreReader> {
        let reader_bytes = get_reader_bytes(&mut self.reader)?;

        CoreReader::new(
            reader_bytes,
            self.options.n_lines,
            self.options.skip_lines,
            self.options.n_threads,
            self.options.low_memory,
            self.options.chunk_size,
            1024,
            self.options.eol_char,
            self.options.encoding,
            self.predicate.clone(),
        )
    }

    /// Sets custom read options.
    pub fn with_options(mut self, options: LineReadOptions) -> Self {
        self.options = options;
        self
    }
}

impl LineReadOptions {
    /// Creates a line reader using a file path.
    ///
    /// # Panics
    /// If both self.path and the path parameter are non-null. Only one of them is
    /// to be non-null.
    pub fn try_into_reader_with_file_path(
        mut self,
        path: Option<PathBuf>,
    ) -> PolarsResult<LineReader<File>> {
        if self.path.is_some() {
            assert!(
                path.is_none(),
                "impl error: only 1 of self.path or the path parameter is to be non-null"
            );
        } else {
            self.path = path;
        };

        assert!(
            self.path.is_some(),
            "impl error: either one of self.path or the path parameter is to be non-null"
        );

        let path = resolve_homedir(self.path.as_ref().unwrap());
        let reader = polars_utils::open_file(&path)?;
        let options = self;

        Ok(LineReader {
            reader,
            options,
            predicate: None,
        })
    }

    /// Creates a line reader using a file handle.
    pub fn into_reader_with_file_handle<R: MmapBytesReader>(self, reader: R) -> LineReader<R> {
        let options = self;

        LineReader {
            reader,
            options,
            predicate: Default::default(),
        }
    }
}

impl<R> SerReader<R> for LineReader<R>
where
    R: MmapBytesReader,
{
    /// Create a new LineReader from a file/stream using default read options. To
    /// use non-default read options, first construct [LineReadOptions] and then use
    /// any of the `(try)_into_` methods.
    fn new(reader: R) -> Self {
        LineReader {
            reader,
            options: Default::default(),
            predicate: None,
        }
    }

    /// Read the file and create the DataFrame.
    fn finish(mut self) -> PolarsResult<DataFrame> {
        let reader = self.core_reader()?;
        let df = reader.finish()?;

        Ok(df)
    }
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use polars_core::prelude::DataType;
    use tempfile::NamedTempFile;

    use super::{LineReadOptions, LineReader};
    use crate::SerReader;

    #[test]
    fn option_try_into_reader_with_file_path() {
        let tmp_file = NamedTempFile::new().unwrap();
        let path = tmp_file.path().to_path_buf();
        // should not set path twice
        assert!(
            std::panic::catch_unwind(|| {
                LineReadOptions::default()
                    .with_path(Some(path.clone()))
                    .try_into_reader_with_file_path(Some(path.clone()))
            })
            .is_err()
        );

        // set path ok
        let reader = LineReadOptions::default()
            .try_into_reader_with_file_path(Some(path.clone()))
            .unwrap();
        assert_eq!(reader.options.path, Some(path.clone()));

        let reader = LineReadOptions::default()
            .with_path(Some(path.clone()))
            .try_into_reader_with_file_path(None)
            .unwrap();
        assert_eq!(reader.options.path, Some(path));

        // should set path at least once
        assert!(
            std::panic::catch_unwind(|| {
                LineReadOptions::default().try_into_reader_with_file_path(None)
            })
            .is_err()
        );
    }

    #[test]
    fn reader_with_options() {
        let mut tmp_file = NamedTempFile::new().unwrap();
        tmp_file.write_all(b"a").unwrap();
        let file = polars_utils::open_file(tmp_file.path()).unwrap();
        let mut reader = LineReader::new(file);
        assert_eq!(reader.options.eol_char, b'\n');
        reader = reader.with_options(LineReadOptions::default().with_eol_char(b'!'));
        assert_eq!(reader.options.eol_char, b'!');
    }

    #[test]
    fn read() {
        let mut tmp_file = NamedTempFile::new().unwrap();
        tmp_file.write_all("a\nb\nc".as_bytes()).unwrap();
        let reader = LineReadOptions::default()
            .try_into_reader_with_file_path(Some(tmp_file.path().to_path_buf()))
            .unwrap();
        let df = reader.finish().unwrap();
        assert_eq!(df.size(), 3);
        let cols = df.get_columns();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].dtype(), &DataType::String);
        assert_eq!(cols[0].name(), "lines");
    }
}
