#[cfg(any(feature = "ipc_streaming", feature = "parquet"))]
use std::borrow::Cow;
use std::io::Read;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use polars_core::prelude::*;
#[cfg(any(feature = "ipc_streaming", feature = "parquet"))]
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df_as_ref};
use regex::{Regex, RegexBuilder};

use crate::mmap::{MmapBytesReader, ReaderBytes};

pub static POLARS_TEMP_DIR_BASE_PATH: Lazy<Box<Path>> = Lazy::new(|| {
    let path = std::env::var("POLARS_TEMP_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            PathBuf::from(std::env::temp_dir().to_string_lossy().as_ref()).join("polars/")
        })
        .into_boxed_path();

    if let Err(err) = std::fs::create_dir_all(path.as_ref()) {
        if !path.is_dir() {
            panic!("failed to create temporary directory: {}", err);
        }
    }

    path
});

pub fn get_reader_bytes<'a, R: Read + MmapBytesReader + ?Sized>(
    reader: &'a mut R,
) -> PolarsResult<ReaderBytes<'a>> {
    // we have a file so we can mmap
    if let Some(file) = reader.to_file() {
        let mmap = unsafe { memmap::Mmap::map(file)? };

        // somehow bck thinks borrows alias
        // this is sound as file was already bound to 'a
        use std::fs::File;
        let file = unsafe { std::mem::transmute::<&File, &'a File>(file) };
        Ok(ReaderBytes::Mapped(mmap, file))
    } else {
        // we can get the bytes for free
        if reader.to_bytes().is_some() {
            // duplicate .to_bytes() is necessary to satisfy the borrow checker
            Ok(ReaderBytes::Borrowed((*reader).to_bytes().unwrap()))
        } else {
            // we have to read to an owned buffer to get the bytes.
            let mut bytes = Vec::with_capacity(1024 * 128);
            reader.read_to_end(&mut bytes)?;
            Ok(ReaderBytes::Owned(bytes))
        }
    }
}

// used by python polars
pub fn resolve_homedir(path: &Path) -> PathBuf {
    // replace "~" with home directory
    if path.starts_with("~") {
        // home crate does not compile on wasm https://github.com/rust-lang/cargo/issues/12297
        #[cfg(not(target_family = "wasm"))]
        if let Some(homedir) = home::home_dir() {
            return homedir.join(path.strip_prefix("~").unwrap());
        }
    }

    path.into()
}

#[cfg(any(
    feature = "ipc",
    feature = "ipc_streaming",
    feature = "parquet",
    feature = "avro"
))]
pub(crate) fn apply_projection(schema: &ArrowSchema, projection: &[usize]) -> ArrowSchema {
    let fields = &schema.fields;
    let fields = projection
        .iter()
        .map(|idx| fields[*idx].clone())
        .collect::<Vec<_>>();
    ArrowSchema::from(fields)
}

#[cfg(any(
    feature = "ipc",
    feature = "ipc_streaming",
    feature = "avro",
    feature = "parquet"
))]
pub(crate) fn columns_to_projection(
    columns: &[String],
    schema: &ArrowSchema,
) -> PolarsResult<Vec<usize>> {
    let mut prj = Vec::with_capacity(columns.len());
    if columns.len() > 100 {
        let mut column_names = PlHashMap::with_capacity(schema.fields.len());
        schema.fields.iter().enumerate().for_each(|(i, c)| {
            column_names.insert(c.name.as_str(), i);
        });

        for column in columns.iter() {
            let Some(&i) = column_names.get(column.as_str()) else {
                polars_bail!(
                    ColumnNotFound:
                    "unable to find column {:?}; valid columns: {:?}", column, schema.get_names(),
                );
            };
            prj.push(i);
        }
    } else {
        for column in columns.iter() {
            let i = schema.try_index_of(column)?;
            prj.push(i);
        }
    }

    Ok(prj)
}

/// Because of threading every row starts from `0` or from `offset`.
/// We must correct that so that they are monotonically increasing.
#[cfg(any(feature = "csv", feature = "json"))]
pub(crate) fn update_row_counts(dfs: &mut [(DataFrame, IdxSize)], offset: IdxSize) {
    if !dfs.is_empty() {
        let mut previous = dfs[0].1 + offset;
        for (df, n_read) in &mut dfs[1..] {
            if let Some(s) = unsafe { df.get_columns_mut() }.get_mut(0) {
                *s = &*s + previous;
            }
            previous += *n_read;
        }
    }
}

/// Because of threading every row starts from `0` or from `offset`.
/// We must correct that so that they are monotonically increasing.
#[cfg(any(feature = "csv", feature = "json"))]
pub(crate) fn update_row_counts2(dfs: &mut [DataFrame], offset: IdxSize) {
    if !dfs.is_empty() {
        let mut previous = dfs[0].height() as IdxSize + offset;
        for df in &mut dfs[1..] {
            let n_read = df.height() as IdxSize;
            if let Some(s) = unsafe { df.get_columns_mut() }.get_mut(0) {
                *s = &*s + previous;
            }
            previous += n_read;
        }
    }
}

/// Compute `remaining_rows_to_read` to be taken per file up front, so we can actually read
/// concurrently/parallel
///
/// This takes an iterator over the number of rows per file.
pub fn get_sequential_row_statistics<I>(
    iter: I,
    mut total_rows_to_read: usize,
) -> Vec<(usize, usize)>
where
    I: Iterator<Item = usize>,
{
    let mut cumulative_read = 0;
    iter.map(|rows_this_file| {
        let remaining_rows_to_read = total_rows_to_read;
        total_rows_to_read = total_rows_to_read.saturating_sub(rows_this_file);

        let current_cumulative_read = cumulative_read;
        cumulative_read += rows_this_file;

        (remaining_rows_to_read, current_cumulative_read)
    })
    .collect()
}

#[cfg(feature = "json")]
pub(crate) fn overwrite_schema(
    schema: &mut Schema,
    overwriting_schema: &Schema,
) -> PolarsResult<()> {
    for (k, value) in overwriting_schema.iter() {
        *schema.try_get_mut(k)? = value.clone();
    }
    Ok(())
}

pub static FLOAT_RE: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[-+]?((\d*\.\d+)([eE][-+]?\d+)?|inf|NaN|(\d+)[eE][-+]?\d+|\d+\.)$").unwrap()
});

pub static FLOAT_RE_DECIMAL: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[-+]?((\d*,\d+)([eE][-+]?\d+)?|inf|NaN|(\d+)[eE][-+]?\d+|\d+,)$").unwrap()
});

pub static INTEGER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^-?(\d+)$").unwrap());

pub static BOOLEAN_RE: Lazy<Regex> = Lazy::new(|| {
    RegexBuilder::new(r"^(true|false)$")
        .case_insensitive(true)
        .build()
        .unwrap()
});

pub fn materialize_projection(
    with_columns: Option<&[String]>,
    schema: &Schema,
    hive_partitions: Option<&[Series]>,
    has_row_index: bool,
) -> Option<Vec<usize>> {
    match hive_partitions {
        None => with_columns.map(|with_columns| {
            with_columns
                .iter()
                .map(|name| schema.index_of(name).unwrap() - has_row_index as usize)
                .collect()
        }),
        Some(part_cols) => {
            with_columns.map(|with_columns| {
                with_columns
                    .iter()
                    .flat_map(|name| {
                        // the hive partitions are added at the end of the schema, but we don't want to project
                        // them from the file
                        if part_cols.iter().any(|s| s.name() == name.as_str()) {
                            None
                        } else {
                            Some(schema.index_of(name).unwrap() - has_row_index as usize)
                        }
                    })
                    .collect()
            })
        },
    }
}

pub fn check_projected_schema_impl(
    a: &Schema,
    b: &Schema,
    projected_names: Option<&[String]>,
    msg: &str,
) -> PolarsResult<()> {
    if !projected_names
        .map(|projected_names| {
            projected_names
                .iter()
                .all(|name| a.get(name) == b.get(name))
        })
        .unwrap_or_else(|| a == b)
    {
        polars_bail!(ComputeError: "{msg}\n\n\
                    Expected: {:?}\n\n\
                    Got: {:?}", a, b)
    }
    Ok(())
}

/// Checks if the projected columns are equal
pub fn check_projected_arrow_schema(
    a: &ArrowSchema,
    b: &ArrowSchema,
    projected_names: Option<&[String]>,
    msg: &str,
) -> PolarsResult<()> {
    if a != b {
        let a = Schema::from(a);
        let b = Schema::from(b);
        check_projected_schema_impl(&a, &b, projected_names, msg)
    } else {
        Ok(())
    }
}

/// Checks if the projected columns are equal
pub fn check_projected_schema(
    a: &Schema,
    b: &Schema,
    projected_names: Option<&[String]>,
    msg: &str,
) -> PolarsResult<()> {
    check_projected_schema_impl(a, b, projected_names, msg)
}

/// Split DataFrame into chunks in preparation for writing. The chunks have a
/// maximum number of rows per chunk to ensure reasonable memory efficiency when
/// reading the resulting file, and a minimum size per chunk to ensure
/// reasonable performance when writing.
#[cfg(any(feature = "ipc_streaming", feature = "parquet"))]
pub(crate) fn chunk_df_for_writing(
    df: &mut DataFrame,
    row_group_size: usize,
) -> PolarsResult<Cow<DataFrame>> {
    // ensures all chunks are aligned.
    df.align_chunks();

    // Accumulate many small chunks to the row group size.
    // See: #16403
    if !df.get_columns().is_empty()
        && df.get_columns()[0]
            .chunk_lengths()
            .take(5)
            .all(|len| len < row_group_size)
    {
        fn finish(scratch: &mut Vec<DataFrame>, new_chunks: &mut Vec<DataFrame>) {
            let mut new = accumulate_dataframes_vertical_unchecked(scratch.drain(..));
            new.as_single_chunk_par();
            new_chunks.push(new);
        }

        let mut new_chunks = Vec::with_capacity(df.n_chunks()); // upper limit;
        let mut scratch = vec![];
        let mut remaining = row_group_size;

        for df in df.split_chunks() {
            remaining = remaining.saturating_sub(df.height());
            scratch.push(df);

            if remaining == 0 {
                remaining = row_group_size;
                finish(&mut scratch, &mut new_chunks);
            }
        }
        if !scratch.is_empty() {
            finish(&mut scratch, &mut new_chunks);
        }
        return Ok(Cow::Owned(accumulate_dataframes_vertical_unchecked(
            new_chunks,
        )));
    }

    let n_splits = df.height() / row_group_size;
    let result = if n_splits > 0 {
        let mut splits = split_df_as_ref(df, n_splits, false);

        for df in splits.iter_mut() {
            // If the chunks are small enough, writing many small chunks
            // leads to slow writing performance, so in that case we
            // merge them.
            let n_chunks = df.n_chunks();
            if n_chunks > 1 && (df.estimated_size() / n_chunks < 128 * 1024) {
                df.as_single_chunk_par();
            }
        }

        Cow::Owned(accumulate_dataframes_vertical_unchecked(splits))
    } else {
        Cow::Borrowed(df)
    };
    Ok(result)
}

static CLOUD_URL: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^(s3a?|gs|gcs|file|abfss?|azure|az|adl|https?)://").unwrap());

/// Check if the path is a cloud url.
pub fn is_cloud_url<P: AsRef<Path>>(p: P) -> bool {
    match p.as_ref().as_os_str().to_str() {
        Some(s) => CLOUD_URL.is_match(s),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::{resolve_homedir, FLOAT_RE};

    #[test]
    fn test_float_parse() {
        assert!(FLOAT_RE.is_match("0.1"));
        assert!(FLOAT_RE.is_match("3.0"));
        assert!(FLOAT_RE.is_match("3.00001"));
        assert!(FLOAT_RE.is_match("-9.9990e-003"));
        assert!(FLOAT_RE.is_match("9.9990e+003"));
        assert!(FLOAT_RE.is_match("9.9990E+003"));
        assert!(FLOAT_RE.is_match("9.9990E+003"));
        assert!(FLOAT_RE.is_match(".5"));
        assert!(FLOAT_RE.is_match("2.5E-10"));
        assert!(FLOAT_RE.is_match("2.5e10"));
        assert!(FLOAT_RE.is_match("NaN"));
        assert!(FLOAT_RE.is_match("-NaN"));
        assert!(FLOAT_RE.is_match("-inf"));
        assert!(FLOAT_RE.is_match("inf"));
        assert!(FLOAT_RE.is_match("-7e-05"));
        assert!(FLOAT_RE.is_match("7e-05"));
        assert!(FLOAT_RE.is_match("+7e+05"));
    }

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_resolve_homedir() {
        let paths: Vec<PathBuf> = vec![
            "~/dir1/dir2/test.csv".into(),
            "/abs/path/test.csv".into(),
            "rel/path/test.csv".into(),
            "/".into(),
            "~".into(),
        ];

        let resolved: Vec<PathBuf> = paths.iter().map(|x| resolve_homedir(x)).collect();

        assert_eq!(resolved[0].file_name(), paths[0].file_name());
        assert!(resolved[0].is_absolute());
        assert_eq!(resolved[1], paths[1]);
        assert_eq!(resolved[2], paths[2]);
        assert_eq!(resolved[3], paths[3]);
        assert!(resolved[4].is_absolute());
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_resolve_homedir_windows() {
        let paths: Vec<PathBuf> = vec![
            r#"c:\Users\user1\test.csv"#.into(),
            r#"~\user1\test.csv"#.into(),
            "~".into(),
        ];

        let resolved: Vec<PathBuf> = paths.iter().map(|x| resolve_homedir(x)).collect();

        assert_eq!(resolved[0], paths[0]);
        assert_eq!(resolved[1].file_name(), paths[1].file_name());
        assert!(resolved[1].is_absolute());
        assert!(resolved[2].is_absolute());
    }
}
