use std::io::Read;
use std::path::{Path, PathBuf};

use once_cell::sync::Lazy;
use polars_core::frame::DataFrame;
use polars_core::prelude::*;
use regex::{Regex, RegexBuilder};

use crate::mmap::{MmapBytesReader, ReaderBytes};
#[cfg(any(
    feature = "ipc",
    feature = "ipc_streaming",
    feature = "parquet",
    feature = "avro"
))]
use crate::ArrowSchema;

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
            if !bytes.is_empty()
                && (bytes[bytes.len() - 1] != b'\n' || bytes[bytes.len() - 1] != b'\r')
            {
                bytes.push(b'\n')
            }
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

#[cfg(feature = "parquet")]
pub(crate) fn apply_projection_pl_schema(schema: &Schema, projection: &[usize]) -> Schema {
    Schema::from_iter(projection.iter().map(|idx| {
        let (name, dt) = schema.get_at_index(*idx).unwrap();
        Field::new(name.as_str(), dt.clone())
    }))
}

#[cfg(feature = "parquet")]
pub(crate) fn columns_to_projection_pl_schema(
    columns: &[String],
    schema: &Schema,
) -> PolarsResult<Vec<usize>> {
    columns
        .iter()
        .map(|name| schema.try_get_full(name).map(|(idx, _, _)| idx))
        .collect()
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
    Regex::new(r"^\s*[-+]?((\d*\.\d+)([eE][-+]?\d+)?|inf|NaN|(\d+)[eE][-+]?\d+|\d+\.)$").unwrap()
});

pub static INTEGER_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"^\s*-?(\d+)$").unwrap());

pub static BOOLEAN_RE: Lazy<Regex> = Lazy::new(|| {
    RegexBuilder::new(r"^\s*(true)$|^(false)$")
        .case_insensitive(true)
        .build()
        .unwrap()
});

pub fn materialize_projection(
    with_columns: Option<&[String]>,
    schema: &Schema,
    hive_partitions: Option<&[Series]>,
    has_row_count: bool,
) -> Option<Vec<usize>> {
    match hive_partitions {
        None => with_columns.map(|with_columns| {
            with_columns
                .iter()
                .map(|name| schema.index_of(name).unwrap() - has_row_count as usize)
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
                            Some(schema.index_of(name).unwrap() - has_row_count as usize)
                        }
                    })
                    .collect()
            })
        },
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::resolve_homedir;

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
