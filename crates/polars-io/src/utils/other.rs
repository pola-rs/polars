#[cfg(any(feature = "ipc_streaming", feature = "parquet"))]
use std::borrow::Cow;
use std::io::Read;

use once_cell::sync::Lazy;
use polars_core::prelude::*;
#[cfg(any(feature = "ipc_streaming", feature = "parquet"))]
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df_as_ref};
use polars_utils::mmap::{MMapSemaphore, MemSlice};
use regex::{Regex, RegexBuilder};

use crate::mmap::{MmapBytesReader, ReaderBytes};

pub fn get_reader_bytes<R: Read + MmapBytesReader + ?Sized>(
    reader: &mut R,
) -> PolarsResult<ReaderBytes<'_>> {
    // we have a file so we can mmap
    // only seekable files are mmap-able
    if let Some((file, offset)) = reader
        .stream_position()
        .ok()
        .and_then(|offset| Some((reader.to_file()?, offset)))
    {
        let mut options = memmap::MmapOptions::new();
        options.offset(offset);
        let mmap = MMapSemaphore::new_from_file_with_options(file, options)?;
        Ok(ReaderBytes::Owned(MemSlice::from_mmap(Arc::new(mmap))))
    } else {
        // we can get the bytes for free
        if reader.to_bytes().is_some() {
            // duplicate .to_bytes() is necessary to satisfy the borrow checker
            Ok(ReaderBytes::Borrowed((*reader).to_bytes().unwrap()))
        } else {
            // we have to read to an owned buffer to get the bytes.
            let mut bytes = Vec::with_capacity(1024 * 128);
            reader.read_to_end(&mut bytes)?;
            Ok(ReaderBytes::Owned(bytes.into()))
        }
    }
}

#[cfg(any(
    feature = "ipc",
    feature = "ipc_streaming",
    feature = "parquet",
    feature = "avro"
))]
pub fn apply_projection(schema: &ArrowSchema, projection: &[usize]) -> ArrowSchema {
    projection
        .iter()
        .map(|idx| schema.get_at_index(*idx).unwrap())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect()
}

#[cfg(any(
    feature = "ipc",
    feature = "ipc_streaming",
    feature = "avro",
    feature = "parquet"
))]
pub fn columns_to_projection<T: AsRef<str>>(
    columns: &[T],
    schema: &ArrowSchema,
) -> PolarsResult<Vec<usize>> {
    let mut prj = Vec::with_capacity(columns.len());

    for column in columns {
        let i = schema.try_index_of(column.as_ref())?;
        prj.push(i);
    }

    Ok(prj)
}

#[cfg(debug_assertions)]
fn check_offsets(dfs: &[DataFrame]) {
    dfs.windows(2).for_each(|s| {
        let a = &s[0].get_columns()[0];
        let b = &s[1].get_columns()[0];

        let prev = a.get(a.len() - 1).unwrap().extract::<usize>().unwrap();
        let next = b.get(0).unwrap().extract::<usize>().unwrap();
        assert_eq!(prev + 1, next);
    })
}

/// Because of threading every row starts from `0` or from `offset`.
/// We must correct that so that they are monotonically increasing.
#[cfg(any(feature = "csv", feature = "json"))]
pub(crate) fn update_row_counts2(dfs: &mut [DataFrame], offset: IdxSize) {
    if !dfs.is_empty() {
        let mut previous = offset;
        for df in &mut *dfs {
            if df.is_empty() {
                continue;
            }
            let n_read = df.height() as IdxSize;
            if let Some(s) = unsafe { df.get_columns_mut() }.get_mut(0) {
                if let Ok(v) = s.get(0) {
                    if v.extract::<usize>().unwrap() != previous as usize {
                        *s = &*s + previous;
                    }
                }
            }
            previous += n_read;
        }
    }
    #[cfg(debug_assertions)]
    {
        check_offsets(dfs)
    }
}

/// Because of threading every row starts from `0` or from `offset`.
/// We must correct that so that they are monotonically increasing.
#[cfg(feature = "json")]
pub(crate) fn update_row_counts3(dfs: &mut [DataFrame], heights: &[IdxSize], offset: IdxSize) {
    assert_eq!(dfs.len(), heights.len());
    if !dfs.is_empty() {
        let mut previous = offset;
        for i in 0..dfs.len() {
            let df = &mut dfs[i];
            if df.is_empty() {
                continue;
            }

            if let Some(s) = unsafe { df.get_columns_mut() }.get_mut(0) {
                if let Ok(v) = s.get(0) {
                    if v.extract::<usize>().unwrap() != previous as usize {
                        *s = &*s + previous;
                    }
                }
            }
            let n_read = heights[i];
            previous += n_read;
        }
    }
}

#[cfg(feature = "json")]
pub fn overwrite_schema(schema: &mut Schema, overwriting_schema: &Schema) -> PolarsResult<()> {
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
    with_columns: Option<&[PlSmallStr]>,
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
    df.align_chunks_par();

    // Accumulate many small chunks to the row group size.
    // See: #16403
    if !df.get_columns().is_empty()
        && df.get_columns()[0]
            .as_materialized_series()
            .chunk_lengths()
            .take(5)
            .all(|len| len < row_group_size)
    {
        fn finish(scratch: &mut Vec<DataFrame>, new_chunks: &mut Vec<DataFrame>) {
            let mut new = accumulate_dataframes_vertical_unchecked(scratch.drain(..));
            new.as_single_chunk_par();
            new_chunks.push(new);
        }

        let mut new_chunks = Vec::with_capacity(df.first_col_n_chunks()); // upper limit;
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
            let n_chunks = df.first_col_n_chunks();
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

#[cfg(test)]
mod tests {
    use super::FLOAT_RE;

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
}
