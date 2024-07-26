#[cfg(any(feature = "ipc_streaming", feature = "parquet"))]
use std::borrow::Cow;
use std::io::Read;

use once_cell::sync::Lazy;
use polars_core::prelude::*;
#[cfg(any(feature = "ipc_streaming", feature = "parquet"))]
use polars_core::utils::{accumulate_dataframes_vertical_unchecked, split_df_as_ref};
use polars_error::to_compute_err;
use regex::{Regex, RegexBuilder};

use crate::mmap::{MmapBytesReader, ReaderBytes};

pub fn get_reader_bytes<'a, R: Read + MmapBytesReader + ?Sized>(
    reader: &'a mut R,
) -> PolarsResult<ReaderBytes<'a>> {
    // we have a file so we can mmap
    // only seekable files are mmap-able
    if let Some((file, offset)) = reader
        .stream_position()
        .ok()
        .and_then(|offset| Some((reader.to_file()?, offset)))
    {
        let mmap = unsafe { memmap::MmapOptions::new().offset(offset).map(file)? };

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

/// Decompress `bytes` if compression is detected, otherwise simply return it.
/// An `out` vec must be given for ownership of the decompressed data.
///
/// # Safety
/// The `out` vec outlives `bytes` (declare `out` first).
pub unsafe fn maybe_decompress_bytes<'a>(
    bytes: &'a [u8],
    out: &'a mut Vec<u8>,
) -> PolarsResult<&'a [u8]> {
    assert!(out.is_empty());
    use crate::prelude::is_compressed;
    let is_compressed = bytes.len() >= 4 && is_compressed(bytes);

    if is_compressed {
        #[cfg(any(feature = "decompress", feature = "decompress-fast"))]
        {
            use crate::utils::compression::magic::*;

            if bytes.starts_with(&GZIP) {
                flate2::read::MultiGzDecoder::new(bytes)
                    .read_to_end(out)
                    .map_err(to_compute_err)?;
            } else if bytes.starts_with(&ZLIB0)
                || bytes.starts_with(&ZLIB1)
                || bytes.starts_with(&ZLIB2)
            {
                flate2::read::ZlibDecoder::new(bytes)
                    .read_to_end(out)
                    .map_err(to_compute_err)?;
            } else if bytes.starts_with(&ZSTD) {
                zstd::Decoder::new(bytes)?.read_to_end(out)?;
            } else {
                polars_bail!(ComputeError: "unimplemented compression format")
            }

            Ok(out)
        }
        #[cfg(not(any(feature = "decompress", feature = "decompress-fast")))]
        {
            panic!("cannot decompress without 'decompress' or 'decompress-fast' feature")
        }
    } else {
        Ok(bytes)
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

/// Because of threading every row starts from `0` or from `offset`.
/// We must correct that so that they are monotonically increasing.
#[cfg(feature = "json")]
pub(crate) fn update_row_counts3(dfs: &mut [DataFrame], heights: &[IdxSize], offset: IdxSize) {
    assert_eq!(dfs.len(), heights.len());
    if !dfs.is_empty() {
        let mut previous = heights[0] + offset;
        for i in 1..dfs.len() {
            let df = &mut dfs[i];
            let n_read = heights[i];

            if let Some(s) = unsafe { df.get_columns_mut() }.get_mut(0) {
                *s = &*s + previous;
            }

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
