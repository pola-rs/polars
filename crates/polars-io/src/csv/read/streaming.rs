use std::cmp;
use std::iter::Iterator;
use std::sync::Arc;

use polars_buffer::Buffer;
use polars_core::prelude::Schema;
use polars_core::schema::SchemaRef;
use polars_error::{PolarsResult, polars_bail, polars_ensure};

use crate::csv::read::schema_inference::infer_file_schema_impl;
use crate::prelude::_csv_read_internal::{SplitLines, is_comment_line};
use crate::prelude::{CsvParseOptions, CsvReadOptions};
use crate::utils::compression::CompressedReader;

pub type InspectContentFn<'a> = Box<dyn FnMut(&[u8]) + 'a>;

/// Reads bytes from `reader` until the CSV starting point is reached depending on the options.
///
/// Returns the inferred schema and leftover bytes not yet consumed, which may be empty. The
/// leftover bytes + `reader.read_next_slice` is guaranteed to start at first real content row.
///
/// `inspect_first_content_row_fn` allows looking at the first content row, this is where parsing
/// will start. Beware even if the function is provided it's *not* guaranteed that the returned
/// value will be `Some`, since it the CSV may be incomplete.
///
/// The reading is done in an iterative streaming fashion
///
/// This function isn't perf critical but would increase binary-size so don't inline it.
#[inline(never)]
pub fn read_until_start_and_infer_schema(
    options: &CsvReadOptions,
    projected_schema: Option<SchemaRef>,
    mut inspect_first_content_row_fn: Option<InspectContentFn<'_>>,
    reader: &mut CompressedReader,
) -> PolarsResult<(Schema, Buffer<u8>)> {
    #[derive(Copy, Clone)]
    enum State {
        // Ordered so that all states only happen after the ones before it.
        SkipEmpty,
        SkipRowsBeforeHeader(usize),
        SkipHeader(bool),
        SkipRowsAfterHeader(usize),
        ContentInspect,
        InferCollect,
        Done,
    }

    polars_ensure!(
        !(options.skip_lines != 0 && options.skip_rows != 0),
        InvalidOperation: "only one of 'skip_rows'/'skip_lines' may be set"
    );

    // We have to treat skip_lines differently since the lines it skips may not follow regular CSV
    // quote escape rules.
    let prev_leftover = skip_lines_naive(
        options.parse_options.eol_char,
        options.skip_lines,
        options.raise_if_empty,
        reader,
    )?;

    let mut state = if options.has_header {
        State::SkipEmpty
    } else if options.skip_lines != 0 {
        // skip_lines shouldn't skip extra comments before the header, so directly go to SkipHeader
        // state.
        State::SkipHeader(false)
    } else {
        State::SkipRowsBeforeHeader(options.skip_rows)
    };

    let comment_prefix = options.parse_options.comment_prefix.as_ref();
    let infer_schema_length = options.infer_schema_length.unwrap_or(usize::MAX);

    let mut header_line = None;
    let mut content_lines = Vec::with_capacity(options.infer_schema_length.unwrap_or(256));

    let leftover = for_each_line_from_reader(
        &options.parse_options,
        true,
        prev_leftover,
        reader,
        |mem_slice_line| {
            let line = &*mem_slice_line;

            let done = loop {
                match &mut state {
                    State::SkipEmpty => {
                        if line.is_empty() || line == b"\r" {
                            break LineUse::ConsumeDiscard;
                        }

                        state = State::SkipRowsBeforeHeader(options.skip_rows);
                    },
                    State::SkipRowsBeforeHeader(remaining) => {
                        let is_comment = is_comment_line(line, comment_prefix);

                        if *remaining == 0 && !is_comment {
                            state = State::SkipHeader(false);
                            continue;
                        }

                        *remaining -= !is_comment as usize;
                        break LineUse::ConsumeDiscard;
                    },
                    State::SkipHeader(did_skip) => {
                        if !options.has_header || *did_skip {
                            state = State::SkipRowsAfterHeader(options.skip_rows_after_header);
                            continue;
                        }

                        header_line = Some(mem_slice_line.clone());
                        *did_skip = true;
                        break LineUse::ConsumeDiscard;
                    },
                    State::SkipRowsAfterHeader(remaining) => {
                        let is_comment = is_comment_line(line, comment_prefix);

                        if *remaining == 0 && !is_comment {
                            state = State::ContentInspect;
                            continue;
                        }

                        *remaining -= !is_comment as usize;
                        break LineUse::ConsumeDiscard;
                    },
                    State::ContentInspect => {
                        if let Some(func) = &mut inspect_first_content_row_fn {
                            func(line);
                        }

                        state = State::InferCollect;
                    },
                    State::InferCollect => {
                        if !is_comment_line(line, comment_prefix) {
                            content_lines.push(mem_slice_line.clone());
                            if content_lines.len() >= infer_schema_length {
                                state = State::Done;
                                continue;
                            }
                        }

                        break LineUse::ConsumeKeep;
                    },
                    State::Done => {
                        break LineUse::Done;
                    },
                }
            };

            Ok(done)
        },
    )?;

    let infer_all_as_str = infer_schema_length == 0;

    let inferred_schema = infer_schema(
        &header_line,
        &content_lines,
        infer_all_as_str,
        options,
        projected_schema,
    )?;

    Ok((inferred_schema, leftover))
}

enum LineUse {
    ConsumeDiscard,
    ConsumeKeep,
    Done,
}

/// Iterate over valid CSV lines produced by reader.
///
/// Returning `ConsumeDiscard` after `ConsumeKeep` is a logic error, since a segmented `Buffer`
/// can't be constructed.
fn for_each_line_from_reader(
    parse_options: &CsvParseOptions,
    is_file_start: bool,
    mut prev_leftover: Buffer<u8>,
    reader: &mut CompressedReader,
    mut line_fn: impl FnMut(Buffer<u8>) -> PolarsResult<LineUse>,
) -> PolarsResult<Buffer<u8>> {
    let mut is_first_line = is_file_start;

    let mut read_size = CompressedReader::initial_read_size();
    let mut retain_offset = None;

    loop {
        let (mut slice, bytes_read) = reader.read_next_slice(&prev_leftover, read_size)?;
        if slice.is_empty() {
            return Ok(Buffer::new());
        }

        if is_first_line {
            is_first_line = false;
            const UTF8_BOM_MARKER: Option<&[u8]> = Some(b"\xef\xbb\xbf");
            if slice.get(0..3) == UTF8_BOM_MARKER {
                slice = slice.sliced(3..);
            }
        }

        let line_to_sub_slice = |line: &[u8]| {
            let start = line.as_ptr() as usize - slice.as_ptr() as usize;
            slice.clone().sliced(start..(start + line.len()))
        };

        // When reading a CSV with `has_header=False` we need to read up to `infer_schema_length` lines, but we only want to decompress the input once, so we grow a `Buffer` that will be returned as leftover.
        let effective_slice = if let Some(offset) = retain_offset {
            slice.clone().sliced(offset..)
        } else {
            slice.clone()
        };

        let mut lines = SplitLines::new(
            &effective_slice,
            parse_options.quote_char,
            parse_options.eol_char,
            parse_options.comment_prefix.as_ref(),
        );
        let Some(mut prev_line) = lines.next() else {
            read_size = read_size.saturating_mul(2);
            prev_leftover = slice;
            continue;
        };

        let mut should_ret = false;

        // The last line in `SplitLines` may be incomplete if `slice` ends before the file does, so
        // we iterate everything except the last line.
        for next_line in lines {
            match line_fn(line_to_sub_slice(prev_line))? {
                LineUse::ConsumeDiscard => debug_assert!(retain_offset.is_none()),
                LineUse::ConsumeKeep => {
                    retain_offset
                        .get_or_insert(prev_line.as_ptr() as usize - slice.as_ptr() as usize);
                },
                LineUse::Done => {
                    should_ret = true;
                    break;
                },
            }
            prev_line = next_line;
        }

        let mut unconsumed_offset = prev_line.as_ptr() as usize - slice.as_ptr() as usize;

        // EOF file reached, the last line will have no continuation on the next call to
        // `read_next_slice`.
        if bytes_read == 0 {
            match line_fn(line_to_sub_slice(prev_line))? {
                LineUse::ConsumeDiscard => {
                    unconsumed_offset += prev_line.len();
                    if slice.get(unconsumed_offset) == Some(&parse_options.eol_char) {
                        unconsumed_offset += 1;
                    }
                },
                LineUse::ConsumeKeep | LineUse::Done => (),
            }
            should_ret = true;
        }

        if retain_offset.is_some() {
            prev_leftover = slice;
        } else {
            // Since `read_next_slice` has to copy the leftover bytes in the decompression case,
            // it's more efficient to hand in as little as possible.
            prev_leftover = slice.sliced(unconsumed_offset..);
        }

        if should_ret {
            let leftover = prev_leftover.sliced(retain_offset.unwrap_or(0)..);
            return Ok(leftover);
        }

        if read_size < CompressedReader::ideal_read_size() {
            read_size *= 4;
        }
    }
}

fn skip_lines_naive(
    eol_char: u8,
    skip_lines: usize,
    raise_if_empty: bool,
    reader: &mut CompressedReader,
) -> PolarsResult<Buffer<u8>> {
    let mut prev_leftover = Buffer::new();

    if skip_lines == 0 {
        return Ok(prev_leftover);
    }

    let mut remaining = skip_lines;
    let mut read_size = CompressedReader::initial_read_size();

    loop {
        let (slice, bytes_read) = reader.read_next_slice(&prev_leftover, read_size)?;
        let mut bytes: &[u8] = &slice;

        'inner: loop {
            let Some(mut pos) = memchr::memchr(eol_char, bytes) else {
                read_size = read_size.saturating_mul(2);
                break 'inner;
            };
            pos = cmp::min(pos + 1, bytes.len());

            bytes = &bytes[pos..];
            remaining -= 1;

            if remaining == 0 {
                let unconsumed_offset = bytes.as_ptr() as usize - slice.as_ptr() as usize;
                prev_leftover = slice.sliced(unconsumed_offset..);
                return Ok(prev_leftover);
            }
        }

        if bytes_read == 0 {
            if raise_if_empty {
                polars_bail!(NoData: "specified skip_lines is larger than total number of lines.");
            } else {
                return Ok(Buffer::new());
            }
        }

        // No need to search for naive eol twice in the leftover.
        prev_leftover = Buffer::new();

        if read_size < CompressedReader::ideal_read_size() {
            read_size *= 4;
        }
    }
}

fn infer_schema(
    header_line: &Option<Buffer<u8>>,
    content_lines: &[Buffer<u8>],
    infer_all_as_str: bool,
    options: &CsvReadOptions,
    projected_schema: Option<SchemaRef>,
) -> PolarsResult<Schema> {
    let has_no_inference_data = if options.has_header {
        header_line.is_none()
    } else {
        content_lines.is_empty()
    };

    if options.raise_if_empty && has_no_inference_data {
        polars_bail!(NoData: "empty CSV");
    }

    let mut inferred_schema = if has_no_inference_data {
        Schema::default()
    } else {
        infer_file_schema_impl(
            header_line,
            content_lines,
            infer_all_as_str,
            &options.parse_options,
            options.schema_overwrite.as_deref(),
        )
    };

    if let Some(schema) = &options.schema {
        // Note: User can provide schema with more columns, they will simply
        // be projected as NULL.
        // TODO: Should maybe expose a missing_columns parameter to the API for this.
        if schema.len() < inferred_schema.len() && !options.parse_options.truncate_ragged_lines {
            polars_bail!(
                SchemaMismatch:
                "provided schema does not match number of columns in file ({} != {} in file)",
                schema.len(),
                inferred_schema.len(),
            );
        }

        if options.parse_options.truncate_ragged_lines {
            inferred_schema = Arc::unwrap_or_clone(schema.clone());
        } else {
            inferred_schema = schema
                .iter_names()
                .zip(inferred_schema.into_iter().map(|(_, dtype)| dtype))
                .map(|(name, dtype)| (name.clone(), dtype))
                .collect();
        }
    }

    if let Some(dtypes) = options.dtype_overwrite.as_deref() {
        for (i, dtype) in dtypes.iter().enumerate() {
            inferred_schema.set_dtype_at_index(i, dtype.clone());
        }
    }

    // TODO: We currently always override with the projected dtype, but this may cause issues e.g.
    // with temporal types. This can be improved to better choose between the 2 dtypes.
    if let Some(projected_schema) = projected_schema {
        for (name, inferred_dtype) in inferred_schema.iter_mut() {
            if let Some(projected_dtype) = projected_schema.get(name) {
                *inferred_dtype = projected_dtype.clone();
            }
        }
    }

    Ok(inferred_schema)
}
