#![allow(unsafe_op_in_unsafe_fn)]
#[cfg(feature = "decompress")]
use std::io::Read;
use std::mem::MaybeUninit;

use super::parser::next_line_position;
#[cfg(feature = "decompress")]
use super::parser::next_line_position_naive;
use super::splitfields::SplitFields;

#[cfg(feature = "decompress")]
fn decompress_impl<R: Read>(
    decoder: &mut R,
    n_rows: Option<usize>,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) -> Option<Vec<u8>> {
    let chunk_size = 4096;
    Some(match n_rows {
        None => {
            // decompression in a preallocated buffer does not work with zlib-ng
            // and will put the original compressed data in the buffer.
            let mut out = Vec::new();
            decoder.read_to_end(&mut out).ok()?;
            out
        },
        Some(n_rows) => {
            // we take the first rows first '\n\
            let mut out = vec![];
            let mut expected_fields = 0;
            // make sure that we have enough bytes to decode the header (even if it has embedded new line chars)
            // those extra bytes in the buffer don't matter, we don't need to track them
            loop {
                let read = decoder.take(chunk_size).read_to_end(&mut out).ok()?;
                if read == 0 {
                    break;
                }
                if next_line_position_naive(&out, eol_char).is_some() {
                    // an extra shot
                    let read = decoder.take(chunk_size).read_to_end(&mut out).ok()?;
                    if read == 0 {
                        break;
                    }
                    // now that we have enough, we compute the number of fields (also takes embedding into account)
                    expected_fields =
                        SplitFields::new(&out, separator, quote_char, eol_char).count();
                    break;
                }
            }

            let mut line_count = 0;
            let mut buf_pos = 0;
            // keep decoding bytes and count lines
            // keep track of the n_rows we read
            while line_count < n_rows {
                match next_line_position(
                    &out[buf_pos + 1..],
                    Some(expected_fields),
                    separator,
                    quote_char,
                    eol_char,
                ) {
                    Some(pos) => {
                        line_count += 1;
                        buf_pos += pos;
                    },
                    None => {
                        // take more bytes so that we might find a new line the next iteration
                        let read = decoder.take(chunk_size).read_to_end(&mut out).ok()?;
                        // we depleted the reader
                        if read == 0 {
                            break;
                        }
                        continue;
                    },
                };
            }
            if line_count == n_rows {
                out.truncate(buf_pos); // retain only first n_rows in out
            }
            out
        },
    })
}

#[cfg(feature = "decompress")]
pub(crate) fn decompress(
    bytes: &[u8],
    n_rows: Option<usize>,
    separator: u8,
    quote_char: Option<u8>,
    eol_char: u8,
) -> Option<Vec<u8>> {
    use crate::utils::compression::SupportedCompression;

    let algo = SupportedCompression::check(bytes)?;

    match algo {
        SupportedCompression::GZIP => {
            let mut decoder = flate2::read::MultiGzDecoder::new(bytes);
            decompress_impl(&mut decoder, n_rows, separator, quote_char, eol_char)
        },
        SupportedCompression::ZLIB => {
            let mut decoder = flate2::read::ZlibDecoder::new(bytes);
            decompress_impl(&mut decoder, n_rows, separator, quote_char, eol_char)
        },
        SupportedCompression::ZSTD => {
            let mut decoder = zstd::Decoder::with_buffer(bytes).ok()?;
            decompress_impl(&mut decoder, n_rows, separator, quote_char, eol_char)
        },
    }
}

/// replace double quotes by single ones
///
/// This function assumes that bytes is wrapped in the quoting character.
///
/// # Safety
///
/// The caller must ensure that:
///     - Output buffer must have enough capacity to hold `bytes.len()`
///     - bytes ends with the quote character e.g.: `"`
///     - bytes length > 1.
pub(super) unsafe fn escape_field(bytes: &[u8], quote: u8, buf: &mut [MaybeUninit<u8>]) -> usize {
    debug_assert!(bytes.len() > 1);
    let mut prev_quote = false;

    let mut count = 0;
    for c in bytes.get_unchecked(1..bytes.len() - 1) {
        if *c == quote {
            if prev_quote {
                prev_quote = false;
                buf.get_unchecked_mut(count).write(*c);
                count += 1;
            } else {
                prev_quote = true;
            }
        } else {
            prev_quote = false;
            buf.get_unchecked_mut(count).write(*c);
            count += 1;
        }
    }
    count
}
