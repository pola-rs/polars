use arrow::array::MutableBinaryViewArray;
use num_traits::pow::Pow;
use polars_core::POOL;
use polars_core::prelude::{Arc, DataFrame, IntoColumn, PolarsResult, StringChunked, polars_bail};
use polars_core::utils::accumulate_dataframes_vertical;
use rayon::prelude::*;

use crate::line::options::TextEncoding;
use crate::mmap::ReaderBytes;
use crate::predicates::PhysicalIoExpr;

pub(crate) struct CoreReader<'a> {
    reader_bytes: Option<ReaderBytes<'a>>,
    n_lines: Option<usize>,
    skip_lines: usize,
    n_threads: Option<usize>,
    low_memory: bool,
    chunk_size: usize,
    eol_char: u8,
    encoding: TextEncoding,
    predicate: Option<Arc<dyn PhysicalIoExpr>>,
    sample_size: usize,
}
impl<'a> CoreReader<'a> {
    pub(crate) fn new(
        reader_bytes: ReaderBytes<'a>,
        n_lines: Option<usize>,
        skip_lines: usize,
        n_threads: Option<usize>,
        low_memory: bool,
        chunk_size: usize,
        sample_size: usize,
        eol_char: u8,
        encoding: TextEncoding,
        predicate: Option<Arc<dyn PhysicalIoExpr>>,
    ) -> PolarsResult<CoreReader<'a>> {
        let reader_bytes = reader_bytes;

        Ok(Self {
            reader_bytes: Some(reader_bytes),
            n_lines,
            skip_lines,
            n_threads,
            low_memory,
            chunk_size,
            eol_char,
            encoding,
            predicate,
            sample_size,
        })
    }

    fn read_lines(&mut self, mut n_threads: usize, bytes: &[u8]) -> PolarsResult<DataFrame> {
        let mut bytes = bytes;
        let mut total_lines = 128;

        if let Some((mean, std)) = get_line_stats(bytes, self.eol_char, self.sample_size) {
            let line_length_upper_bound = mean + 1.1 * std;

            total_lines = (bytes.len() as f32 / (mean - 0.01 * std)) as usize;
            if let Some(n_lines) = self.n_lines {
                total_lines = std::cmp::min(n_lines, total_lines);
                // the guessed upper bound of  the no. of bytes in the file
                let n_bytes = (line_length_upper_bound * (n_lines as f32)) as usize;

                if n_bytes < bytes.len() {
                    if let Some(pos) = next_line_position_naive(&bytes[n_bytes..], self.eol_char) {
                        bytes = &bytes[..n_bytes + pos]
                    }
                }
            }
        }

        if total_lines <= 128 {
            n_threads = 1;
        }

        let lines_per_thread = total_lines / n_threads;

        let max_proxy = bytes.len() / n_threads / 2;
        let capacity = if self.low_memory {
            usize::from(self.chunk_size)
        } else {
            std::cmp::min(lines_per_thread, max_proxy)
        };
        let file_chunks = get_file_chunks(bytes, self.eol_char, n_threads);

        let mut vs = POOL.install(|| {
            file_chunks
                .into_par_iter()
                .map(|(start_pos, stop_at_nbytes)| {
                    let mut local_df =
                        self.parse_lines(&bytes[start_pos..stop_at_nbytes], capacity, true)?;
                    if let Some(predicate) = &self.predicate {
                        let s = predicate.evaluate_io(&local_df)?;
                        let mask = s.bool()?;
                        local_df = local_df.filter(mask)?;
                    }
                    Ok((local_df, start_pos))
                })
                .collect::<PolarsResult<Vec<(DataFrame, usize)>>>()
        })?;
        vs.sort_by_key(|v| v.1);

        accumulate_dataframes_vertical(vs.into_iter().map(|v| v.0))
    }

    pub fn finish(mut self) -> PolarsResult<DataFrame> {
        let n_threads = self.n_threads.unwrap_or_else(|| POOL.current_num_threads());

        let reader_bytes = self.reader_bytes.take().unwrap();
        let reader_bytes = skip_lines_naive(&reader_bytes, self.eol_char, self.skip_lines);

        let mut df = self.read_lines(n_threads, &reader_bytes)?;

        // if multi-threaded the n_lines was probabilistically determined.
        // Let's slice to correct number of rows if possible.
        if let Some(n_lines) = self.n_lines {
            if n_lines < df.height() {
                df = df.slice(0, n_lines)
            }
        }
        Ok(df)
    }

    fn parse_lines(
        &self,
        bytes: &[u8],
        capacity: usize,
        ignore_errors: bool,
    ) -> PolarsResult<DataFrame> {
        let mut buf = MutableBinaryViewArray::<[u8]>::with_capacity(capacity);
        let iter = bytes.split(|&byte| byte == self.eol_char);
        for line in iter {
            if line.is_empty() {
                buf.push(Some([]));
                continue;
            }

            if matches!(self.encoding, TextEncoding::LossyUtf8) | ignore_errors {
                let parse_result = simdutf8::basic::from_utf8(bytes).is_ok();

                match parse_result {
                    true => {
                        let value = line;
                        buf.push_value(value)
                    },
                    false => {
                        if matches!(self.encoding, TextEncoding::LossyUtf8) {
                            // TODO! do this without allocating
                            let s = String::from_utf8_lossy(line);
                            buf.push_value(s.as_ref().as_bytes())
                        } else if ignore_errors {
                            buf.push_null()
                        } else {
                            polars_bail!(ComputeError: "invalid utf-8 sequence");
                        }
                    },
                }
            } else {
                buf.push_value(line)
            }
        }
        Ok(StringChunked::with_chunk("lines".into(), unsafe {
            buf.freeze().to_utf8view_unchecked()
        })
        .into_column()
        .into_frame())
    }
}

fn skip_lines_naive(mut input: &[u8], eol_char: u8, skip: usize) -> &[u8] {
    for _ in 0..skip {
        if let Some(pos) = next_line_position_naive(input, eol_char) {
            input = &input[pos..];
        } else {
            return input;
        }
    }
    input
}

/// Find the nearest next line position.
fn next_line_position_naive(input: &[u8], eol_char: u8) -> Option<usize> {
    let pos = memchr::memchr(eol_char, input)? + 1;
    if input.len() - pos == 0 {
        return None;
    }
    Some(pos)
}

fn get_line_stats(bytes: &[u8], eol_char: u8, sample_size: usize) -> Option<(f32, f32)> {
    let mut lengths = Vec::with_capacity(sample_size);

    let mut bytes_trunc;
    let n_lines_per_iter = sample_size / 2;

    let mut n_read = 0;

    // sample from start and 75% in the file
    for offset in [0, (bytes.len() as f32 * 0.75) as usize] {
        bytes_trunc = &bytes[offset..];

        for _ in offset..(offset + n_lines_per_iter) {
            let pos = next_line_position_naive(bytes_trunc, eol_char)?;
            // `pos` should point to the beginning of the next line, so its previous slot
            // would be eol_char. The actual line is hence [0..=pos-2] with length pos-1
            n_read += pos - 1;
            lengths.push(pos - 1);
            bytes_trunc = &bytes_trunc[pos..];
        }
    }

    let n_samples = lengths.len();

    let mean = (n_read as f32) / (n_samples as f32);
    let mut std = 0.0;
    for &len in lengths.iter() {
        std += (len as f32 - mean).pow(2.0)
    }
    std = (std / n_samples as f32).sqrt();
    Some((mean, std))
}

fn get_file_chunks(bytes: &[u8], eol_char: u8, n_threads: usize) -> Vec<(usize, usize)> {
    let mut last_pos = 0;
    let total_len = bytes.len();
    let chunk_size = total_len / n_threads;
    let mut offsets = Vec::with_capacity(n_threads);
    for _ in 0..n_threads {
        let search_pos = last_pos + chunk_size;

        if search_pos >= bytes.len() {
            break;
        }

        let end_pos = match next_line_position_naive(&bytes[search_pos..], eol_char) {
            Some(pos) => search_pos + pos,
            None => {
                break;
            },
        };
        offsets.push((last_pos, end_pos));
        last_pos = end_pos;
    }
    offsets.push((last_pos, total_len));
    offsets
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use polars_core::prelude::DataType;
    use tempfile::NamedTempFile;

    use crate::line::options::TextEncoding;
    use crate::line::read_impl::CoreReader;
    use crate::utils::get_reader_bytes;

    #[test]
    fn read() {
        let mut tmp_file = NamedTempFile::new().unwrap();
        let mut text = vec![];
        for i in 0..26u8 {
            for j in 0..26u8 {
                let c = if j == 0 { "!" } else { "" };
                text.push(format!("{}{}{c}", (97 + i) as char, (97 + j) as char));
            }
        }
        tmp_file.write_all(text.join("\n").as_bytes()).unwrap();
        let mut file = polars_utils::open_file(tmp_file.path()).unwrap();
        let bytes = get_reader_bytes(&mut file).unwrap();
        let reader = CoreReader::new(
            bytes,
            None,
            0,
            None,
            false,
            1024,
            1024,
            b'\n',
            TextEncoding::Utf8,
            None,
        )
        .unwrap();

        let df = reader.finish().unwrap();
        assert_eq!(df.size(), 676);
        let cols = df.get_columns();
        assert_eq!(cols.len(), 1);
        assert_eq!(cols[0].dtype(), &DataType::String);
        assert_eq!(cols[0].name(), "lines");

        // collect all rows sequentially and check if they are in the original order
        let raw_values = df
            .iter()
            .next()
            .unwrap()
            .str()
            .iter()
            .map(|v| {
                v.into_iter()
                    .map(|x| x.unwrap_or("").to_string())
                    .collect::<Vec<String>>()
                    .join("\n")
            })
            .collect::<Vec<String>>()
            .join("")
            .split("\n")
            .map(|v| v.to_string())
            .collect::<Vec<String>>();
        assert_eq!(raw_values, text);
    }

    #[test]
    fn get_file_chunks() {
        let s = vec!["a".repeat(3), "a".repeat(10), "a".repeat(2)].join("\n");
        let res = super::get_file_chunks(s.as_bytes(), b'\n', 1);
        assert_eq!(res, vec![(0, 17)]);

        // the split point is at index 8, so the first thread should look for the
        // immediate next eol_char after that
        let res = super::get_file_chunks(s.as_bytes(), b'\n', 2);
        assert_eq!(res, vec![(0, 15), (15, 17)]);

        // there is no eol_char after the split point, so the second thread gets nothing
        let s = vec!["a".repeat(3), "a".repeat(10)].join("\n");
        let res = super::get_file_chunks(s.as_bytes(), b'\n', 2);
        assert_eq!(res, vec![(0, 14)]);
    }

    #[test]
    fn next_line_position_naive() {
        assert!(super::next_line_position_naive("aaa".as_bytes(), b'\n').is_none());
        assert_eq!(
            super::next_line_position_naive("a\nb".as_bytes(), b'\n'),
            Some(2)
        );
        assert_eq!(
            super::next_line_position_naive("abc\n\n".as_bytes(), b'\n'),
            Some(4)
        );
        // trailing eol_char does not count as a new line
        assert!(super::next_line_position_naive("a\n".as_bytes(), b'\n').is_none());
    }

    #[test]
    fn skip_lines_naive() {
        let s = "a!b!c!!d!";
        assert_eq!(super::skip_lines_naive(s.as_bytes(), b'!', 0), b"a!b!c!!d!");
        assert_eq!(super::skip_lines_naive(s.as_bytes(), b'!', 1), b"b!c!!d!");
        assert_eq!(super::skip_lines_naive(s.as_bytes(), b'!', 2), b"c!!d!");
        assert_eq!(super::skip_lines_naive(s.as_bytes(), b'!', 3), b"!d!");
        assert_eq!(super::skip_lines_naive(s.as_bytes(), b'!', 4), b"d!");
        // no more new lines so it returns the same thing
        assert_eq!(super::skip_lines_naive(s.as_bytes(), b'!', 5), b"d!");
    }

    #[test]
    fn get_line_stats() {
        let mut tokens = vec![];
        for i in 0..100 {
            tokens.push("a".repeat(i % 4 + 5));
        }
        let s = tokens.join("\n");
        let (mean, std) = super::get_line_stats(s.as_bytes(), b'\n', 8).unwrap();
        assert!((std - 1.118034).abs() < 1e-6);
        assert_eq!(mean, 6.5);

        let s = "a\n".repeat(100);
        let (mean, std) = super::get_line_stats(s.as_bytes(), b'\n', 8).unwrap();
        assert_eq!(std, 0.0);
        assert_eq!(mean, 1.0);
    }
}
