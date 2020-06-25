// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use std::{cell::RefCell, cmp, io::*};

use crate::file::{reader::ParquetReader, writer::ParquetWriter};

const DEFAULT_BUF_SIZE: usize = 8 * 1024;

// ----------------------------------------------------------------------
// Read/Write wrappers for `File`.

/// Position trait returns the current position in the stream.
/// Should be viewed as a lighter version of `Seek` that does not allow seek operations,
/// and does not require mutable reference for the current position.
pub trait Position {
    /// Returns position in the stream.
    fn pos(&self) -> u64;
}

/// Struct that represents a slice of a file data with independent start position and
/// length. Internally clones provided file handle, wraps with a custom implementation
/// of BufReader that resets position before any read.
///
/// This is workaround and alternative for `file.try_clone()` method. It clones `File`
/// while preserving independent position, which is not available with `try_clone()`.
///
/// Designed after `some::io::RandomAccessFile` and `std::io::BufReader`
pub struct FileSource<R: ParquetReader> {
    reader: RefCell<R>,
    start: u64,     // start position in a file
    end: u64,       // end position in a file
    buf: Vec<u8>,   // buffer where bytes read in advance are stored
    buf_pos: usize, // current position of the reader in the buffer
    buf_cap: usize, // current number of bytes read into the buffer
}

impl<R: ParquetReader> FileSource<R> {
    /// Creates new file reader with start and length from a file handle
    pub fn new(fd: &R, start: u64, length: usize) -> Self {
        let reader = RefCell::new(fd.try_clone().unwrap());
        Self {
            reader,
            start,
            end: start + length as u64,
            buf: vec![0 as u8; DEFAULT_BUF_SIZE],
            buf_pos: 0,
            buf_cap: 0,
        }
    }

    fn fill_inner_buf(&mut self) -> Result<&[u8]> {
        if self.buf_pos >= self.buf_cap {
            // If we've reached the end of our internal buffer then we need to fetch
            // some more data from the underlying reader.
            // Branch using `>=` instead of the more correct `==`
            // to tell the compiler that the pos..cap slice is always valid.
            debug_assert!(self.buf_pos == self.buf_cap);
            let mut reader = self.reader.borrow_mut();
            reader.seek(SeekFrom::Start(self.start))?; // always seek to start before reading
            self.buf_cap = reader.read(&mut self.buf)?;
            self.buf_pos = 0;
        }
        Ok(&self.buf[self.buf_pos..self.buf_cap])
    }

    fn skip_inner_buf(&mut self, buf: &mut [u8]) -> Result<usize> {
        // discard buffer
        self.buf_pos = 0;
        self.buf_cap = 0;
        // read directly into param buffer
        let mut reader = self.reader.borrow_mut();
        reader.seek(SeekFrom::Start(self.start))?; // always seek to start before reading
        let nread = reader.read(buf)?;
        self.start += nread as u64;
        Ok(nread)
    }
}

impl<R: ParquetReader> Read for FileSource<R> {
    fn read(&mut self, buf: &mut [u8]) -> Result<usize> {
        let bytes_to_read = cmp::min(buf.len(), (self.end - self.start) as usize);
        let buf = &mut buf[0..bytes_to_read];

        // If we don't have any buffered data and we're doing a massive read
        // (larger than our internal buffer), bypass our internal buffer
        // entirely.
        if self.buf_pos == self.buf_cap && buf.len() >= self.buf.len() {
            return self.skip_inner_buf(buf);
        }
        let nread = {
            let mut rem = self.fill_inner_buf()?;
            // copy the data from the inner buffer to the param buffer
            rem.read(buf)?
        };
        // consume from buffer
        self.buf_pos = cmp::min(self.buf_pos + nread, self.buf_cap);

        self.start += nread as u64;
        Ok(nread)
    }
}

impl<R: ParquetReader> Position for FileSource<R> {
    fn pos(&self) -> u64 {
        self.start
    }
}

/// Struct that represents `File` output stream with position tracking.
/// Used as a sink in file writer.
pub struct FileSink<W: ParquetWriter> {
    buf: BufWriter<W>,
    // This is not necessarily position in the underlying file,
    // but rather current position in the sink.
    pos: u64,
}

impl<W: ParquetWriter> FileSink<W> {
    /// Creates new file sink.
    /// Position is set to whatever position file has.
    pub fn new(buf: &W) -> Self {
        let mut owned_buf = buf.try_clone().unwrap();
        let pos = owned_buf.seek(SeekFrom::Current(0)).unwrap();
        Self {
            buf: BufWriter::new(owned_buf),
            pos,
        }
    }
}

impl<W: ParquetWriter> Write for FileSink<W> {
    fn write(&mut self, buf: &[u8]) -> Result<usize> {
        let num_bytes = self.buf.write(buf)?;
        self.pos += num_bytes as u64;
        Ok(num_bytes)
    }

    fn flush(&mut self) -> Result<()> {
        self.buf.flush()
    }
}

impl<W: ParquetWriter> Position for FileSink<W> {
    fn pos(&self) -> u64 {
        self.pos
    }
}

// Position implementation for Cursor to use in various tests.
impl<'a> Position for Cursor<&'a mut Vec<u8>> {
    fn pos(&self) -> u64 {
        self.position()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::iter;

    use crate::util::test_common::{get_temp_file, get_test_file};

    #[test]
    fn test_io_read_fully() {
        let mut buf = vec![0; 8];
        let mut src = FileSource::new(&get_test_file("alltypes_plain.parquet"), 0, 4);

        let bytes_read = src.read(&mut buf[..]).unwrap();
        assert_eq!(bytes_read, 4);
        assert_eq!(buf, vec![b'P', b'A', b'R', b'1', 0, 0, 0, 0]);
    }

    #[test]
    fn test_io_read_in_chunks() {
        let mut buf = vec![0; 4];
        let mut src = FileSource::new(&get_test_file("alltypes_plain.parquet"), 0, 4);

        let bytes_read = src.read(&mut buf[0..2]).unwrap();
        assert_eq!(bytes_read, 2);
        let bytes_read = src.read(&mut buf[2..]).unwrap();
        assert_eq!(bytes_read, 2);
        assert_eq!(buf, vec![b'P', b'A', b'R', b'1']);
    }

    #[test]
    fn test_io_read_pos() {
        let mut src = FileSource::new(&get_test_file("alltypes_plain.parquet"), 0, 4);

        src.read(&mut vec![0; 1]).unwrap();
        assert_eq!(src.pos(), 1);

        src.read(&mut vec![0; 4]).unwrap();
        assert_eq!(src.pos(), 4);
    }

    #[test]
    fn test_io_read_over_limit() {
        let mut src = FileSource::new(&get_test_file("alltypes_plain.parquet"), 0, 4);

        // Read all bytes from source
        src.read(&mut vec![0; 128]).unwrap();
        assert_eq!(src.pos(), 4);

        // Try reading again, should return 0 bytes.
        let bytes_read = src.read(&mut vec![0; 128]).unwrap();
        assert_eq!(bytes_read, 0);
        assert_eq!(src.pos(), 4);
    }

    #[test]
    fn test_io_seek_switch() {
        let mut buf = vec![0; 4];
        let mut file = get_test_file("alltypes_plain.parquet");
        let mut src = FileSource::new(&file, 0, 4);

        file.seek(SeekFrom::Start(5 as u64))
            .expect("File seek to a position");

        let bytes_read = src.read(&mut buf[..]).unwrap();
        assert_eq!(bytes_read, 4);
        assert_eq!(buf, vec![b'P', b'A', b'R', b'1']);
    }

    #[test]
    fn test_io_write_with_pos() {
        let mut file = get_temp_file("file_sink_test", &[b'a', b'b', b'c']);
        file.seek(SeekFrom::Current(3)).unwrap();

        // Write into sink
        let mut sink = FileSink::new(&file);
        assert_eq!(sink.pos(), 3);

        sink.write(&[b'd', b'e', b'f', b'g']).unwrap();
        assert_eq!(sink.pos(), 7);

        sink.flush().unwrap();
        assert_eq!(sink.pos(), file.seek(SeekFrom::Current(0)).unwrap());

        // Read data using file chunk
        let mut res = vec![0u8; 7];
        let mut chunk =
            FileSource::new(&file, 0, file.metadata().unwrap().len() as usize);
        chunk.read(&mut res[..]).unwrap();
        assert_eq!(res, vec![b'a', b'b', b'c', b'd', b'e', b'f', b'g']);
    }

    #[test]
    fn test_io_large_read() {
        // Generate repeated 'abcdef' pattern and write it into a file
        let patterned_data: Vec<u8> = iter::repeat(vec![0, 1, 2, 3, 4, 5])
            .flatten()
            .take(3 * DEFAULT_BUF_SIZE)
            .collect();
        // always use different temp files as test might be run in parallel
        let mut file = get_temp_file("large_file_sink_test", &patterned_data);

        // seek the underlying file to the first 'd'
        file.seek(SeekFrom::Start(3)).unwrap();

        // create the FileSource reader that starts at pos 1 ('b')
        let mut chunk = FileSource::new(&file, 1, patterned_data.len() - 1);

        // read the 'b' at pos 1
        let mut res = vec![0u8; 1];
        chunk.read_exact(&mut res).unwrap();
        assert_eq!(res, &[1]);

        // the underlying file is sought to 'e'
        file.seek(SeekFrom::Start(4)).unwrap();

        // now read large chunk that starts with 'c' (after 'b')
        let mut res = vec![0u8; 2 * DEFAULT_BUF_SIZE];
        chunk.read_exact(&mut res).unwrap();
        assert_eq!(
            res,
            &patterned_data[2..2 + 2 * DEFAULT_BUF_SIZE],
            "read buf and original data are not equal"
        );
    }
}
