//! APIs to read from Avro format to arrow.
use std::io::Read;

use avro_schema::file::FileMetadata;
use avro_schema::read::fallible_streaming_iterator::FallibleStreamingIterator;
use avro_schema::read::{BlockStreamingIterator, block_iterator};
use avro_schema::schema::Field as AvroField;

mod deserialize;
pub use deserialize::deserialize;
use polars_error::{PolarsError, PolarsResult, polars_bail, polars_err};

mod nested;
mod schema;
mod util;

pub use schema::infer_schema;

use crate::array::Array;
use crate::datatypes::ArrowSchema;
use crate::record_batch::RecordBatchT;

/// Reads the metadata of an Avro file, like [`avro_schema::read::read_metadata`], but
/// errors on codecs that can't be decompressed.
///
/// `avro_schema` treats any codec other than `deflate` and `snappy` as uncompressed, which
/// would decode compressed blocks as raw records.
pub fn read_metadata<R: Read>(reader: &mut R) -> PolarsResult<FileMetadata> {
    let mut header = RecordingReader {
        reader,
        bytes: Vec::new(),
    };
    // Like `avro_schema`, treat a header cut short as an invalid file, not an IO error.
    let codec = read_codec(&mut header).map_err(|err| match err {
        PolarsError::IO { error, .. } if error.kind() == std::io::ErrorKind::UnexpectedEof => {
            polars_err!(oos = "Avro header ends early")
        },
        err => err,
    })?;
    if let Some(codec) = codec
        && !matches!(codec.as_slice(), b"null" | b"deflate" | b"snappy")
    {
        polars_bail!(
            ComputeError: "unsupported Avro codec {:?}; supported codecs are null, deflate and snappy",
            String::from_utf8_lossy(&codec)
        );
    }
    Ok(avro_schema::read::read_metadata(
        &mut header.bytes.as_slice(),
    )?)
}

/// Reads the magic, header and marker of an Avro file and returns its `avro.codec`.
fn read_codec<R: Read>(reader: &mut R) -> PolarsResult<Option<Vec<u8>>> {
    let mut magic = [0u8; 4];
    reader.read_exact(&mut magic)?;
    if magic != [b'O', b'b', b'j', 1u8] {
        polars_bail!(oos = "Avro file does not start with the Avro magic bytes");
    }

    let mut codec = None;
    loop {
        let len = util::zigzag_i64(reader)?;
        if len == 0 {
            break;
        }
        // `avro_schema` doesn't support header blocks with a byte size (negative counts).
        if len < 0 {
            polars_bail!(oos = "Avro header block with a negative count is not supported");
        }
        for _ in 0..len {
            let key = read_bytes(reader)?;
            let value = read_bytes(reader)?;
            if key == b"avro.codec" {
                codec = Some(value);
            }
        }
    }

    let mut marker = [0u8; 16];
    reader.read_exact(&mut marker)?;
    Ok(codec)
}

fn read_bytes<R: Read>(reader: &mut R) -> PolarsResult<Vec<u8>> {
    let len = util::zigzag_i64(reader)?;
    if len < 0 {
        polars_bail!(oos = "Avro header contains a negative length");
    }
    let mut bytes = Vec::new();
    reader.take(len as u64).read_to_end(&mut bytes)?;
    if bytes.len() != len as usize {
        polars_bail!(oos = "Avro header ends early");
    }
    Ok(bytes)
}

/// Keeps a copy of everything read through it.
struct RecordingReader<'a, R> {
    reader: &'a mut R,
    bytes: Vec<u8>,
}

impl<R: Read> Read for RecordingReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = self.reader.read(buf)?;
        self.bytes.extend_from_slice(&buf[..n]);
        Ok(n)
    }
}

/// Single threaded, blocking reader of Avro; [`Iterator`] of [`RecordBatchT`].
pub struct Reader<R: Read> {
    iter: BlockStreamingIterator<R>,
    avro_fields: Vec<AvroField>,
    fields: ArrowSchema,
    projection: Vec<bool>,
}

impl<R: Read> Reader<R> {
    /// Creates a new [`Reader`].
    pub fn new(
        reader: R,
        metadata: FileMetadata,
        fields: ArrowSchema,
        projection: Option<Vec<bool>>,
    ) -> Self {
        let projection = projection.unwrap_or_else(|| fields.iter().map(|_| true).collect());

        Self {
            iter: block_iterator(reader, metadata.compression, metadata.marker),
            avro_fields: metadata.record.fields,
            fields,
            projection,
        }
    }

    /// Deconstructs itself into its internal reader
    pub fn into_inner(self) -> R {
        self.iter.into_inner()
    }
}

impl<R: Read> Iterator for Reader<R> {
    type Item = PolarsResult<RecordBatchT<Box<dyn Array>>>;

    fn next(&mut self) -> Option<Self::Item> {
        let fields = &self.fields;
        let avro_fields = &self.avro_fields;
        let projection = &self.projection;

        self.iter
            .next()
            .transpose()
            .map(|maybe_block| deserialize(maybe_block?, fields, avro_fields, projection))
    }
}
