//! Minimal Avro object container reader, driven by the writer schema.
//!
//! Iceberg manifest lists and manifests are Avro files whose schema is embedded in the file
//! header. The decoders in [`crate::iceberg::manifest`] walk that writer schema, extracting the fields they
//! need and skipping the rest, so no generic value tree is built.
use std::io::Read;

use polars_io_ext_ffi::common::FfiError;
use polars_utils::aliases::PlHashMap;
use serde_json::Value as JsonValue;

use crate::iceberg::error::{IcebergResult, err_invalid_data, err_not_implemented};

const MAGIC: &[u8; 4] = b"Obj\x01";
const SYNC_LEN: usize = 16;

/// An Avro schema. Named types are resolved when parsing (Iceberg files never use recursive
/// types).
#[derive(Clone, Debug)]
pub enum Schema {
    Null,
    Boolean,
    Int,
    Long,
    Float,
    Double,
    Bytes,
    String,
    Record(Record),
    Enum,
    Array(Box<Schema>),
    Map(Box<Schema>),
    Union(Vec<Schema>),
    Fixed(usize),
}

#[derive(Clone, Debug)]
pub struct Record {
    pub fields: Vec<RecordField>,
}

#[derive(Clone, Debug)]
pub struct RecordField {
    pub name: String,
    pub field_id: Option<i32>,
    pub schema: Schema,
}

impl Schema {
    /// The non-null branch of a `[null, T]` union, or the schema itself.
    pub fn non_null(&self) -> &Schema {
        match self {
            Schema::Union(branches) => branches
                .iter()
                .find(|b| !matches!(b, Schema::Null))
                .or(branches.first())
                .unwrap_or(self),
            s => s,
        }
    }
}

/// A decoded Avro container file.
pub struct AvroFile {
    pub schema: Schema,
    /// Decompressed blocks, each with its object count.
    pub blocks: Vec<(usize, Vec<u8>)>,
}

impl AvroFile {
    pub fn parse(bytes: &[u8]) -> IcebergResult<Self> {
        let mut buf = bytes;

        if buf.len() < MAGIC.len() || &buf[..MAGIC.len()] != MAGIC {
            return Err(err_invalid_data("not an Avro object container file"));
        }
        buf = &buf[MAGIC.len()..];

        let mut metadata = PlHashMap::default();
        loop {
            let mut count = read_long(&mut buf)?;
            if count == 0 {
                break;
            }
            if count < 0 {
                count = count
                    .checked_neg()
                    .ok_or_else(|| err_invalid_data("invalid Avro map block count"))?;
                read_long(&mut buf)?;
            }
            check_item_count(count, buf)?;
            for _ in 0..count {
                let key = read_str(&mut buf)?.to_owned();
                let value = read_bytes(&mut buf)?.to_vec();
                metadata.insert(key, value);
            }
        }

        let sync = take(&mut buf, SYNC_LEN)?;

        let schema_json: JsonValue = serde_json::from_slice(
            metadata
                .get("avro.schema")
                .ok_or_else(|| err_invalid_data("Avro file without schema"))?,
        )
        .map_err(|e| err_invalid_data(format!("invalid Avro schema: {e}")))?;
        let schema = SchemaParser::default().parse(&schema_json, None)?;

        let codec = match metadata.get("avro.codec").map(|v| v.as_slice()) {
            None | Some(b"null") => Codec::Null,
            Some(b"deflate") => Codec::Deflate,
            Some(b"snappy") => Codec::Snappy,
            Some(b"zstandard") => Codec::Zstd,
            Some(other) => {
                return Err(err_not_implemented(format!(
                    "Avro codec '{}'",
                    String::from_utf8_lossy(other)
                )));
            },
        };

        let mut blocks = vec![];
        let mut budget = MAX_DECOMPRESSED_BYTES;
        while !buf.is_empty() {
            let count = read_long(&mut buf)?;
            let size = read_long(&mut buf)?;
            if count < 0 || size < 0 {
                return Err(err_invalid_data("negative Avro block count or size"));
            }
            let data = take(&mut buf, size as usize)?;
            let block_sync = take(&mut buf, SYNC_LEN)?;
            if block_sync != sync {
                return Err(err_invalid_data("Avro block sync marker mismatch"));
            }
            let data = codec.decompress(data, budget)?;
            budget -= data.len();
            // Objects take at least one byte in Iceberg files, which bounds the work done for a
            // corrupt count.
            if count as u64 > data.len() as u64 {
                return Err(err_invalid_data("Avro block count exceeds its size"));
            }
            blocks.push((count as usize, data));
        }

        Ok(Self { schema, blocks })
    }

    /// The top-level record schema.
    pub fn record(&self) -> IcebergResult<&Record> {
        match &self.schema {
            Schema::Record(r) => Ok(r),
            _ => Err(err_invalid_data("Avro file schema is not a record")),
        }
    }

    /// Call `f` for every object in the file, with the buffer positioned at its start.
    pub fn for_each_object(
        &self,
        mut f: impl FnMut(&mut &[u8]) -> IcebergResult<()>,
    ) -> IcebergResult<()> {
        for (count, data) in &self.blocks {
            let mut buf = data.as_slice();
            for _ in 0..*count {
                f(&mut buf)?;
            }
        }
        Ok(())
    }

    pub fn num_objects(&self) -> usize {
        self.blocks.iter().map(|(n, _)| n).sum()
    }

    /// [`Self::num_objects`] bounded by the decoded size, as a capacity hint that a corrupt
    /// block count cannot inflate.
    pub fn objects_capacity_hint(&self) -> usize {
        // Smaller than any manifest (list) entry.
        const MIN_OBJECT_BYTES: usize = 32;
        let bytes: usize = self.blocks.iter().map(|(_, data)| data.len()).sum();
        self.num_objects().min(bytes / MIN_OBJECT_BYTES)
    }
}

enum Codec {
    Null,
    Deflate,
    Snappy,
    Zstd,
}

/// Limit of the decompressed size of an Avro file. Larger (or malicious) files are reported as
/// unsupported, so that Polars falls back to PyIceberg.
pub const MAX_DECOMPRESSED_BYTES: usize = 2 << 30;

pub fn err_too_large(what: &str) -> FfiError {
    err_not_implemented(format!(
        "{what} larger than {MAX_DECOMPRESSED_BYTES} bytes when decompressed"
    ))
}

/// Read all of `reader`, up to `limit` bytes.
pub fn read_to_end_limited(
    reader: impl Read,
    capacity: usize,
    limit: usize,
    what: &str,
) -> IcebergResult<Vec<u8>> {
    let mut out = Vec::with_capacity(capacity.min(limit));
    reader
        .take(limit as u64 + 1)
        .read_to_end(&mut out)
        .map_err(|e| err_invalid_data(format!("{what} decompression: {e}")))?;
    if out.len() > limit {
        return Err(err_too_large(what));
    }
    Ok(out)
}

impl Codec {
    /// Decompress a block of at most `limit` bytes.
    fn decompress(&self, data: &[u8], limit: usize) -> IcebergResult<Vec<u8>> {
        const WHAT: &str = "Avro block";
        let err = |e: std::io::Error| err_invalid_data(format!("Avro block decompression: {e}"));
        match self {
            Codec::Null => {
                if data.len() > limit {
                    return Err(err_too_large(WHAT));
                }
                Ok(data.to_vec())
            },
            Codec::Deflate => read_to_end_limited(
                flate2::read::DeflateDecoder::new(data),
                data.len() * 4,
                limit,
                WHAT,
            ),
            Codec::Snappy => {
                // Snappy blocks are followed by the big-endian CRC32 of the uncompressed data.
                let Some((data, checksum)) = data.split_last_chunk::<4>() else {
                    return Err(err_invalid_data("truncated snappy Avro block"));
                };
                let len = snap::raw::decompress_len(data)
                    .map_err(|e| err_invalid_data(format!("Avro block decompression: {e}")))?;
                if len > limit {
                    return Err(err_too_large(WHAT));
                }
                let out = snap::raw::Decoder::new()
                    .decompress_vec(data)
                    .map_err(|e| err_invalid_data(format!("Avro block decompression: {e}")))?;
                let mut crc = flate2::Crc::new();
                crc.update(&out);
                if crc.sum() != u32::from_be_bytes(*checksum) {
                    return Err(err_invalid_data("snappy Avro block checksum mismatch"));
                }
                Ok(out)
            },
            Codec::Zstd => read_to_end_limited(
                zstd::stream::read::Decoder::new(data).map_err(err)?,
                data.len() * 4,
                limit,
                WHAT,
            ),
        }
    }
}

/// Limits of parsed schemas: a reference to a named type copies it, so that a small schema could
/// otherwise expand exponentially. Iceberg's schemas are far smaller.
const MAX_SCHEMA_NODES: usize = 1 << 20;
const MAX_SCHEMA_DEPTH: usize = 64;

#[derive(Default)]
struct SchemaParser {
    /// Named types, with their node counts.
    named: PlHashMap<String, (Schema, usize)>,
    /// Nodes parsed so far, counting copies of named types.
    nodes: usize,
}

impl SchemaParser {
    fn add_nodes(&mut self, n: usize) -> IcebergResult<()> {
        self.nodes += n;
        if self.nodes > MAX_SCHEMA_NODES {
            return Err(err_invalid_data("Avro schema too large"));
        }
        Ok(())
    }

    fn parse(&mut self, json: &JsonValue, namespace: Option<&str>) -> IcebergResult<Schema> {
        self.add_nodes(1)?;
        match json {
            JsonValue::String(name) => self.parse_named_or_primitive(name, namespace),
            JsonValue::Array(branches) if branches.is_empty() => {
                Err(err_invalid_data("empty Avro union"))
            },
            JsonValue::Array(branches) => Ok(Schema::Union(
                branches
                    .iter()
                    .map(|b| self.parse(b, namespace))
                    .collect::<IcebergResult<_>>()?,
            )),
            JsonValue::Object(obj) => {
                let ty = obj
                    .get("type")
                    .ok_or_else(|| err_invalid_data("Avro schema object without 'type'"))?;

                let JsonValue::String(ty) = ty else {
                    // e.g. {"type": {"type": "array", ...}}
                    return self.parse(ty, namespace);
                };

                let name_of = |obj: &serde_json::Map<String, JsonValue>| -> Option<String> {
                    let name = obj.get("name")?.as_str()?;
                    let ns = obj.get("namespace").and_then(|v| v.as_str()).or(namespace);
                    Some(match ns {
                        Some(ns) if !name.contains('.') && !ns.is_empty() => format!("{ns}.{name}"),
                        _ => name.to_owned(),
                    })
                };

                let schema = match ty.as_str() {
                    "record" | "error" => {
                        let nodes_before = self.nodes;
                        let fullname = name_of(obj);
                        let inner_ns = fullname
                            .as_deref()
                            .and_then(|n| n.rsplit_once('.').map(|(ns, _)| ns))
                            .map(str::to_owned);
                        let fields = obj
                            .get("fields")
                            .and_then(|v| v.as_array())
                            .ok_or_else(|| err_invalid_data("Avro record without fields"))?;
                        let fields = fields
                            .iter()
                            .map(|f| {
                                let name = f
                                    .get("name")
                                    .and_then(|v| v.as_str())
                                    .ok_or_else(|| err_invalid_data("Avro field without name"))?
                                    .to_owned();
                                let field_id =
                                    f.get("field-id").and_then(|v| v.as_i64()).map(|v| v as i32);
                                let schema = self.parse(
                                    f.get("type").ok_or_else(|| {
                                        err_invalid_data("Avro field without type")
                                    })?,
                                    inner_ns.as_deref().or(namespace),
                                )?;
                                Ok(RecordField {
                                    name,
                                    field_id,
                                    schema,
                                })
                            })
                            .collect::<IcebergResult<_>>()?;
                        let schema = Schema::Record(Record { fields });
                        if let Some(n) = fullname {
                            // Before it is copied by references.
                            if depth(&schema) > MAX_SCHEMA_DEPTH {
                                return Err(err_invalid_data("Avro schema too deeply nested"));
                            }
                            let nodes = self.nodes - nodes_before;
                            self.register(&n, schema.clone(), nodes);
                        }
                        schema
                    },
                    "enum" => {
                        let schema = Schema::Enum;
                        if let Some(name) = name_of(obj) {
                            self.register(&name, schema.clone(), 1);
                        }
                        schema
                    },
                    "array" => Schema::Array(Box::new(
                        self.parse(
                            obj.get("items")
                                .ok_or_else(|| err_invalid_data("Avro array without items"))?,
                            namespace,
                        )?,
                    )),
                    "map" => Schema::Map(Box::new(
                        self.parse(
                            obj.get("values")
                                .ok_or_else(|| err_invalid_data("Avro map without values"))?,
                            namespace,
                        )?,
                    )),
                    "fixed" => {
                        let size = obj
                            .get("size")
                            .and_then(|v| v.as_u64())
                            .ok_or_else(|| err_invalid_data("Avro fixed without size"))?;
                        let schema = Schema::Fixed(size as usize);
                        if let Some(name) = name_of(obj) {
                            self.register(&name, schema.clone(), 1);
                        }
                        schema
                    },
                    other => self.parse_named_or_primitive(other, namespace)?,
                };
                Ok(schema)
            },
            _ => Err(err_invalid_data(format!("invalid Avro schema: {json}"))),
        }
    }

    fn register(&mut self, fullname: &str, schema: Schema, nodes: usize) {
        if let Some((_, short)) = fullname.rsplit_once('.') {
            self.named
                .entry(short.to_owned())
                .or_insert_with(|| (schema.clone(), nodes));
        }
        self.named.insert(fullname.to_owned(), (schema, nodes));
    }

    fn parse_named_or_primitive(
        &mut self,
        name: &str,
        namespace: Option<&str>,
    ) -> IcebergResult<Schema> {
        Ok(match name {
            "null" => Schema::Null,
            "boolean" => Schema::Boolean,
            "int" => Schema::Int,
            "long" => Schema::Long,
            "float" => Schema::Float,
            "double" => Schema::Double,
            "bytes" => Schema::Bytes,
            "string" => Schema::String,
            name => {
                let qualified = namespace.map(|ns| format!("{ns}.{name}"));
                let (schema, nodes) = qualified
                    .and_then(|q| self.named.get(&q))
                    .or_else(|| self.named.get(name))
                    .ok_or_else(|| err_invalid_data(format!("unknown Avro type '{name}'")))?;
                let (schema, nodes) = (schema.clone(), *nodes);
                self.add_nodes(nodes)?;
                schema
            },
        })
    }
}

/// Nesting depth of a schema, without recursion.
fn depth(schema: &Schema) -> usize {
    let mut max = 0;
    let mut stack = vec![(schema, 1)];
    while let Some((schema, d)) = stack.pop() {
        max = max.max(d);
        match schema {
            Schema::Record(r) => stack.extend(r.fields.iter().map(|f| (&f.schema, d + 1))),
            Schema::Array(s) | Schema::Map(s) => stack.push((s, d + 1)),
            Schema::Union(branches) => stack.extend(branches.iter().map(|b| (b, d + 1))),
            _ => {},
        }
    }
    max
}

// Primitive decoding.

pub fn take<'a>(buf: &mut &'a [u8], n: usize) -> IcebergResult<&'a [u8]> {
    if buf.len() < n {
        return Err(err_invalid_data("unexpected end of Avro data"));
    }
    let (head, tail) = buf.split_at(n);
    *buf = tail;
    Ok(head)
}

pub fn read_long(buf: &mut &[u8]) -> IcebergResult<i64> {
    let mut value: u64 = 0;
    let mut shift = 0;
    loop {
        let Some((&byte, rest)) = buf.split_first() else {
            return Err(err_invalid_data("unexpected end of Avro data"));
        };
        *buf = rest;
        // The 10th byte holds bit 63 only.
        if shift == 63 && byte & 0x7E != 0 {
            return Err(err_invalid_data("invalid Avro varint"));
        }
        value |= u64::from(byte & 0x7F) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift > 63 {
            return Err(err_invalid_data("invalid Avro varint"));
        }
    }
    Ok(((value >> 1) as i64) ^ -((value & 1) as i64))
}

pub fn read_bytes<'a>(buf: &mut &'a [u8]) -> IcebergResult<&'a [u8]> {
    let len = read_long(buf)?;
    if len < 0 {
        return Err(err_invalid_data("negative Avro bytes length"));
    }
    take(buf, len as usize)
}

pub fn read_str<'a>(buf: &mut &'a [u8]) -> IcebergResult<&'a str> {
    std::str::from_utf8(read_bytes(buf)?).map_err(|_| err_invalid_data("invalid UTF-8 in Avro"))
}

/// Read the branch index of a union and return the selected branch.
pub fn read_union_branch<'s>(branches: &'s [Schema], buf: &mut &[u8]) -> IcebergResult<&'s Schema> {
    let idx = read_long(buf)?;
    branches
        .get(usize::try_from(idx).map_err(|_| err_invalid_data("negative Avro union index"))?)
        .ok_or_else(|| err_invalid_data("Avro union index out of range"))
}

/// Resolve unions, returning `None` for null values.
pub fn resolve<'s>(schema: &'s Schema, buf: &mut &[u8]) -> IcebergResult<Option<&'s Schema>> {
    match schema {
        Schema::Union(branches) => {
            let branch = read_union_branch(branches, buf)?;
            resolve(branch, buf)
        },
        Schema::Null => Ok(None),
        s => Ok(Some(s)),
    }
}

pub fn read_opt_long(schema: &Schema, buf: &mut &[u8]) -> IcebergResult<Option<i64>> {
    match resolve(schema, buf)? {
        None => Ok(None),
        Some(Schema::Int | Schema::Long) => read_long(buf).map(Some),
        Some(other) => {
            skip(other, buf)?;
            Err(err_invalid_data(format!(
                "expected an Avro int or long, found {other:?}"
            )))
        },
    }
}

pub fn read_opt_str<'a>(schema: &Schema, buf: &mut &'a [u8]) -> IcebergResult<Option<&'a str>> {
    match resolve(schema, buf)? {
        None => Ok(None),
        Some(Schema::String) => read_str(buf).map(Some),
        Some(Schema::Bytes) => std::str::from_utf8(read_bytes(buf)?)
            .map(Some)
            .map_err(|_| err_invalid_data("invalid UTF-8 in Avro")),
        Some(other) => Err(err_invalid_data(format!(
            "expected an Avro string, found {other:?}"
        ))),
    }
}

pub fn read_opt_bytes<'a>(schema: &Schema, buf: &mut &'a [u8]) -> IcebergResult<Option<&'a [u8]>> {
    match resolve(schema, buf)? {
        None => Ok(None),
        Some(Schema::String | Schema::Bytes) => read_bytes(buf).map(Some),
        Some(Schema::Fixed(n)) => take(buf, *n).map(Some),
        Some(other) => Err(err_invalid_data(format!(
            "expected Avro bytes, found {other:?}"
        ))),
    }
}

pub fn read_opt_bool(schema: &Schema, buf: &mut &[u8]) -> IcebergResult<Option<bool>> {
    match resolve(schema, buf)? {
        None => Ok(None),
        Some(Schema::Boolean) => Ok(Some(take(buf, 1)?[0] != 0)),
        Some(other) => Err(err_invalid_data(format!(
            "expected an Avro boolean, found {other:?}"
        ))),
    }
}

/// Iterate the items of an Avro array (or map) block sequence, calling `f` for each item.
pub fn for_each_item(
    buf: &mut &[u8],
    mut f: impl FnMut(&mut &[u8]) -> IcebergResult<()>,
) -> IcebergResult<()> {
    loop {
        let mut count = read_long(buf)?;
        if count == 0 {
            return Ok(());
        }
        if count < 0 {
            count = count
                .checked_neg()
                .ok_or_else(|| err_invalid_data("invalid Avro block count"))?;
            read_long(buf)?;
        }
        check_item_count(count, buf)?;
        for _ in 0..count {
            f(buf)?;
        }
    }
}

/// Array and map items take at least one byte in Iceberg files, which bounds the work done for
/// a corrupt count.
fn check_item_count(count: i64, buf: &[u8]) -> IcebergResult<()> {
    if count as u64 > buf.len() as u64 {
        return Err(err_invalid_data("Avro block count exceeds the data size"));
    }
    Ok(())
}

/// Skip a value of the given schema.
pub fn skip(schema: &Schema, buf: &mut &[u8]) -> IcebergResult<()> {
    match schema {
        Schema::Null => {},
        Schema::Boolean => {
            take(buf, 1)?;
        },
        Schema::Int | Schema::Long | Schema::Enum => {
            read_long(buf)?;
        },
        Schema::Float => {
            take(buf, 4)?;
        },
        Schema::Double => {
            take(buf, 8)?;
        },
        Schema::Bytes | Schema::String => {
            read_bytes(buf)?;
        },
        Schema::Fixed(n) => {
            take(buf, *n)?;
        },
        Schema::Record(r) => {
            for f in &r.fields {
                skip(&f.schema, buf)?;
            }
        },
        Schema::Array(items) => skip_blocks(buf, |buf| skip(items, buf))?,
        Schema::Map(values) => skip_blocks(buf, |buf| {
            read_bytes(buf)?;
            skip(values, buf)
        })?,
        Schema::Union(branches) => {
            let branch = read_union_branch(branches, buf)?;
            skip(branch, buf)?;
        },
    }
    Ok(())
}

fn skip_blocks(
    buf: &mut &[u8],
    mut f: impl FnMut(&mut &[u8]) -> IcebergResult<()>,
) -> IcebergResult<()> {
    loop {
        let count = read_long(buf)?;
        if count == 0 {
            return Ok(());
        }
        if count < 0 {
            // A negative count is followed by the block size in bytes, so it can be skipped
            // without decoding.
            let size = read_long(buf)?;
            take(
                buf,
                usize::try_from(size).map_err(|_| err_invalid_data("bad block size"))?,
            )?;
        } else {
            check_item_count(count, buf)?;
            for _ in 0..count {
                f(buf)?;
            }
        }
    }
}

/// A decoded primitive Avro value (used for partition tuples and partition summaries).
#[derive(Clone, Debug, PartialEq)]
pub enum Datum {
    Bool(bool),
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
    String(String),
    Bytes(Vec<u8>),
}

/// Decode a primitive value, or `None` for null.
pub fn read_datum(schema: &Schema, buf: &mut &[u8]) -> IcebergResult<Option<Datum>> {
    let Some(schema) = resolve(schema, buf)? else {
        return Ok(None);
    };
    Ok(Some(match schema {
        Schema::Boolean => Datum::Bool(take(buf, 1)?[0] != 0),
        Schema::Int => Datum::Int(read_long(buf)? as i32),
        Schema::Long => Datum::Long(read_long(buf)?),
        Schema::Float => Datum::Float(f32::from_le_bytes(take(buf, 4)?.try_into().unwrap())),
        Schema::Double => Datum::Double(f64::from_le_bytes(take(buf, 8)?.try_into().unwrap())),
        Schema::String => Datum::String(read_str(buf)?.to_owned()),
        Schema::Bytes => Datum::Bytes(read_bytes(buf)?.to_vec()),
        Schema::Fixed(n) => Datum::Bytes(take(buf, *n)?.to_vec()),
        other => {
            skip(other, buf)?;
            return Err(err_not_implemented(format!(
                "non-primitive Avro value {other:?}"
            )));
        },
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snappy_block(data: &[u8]) -> Vec<u8> {
        let mut block = snap::raw::Encoder::new().compress_vec(data).unwrap();
        let mut crc = flate2::Crc::new();
        crc.update(data);
        block.extend_from_slice(&crc.sum().to_be_bytes());
        block
    }

    #[test]
    fn overlong_varint_is_rejected() {
        // i64::MIN zigzag-encodes to u64::MAX: nine 0xFF bytes, then 0x01.
        let mut buf: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x01];
        assert_eq!(read_long(&mut buf).unwrap(), i64::MIN);

        // Bits beyond 64 in the 10th byte.
        let mut buf: &[u8] = &[0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0x03];
        assert!(read_long(&mut buf).is_err());
    }

    #[test]
    fn empty_union_is_rejected() {
        let json: JsonValue = serde_json::from_str("[]").unwrap();
        assert!(SchemaParser::default().parse(&json, None).is_err());
    }

    #[test]
    fn snappy_checksum_is_verified() {
        let data = b"some manifest entries some manifest entries";
        let block = snappy_block(data);
        assert_eq!(
            Codec::Snappy
                .decompress(&block, MAX_DECOMPRESSED_BYTES)
                .unwrap(),
            data
        );

        let mut corrupt = block.clone();
        *corrupt.last_mut().unwrap() ^= 1;
        let err = Codec::Snappy
            .decompress(&corrupt, MAX_DECOMPRESSED_BYTES)
            .unwrap_err();
        assert!(err.message().contains("checksum mismatch"), "{err:?}");

        assert!(
            Codec::Snappy
                .decompress(&block[..3], MAX_DECOMPRESSED_BYTES)
                .is_err()
        );
    }

    /// An Avro container file with one block.
    fn container(schema: &str, count: i64, data: &[u8]) -> Vec<u8> {
        fn long(out: &mut Vec<u8>, v: i64) {
            let mut z = ((v << 1) ^ (v >> 63)) as u64;
            while z >= 0x80 {
                out.push((z as u8) | 0x80);
                z >>= 7;
            }
            out.push(z as u8);
        }
        let mut out = MAGIC.to_vec();
        long(&mut out, 1);
        long(&mut out, "avro.schema".len() as i64);
        out.extend_from_slice(b"avro.schema");
        long(&mut out, schema.len() as i64);
        out.extend_from_slice(schema.as_bytes());
        long(&mut out, 0);
        out.extend_from_slice(&[7; SYNC_LEN]);
        long(&mut out, count);
        long(&mut out, data.len() as i64);
        out.extend_from_slice(data);
        out.extend_from_slice(&[7; SYNC_LEN]);
        out
    }

    #[test]
    fn counts_beyond_the_data_size_are_rejected() {
        // Objects that decode to no bytes.
        let schema = r#"{"type": "record", "name": "r", "fields": []}"#;
        assert!(AvroFile::parse(&container(schema, 100_000_000, &[0])).is_err());
        assert!(AvroFile::parse(&container(schema, 1, &[0])).is_ok());

        // An array of nulls.
        let items = Schema::Null;
        let mut buf: &[u8] = &[0xFE, 0xFF, 0xFF, 0xFF, 0x0F, 0x00];
        assert!(skip(&Schema::Array(Box::new(items)), &mut buf).is_err());
    }

    #[test]
    fn decompressed_size_is_limited() {
        let data = vec![0u8; 1 << 20];
        let compressed = zstd::stream::encode_all(data.as_slice(), 3).unwrap();
        assert_eq!(Codec::Zstd.decompress(&compressed, 1 << 20).unwrap(), data);
        let err = Codec::Zstd.decompress(&compressed, 1000).unwrap_err();
        assert!(err.message().contains("unsupported"), "{err:?}");

        let mut deflated = vec![];
        flate2::read::DeflateEncoder::new(data.as_slice(), flate2::Compression::fast())
            .read_to_end(&mut deflated)
            .unwrap();
        assert!(Codec::Deflate.decompress(&deflated, 1000).is_err());
        assert!(
            Codec::Snappy
                .decompress(&snappy_block(&data), 1000)
                .is_err()
        );
    }

    #[test]
    fn named_type_expansion_is_limited() {
        // `t{i}` has two fields of type `t{i-1}`: 2^40 nodes.
        let mut types = vec![r#"{"type": "record", "name": "t0", "fields": []}"#.to_string()];
        for i in 1..40 {
            types.push(format!(
                r#"{{"type": "record", "name": "t{i}", "fields": [{{"name": "a", "type": "t{p}"}}, {{"name": "b", "type": "t{p}"}}]}}"#,
                p = i - 1
            ));
        }
        let json: JsonValue = serde_json::from_str(&format!(
            r#"{{"type": "record", "name": "top", "fields": [{}]}}"#,
            types
                .iter()
                .enumerate()
                .map(|(i, t)| format!(r#"{{"name": "f{i}", "type": {t}}}"#))
                .collect::<Vec<_>>()
                .join(", ")
        ))
        .unwrap();
        let err = SchemaParser::default().parse(&json, None).unwrap_err();
        assert!(err.message().contains("too large"), "{err:?}");
    }

    #[test]
    fn named_type_depth_is_limited() {
        // `t{i}` has one field of type `t{i-1}`.
        let mut fields = vec![
            r#"{"name": "f0", "type": {"type": "record", "name": "t0", "fields": []}}"#.to_string(),
        ];
        for i in 1..100 {
            fields.push(format!(
                r#"{{"name": "f{i}", "type": {{"type": "record", "name": "t{i}", "fields": [{{"name": "a", "type": "t{p}"}}]}}}}"#,
                p = i - 1
            ));
        }
        let json: JsonValue = serde_json::from_str(&format!(
            r#"{{"type": "record", "name": "top", "fields": [{}]}}"#,
            fields.join(", ")
        ))
        .unwrap();
        let err = SchemaParser::default().parse(&json, None).unwrap_err();
        assert!(err.message().contains("deeply nested"), "{err:?}");
    }
}
