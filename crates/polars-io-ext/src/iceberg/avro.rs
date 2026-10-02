//! Minimal Avro object container reader, driven by the writer schema.
//!
//! Iceberg manifest lists and manifests are Avro files whose schema is embedded in the file
//! header. The decoders in [`crate::iceberg::manifest`] walk that writer schema, extracting the fields they
//! need and skipping the rest, so no generic value tree is built.
use std::io::Read;

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
                .unwrap_or(&branches[0]),
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
                count = -count;
                read_long(&mut buf)?;
            }
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
            blocks.push((count as usize, codec.decompress(data)?));
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
}

enum Codec {
    Null,
    Deflate,
    Snappy,
    Zstd,
}

impl Codec {
    fn decompress(&self, data: &[u8]) -> IcebergResult<Vec<u8>> {
        let err = |e: std::io::Error| err_invalid_data(format!("Avro block decompression: {e}"));
        match self {
            Codec::Null => Ok(data.to_vec()),
            Codec::Deflate => {
                let mut out = Vec::with_capacity(data.len() * 4);
                flate2::read::DeflateDecoder::new(data)
                    .read_to_end(&mut out)
                    .map_err(err)?;
                Ok(out)
            },
            Codec::Snappy => {
                // Snappy blocks are followed by a 4-byte CRC32 of the uncompressed data.
                let data = data
                    .get(..data.len().saturating_sub(4))
                    .ok_or_else(|| err_invalid_data("truncated snappy Avro block"))?;
                snap::raw::Decoder::new()
                    .decompress_vec(data)
                    .map_err(|e| err_invalid_data(format!("Avro block decompression: {e}")))
            },
            Codec::Zstd => zstd::stream::decode_all(data).map_err(err),
        }
    }
}

#[derive(Default)]
struct SchemaParser {
    named: PlHashMap<String, Schema>,
}

impl SchemaParser {
    fn parse(&mut self, json: &JsonValue, namespace: Option<&str>) -> IcebergResult<Schema> {
        match json {
            JsonValue::String(name) => self.parse_named_or_primitive(name, namespace),
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
                            self.register(&n, schema.clone());
                        }
                        schema
                    },
                    "enum" => {
                        let schema = Schema::Enum;
                        if let Some(name) = name_of(obj) {
                            self.register(&name, schema.clone());
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
                            self.register(&name, schema.clone());
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

    fn register(&mut self, fullname: &str, schema: Schema) {
        if let Some((_, short)) = fullname.rsplit_once('.') {
            self.named
                .entry(short.to_owned())
                .or_insert_with(|| schema.clone());
        }
        self.named.insert(fullname.to_owned(), schema);
    }

    fn parse_named_or_primitive(
        &self,
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
                qualified
                    .and_then(|q| self.named.get(&q))
                    .or_else(|| self.named.get(name))
                    .cloned()
                    .ok_or_else(|| err_invalid_data(format!("unknown Avro type '{name}'")))?
            },
        })
    }
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
            count = -count;
            read_long(buf)?;
        }
        for _ in 0..count {
            f(buf)?;
        }
    }
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

    fn zigzag(v: i64) -> Vec<u8> {
        let mut n = ((v << 1) ^ (v >> 63)) as u64;
        let mut out = vec![];
        loop {
            let b = (n & 0x7F) as u8;
            n >>= 7;
            if n == 0 {
                out.push(b);
                return out;
            }
            out.push(b | 0x80);
        }
    }

    #[test]
    fn test_read_long() {
        for v in [0, 1, -1, 63, -64, 64, 300, i64::MAX, i64::MIN, 1 << 40] {
            let bytes = zigzag(v);
            let mut buf = bytes.as_slice();
            assert_eq!(read_long(&mut buf).unwrap(), v);
            assert!(buf.is_empty());
        }
    }

    #[test]
    fn test_parse_named_reference() {
        let json: JsonValue = serde_json::from_str(
            r#"{"type": "record", "name": "a", "fields": [
                {"name": "x", "type": {"type": "fixed", "name": "f16", "size": 16}},
                {"name": "y", "type": ["null", "f16"]}
            ]}"#,
        )
        .unwrap();
        let schema = SchemaParser::default().parse(&json, None).unwrap();
        let Schema::Record(r) = schema else { panic!() };
        assert!(matches!(r.fields[1].schema.non_null(), Schema::Fixed(16)));
    }
}
