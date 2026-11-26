//! Per-file statistics: a scanner for the `stats` JSON string that extracts `numRecords` and
//! only the requested columns, and the conversion of checkpoint `stats_parsed` values.
use std::borrow::Cow;

use polars_arrow::array::*;
use polars_arrow::datatypes::{ArrowDataType, TimeUnit};

use super::values::Sv;

/// What one file's statistics hold for the requested columns, in request order.
#[derive(Debug, Default)]
pub struct FileStats<'a> {
    pub num_records: Option<u64>,
    pub tight_bounds: Option<bool>,
    pub min: Vec<Sv<'a>>,
    pub max: Vec<Sv<'a>>,
    pub null_count: Vec<Sv<'a>>,
}
impl<'a> FileStats<'a> {
    pub fn new(columns: usize) -> Self {
        Self {
            min: vec![Sv::Null; columns],
            max: vec![Sv::Null; columns],
            null_count: vec![Sv::Null; columns],
            ..Self::default()
        }
    }
}
// Bound stack use for both skipped values and materialized statistics.
const MAX_JSON_DEPTH: u8 = 128;

struct Scanner<'a> {
    text: &'a str,
    bytes: &'a [u8],
    pos: usize,
}
impl<'a> Scanner<'a> {
    fn ws(&mut self) {
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(b, b' ' | b'\n' | b'\r' | b'\t'))
        {
            self.pos += 1;
        }
    }
    fn eat(&mut self, byte: u8) -> Option<()> {
        self.ws();
        (self.bytes.get(self.pos) == Some(&byte)).then(|| self.pos += 1)
    }
    fn peek(&mut self) -> Option<u8> {
        self.ws();
        self.bytes.get(self.pos).copied()
    }
    /// A JSON string; borrowed unless it has escapes.
    fn string(&mut self) -> Option<Cow<'a, str>> {
        self.eat(b'"')?;
        let start = self.pos;
        let mut escaped = false;
        loop {
            match *self.bytes.get(self.pos)? {
                b'"' => break,
                b'\\' => {
                    escaped = true;
                    self.pos += 2
                },
                0..=0x1f => return None,
                _ => self.pos += 1,
            }
        }
        let raw = self.text.get(start..self.pos)?;
        self.pos += 1;
        if escaped {
            serde_json::from_str::<String>(&self.text[start - 1..self.pos])
                .ok()
                .map(Cow::Owned)
        } else {
            Some(Cow::Borrowed(raw))
        }
    }
    /// A number token that follows the JSON number grammar.
    fn number(&mut self) -> Option<&'a str> {
        let start = self.pos;
        while self
            .bytes
            .get(self.pos)
            .is_some_and(|b| matches!(b, b'-' | b'+' | b'.' | b'0'..=b'9' | b'e' | b'E'))
        {
            self.pos += 1;
        }
        let token = &self.text[start..self.pos];
        is_json_number(token).then_some(token)
    }
    fn literal(&mut self, word: &str) -> Option<()> {
        self.bytes
            .get(self.pos..)?
            .starts_with(word.as_bytes())
            .then(|| self.pos += word.len())
    }
    fn skip(&mut self, depth: u8) -> Option<()> {
        match self.peek()? {
            b'"' => {
                self.string()?;
            },
            start @ (b'{' | b'[') => {
                let depth = depth.checked_sub(1)?;
                let end = if start == b'{' { b'}' } else { b']' };
                self.pos += 1;
                if self.eat(end).is_some() {
                    return Some(());
                }
                loop {
                    if start == b'{' {
                        self.string()?;
                        self.eat(b':')?;
                    }
                    self.skip(depth)?;
                    if self.eat(end).is_some() {
                        break;
                    }
                    self.eat(b',')?;
                }
            },
            b't' => self.literal("true")?,
            b'f' => self.literal("false")?,
            b'n' => self.literal("null")?,
            _ => {
                self.number()?;
            },
        }
        Some(())
    }
    fn value(&mut self, depth: u8) -> Option<Sv<'a>> {
        Some(match self.peek()? {
            b'"' => Sv::Str(self.string()?),
            b'{' => {
                let depth = depth.checked_sub(1)?;
                self.pos += 1;
                let mut entries = Vec::new();
                if self.peek()? == b'}' {
                    self.pos += 1;
                    return Some(Sv::Obj(entries));
                }
                loop {
                    let key = self.string()?;
                    self.eat(b':')?;
                    entries.push((key, self.value(depth)?));
                    match self.peek()? {
                        b',' => self.pos += 1,
                        b'}' => {
                            self.pos += 1;
                            break;
                        },
                        _ => return None,
                    }
                }
                Sv::Obj(entries)
            },
            b'[' => {
                // Lists have no usable bounds.
                self.skip(depth)?;
                Sv::Null
            },
            b't' => {
                self.literal("true")?;
                Sv::Bool(true)
            },
            b'f' => {
                self.literal("false")?;
                Sv::Bool(false)
            },
            b'n' => {
                self.literal("null")?;
                Sv::Null
            },
            _ => Sv::Num(Cow::Borrowed(self.number()?)),
        })
    }
    /// An object of per-column statistics: keep the requested columns.
    fn columns(&mut self, wanted: &Wanted, out: &mut [Sv<'a>]) -> Option<()> {
        if self.peek()? != b'{' {
            // `null` or another type: no statistics of this kind.
            return self.skip(MAX_JSON_DEPTH - 1);
        }
        self.pos += 1;
        if self.peek()? == b'}' {
            self.pos += 1;
            return Some(());
        }
        loop {
            let key = self.string()?;
            self.eat(b':')?;
            match wanted.position(&key) {
                Some(i) => out[i] = self.value(MAX_JSON_DEPTH - 2)?,
                None => self.skip(MAX_JSON_DEPTH - 2)?,
            }
            match self.peek()? {
                b',' => self.pos += 1,
                b'}' => {
                    self.pos += 1;
                    return Some(());
                },
                _ => return None,
            }
        }
    }
}
/// The requested columns (physical names), indexed once for all files of a request.
pub struct Wanted(polars_utils::aliases::PlHashMap<Box<str>, usize>);
impl Wanted {
    pub fn new(names: &[&str]) -> Self {
        let mut index = polars_utils::aliases::PlHashMap::default();
        for (i, name) in names.iter().enumerate() {
            // The first of duplicate names wins.
            index.entry(Box::<str>::from(*name)).or_insert(i);
        }
        Self(index)
    }
    fn position(&self, name: &str) -> Option<usize> {
        self.0.get(name).copied()
    }
    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }
}
/// [`scan_with`] for a one-off list of columns.
#[cfg(test)]
pub fn scan<'a>(stats: &'a str, wanted: &[&str]) -> Option<FileStats<'a>> {
    scan_with(stats, wanted.len(), &Wanted::new(wanted))
}
/// Scan a `stats` string. `None` if it is not a JSON object (the file then has no statistics).
/// Skips everything but `numRecords`, `tightBounds` and the `wanted` columns, of which there
/// are `columns`.
pub fn scan_with<'a>(stats: &'a str, columns: usize, wanted: &Wanted) -> Option<FileStats<'a>> {
    let mut s = Scanner {
        text: stats,
        bytes: stats.as_bytes(),
        pos: 0,
    };
    let mut out = FileStats::new(columns);
    s.eat(b'{')?;
    if s.peek()? == b'}' {
        s.pos += 1;
        return s.peek().is_none().then_some(out);
    }
    loop {
        let key = s.string()?;
        s.eat(b':')?;
        match &*key {
            "numRecords" => {
                out.num_records = match s.value(MAX_JSON_DEPTH - 1)? {
                    Sv::Num(n) => n.parse().ok(),
                    _ => None,
                }
            },
            "tightBounds" => {
                out.tight_bounds = match s.value(MAX_JSON_DEPTH - 1)? {
                    Sv::Bool(b) => Some(b),
                    _ => None,
                }
            },
            "minValues" if !wanted.is_empty() => s.columns(wanted, &mut out.min)?,
            "maxValues" if !wanted.is_empty() => s.columns(wanted, &mut out.max)?,
            "nullCount" if !wanted.is_empty() => s.columns(wanted, &mut out.null_count)?,
            _ => s.skip(MAX_JSON_DEPTH - 1)?,
        }
        match s.peek()? {
            b',' => s.pos += 1,
            b'}' => {
                s.pos += 1;
                return s.peek().is_none().then_some(out);
            },
            _ => return None,
        }
    }
}
/// `-? (0 | [1-9][0-9]*) (. [0-9]+)? ([eE] [+-]? [0-9]+)?`, the JSON number grammar.
fn is_json_number(n: &str) -> bool {
    let b = n.as_bytes();
    let digits = |mut i: usize| {
        let start = i;
        while i < b.len() && b[i].is_ascii_digit() {
            i += 1;
        }
        (i, i > start)
    };
    let mut i = usize::from(b.first() == Some(&b'-'));
    match b.get(i) {
        Some(b'0') => i += 1,
        Some(b'1'..=b'9') => i = digits(i).0,
        _ => return false,
    }
    if b.get(i) == Some(&b'.') {
        let (j, ok) = digits(i + 1);
        if !ok {
            return false;
        }
        i = j;
    }
    if matches!(b.get(i), Some(b'e' | b'E')) {
        i += 1;
        if matches!(b.get(i), Some(b'+' | b'-')) {
            i += 1;
        }
        let (j, ok) = digits(i);
        if !ok {
            return false;
        }
        i = j;
    }
    i == b.len()
}
/// Delta timestamps are microseconds whatever the Parquet unit (millisecond, INT96, ...).
pub fn timestamp_micros(v: i64, unit: TimeUnit) -> Option<i64> {
    match unit {
        TimeUnit::Second => v.checked_mul(1_000_000),
        TimeUnit::Millisecond => v.checked_mul(1_000),
        TimeUnit::Microsecond => Some(v),
        TimeUnit::Nanosecond => Some(v.div_euclid(1_000)),
    }
}
/// Element `i` of a checkpoint statistics column. `None` for a type that holds no usable
/// bounds (binary, lists, ...): the caller then treats the column's statistics as unknown.
pub fn array_value(a: &dyn Array, i: usize) -> Option<Sv<'_>> {
    if a.is_null(i) {
        return Some(Sv::Null);
    }
    macro_rules! at {
        ($t:ty) => {
            a.as_any().downcast_ref::<PrimitiveArray<$t>>()?.value(i)
        };
    }
    Some(match a.dtype() {
        ArrowDataType::Null => Sv::Null,
        ArrowDataType::Boolean => Sv::Bool(a.as_any().downcast_ref::<BooleanArray>()?.value(i)),
        ArrowDataType::Int8 => Sv::Int(at!(i8).into()),
        ArrowDataType::Int16 => Sv::Int(at!(i16).into()),
        ArrowDataType::Int32 => Sv::Int(at!(i32).into()),
        ArrowDataType::Date32 => Sv::Date(at!(i32)),
        ArrowDataType::Int64 | ArrowDataType::Date64 => Sv::Int(at!(i64)),
        ArrowDataType::UInt8 => Sv::UInt(at!(u8).into()),
        ArrowDataType::UInt16 => Sv::UInt(at!(u16).into()),
        ArrowDataType::UInt32 => Sv::UInt(at!(u32).into()),
        ArrowDataType::UInt64 => Sv::UInt(at!(u64)),
        ArrowDataType::Float32 => finite(at!(f32).into()),
        ArrowDataType::Float64 => finite(at!(f64)),
        ArrowDataType::Timestamp(unit, _) => {
            timestamp_micros(at!(i64), *unit).map_or(Sv::Null, Sv::Int)
        },
        ArrowDataType::Decimal(_, scale) => Sv::Decimal(at!(i128), *scale),
        ArrowDataType::Utf8View => Sv::Str(Cow::Borrowed(
            a.as_any().downcast_ref::<Utf8ViewArray>()?.value(i),
        )),
        ArrowDataType::Utf8 => Sv::Str(Cow::Borrowed(
            a.as_any().downcast_ref::<Utf8Array<i32>>()?.value(i),
        )),
        ArrowDataType::LargeUtf8 => Sv::Str(Cow::Borrowed(
            a.as_any().downcast_ref::<Utf8Array<i64>>()?.value(i),
        )),
        ArrowDataType::Struct(fields) => {
            let a = a.as_any().downcast_ref::<StructArray>()?;
            let mut entries = Vec::with_capacity(fields.len());
            for (f, child) in fields.iter().zip(a.values()) {
                // A child of an unusable type leaves its own bounds unknown only.
                let v = array_value(child.as_ref(), i).unwrap_or(Sv::Null);
                entries.push((Cow::Borrowed(f.name.as_str()), v));
            }
            Sv::Obj(entries)
        },
        _ => return None,
    })
}
/// Non-finite floats have no JSON form and carry no bound.
fn finite<'a>(v: f64) -> Sv<'a> {
    if v.is_finite() {
        Sv::Float(v)
    } else {
        Sv::Null
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    fn num(v: &Sv) -> String {
        match v {
            Sv::Num(n) => n.to_string(),
            Sv::Str(n) => format!("s:{n}"),
            other => format!("{other:?}"),
        }
    }
    #[test]
    fn scan_keeps_exact_number_text_and_skips_the_rest() {
        let s = scan(
            r#"{"a\"1.5":"x 1.5","numRecords":3,"tightBounds":true,"minValues":{"skip":{"x":[1,"}"]},"d":12345678901234567.89,"e":-1.5E+3,"w":123456789012345678901234567890,"i":9007199254740993,"s":"a\"b"},"maxValues":{"d":null},"nullCount":{"d":0}}"#,
            &["d", "e", "w", "i", "s", "absent"],
        )
        .unwrap();
        assert_eq!(s.num_records, Some(3));
        assert_eq!(s.tight_bounds, Some(true));
        let m: Vec<_> = s.min.iter().map(num).collect();
        assert_eq!(
            m,
            [
                "12345678901234567.89",
                "-1.5E+3",
                "123456789012345678901234567890",
                "9007199254740993",
                "s:a\"b",
                "Null"
            ]
        );
        assert!(s.max[0].is_null());
        assert_eq!(num(&s.null_count[0]), "0");
        // Nothing requested: validate nested bounds without retaining them.
        let s = scan(r#"{"numRecords":7,"minValues":{"a":{"b":[{}]}}}"#, &[]).unwrap();
        assert_eq!(s.num_records, Some(7));
    }
    #[test]
    fn scan_rejects_invalid_text() {
        for n in [
            "-", "1e", "--5", "-.5", "01.5", "1.", "1e+", "+1", "1.5.2", "-e5",
        ] {
            assert!(
                scan(&format!(r#"{{"minValues":{{"a":{n}}}}}"#), &["a"]).is_none(),
                "{n}"
            );
        }
        for n in ["0", "-0", "0.5", "-0.5e-3", "1E5", "10"] {
            assert!(
                scan(&format!(r#"{{"minValues":{{"a":{n}}}}}"#), &["a"]).is_some(),
                "{n}"
            );
        }
        for text in ["", "[]", "{", r#"{"numRecords":"#, r#"{"a":1,}"#, "null"] {
            assert!(scan(text, &["a"]).is_none(), "{text}");
        }
        assert_eq!(scan("{}", &[]).unwrap().num_records, None);
        // A fractional or non-numeric count is unknown, not an error.
        assert_eq!(
            scan(r#"{"numRecords":5.0}"#, &[]).unwrap().num_records,
            None
        );
    }
    #[test]
    fn scan_requires_the_statistics_object_to_consume_the_input() {
        for bad in [
            r#"{"numRecords":5}garbage"#,
            r#"{"numRecords":5} {}"#,
            "{}[]",
            "{}\u{000C}",
        ] {
            assert!(scan(bad, &[]).is_none(), "{bad}");
        }
        assert_eq!(
            scan("{\"numRecords\":5} \t\r\n", &[]).unwrap().num_records,
            Some(5)
        );
        assert!(scan("{} \t\r\n", &[]).is_some());
    }
    #[test]
    fn scan_validates_skipped_json_and_borrowed_strings() {
        for ignored in [
            r#"[oops]"#,
            r#"{"key":}"#,
            r#"{"key":[null,]}"#,
            r#"{"key":[1,2}}"#,
            r#"{"key":1 "other":2}"#,
            r#"["\q"]"#,
            r#"["\uD800"]"#,
            "\"raw\nnewline\"",
            "{\"raw\u{0001}control\":0}",
        ] {
            let stats = format!(r#"{{"numRecords":5,"ignored":{ignored}}}"#);
            assert!(scan(&stats, &[]).is_none(), "{stats:?}");
        }
        assert!(scan("{\"numRecords\":5,\"bad\u{0000}key\":0}", &[]).is_none());
        let stats = scan(
            r#"{"numRecords":5,"ignored":[{"key": [true, {}, [], null, "\uD83D\uDCA1", 1e9999]}]}"#,
            &[],
        )
        .unwrap();
        assert_eq!(stats.num_records, Some(5));
    }

    #[test]
    fn scan_bounds_nesting_for_skipped_and_requested_data() {
        let depth = 10_000;
        let malformed = format!(r#"{{"numRecords":5,"ignored":{}oops"#, "[".repeat(depth));
        assert!(scan(&malformed, &[]).is_none());
        let level = r#"{"child":"#;
        let prefix = r#"{"numRecords":5,"minValues":{"wanted":"#;
        let too_deep = format!("{prefix}{}0{}", level.repeat(depth), "}".repeat(depth + 2));
        assert!(scan(&too_deep, &["wanted"]).is_none());
    }

    #[test]
    fn nested_structs_are_objects() {
        let s = scan(r#"{"minValues":{"p":{"q":1,"r":{"z":"x"}}}}"#, &["p"]).unwrap();
        assert_eq!(num(s.min[0].get("q")), "1");
        assert_eq!(num(s.min[0].get("r").get("z")), "s:x");
    }
}
