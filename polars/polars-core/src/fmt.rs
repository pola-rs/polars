use crate::prelude::*;

#[cfg(feature = "temporal")]
use crate::chunked_array::temporal::{
    date32_as_datetime, date64_as_datetime, time64_nanosecond_as_time,
};
use comfy_table::modifiers::UTF8_ROUND_CORNERS;
use comfy_table::presets::UTF8_FULL;
use comfy_table::*;
use num::{Num, NumCast};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};
const LIMIT: usize = 10;

/// Some unit functions that just pass the integer values if we don't want all chrono functionality
#[cfg(not(feature = "temporal"))]
mod temporal {
    pub struct DateTime<T>(T)
    where
        T: Copy;

    impl<T> DateTime<T>
    where
        T: Copy,
    {
        pub fn date(&self) -> T {
            self.0
        }
    }

    pub fn date32_as_datetime(v: i32) -> DateTime<i32> {
        DateTime(v)
    }
    pub fn date64_as_datetime(v: i64) -> DateTime<i64> {
        DateTime(v)
    }
    pub fn time32_millisecond_as_time(v: i32) -> i32 {
        v
    }
    pub fn time32_second_as_time(v: i32) -> i32 {
        v
    }
    pub fn time64_nanosecond_as_time(v: i64) -> i64 {
        v
    }
    pub fn time64_microsecond_as_time(v: i64) -> i64 {
        v
    }
    pub fn timestamp_nanoseconds_as_datetime(v: i64) -> i64 {
        v
    }
    pub fn timestamp_microseconds_as_datetime(v: i64) -> i64 {
        v
    }
    pub fn timestamp_milliseconds_as_datetime(v: i64) -> i64 {
        v
    }
    pub fn timestamp_seconds_as_datetime(v: i64) -> i64 {
        v
    }
}
#[cfg(not(feature = "temporal"))]
use temporal::*;

macro_rules! format_array {
    ($limit:ident, $f:ident, $a:expr, $dtype:expr, $name:expr, $array_type:expr) => {{
        write![$f, "{}: '{}' [{}]\n[\n", $array_type, $name, $dtype]?;

        for i in 0..$limit {
            let v = $a.get_any_value(i);
            write!($f, "\t{}\n", v)?;
        }

        write![$f, "]"]
    }};
}

macro_rules! format_utf8_array {
    ($limit:expr, $f:expr, $a:expr, $name:expr, $array_type:expr) => {{
        write![$f, "{}: '{}' [str]\n[\n", $array_type, $name]?;
        $a.into_iter().take($limit).for_each(|opt_s| match opt_s {
            None => {
                write!($f, "\tnull\n").ok();
            }
            Some(s) => {
                if s.len() >= $limit {
                    write!($f, "\t\"{}...\"\n", &s[..$limit]).ok();
                } else {
                    write!($f, "\t\"{}\"\n", &s).ok();
                }
            }
        });
        write![$f, "]"]
    }};
}
macro_rules! format_list_array {
    ($limit:ident, $f:ident, $a:expr, $name:expr, $array_type:expr) => {{
        write![$f, "{}: '{}' [list]\n[\n", $array_type, $name]?;

        for i in 0..$limit {
            let opt_v = $a.get(i);
            match opt_v {
                Some(v) => write!($f, "\t{}\n", v.fmt_list())?,
                None => write!($f, "\tnull\n")?,
            }
        }

        write![$f, "]"]
    }};
}

#[cfg(feature = "object")]
fn format_object_array(
    limit: usize,
    f: &mut Formatter<'_>,
    object: &dyn SeriesTrait,
    name: &str,
    array_type: &str,
) -> fmt::Result {
    write![f, "{}: '{}' [object]\n[\n", array_type, name]?;

    for i in 0..limit {
        let v = object.get(i);
        match v {
            AnyValue::Null => writeln!(f, "\tnull")?,
            _ => writeln!(f, "\tobject")?,
        }
    }

    write![f, "]"]
}

#[cfg(feature = "object")]
macro_rules! format_object_array {
    ($limit:expr, $f:expr, $object:expr, $name:expr, $array_type: expr) => {{
        write![$f, "{}: '{}' [object]\n[\n", $array_type, $name]?;

        for i in 0..$limit {
            let v = $object.get_any_value(i);
            match v {
                AnyValue::Null => writeln!($f, "\tnull")?,
                _ => writeln!($f, "\tobject")?,
            }
        }

        write![$f, "]"]
    }};
}

macro_rules! set_limit {
    ($self:ident) => {
        std::cmp::min($self.len(), LIMIT)
    };
}

impl<T> Debug for ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);
        let dtype = format!("{:?}", T::get_dtype());
        format_array!(limit, f, self, dtype, self.name(), "ChunkedArray")
    }
}

impl Debug for ChunkedArray<BooleanType> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);
        let dtype = format!("{:?}", DataType::Boolean);
        format_array!(limit, f, self, dtype, self.name(), "ChunkedArray")
    }
}

impl Debug for Utf8Chunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        format_utf8_array!(80, f, self, self.name(), "ChunkedArray")
    }
}

impl Debug for ListChunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);
        format_list_array!(limit, f, self, self.name(), "ChunkedArray")
    }
}

impl Debug for CategoricalChunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);
        let dtype = format!("{:?}", DataType::Categorical);
        format_array!(limit, f, self, dtype, self.name(), "ChunkedArray")
    }
}

#[cfg(feature = "object")]
impl<T> Debug for ObjectChunked<T>
where
    T: 'static + Debug + Clone + Send + Sync + Default,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);
        format_object_array!(limit, f, self, self.name(), "ChunkedArray")
    }
}

impl Debug for Series {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);

        match self.dtype() {
            DataType::Boolean => format_array!(
                limit,
                f,
                self.bool().unwrap(),
                "bool",
                self.name(),
                "Series"
            ),
            DataType::Utf8 => {
                format_utf8_array!(limit, f, self.utf8().unwrap(), self.name(), "Series")
            }
            DataType::UInt8 => {
                format_array!(limit, f, self.u8().unwrap(), "u8", self.name(), "Series")
            }
            DataType::UInt16 => {
                format_array!(limit, f, self.u16().unwrap(), "u6", self.name(), "Series")
            }
            DataType::UInt32 => {
                format_array!(limit, f, self.u32().unwrap(), "u32", self.name(), "Series")
            }
            DataType::UInt64 => {
                format_array!(limit, f, self.u64().unwrap(), "u64", self.name(), "Series")
            }
            DataType::Int8 => {
                format_array!(limit, f, self.i8().unwrap(), "i8", self.name(), "Series")
            }
            DataType::Int16 => {
                format_array!(limit, f, self.i16().unwrap(), "i16", self.name(), "Series")
            }
            DataType::Int32 => {
                format_array!(limit, f, self.i32().unwrap(), "i32", self.name(), "Series")
            }
            DataType::Int64 => {
                format_array!(limit, f, self.i64().unwrap(), "i64", self.name(), "Series")
            }
            DataType::Float32 => {
                format_array!(limit, f, self.f32().unwrap(), "f32", self.name(), "Series")
            }
            DataType::Float64 => {
                format_array!(limit, f, self.f64().unwrap(), "f64", self.name(), "Series")
            }
            DataType::Date32 => format_array!(
                limit,
                f,
                self.date32().unwrap(),
                "date32",
                self.name(),
                "Series"
            ),
            DataType::Date64 => format_array!(
                limit,
                f,
                self.date64().unwrap(),
                "date64",
                self.name(),
                "Series"
            ),
            DataType::Time64(TimeUnit::Nanosecond) => format_array!(
                limit,
                f,
                self.time64_nanosecond().unwrap(),
                "time64(ns)",
                self.name(),
                "Series"
            ),
            DataType::Duration(TimeUnit::Nanosecond) => format_array!(
                limit,
                f,
                self.duration_nanosecond().unwrap(),
                "duration(ns)",
                self.name(),
                "Series"
            ),
            DataType::Duration(TimeUnit::Millisecond) => format_array!(
                limit,
                f,
                self.duration_millisecond().unwrap(),
                "duration(ms)",
                self.name(),
                "Series"
            ),
            DataType::List(_) => {
                format_list_array!(limit, f, self.list().unwrap(), self.name(), "Series")
            }
            #[cfg(feature = "object")]
            DataType::Object => format_object_array(limit, f, self.as_ref(), self.name(), "Series"),
            DataType::Categorical => format_array!(
                limit,
                f,
                self.categorical().unwrap(),
                "cat",
                self.name(),
                "Series"
            ),
            _ => unimplemented!(),
        }
    }
}

impl Display for Series {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Debug::fmt(self, f)
    }
}

impl Debug for DataFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Display::fmt(self, f)
    }
}

fn prepare_row(row: Vec<AnyValue>, n_first: usize, n_last: usize) -> Vec<String> {
    fn make_str_val(v: &AnyValue) -> String {
        let string_limit = 32;
        if let AnyValue::Utf8(s) = v {
            if s.len() > string_limit {
                format!("{}...", &s[..string_limit])
            } else {
                s.to_string()
            }
        } else {
            format!("{}", v)
        }
    }

    let reduce_columns = n_first + n_last < row.len();
    let mut row_str = Vec::with_capacity(n_first + n_last + reduce_columns as usize);
    for v in row[0..n_first].iter() {
        row_str.push(make_str_val(v));
    }
    if reduce_columns {
        row_str.push("...".to_string());
    }
    for v in row[row.len() - n_last..].iter() {
        row_str.push(make_str_val(v));
    }
    row_str
}

impl Display for DataFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let max_n_cols = std::env::var("POLARS_FMT_MAX_COLS")
            .unwrap_or_else(|_| "8".to_string())
            .parse()
            .unwrap_or(8);
        let max_n_rows = std::env::var("POLARS_FMT_MAX_ROWS")
            .unwrap_or_else(|_| "8".to_string())
            .parse()
            .unwrap_or(8);

        let (n_first, n_last) = if self.width() > max_n_cols {
            ((max_n_cols + 1) / 2, max_n_cols / 2)
        } else {
            (self.width(), 0)
        };
        let reduce_columns = n_first + n_last < self.width();

        let field_to_str = |f: &Field| format!("{}\n---\n{}", f.name(), f.data_type());

        let mut names = Vec::with_capacity(n_first + n_last + reduce_columns as usize);
        let schema = self.schema();
        let fields = schema.fields();
        for field in fields[0..n_first].iter() {
            names.push(field_to_str(field))
        }
        if reduce_columns {
            names.push("...".to_string())
        }
        for field in fields[self.width() - n_last..].iter() {
            names.push(field_to_str(field))
        }
        let mut table = Table::new();
        table
            .load_preset(UTF8_FULL)
            .set_content_arrangement(ContentArrangement::Dynamic)
            .apply_modifier(UTF8_ROUND_CORNERS)
            .set_table_width(
                std::env::var("POLARS_TABLE_WIDTH")
                    .map(|s| {
                        s.parse::<u16>()
                            .expect("could not parse table width argument")
                    })
                    .unwrap_or(100),
            )
            .set_header(names);
        let mut rows = Vec::with_capacity(max_n_rows);
        if self.height() > max_n_rows {
            for i in 0..(max_n_rows / 2) {
                let row = self.get(i).unwrap();
                rows.push(prepare_row(row, n_first, n_last));
            }
            let dots = rows[0].iter().map(|_| "...".to_string()).collect();
            rows.push(dots);
            for i in (self.height() - max_n_cols / 2 - 1)..self.height() {
                let row = self.get(i).unwrap();
                rows.push(prepare_row(row, n_first, n_last));
            }
            for row in rows {
                table.add_row(row);
            }
        } else {
            for i in 0..max_n_rows {
                let opt = self.get(i);
                if let Some(row) = opt {
                    table.add_row(prepare_row(row, n_first, n_last));
                } else {
                    break;
                }
            }
        }

        write!(f, "shape: {:?}\n{}", self.shape(), table)?;
        Ok(())
    }
}

fn fmt_integer<T: Num + NumCast + Display>(
    f: &mut Formatter<'_>,
    width: usize,
    v: T,
) -> fmt::Result {
    write!(f, "{:>width$}", v, width = width)
}

fn fmt_float<T: Num + NumCast>(f: &mut Formatter<'_>, width: usize, v: T) -> fmt::Result {
    let v: f64 = NumCast::from(v).unwrap();
    let v = (v * 1000.).round() / 1000.;
    if v == 0.0 {
        write!(f, "{:>width$.1}", v, width = width)
    } else if !(0.0001..=9999.).contains(&v) {
        write!(f, "{:>width$e}", v, width = width)
    } else {
        write!(f, "{:>width$}", v, width = width)
    }
}

impl Display for AnyValue<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 0;
        match self {
            AnyValue::Null => write!(f, "null"),
            AnyValue::UInt8(v) => write!(f, "{}", v),
            AnyValue::UInt16(v) => write!(f, "{}", v),
            AnyValue::UInt32(v) => write!(f, "{}", v),
            AnyValue::UInt64(v) => write!(f, "{}", v),
            AnyValue::Int8(v) => fmt_integer(f, width, *v),
            AnyValue::Int16(v) => fmt_integer(f, width, *v),
            AnyValue::Int32(v) => fmt_integer(f, width, *v),
            AnyValue::Int64(v) => fmt_integer(f, width, *v),
            AnyValue::Float32(v) => fmt_float(f, width, *v),
            AnyValue::Float64(v) => fmt_float(f, width, *v),
            AnyValue::Boolean(v) => write!(f, "{}", *v),
            AnyValue::Utf8(v) => write!(f, "{}", format!("\"{}\"", v)),
            AnyValue::Date32(v) => write!(f, "{}", date32_as_datetime(*v).date()),
            #[cfg(feature = "temporal")]
            AnyValue::Date64(v) => write!(f, "{}", date64_as_datetime(*v)),
            AnyValue::Time64(v, TimeUnit::Nanosecond) => {
                write!(f, "{}", time64_nanosecond_as_time(*v))
            }
            AnyValue::Duration(v, TimeUnit::Nanosecond) => write!(f, "{}", v),
            AnyValue::Duration(v, TimeUnit::Millisecond) => write!(f, "{}", v),
            AnyValue::List(s) => write!(f, "{:?}", s.fmt_list()),
            #[cfg(feature = "object")]
            AnyValue::Object(_) => write!(f, "object"),
            _ => unimplemented!(),
        }
    }
}

macro_rules! fmt_option {
    ($opt:expr) => {{
        match $opt {
            Some(v) => format!("{:?}", v),
            None => "null".to_string(),
        }
    }};
}

macro_rules! impl_fmt_list {
    ($self:ident) => {{
        match $self.len() {
            0 => format!("[]"),
            1 => format!("[{}]", fmt_option!($self.get(0))),
            2 => format!(
                "[{}, {}]",
                fmt_option!($self.get(0)),
                fmt_option!($self.get(1))
            ),
            3 => format!(
                "[{}, {}, {}]",
                fmt_option!($self.get(0)),
                fmt_option!($self.get(1)),
                fmt_option!($self.get(2))
            ),
            _ => format!(
                "[{}, {}, ... {}]",
                fmt_option!($self.get(0)),
                fmt_option!($self.get(1)),
                fmt_option!($self.get($self.len() - 1))
            ),
        }
    }};
}

pub(crate) trait FmtList {
    fn fmt_list(&self) -> String;
}

impl<T> FmtList for ChunkedArray<T>
where
    T: PolarsPrimitiveType,
{
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

impl FmtList for BooleanChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

impl FmtList for Utf8Chunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

impl FmtList for ListChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

impl FmtList for CategoricalChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "object")]
impl<T> FmtList for ObjectChunked<T> {
    fn fmt_list(&self) -> String {
        todo!()
    }
}

#[cfg(all(test, feature = "temporal"))]
mod test {
    use crate::prelude::*;
    use polars_arrow::prelude::PrimitiveArrayBuilder;

    #[test]
    fn list() {
        let values_builder = PrimitiveArrayBuilder::<UInt32Type>::new(10);
        let mut builder = ListPrimitiveChunkedBuilder::new("a", values_builder, 10);
        builder.append_slice(Some(&[1, 2, 3]));
        builder.append_slice(None);
        let list = builder.finish().into_series();

        println!("{:?}", list);
        assert_eq!(
            r#"Series: 'a' [list]
[
	[1, 2, 3]
	null
]"#,
            format!("{:?}", list)
        );
    }

    #[test]
    fn temporal() {
        let s = Date32Chunked::new_from_opt_slice("date32", &[Some(1), None, Some(3)]);
        assert_eq!(
            r#"Series: 'date32' [date32]
[
	1970-01-02
	null
	1970-01-04
]"#,
            format!("{:?}", s.into_series())
        );

        let s = Date64Chunked::new_from_opt_slice("", &[Some(1), None, Some(1_000_000_000_000)]);
        assert_eq!(
            r#"Series: '' [date64]
[
	1970-01-01 00:00:00.001
	null
	2001-09-09 01:46:40
]"#,
            format!("{:?}", s.into_series())
        );
        let s = Time64NanosecondChunked::new_from_slice(
            "",
            &[1_000_000, 37_800_005_000_000, 86_399_210_000_000],
        );
        assert_eq!(
            r#"Series: '' [time64(ns)]
[
	00:00:00.001
	10:30:00.005
	23:59:59.210
]"#,
            format!("{:?}", s.into_series())
        )
    }
    #[test]
    fn test_fmt_chunkedarray() {
        let ca = Int32Chunked::new_from_opt_slice("date32", &[Some(1), None, Some(3)]);
        println!("{:?}", ca);
        assert_eq!(
            r#"ChunkedArray: 'date32' [Int32]
[
	1
	null
	3
]"#,
            format!("{:?}", ca)
        );
        let ca = Utf8Chunked::new_from_slice("name", &["a", "b"]);
        println!("{:?}", ca);
        assert_eq!(
            r#"ChunkedArray: 'name' [str]
[
	"a"
	"b"
]"#,
            format!("{:?}", ca)
        );
    }

    #[test]
    fn test_series() {
        let s = Series::new("foo", &["Somelongstringto eeat wit me oundaf"]);
        dbg!(s);
    }
}
