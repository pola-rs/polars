use crate::datatypes::{AnyType, ToStr};
use crate::prelude::*;

#[cfg(feature = "temporal")]
use crate::chunked_array::temporal::{
    date32_as_datetime, date64_as_datetime, time32_millisecond_as_time, time32_second_as_time,
    time64_microsecond_as_time, time64_nanosecond_as_time, timestamp_microseconds_as_datetime,
    timestamp_milliseconds_as_datetime, timestamp_nanoseconds_as_datetime,
    timestamp_seconds_as_datetime,
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
use crate::series::ops::SeriesOps;
#[cfg(not(feature = "temporal"))]
use temporal::*;

macro_rules! format_array {
    ($limit:ident, $f:ident, $a:ident, $dtype:expr, $name:expr, $array_type:expr) => {{
        write![$f, "{}: '{}' [{}]\n[\n", $array_type, $name, $dtype]?;

        for i in 0..$limit {
            let v = $a.get_any(i);
            write!($f, "\t{}\n", v)?;
        }

        write![$f, "]"]
    }};
}

macro_rules! format_utf8_array {
    ($limit:expr, $f:expr, $a:ident, $name:expr, $array_type:expr) => {{
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
    ($limit:ident, $f:ident, $a:ident, $name:expr, $array_type:expr) => {{
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

fn format_object_array(
    limit: usize,
    f: &mut Formatter<'_>,
    ca: &dyn SeriesOps,
    name: &str,
    array_type: &str,
) -> fmt::Result {
    write![f, "{}: '{}' [object]\n[\n", array_type, name]?;

    for i in 0..limit {
        let v = ca.get_any(i);
        match v {
            AnyType::Null => writeln!(f, "\tnull")?,
            _ => writeln!(f, "\tobject")?,
        }
    }

    write![f, "]"]
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
        let dtype = format!("{:?}", T::get_data_type());
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

impl<T> Debug for ObjectChunked<T>
where
    T: 'static + Debug + Clone + Send + Sync + Default,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);
        format_object_array(limit, f, self, self.name(), "ChunkedArray")
    }
}

impl Debug for Series {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);

        match self {
            Series::UInt8(a) => format_array!(limit, f, a, "u8", a.name(), "Series"),
            Series::UInt16(a) => format_array!(limit, f, a, "u16", a.name(), "Series"),
            Series::UInt32(a) => format_array!(limit, f, a, "u32", a.name(), "Series"),
            Series::UInt64(a) => format_array!(limit, f, a, "u64", a.name(), "Series"),
            Series::Int8(a) => format_array!(limit, f, a, "i8", a.name(), "Series"),
            Series::Int16(a) => format_array!(limit, f, a, "i16", a.name(), "Series"),
            Series::Int32(a) => format_array!(limit, f, a, "i32", a.name(), "Series"),
            Series::Int64(a) => format_array!(limit, f, a, "i64", a.name(), "Series"),
            Series::Bool(a) => format_array!(limit, f, a, "bool", a.name(), "Series"),
            Series::Float32(a) => format_array!(limit, f, a, "f32", a.name(), "Series"),
            Series::Float64(a) => format_array!(limit, f, a, "f64", a.name(), "Series"),
            Series::Date32(a) => format_array!(limit, f, a, "date32(day)", a.name(), "Series"),
            Series::Date64(a) => format_array!(limit, f, a, "date64(ms)", a.name(), "Series"),
            Series::Time32Millisecond(a) => {
                format_array!(limit, f, a, "time32(ms)", a.name(), "Series")
            }
            Series::Time32Second(a) => format_array!(limit, f, a, "time32(s)", a.name(), "Series"),
            Series::Time64Nanosecond(a) => {
                format_array!(limit, f, a, "time64(ns)", a.name(), "Series")
            }
            Series::Time64Microsecond(a) => {
                format_array!(limit, f, a, "time64(μs)", a.name(), "Series")
            }
            Series::DurationNanosecond(a) => {
                format_array!(limit, f, a, "duration(ns)", a.name(), "Series")
            }
            Series::DurationMicrosecond(a) => {
                format_array!(limit, f, a, "duration(μs)", a.name(), "Series")
            }
            Series::DurationMillisecond(a) => {
                format_array!(limit, f, a, "duration(ms)", a.name(), "Series")
            }
            Series::DurationSecond(a) => {
                format_array!(limit, f, a, "duration(s)", a.name(), "Series")
            }
            #[cfg(feature = "dtype-interval")]
            Series::IntervalDayTime(a) => {
                format_array!(limit, f, a, "interval(daytime)", a.name(), "Series")
            }
            #[cfg(feature = "dtype-interval")]
            Series::IntervalYearMonth(a) => {
                format_array!(limit, f, a, "interval(year-month)", a.name(), "Series")
            }
            Series::TimestampNanosecond(a) => {
                format_array!(limit, f, a, "timestamp(ns)", a.name(), "Series")
            }
            Series::TimestampMicrosecond(a) => {
                format_array!(limit, f, a, "timestamp(μs)", a.name(), "Series")
            }
            Series::TimestampMillisecond(a) => {
                format_array!(limit, f, a, "timestamp(ms)", a.name(), "Series")
            }
            Series::TimestampSecond(a) => {
                format_array!(limit, f, a, "timestamp(s)", a.name(), "Series")
            }
            Series::Utf8(a) => format_utf8_array!(80, f, a, a.name(), "Series"),
            Series::List(a) => format_list_array!(limit, f, a, a.name(), "Series"),
            Series::Object(a) => format_object_array(limit, f, &**a, a.name(), "Series"),
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

fn prepare_row(
    row: Vec<AnyType>,
    insert_idx: usize,
    reduce_columns: bool,
    max_n_cols: usize,
) -> Vec<String> {
    let string_limit = 32;
    let mut row_str = Vec::with_capacity(row.len());
    for v in row.iter().take(max_n_cols) {
        let str_val = if let AnyType::Utf8(s) = v {
            if s.len() > string_limit {
                format!("{}...", &s[..string_limit])
            } else {
                s.to_string()
            }
        } else {
            format!("{}", v)
        };
        row_str.push(str_val);
    }
    if reduce_columns {
        row_str.insert(insert_idx, "...".to_string());
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
        let insert_idx = max_n_cols / 2;
        let reduce_columns = self.width() > max_n_cols;

        let mut names: Vec<_> = self
            .schema()
            .fields()
            .iter()
            .take(max_n_cols)
            .map(|f| format!("{}\n---\n{}", f.name(), f.data_type().to_str()))
            .collect();
        if reduce_columns {
            names.insert(insert_idx, "...".to_string())
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
                rows.push(prepare_row(row, insert_idx, reduce_columns, max_n_cols));
            }
            let dots = rows[0].iter().map(|_| "...".to_string()).collect();
            rows.push(dots);
            for i in (self.height() - max_n_cols / 2 - 1)..self.height() {
                let row = self.get(i).unwrap();
                rows.push(prepare_row(row, insert_idx, reduce_columns, max_n_cols));
            }
            for row in rows {
                table.add_row(row);
            }
        } else {
            for i in 0..10 {
                let opt = self.get(i);
                if let Some(row) = opt {
                    table.add_row(prepare_row(row, insert_idx, reduce_columns, max_n_cols));
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

impl Display for AnyType<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 0;
        match self {
            AnyType::Null => write!(f, "null"),
            AnyType::UInt8(v) => write!(f, "{}", v),
            AnyType::UInt16(v) => write!(f, "{}", v),
            AnyType::UInt32(v) => write!(f, "{}", v),
            AnyType::UInt64(v) => write!(f, "{}", v),
            AnyType::Int8(v) => fmt_integer(f, width, *v),
            AnyType::Int16(v) => fmt_integer(f, width, *v),
            AnyType::Int32(v) => fmt_integer(f, width, *v),
            AnyType::Int64(v) => fmt_integer(f, width, *v),
            AnyType::Float32(v) => fmt_float(f, width, *v),
            AnyType::Float64(v) => fmt_float(f, width, *v),
            AnyType::Boolean(v) => write!(f, "{}", *v),
            AnyType::Utf8(v) => write!(f, "{}", format!("\"{}\"", v)),
            AnyType::Date32(v) => write!(f, "{}", date32_as_datetime(*v).date()),
            AnyType::Date64(v) => write!(f, "{}", date64_as_datetime(*v).date()),
            AnyType::Time32(v, TimeUnit::Millisecond) => {
                write!(f, "{}", time32_millisecond_as_time(*v))
            }
            AnyType::Time32(v, TimeUnit::Second) => write!(f, "{}", time32_second_as_time(*v)),
            AnyType::Time64(v, TimeUnit::Nanosecond) => {
                write!(f, "{}", time64_nanosecond_as_time(*v))
            }
            AnyType::Time64(v, TimeUnit::Microsecond) => {
                write!(f, "{}", time64_microsecond_as_time(*v))
            }
            AnyType::Duration(v, TimeUnit::Nanosecond) => write!(f, "{}", v),
            AnyType::Duration(v, TimeUnit::Microsecond) => write!(f, "{}", v),
            AnyType::Duration(v, TimeUnit::Millisecond) => write!(f, "{}", v),
            AnyType::Duration(v, TimeUnit::Second) => write!(f, "{}", v),
            AnyType::TimeStamp(v, TimeUnit::Nanosecond) => {
                write!(f, "{}", timestamp_nanoseconds_as_datetime(*v))
            }
            AnyType::TimeStamp(v, TimeUnit::Microsecond) => {
                write!(f, "{}", timestamp_microseconds_as_datetime(*v))
            }
            AnyType::TimeStamp(v, TimeUnit::Millisecond) => {
                write!(f, "{}", timestamp_milliseconds_as_datetime(*v))
            }
            AnyType::TimeStamp(v, TimeUnit::Second) => {
                write!(f, "{}", timestamp_seconds_as_datetime(*v))
            }
            #[cfg(feature = "dtype-interval")]
            AnyType::IntervalDayTime(v) => write!(f, "{}", v),
            #[cfg(feature = "dtype-interval")]
            AnyType::IntervalYearMonth(v) => write!(f, "{}", v),
            AnyType::List(s) => write!(f, "{:?}", s.fmt_list()),
            AnyType::Object(_) => write!(f, "object"),
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

impl<T> FmtList for ObjectChunked<T> {
    fn fmt_list(&self) -> String {
        todo!()
    }
}

#[cfg(all(test, feature = "temporal"))]
mod test {
    use crate::prelude::*;

    #[test]
    fn list() {
        use arrow::array::Int32Array;
        let values_builder = Int32Array::builder(10);
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
            r#"Series: 'date32' [date32(day)]
[
	1970-01-02
	null
	1970-01-04
]"#,
            format!("{:?}", s.into_series())
        );

        let s = Date64Chunked::new_from_opt_slice("", &[Some(1), None, Some(1_000_000_000_000)]);
        assert_eq!(
            r#"Series: '' [date64(ms)]
[
	1970-01-01
	null
	2001-09-09
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
