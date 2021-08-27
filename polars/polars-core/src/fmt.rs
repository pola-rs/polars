use crate::prelude::*;

#[cfg(feature = "temporal")]
use crate::chunked_array::temporal::{
    date32_as_datetime, date64_as_datetime, time64_nanosecond_as_time,
};
use num::{Num, NumCast};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};
const LIMIT: usize = 25;

#[cfg(feature = "pretty_fmt")]
use comfy_table::presets::{ASCII_FULL, UTF8_FULL};
#[cfg(feature = "pretty_fmt")]
use comfy_table::*;

#[cfg(all(feature = "plain_fmt", not(feature = "pretty_fmt")))]
use prettytable::{Cell, Row, Table};

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
#[cfg(any(feature = "plain_fmt", feature = "pretty_fmt"))]
use std::borrow::Cow;
#[cfg(not(feature = "temporal"))]
use temporal::*;

macro_rules! format_array {
    ($limit:expr, $f:ident, $a:expr, $dtype:expr, $name:expr, $array_type:expr) => {{
        write!(
            $f,
            "shape: ({},)\n{}: '{}' [{}]\n[\n",
            $a.len(),
            $array_type,
            $name,
            $dtype
        )?;
        let truncate = matches!($a.dtype(), DataType::Utf8);
        let limit = std::cmp::min($limit, $a.len());

        let write = |v, f: &mut Formatter| {
            if truncate {
                let v = format!("{}", v);
                let v_trunc = &v[..v
                    .char_indices()
                    .take(15)
                    .last()
                    .map(|(i, c)| i + c.len_utf8())
                    .unwrap_or(0)];
                if v == v_trunc {
                    write!(f, "\t{}\n", v)?;
                } else {
                    write!(f, "\t{}...\n", v_trunc)?;
                }
            } else {
                write!(f, "\t{}\n", v)?;
            };
            Ok(())
        };

        if limit < $a.len() {
            for i in 0..limit / 2 {
                let v = $a.get_any_value(i);
                write(v, $f)?;
            }
            write!($f, "\t...\n")?;
            for i in (0..limit / 2).rev() {
                let v = $a.get_any_value($a.len() - i - 1);
                write(v, $f)?;
            }
        } else {
            for i in 0..limit {
                let v = $a.get_any_value(i);
                write(v, $f)?;
            }
        }

        write!($f, "]")
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
    match object.dtype() {
        DataType::Object(inner_type) => {
            write!(
                f,
                "shape: ({},)\n{}: '{}' [o][{}]\n[\n",
                object.len(),
                array_type,
                name,
                inner_type
            )?;

            for i in 0..limit {
                let v = object.str_value(i);
                writeln!(f, "\t{}", v)?;
            }

            write!(f, "]")
        }
        _ => unreachable!(),
    }
}

macro_rules! set_limit {
    ($self:ident) => {
        std::cmp::min($self.len(), LIMIT)
    };
}

impl<T> Debug for ChunkedArray<T>
where
    T: PolarsNumericType,
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
        format_array!(limit, f, self, "bool", self.name(), "ChunkedArray")
    }
}

impl Debug for Utf8Chunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        format_array!(80, f, self, "str", self.name(), "ChunkedArray")
    }
}

impl Debug for ListChunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);
        format_array!(limit, f, self, "list", self.name(), "ChunkedArray")
    }
}

impl Debug for CategoricalChunked {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);
        format_array!(limit, f, self, "cat", self.name(), "ChunkedArray")
    }
}

#[cfg(feature = "object")]
impl<T> Debug for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let limit = set_limit!(self);

        let taker = self.take_rand();
        let inner_type = T::type_name();
        write!(
            f,
            "ChunkedArray: '{}' [o][{}]\n[\n",
            self.name(),
            inner_type
        )?;

        if limit < self.len() {
            for i in 0..limit / 2 {
                match taker.get(i) {
                    None => writeln!(f, "\tnull")?,
                    Some(val) => writeln!(f, "\t{}", val)?,
                };
            }
            writeln!(f, "\t...")?;
            for i in (0..limit / 2).rev() {
                match taker.get(self.len() - i - 1) {
                    None => writeln!(f, "\tnull")?,
                    Some(val) => writeln!(f, "\t{}", val)?,
                };
            }
        } else {
            for i in 0..limit {
                match taker.get(i) {
                    None => writeln!(f, "\tnull")?,
                    Some(val) => writeln!(f, "\t{}", val)?,
                };
            }
        }
        Ok(())
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
                format_array!(limit, f, self.utf8().unwrap(), "str", self.name(), "Series")
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
            DataType::List(_) => format_array!(
                limit,
                f,
                self.list().unwrap(),
                "list",
                self.name(),
                "Series"
            ),
            #[cfg(feature = "object")]
            DataType::Object(_) => {
                format_object_array(limit, f, self.as_ref(), self.name(), "Series")
            }
            DataType::Categorical => format_array!(
                limit,
                f,
                self.categorical().unwrap(),
                "cat",
                self.name(),
                "Series"
            ),
            dt => panic!("{:?} not impl", dt),
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

#[cfg(any(feature = "plain_fmt", feature = "pretty_fmt"))]
fn prepare_row(row: Vec<Cow<'_, str>>, n_first: usize, n_last: usize) -> Vec<String> {
    fn make_str_val(v: &str) -> String {
        let string_limit = 32;
        let v_trunc = &v[..v
            .char_indices()
            .take(string_limit)
            .last()
            .map(|(i, c)| i + c.len_utf8())
            .unwrap_or(0)];
        if v == v_trunc {
            v.to_string()
        } else {
            format!("{}...", v_trunc)
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
        let height = self.height();
        if !self.columns.iter().all(|s| s.len() == height) {
            panic!("The columns lengths in the DataFrame are not equal.");
        }

        let max_n_cols = std::env::var("POLARS_FMT_MAX_COLS")
            .unwrap_or_else(|_| "8".to_string())
            .parse()
            .unwrap_or(8);
        #[cfg(any(feature = "plain_fmt", feature = "pretty_fmt"))]
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
        #[cfg(feature = "pretty_fmt")]
        {
            let mut table = Table::new();
            let preset = if std::env::var("POLARS_FMT_NO_UTF8").is_ok() {
                ASCII_FULL
            } else {
                UTF8_FULL
            };

            table
                .load_preset(preset)
                .set_content_arrangement(ContentArrangement::Dynamic)
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
                    let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                    rows.push(prepare_row(row, n_first, n_last));
                }
                let dots = rows[0].iter().map(|_| "...".to_string()).collect();
                rows.push(dots);
                for i in (self.height() - max_n_rows / 2 - 1)..self.height() {
                    let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                    rows.push(prepare_row(row, n_first, n_last));
                }
                for row in rows {
                    table.add_row(row);
                }
            } else {
                for i in 0..max_n_rows {
                    if i < self.height() && self.width() > 0 {
                        let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                        table.add_row(prepare_row(row, n_first, n_last));
                    } else {
                        break;
                    }
                }
            }

            write!(f, "shape: {:?}\n{}", self.shape(), table)?;
        }
        #[cfg(not(any(feature = "plain_fmt", feature = "pretty_fmt")))]
        {
            write!(
                f,
                "shape: {:?}\nto see more, compile with 'plain_fmt' or 'pretty_fmt' feature",
                self.shape()
            )?;
        }

        #[cfg(all(feature = "plain_fmt", not(feature = "pretty_fmt")))]
        {
            let mut table = Table::new();
            table.set_titles(Row::new(names.into_iter().map(|s| Cell::new(&s)).collect()));
            let mut rows = Vec::with_capacity(max_n_rows);
            if self.height() > max_n_rows {
                for i in 0..(max_n_rows / 2) {
                    let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                    rows.push(prepare_row(row, n_first, n_last));
                }
                let dots = rows[0].iter().map(|_| "...".to_string()).collect();
                rows.push(dots);
                for i in (self.height() - max_n_rows / 2 - 1)..self.height() {
                    let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                    rows.push(prepare_row(row, n_first, n_last));
                }
                for row in rows {
                    table.add_row(Row::new(row.into_iter().map(|s| Cell::new(&s)).collect()));
                }
            } else {
                for i in 0..max_n_rows {
                    if i < self.height() && self.width() > 0 {
                        let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                        table.add_row(Row::new(
                            prepare_row(row, n_first, n_last)
                                .into_iter()
                                .map(|s| Cell::new(&s))
                                .collect(),
                        ));
                    } else {
                        break;
                    }
                }
            }

            write!(f, "shape: {:?}\n{}", self.shape(), table)?;
        }

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
            #[cfg(feature = "temporal")]
            AnyValue::Date32(v) => write!(f, "{}", date32_as_datetime(*v).date()),
            #[cfg(feature = "temporal")]
            AnyValue::Date64(v) => write!(f, "{}", date64_as_datetime(*v)),
            AnyValue::Time64(v, TimeUnit::Nanosecond) => {
                write!(f, "{}", time64_nanosecond_as_time(*v))
            }
            AnyValue::Duration(v, TimeUnit::Nanosecond) => write!(f, "{}", v),
            AnyValue::Duration(v, TimeUnit::Millisecond) => write!(f, "{}", v),
            AnyValue::List(s) => write!(f, "{}", s.fmt_list()),
            #[cfg(feature = "object")]
            AnyValue::Object(_) => write!(f, "object"),
            _ => unimplemented!(),
        }
    }
}

macro_rules! fmt_option {
    ($opt:expr) => {{
        match $opt {
            Some(v) => format!("{}", v),
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
    T: PolarsNumericType,
    T::Native: fmt::Display,
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

#[cfg(all(
    test,
    feature = "temporal",
    feature = "dtype-date32",
    feature = "dtype-date64"
))]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_fmt_list() {
        let mut builder = ListPrimitiveChunkedBuilder::<Int32Type>::new("a", 10, 10);
        builder.append_slice(Some(&[1, 2, 3]));
        builder.append_slice(None);
        let list = builder.finish().into_series();

        println!("{:?}", list);
        assert_eq!(
            r#"shape: (2,)
Series: 'a' [list]
[
	[1, 2, 3]
	null
]"#,
            format!("{:?}", list)
        );
    }

    #[test]
    #[cfg(feature = "dtype-time64-ns")]
    fn test_fmt_temporal() {
        let s = Date32Chunked::new_from_opt_slice("date32", &[Some(1), None, Some(3)]);
        assert_eq!(
            r#"shape: (3,)
Series: 'date32' [date32]
[
	1970-01-02
	null
	1970-01-04
]"#,
            format!("{:?}", s.into_series())
        );

        let s = Date64Chunked::new_from_opt_slice("", &[Some(1), None, Some(1_000_000_000_000)]);
        assert_eq!(
            r#"shape: (3,)
Series: '' [date64]
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
            r#"shape: (3,)
Series: '' [time64(ns)]
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
            r#"shape: (3,)
ChunkedArray: 'date32' [Int32]
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
            r#"shape: (2,)
ChunkedArray: 'name' [str]
[
	"a"
	"b"
]"#,
            format!("{:?}", ca)
        );
    }

    #[test]
    fn test_fmt_series() {
        let s = Series::new("foo", &["Somelongstringto eeat wit me oundaf"]);
        dbg!(&s);
        assert_eq!(
            r#"shape: (1,)
Series: 'foo' [str]
[
	"Somelongstring...
]"#,
            format!("{:?}", s)
        );

        let s = Series::new("foo", &["😀😁😂😃😄😅😆😇😈😉😊😋😌😎😏😐😑😒😓"]);
        dbg!(&s);
        assert_eq!(
            r#"shape: (1,)
Series: 'foo' [str]
[
	"😀😁😂😃😄😅😆😇😈😉😊😋😌😎...
]"#,
            format!("{:?}", s)
        );

        let s = Series::new("foo", &["yzäöüäöüäöüäö"]);
        dbg!(&s);
        assert_eq!(
            r#"shape: (1,)
Series: 'foo' [str]
[
	"yzäöüäöüäöüäö"
]"#,
            format!("{:?}", s)
        );

        let s = Series::new("foo", (0..100).collect::<Vec<_>>());

        dbg!(&s);
        assert_eq!(
            r#"shape: (100,)
Series: 'foo' [i32]
[
	0
	1
	2
	3
	4
	5
	6
	7
	8
	9
	10
	11
	...
	88
	89
	90
	91
	92
	93
	94
	95
	96
	97
	98
	99
]"#,
            format!("{:?}", s)
        );
    }
}
