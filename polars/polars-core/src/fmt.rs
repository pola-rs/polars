use crate::prelude::*;

#[cfg(any(feature = "dtype-date", feature = "dtype-datetime"))]
use arrow::temporal_conversions::{date32_to_date, timestamp_ns_to_datetime};
use num::{Num, NumCast};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};
const LIMIT: usize = 25;

use arrow::temporal_conversions::{timestamp_ms_to_datetime, timestamp_us_to_datetime};
#[cfg(feature = "fmt")]
use comfy_table::presets::{ASCII_FULL, UTF8_FULL};
#[cfg(feature = "fmt")]
use comfy_table::*;
#[cfg(feature = "fmt")]
use std::borrow::Cow;

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
    object: &Series,
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
            #[cfg(feature = "dtype-date")]
            DataType::Date => format_array!(
                limit,
                f,
                self.date().unwrap(),
                "date",
                self.name(),
                "Series"
            ),
            #[cfg(feature = "dtype-datetime")]
            DataType::Datetime(_, _) => {
                let dt = format!("{}", self.dtype());
                format_array!(
                    limit,
                    f,
                    self.datetime().unwrap(),
                    &dt,
                    self.name(),
                    "Series"
                )
            }
            #[cfg(feature = "dtype-duration")]
            DataType::Duration(_) => {
                let dt = format!("{}", self.dtype());
                format_array!(
                    limit,
                    f,
                    self.duration().unwrap(),
                    &dt,
                    self.name(),
                    "Series"
                )
            }
            DataType::List(_) => format_array!(
                limit,
                f,
                self.list().unwrap(),
                "list",
                self.name(),
                "Series"
            ),
            #[cfg(feature = "object")]
            DataType::Object(_) => format_object_array(limit, f, self, self.name(), "Series"),
            #[cfg(feature = "dtype-categorical")]
            DataType::Categorical(_) => format_array!(
                limit,
                f,
                self.categorical().unwrap(),
                "cat",
                self.name(),
                "Series"
            ),
            #[cfg(feature = "dtype-struct")]
            dt @ DataType::Struct(_) => format_array!(
                limit,
                f,
                self.struct_().unwrap(),
                format!("{}", dt),
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
#[cfg(feature = "fmt")]
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

#[cfg(feature = "fmt")]
fn prepare_row(row: Vec<Cow<'_, str>>, n_first: usize, n_last: usize) -> Vec<String> {
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
        #[cfg(feature = "fmt")]
        {
            let height = self.height();
            assert!(
                self.columns.iter().all(|s| s.len() == height),
                "The columns lengths in the DataFrame are not equal."
            );

            let max_n_cols = std::env::var("POLARS_FMT_MAX_COLS")
                .unwrap_or_else(|_| "8".to_string())
                .parse()
                .unwrap_or(8);

            let max_n_rows = {
                let max_n_rows = std::env::var("POLARS_FMT_MAX_ROWS")
                    .unwrap_or_else(|_| "8".to_string())
                    .parse()
                    .unwrap_or(8);
                if max_n_rows < 2 {
                    2
                } else {
                    max_n_rows
                }
            };
            let (n_first, n_last) = if self.width() > max_n_cols {
                ((max_n_cols + 1) / 2, max_n_cols / 2)
            } else {
                (self.width(), 0)
            };
            let reduce_columns = n_first + n_last < self.width();

            let mut names = Vec::with_capacity(n_first + n_last + reduce_columns as usize);

            let field_to_str = |f: &Field| {
                let name = make_str_val(f.name());
                let lower_bounds = std::cmp::max(5, std::cmp::min(12, name.len()));
                let s = format!("{}\n---\n{}", name, f.data_type());
                (s, lower_bounds)
            };
            let tbl_lower_bounds = |l: usize| {
                comfy_table::ColumnConstraint::LowerBoundary(comfy_table::Width::Fixed(l as u16))
            };
            let mut constraints = Vec::with_capacity(n_first + n_last + reduce_columns as usize);
            let fields = self.fields();
            for field in fields[0..n_first].iter() {
                let (s, l) = field_to_str(field);
                names.push(s);
                constraints.push(tbl_lower_bounds(l));
            }
            if reduce_columns {
                names.push("...".into());
                constraints.push(tbl_lower_bounds(5));
            }
            for field in fields[self.width() - n_last..].iter() {
                let (s, l) = field_to_str(field);
                names.push(s);
                constraints.push(tbl_lower_bounds(l));
            }
            let mut table = Table::new();
            let preset = if std::env::var("POLARS_FMT_NO_UTF8").is_ok() {
                ASCII_FULL
            } else {
                UTF8_FULL
            };

            table
                .load_preset(preset)
                .set_content_arrangement(ContentArrangement::Dynamic);

            let mut rows = Vec::with_capacity(max_n_rows);
            if self.height() > max_n_rows {
                for i in 0..(max_n_rows / 2) {
                    let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                    rows.push(prepare_row(row, n_first, n_last));
                }
                let dots = rows[0].iter().map(|_| "...".to_string()).collect();
                rows.push(dots);
                for i in (self.height() - (max_n_rows + 1) / 2)..self.height() {
                    let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                    rows.push(prepare_row(row, n_first, n_last));
                }
                for row in rows {
                    table.add_row(row);
                }
            } else {
                for i in 0..self.height() {
                    if self.width() > 0 {
                        let row = self.columns.iter().map(|s| s.str_value(i)).collect();
                        table.add_row(prepare_row(row, n_first, n_last));
                    } else {
                        break;
                    }
                }
            }

            table.set_header(names).set_constraints(constraints);

            let tbl_width = std::env::var("POLARS_TABLE_WIDTH")
                .map(|s| {
                    Some(
                        s.parse::<u16>()
                            .expect("could not parse table width argument"),
                    )
                })
                .unwrap_or(None);
            // if tbl_width is explicitly set, use it
            if let Some(w) = tbl_width {
                table.set_table_width(w);
            }

            // if no tbl_width (its not-tty && it is not explicitly set), then set default
            // this is needed to support non-tty applications
            if !table.is_tty() && table.get_table_width().is_none() {
                table.set_table_width(100);
            }

            write!(f, "shape: {:?}\n{}", self.shape(), table)?;
        }

        #[cfg(not(feature = "fmt"))]
        {
            write!(
                f,
                "shape: {:?}\nto see more, compile with 'plain_fmt' or 'pretty_fmt' feature",
                self.shape()
            )?;
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

const SCIENTIFIC_BOUND: f64 = 999999.0;
fn fmt_float<T: Num + NumCast>(f: &mut Formatter<'_>, width: usize, v: T) -> fmt::Result {
    let v: f64 = NumCast::from(v).unwrap();
    // show integers as 0.0, 1.0 ... 101.0
    if v.fract() == 0.0 && v.abs() < SCIENTIFIC_BOUND {
        write!(f, "{:>width$.1}", v, width = width)
    } else if format!("{}", v).len() > 9 {
        // large and small floats in scientific notation
        if !(0.000001..=SCIENTIFIC_BOUND).contains(&v.abs()) | (v.abs() > SCIENTIFIC_BOUND) {
            write!(f, "{:>width$.4e}", v, width = width)
        } else {
            // this makes sure we don't write 12.00000 in case of a long flt that is 12.0000000001
            // instead we write 12.0
            let s = format!("{:>width$.6}", v, width = width);

            if s.ends_with('0') {
                let mut s = s.as_str();
                let mut len = s.len() - 1;

                while s.ends_with('0') {
                    len -= 1;
                    s = &s[..len];
                }
                if s.ends_with('.') {
                    write!(f, "{}0", s)
                } else {
                    write!(f, "{}", s)
                }
            } else {
                // 12.0934509341243124
                // written as
                // 12.09345
                write!(f, "{:>width$.6}", v, width = width)
            }
        }
    } else if v.fract() == 0.0 {
        write!(f, "{:>width$e}", v, width = width)
    } else {
        write!(f, "{:>width$}", v, width = width)
    }
}

const SIZES_NS: [i64; 4] = [
    86_400_000_000_000,
    3_600_000_000_000,
    60_000_000_000,
    1_000_000_000,
];
const NAMES: [&str; 4] = ["day", "hour", "minute", "second"];
const SIZES_US: [i64; 4] = [86_400_000_000, 3_600_000_000, 60_000_000, 1_000_000];
const SIZES_MS: [i64; 4] = [86_400_000, 3_600_000, 60_000, 1_000];

fn fmt_duration_ns(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0 ns");
    }
    format_duration(f, v, SIZES_NS.as_slice(), NAMES.as_slice())?;
    if v % 1000 != 0 {
        write!(f, "{} ns", v % 1_000_000_000)?;
    } else if v % 1_000_000 != 0 {
        write!(f, "{} µs", (v % 1_000_000_000) / 1000)?;
    } else if v % 1_000_000_000 != 0 {
        write!(f, "{} ms", (v % 1_000_000_000) / 1_000_000)?;
    }
    Ok(())
}

fn fmt_duration_us(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0 µs");
    }
    format_duration(f, v, SIZES_US.as_slice(), NAMES.as_slice())?;
    if v % 1000 != 0 {
        write!(f, "{} µs", (v % 1_000_000_000) / 1000)?;
    } else if v % 1_000_000 != 0 {
        write!(f, "{} ms", (v % 1_000_000_000) / 1_000_000)?;
    }
    Ok(())
}

fn fmt_duration_ms(f: &mut Formatter<'_>, v: i64) -> fmt::Result {
    if v == 0 {
        return write!(f, "0 ms");
    }
    format_duration(f, v, SIZES_MS.as_slice(), NAMES.as_slice())?;
    if v % 1_000 != 0 {
        write!(f, "{} ms", (v % 1_000_000_000) / 1_000_000)?;
    }
    Ok(())
}

fn format_duration(f: &mut Formatter, v: i64, sizes: &[i64], names: &[&str]) -> fmt::Result {
    for i in 0..4 {
        let whole_num = if i == 0 {
            v / sizes[i]
        } else {
            (v % sizes[i - 1]) / sizes[i]
        };
        if whole_num <= -1 || whole_num >= 1 {
            write!(f, "{} {}", whole_num, names[i])?;
            if whole_num != 1 {
                write!(f, "s")?;
            }
            if v % sizes[i] != 0 {
                write!(f, " ")?;
            }
        }
    }
    Ok(())
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
            AnyValue::Utf8(v) => write!(f, "{}", format_args!("\"{}\"", v)),
            AnyValue::Utf8Owned(v) => write!(f, "{}", format_args!("\"{}\"", v)),
            #[cfg(feature = "dtype-date")]
            AnyValue::Date(v) => write!(f, "{}", date32_to_date(*v)),
            #[cfg(feature = "dtype-datetime")]
            AnyValue::Datetime(v, tu, _) => match tu {
                TimeUnit::Nanoseconds => write!(f, "{}", timestamp_ns_to_datetime(*v)),
                TimeUnit::Microseconds => write!(f, "{}", timestamp_us_to_datetime(*v)),
                TimeUnit::Milliseconds => write!(f, "{}", timestamp_ms_to_datetime(*v)),
            },
            #[cfg(feature = "dtype-duration")]
            AnyValue::Duration(v, tu) => match tu {
                TimeUnit::Nanoseconds => fmt_duration_ns(f, *v),
                TimeUnit::Microseconds => fmt_duration_us(f, *v),
                TimeUnit::Milliseconds => fmt_duration_ms(f, *v),
            },
            #[cfg(feature = "dtype-time")]
            AnyValue::Time(_) => {
                let nt: chrono::NaiveTime = self.into();
                write!(f, "{}", nt)
            }
            #[cfg(feature = "dtype-categorical")]
            AnyValue::Categorical(idx, rev) => {
                let s = rev.get(*idx);
                write!(f, "\"{}\"", s)
            }
            AnyValue::List(s) => write!(f, "{}", s.fmt_list()),
            #[cfg(feature = "object")]
            AnyValue::Object(v) => write!(f, "{}", v),
            #[cfg(feature = "dtype-struct")]
            AnyValue::Struct(vals) => {
                write!(f, "{{")?;
                if !vals.is_empty() {
                    for v in &vals[..vals.len() - 1] {
                        write!(f, "{},", v)?;
                    }
                    // last value has no trailing comma
                    write!(f, "{}", vals[vals.len() - 1])?;
                }
                write!(f, "}}")
            }
        }
    }
}

macro_rules! impl_fmt_list {
    ($self:ident) => {{
        match $self.len() {
            0 => format!("[]"),
            1 => format!("[{}]", $self.get_any_value(0)),
            2 => format!("[{}, {}]", $self.get_any_value(0), $self.get_any_value(1)),
            3 => format!(
                "[{}, {}, {}]",
                $self.get_any_value(0),
                $self.get_any_value(1),
                $self.get_any_value(2)
            ),
            _ => format!(
                "[{}, {}, ... {}]",
                $self.get_any_value(0),
                $self.get_any_value(1),
                $self.get_any_value($self.len() - 1)
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

#[cfg(feature = "dtype-categorical")]
impl FmtList for CategoricalChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-date")]
impl FmtList for DateChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-datetime")]
impl FmtList for DatetimeChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-duration")]
impl FmtList for DurationChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-time")]
impl FmtList for TimeChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "dtype-struct")]
impl FmtList for StructChunked {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(feature = "object")]
impl<T: PolarsObject> FmtList for ObjectChunked<T> {
    fn fmt_list(&self) -> String {
        impl_fmt_list!(self)
    }
}

#[cfg(all(
    test,
    feature = "temporal",
    feature = "dtype-date",
    feature = "dtype-datetime"
))]
mod test {
    use crate::prelude::*;

    #[test]
    fn test_fmt_list() {
        let mut builder = ListPrimitiveChunkedBuilder::<i32>::new("a", 10, 10, DataType::Int32);
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
    fn test_fmt_temporal() {
        let s = Int32Chunked::new("Date", &[Some(1), None, Some(3)]).into_date();
        assert_eq!(
            r#"shape: (3,)
Series: 'Date' [date]
[
	1970-01-02
	null
	1970-01-04
]"#,
            format!("{:?}", s.into_series())
        );

        let s = Int64Chunked::new("", &[Some(1), None, Some(1_000_000_000_000)])
            .into_datetime(TimeUnit::Nanoseconds, None);
        assert_eq!(
            r#"shape: (3,)
Series: '' [datetime[ns]]
[
	1970-01-01 00:00:00.000000001
	null
	1970-01-01 00:16:40
]"#,
            format!("{:?}", s.into_series())
        );
    }

    #[test]
    fn test_fmt_chunkedarray() {
        let ca = Int32Chunked::new("Date", &[Some(1), None, Some(3)]);
        println!("{:?}", ca);
        assert_eq!(
            r#"shape: (3,)
ChunkedArray: 'Date' [Int32]
[
	1
	null
	3
]"#,
            format!("{:?}", ca)
        );
        let ca = Utf8Chunked::new("name", &["a", "b"]);
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
