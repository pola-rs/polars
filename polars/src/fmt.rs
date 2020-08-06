use crate::chunked_array::temporal::{date64_as_datetime, time64_ns_as_time};
use crate::prelude::*;
use crate::{
    chunked_array::temporal::date32_as_datetime,
    datatypes::{AnyType, ToStr},
};
use num::{Num, NumCast};
#[cfg(feature = "pretty")]
use prettytable::Table;
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};

impl Debug for Series {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        const LIMIT: usize = 10;
        let limit = std::cmp::min(self.len(), LIMIT);

        macro_rules! format_series {
            ($a:ident, $name:expr) => {{
                write![f, "Series: {}\n[\n", $name]?;

                for i in 0..limit {
                    let v = $a.get(i);
                    write!(f, "\t{}\n", v)?;
                }

                write![f, "]"]
            }};
        }

        match self {
            // TODO: insert new datatypes
            Series::UInt8(a) => format_series!(a, "u8"),
            Series::UInt16(a) => format_series!(a, "u16"),
            Series::UInt32(a) => format_series!(a, "u32"),
            Series::UInt64(a) => format_series!(a, "u64"),
            Series::Int8(a) => format_series!(a, "i8"),
            Series::Int16(a) => format_series!(a, "i16"),
            Series::Int32(a) => format_series!(a, "i32"),
            Series::Int64(a) => format_series!(a, "i64"),
            Series::Bool(a) => format_series!(a, "bool"),
            Series::Float32(a) => format_series!(a, "f32"),
            Series::Float64(a) => format_series!(a, "f64"),
            Series::Date32(a) => format_series!(a, "date32(day)"),
            Series::Date64(a) => format_series!(a, "date64(ms)"),
            Series::DurationNs(a) => format_series!(a, "duration64(ns)"),
            Series::Time64Ns(a) => format_series!(a, "time64(ns)"),
            Series::Utf8(a) => {
                write![f, "Series: str \n[\n"]?;
                a.into_iter().take(LIMIT).for_each(|opt_s| match opt_s {
                    None => {
                        write!(f, "\tnull\n").ok();
                    }
                    Some(s) => {
                        write!(f, "\t\"{}\"\n", &s[..std::cmp::min(LIMIT, s.len())]).ok();
                    }
                });
                write![f, "]"]
            }
            _ => write!(f, "no supported"),
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

#[cfg(feature = "pretty")]
impl Display for DataFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let mut table = Table::new();
        let names = self
            .schema()
            .fields()
            .iter()
            .map(|f| format!("{}\n---\n{}", f.name(), f.data_type().to_str()))
            .collect();
        table.set_titles(names);
        for i in 0..10 {
            let opt = self.get(i);
            if let Some(row) = opt {
                let mut row_str = Vec::with_capacity(row.len());
                for v in &row {
                    row_str.push(format!("{}", v));
                }
                table.add_row(row.iter().map(|v| format!("{}", v)).collect());
            } else {
                break;
            }
        }
        write!(f, "{}", table)?;
        Ok(())
    }
}

#[cfg(not(feature = "pretty"))]
impl Display for DataFrame {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for field in self.schema().fields() {
            write!(f, "{:>15}", field.name())?;
        }
        write!(f, "\n")?;
        for field in self.schema().fields() {
            write!(f, "{:>15}", field.data_type().to_str())?;
        }
        write!(f, "\n")?;
        for _ in self.schema().fields() {
            write!(f, "{:>15}", "---")?;
        }
        write!(f, "\n\n")?;

        for i in 0..10 {
            let opt = self.get(i);
            if let Some(row) = opt {
                for v in row {
                    write!(f, "{}", v)?;
                }
                write!(f, "\n")?;
            }
        }
        Ok(())
    }
}

fn fmt_integer<T: Num + NumCast>(f: &mut Formatter<'_>, width: usize, v: T) -> fmt::Result {
    let v: i64 = NumCast::from(v).unwrap();
    if v > 9999 {
        write!(f, "{:>width$e}", v, width = width)
    } else {
        write!(f, "{:>width$}", v, width = width)
    }
}

fn fmt_float<T: Num + NumCast>(f: &mut Formatter<'_>, width: usize, v: T) -> fmt::Result {
    let v: f64 = NumCast::from(v).unwrap();
    let v = (v * 1000.).round() / 1000.;
    if v > 9999. || v < 0.001 {
        write!(f, "{:>width$e}", v, width = width)
    } else {
        write!(f, "{:>width$}", v, width = width)
    }
}

#[cfg(not(feature = "pretty"))]
impl Display for AnyType<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 15;
        match self {
            AnyType::Null => write!(f, "{:>width$}", "null", width = width),
            AnyType::UInt8(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::UInt16(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::UInt32(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::UInt64(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::Int8(v) => fmt_integer(f, width, *v),
            AnyType::Int16(v) => fmt_integer(f, width, *v),
            AnyType::Int32(v) => fmt_integer(f, width, *v),
            AnyType::Int64(v) => fmt_integer(f, width, *v),
            AnyType::Float32(v) => fmt_float(f, width, *v),
            AnyType::Float64(v) => fmt_float(f, width, *v),
            AnyType::Boolean(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::Utf8(v) => write!(f, "{:>width$}", format!("\"{}\"", v), width = width),
            AnyType::Date32(v) => write!(
                f,
                "{:>width$}",
                date32_as_datetime(*v).date(),
                width = width
            ),
            AnyType::Date64(v) => write!(f, "{:>width$}", date64_as_datetime(*v), width = width),
            AnyType::Time64(v, TimeUnit::Nanosecond) => {
                write!(f, "{:>width$}", time_64_ns_as_time(*v), width = width)
            }
            AnyType::Duration(v, TimeUnit::Nanosecond) => write!(f, "{:>width$}", v, width = width),
            _ => unimplemented!(),
        }
    }
}

#[cfg(feature = "pretty")]
impl Display for AnyType<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 0;
        match self {
            AnyType::Null => write!(f, "{}", "null"),
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
            AnyType::Date64(v) => write!(f, "{}", date64_as_datetime(*v)),
            AnyType::Time64(v, TimeUnit::Nanosecond) => write!(f, "{}", time64_ns_as_time(*v)),
            AnyType::Duration(v, TimeUnit::Nanosecond) => write!(f, "{}", v),
            _ => unimplemented!(),
        }
    }
}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn temporal() {
        let s = Date32Chunked::new_from_opt_slice("date32", &[Some(1), None, Some(3)]);
        assert_eq!(
            r#"Series: date32(day)
[
	1970-01-02
	null
	1970-01-04
]"#,
            format!("{:?}", s.into_series())
        );

        let s = Date64Chunked::new_from_opt_slice("", &[Some(1), None, Some(1000_000_000_000)]);
        assert_eq!(
            r#"Series: date64(ms)
[
	1970-01-01 00:00:00.001
	null
	2001-09-09 01:46:40
]"#,
            format!("{:?}", s.into_series())
        );
        let s = Time64NsChunked::new_from_slice(
            "",
            &[1_000_000, 37_800_005_000_000, 86_399_210_000_000],
        );
        assert_eq!(
            r#"Series: time64(ns)
[
	00:00:00.001
	10:30:00.005
	23:59:59.210
]"#,
            format!("{:?}", s.into_series())
        )
    }
}
