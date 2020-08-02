use crate::datatypes::{AnyType, ToStr};
use crate::prelude::*;
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

        macro_rules! format_series {
            ($a:ident, $name:expr) => {{
                write![f, "Series: {} \n[\n", $name]?;
                $a.into_iter().take(LIMIT).for_each(|v| {
                    match v {
                        Some(v) => {
                            write!(f, "\t{}\n", v).ok();
                        }
                        None => {
                            write!(f, "\tnull\n").ok();
                        }
                    };
                });
                write![f, "]"]
            }};
        }

        match self {
            // TODO: insert new datatypes
            Series::Int32(a) => format_series!(a, "i32"),
            Series::Int64(a) => format_series!(a, "i64"),
            Series::UInt32(a) => format_series!(a, "u32"),
            Series::Bool(a) => format_series!(a, "bool"),
            Series::Float32(a) => format_series!(a, "f32"),
            Series::Float64(a) => format_series!(a, "f64"),
            Series::Date32(a) => format_series!(a, "date32"),
            Series::Date64(a) => format_series!(a, "date64"),
            Series::DurationNs(a) => format_series!(a, "date64"),
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
            AnyType::U32(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::I32(v) => fmt_integer(f, width, *v),
            AnyType::I64(v) => fmt_integer(f, width, *v),
            AnyType::F32(v) => fmt_float(f, width, *v),
            AnyType::F64(v) => fmt_float(f, width, *v),
            AnyType::Bool(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::Str(v) => write!(f, "{:>width$}", format!("\"{}\"", v), width = width),
            AnyType::Date64(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::Date32(v) => write!(f, "{:>width$}", v, width = width),
            AnyType::Time64(v, _) => write!(f, "{:>width$}", v, width = width),
            AnyType::Duration(v, _) => write!(f, "{:>width$}", v, width = width),
        }
    }
}

#[cfg(feature = "pretty")]
impl Display for AnyType<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let width = 0;
        match self {
            AnyType::Null => write!(f, "{}", "null"),
            AnyType::U32(v) => write!(f, "{}", v),
            AnyType::I32(v) => fmt_integer(f, width, *v),
            AnyType::I64(v) => fmt_integer(f, width, *v),
            AnyType::F32(v) => fmt_float(f, width, *v),
            AnyType::F64(v) => fmt_float(f, width, *v),
            AnyType::Bool(v) => write!(f, "{}", *v),
            AnyType::Str(v) => write!(f, "{}", format!("\"{}\"", v)),
            AnyType::Date64(v) => write!(f, "{}", v),
            AnyType::Date32(v) => write!(f, "{}", v),
            AnyType::Time64(v, _) => write!(f, "{}", v),
            AnyType::Duration(v, _) => write!(f, "{}", v),
        }
    }
}
