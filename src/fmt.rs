use crate::{
    frame::DataFrame,
    series::{chunked_array::iterator::ChunkIterator, series::Series},
};
use std::{
    fmt,
    fmt::{Debug, Display, Formatter},
};

impl Debug for Series {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        const LIMIT: usize = 10;

        macro_rules! format_series {
            ($a:ident, $name:expr) => {{
                write![f, "Series: {} \n[\n", $name];
                $a.iter().take(LIMIT).for_each(|v| {
                    match v {
                        Some(v) => {
                            write!(f, "\t{}\n", v);
                        }
                        None => {
                            write!(f, "\tnull");
                        }
                    };
                });
                write![f, "]"]
            }};
        }

        match self {
            Series::Int32(a) => format_series!(a, "i32"),
            Series::Int64(a) => format_series!(a, "i64"),
            Series::UInt32(a) => format_series!(a, "u32"),
            Series::Bool(a) => format_series!(a, "bool"),
            Series::Float32(a) => format_series!(a, "f32"),
            Series::Float64(a) => format_series!(a, "f64"),
            Series::Utf8(a) => {
                write![f, "Series: str \n[\n"];
                a.iter().take(LIMIT).for_each(|v| {
                    write!(f, "\t{}\n", &v[..LIMIT]);
                });
                write![f, "]"]
            }
            _ => write!(f, "hello"),
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
        unimplemented!()
    }
}
