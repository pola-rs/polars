//! # (De)serializing CSV files
//!
//! ## Maximal performance
//! Currently [CsvReader::new](CsvReader::new) has an extra copy. If you want optimal performance in CSV parsing/
//! reading, it is advised to use [CsvReader::from_path](CsvReader::from_path).
//!
//! ## Write a DataFrame to a csv file.
//!
//! ## Example
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example(df: &mut DataFrame) -> PolarsResult<()> {
//!     let mut file = File::create("example.csv").expect("could not create file");
//!
//!     CsvWriter::new(&mut file)
//!     .has_header(true)
//!     .with_delimiter(b',')
//!     .finish(df)
//! }
//! ```
//!
//! ## Read a csv file to a DataFrame
//!
//! ## Example
//!
//! ```
//! use polars_core::prelude::*;
//! use polars_io::prelude::*;
//! use std::fs::File;
//!
//! fn example() -> PolarsResult<DataFrame> {
//!     // always prefer `from_path` as that is fastest.
//!     CsvReader::from_path("iris_csv")?
//!             .has_header(true)
//!             .finish()
//! }
//! ```
//!
pub(crate) mod buffer;
pub(crate) mod parser;
pub mod read_impl;

mod read;
#[cfg(not(feature = "private"))]
pub(crate) mod utils;
#[cfg(feature = "private")]
pub mod utils;
mod write;
pub(super) mod write_impl;

use std::borrow::Cow;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

use polars_core::prelude::*;
#[cfg(feature = "temporal")]
use polars_time::prelude::*;
#[cfg(feature = "temporal")]
use rayon::prelude::*;
pub use read::{CsvEncoding, CsvReader, NullValues};
#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};
pub use write::CsvWriter;

use crate::csv::read_impl::CoreReader;
use crate::csv::utils::get_reader_bytes;
use crate::mmap::MmapBytesReader;
use crate::predicates::PhysicalIoExpr;
use crate::utils::resolve_homedir;
use crate::{RowCount, SerReader, SerWriter};

#[test]
fn test_foo() {
    let csv = std::io::Cursor::new(br#"type,dbname,dump
database,connections-prod,"--
x x x.x_x_x_x_x_x x
 x x.x,
    x.x,
    x.x,
    x.""x""
   x ( x x.x,
            'x'::x x x,
            x.x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x (x_x x
             x x_x x x ((x.x = x.x)))
          x (x_x_x(x.x, x.x, 'x,x'::x) x (x.x x ( x x.x_x
                   x x_x.x
                  x (((x.x_x)::x <> 'x'::x) x ((x.x_x)::x = 'x'::x)))))
        x x
         x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x, x,x,x,x,x,x'::x) x x_x_x(x.x, x.x, 'x'::x))
        x x
         x x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x, x,x,x,x,x,x'::x) x x_x_x(x.x, x.x, 'x'::x) x (((x.x)::x)::x <> x (x['x'::x, 'x'::x, 'x'::x, 'x'::x, 'x'::x])))
        x x
         x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x,x'::x) x x_x_x(x.x, x.x, 'x'::x))
        x x
         x x_x.x x x,
            'x'::x x x,
            x_x.x_x x x_x,
            x_x.x_x
           x x_x.x_x
          x ((x_x.x_x)::x x ( x x.x_x
                   x x_x.x
                  x (((x.x_x)::x <> 'x'::x) x ((x.x_x)::x = 'x'::x))))) x
  x x x.x, x.x;


--
-- x: x_x x; x: x; x: x; x: x
--

x x x x.x_x x x x x x x('x.x_x_x_x'::x);
"
database,content-prod,"--
-- x x x
--

-- x x x x 11.16
-- x x x_x x 15.1

x x_x = 0;
x x_x = 0;
x x_x_x_x_x = 0;
x x_x = 'x8';
x x_x_x = x;
x x_x.x_x('x_x', '', x);
x x_x_x = x;
x x = x;
x x_x_x = x;
x x_x = x;

--
--

x x ""x-x"" x x = x0 x = 'x8' ;



\x -x-x=x ""x='x-x'""

x x_x = 0;
x x_x = 0;
x x_x_x_x_x = 0;
x x_x = 'x8';
x x_x_x = x;
x x_x.x_x('x_x', '', x);
x x_x_x = x;
x x = x;
x x_x_x = x;
x x_x = x;

--
--

-- *x* x x, x x x x



x x_x = '';

--
-- x: x; x: x; x: x; x: x
--

x x x.x (
    x x x x,
    x_x x x(50) x x,
    x x x(50) x ''::x x x x,
    x_x x x ''::x x x,
    x_x x x ''::x x x,
    x x x x,
    x_x x x x,
    x_x x x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x,
    x x x(255) x 'x'::x x x x,
    x x x 9999 x x
);


x x x.x x x x;

--
-- x: x_x_x; x: x; x: x; x: x
--

x x x.x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x x x x;

--
-- x: x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x x x x.x.x;


--
-- x: x; x: x; x: x; x: x
--

x x x.x (
    x x x x,
    x_x x x x,
    x x x(50) x ''::x x x x,
    x_x x x x x x x,
    x_x x x x x x x,
    x_x x x(200) x x,
    x_x x x(200) x x,
    x x x '[]'::x x x,
    x_x x x x x x,
    x_x x x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);


x x x.x x x x;

--
-- x: x_x_x; x: x; x: x; x: x
--

x x x.x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x x x x;

--
-- x: x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x x x x.x.x;


--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x (
    x x x x,
    x_x x,
    x_x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);


x x x.x_x x x x;

--
-- x: x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x_x x x x;

--
-- x: x_x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x_x x x x.x_x.x;


--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x (
    x x x x,
    x x x x
);


x x x.x_x x x x;

--
-- x: x_x_x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x_x_x x
 x x.x,
    x.x,
    x.x,
    x.""x""
   x ( x x.x,
            'x'::x x x,
            x.x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x (x_x x
             x x_x x x ((x.x = x.x)))
          x (x_x_x(x.x, x.x, 'x,x'::x) x (x.x x ( x x.x_x
                   x x_x.x
                  x (((x.x_x)::x <> 'x'::x) x ((x.x_x)::x = 'x'::x)))))
        x x
         x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x, x,x,x,x,x,x'::x) x x_x_x(x.x, x.x, 'x'::x))
        x x
         x x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x, x,x,x,x,x,x'::x) x x_x_x(x.x, x.x, 'x'::x) x (((x.x)::x)::x <> x (x['x'::x, 'x'::x, 'x'::x, 'x'::x, 'x'::x])))
        x x
         x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x,x'::x) x x_x_x(x.x, x.x, 'x'::x))
        x x
         x x_x.x x x,
            'x'::x x x,
            x_x.x_x x x_x,
            x_x.x_x
           x x_x.x_x
          x ((x_x.x_x)::x x ( x x.x_x
                   x x_x.x
                  x (((x.x_x)::x <> 'x'::x) x ((x.x_x)::x = 'x'::x))))) x
  x x x.x, x.x;


x x x.x_x_x_x_x_x x x x;

--
-- x: x x; x: x; x: x; x: x
--

x x x x.x x x x x x x('x.x_x_x'::x);


--
-- x: x x; x: x; x: x; x: x
--

x x x x.x x x x x x x('x.x_x_x'::x);


--
-- x: x_x x; x: x; x: x; x: x
--

x x x x.x_x x x x x x x('x.x_x_x_x'::x);


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x x x (x);


--
-- x: x x_x_x_x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x_x_x_x x (x_x);


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x x x (x);


--
-- x: x x_x_x_x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x_x_x_x x (x_x);


--
-- x: x x_x_x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x_x_x x (x);


--
-- x: x x_x_x_x_x_x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x_x_x_x_x_x x (x_x, x_x);


--
-- x: x_x x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x x x (x);


--
-- x: x_x x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x_x x (x_x, x_x);


--
-- x: x_x x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x x x (x);


--
-- x: x_x_x; x: x; x: x; x: x
--

x x x_x_x x x.x x x (x);


--
--

x x,x x x ""x-x"" x ""x-x_x_x_x"";
x x,x x x ""x-x"" x ""x-x_x_x_x"";


--
--

x x x x x x x;
x x x x x x ""x-x_x_x_x"";
x x x x x x ""x-x_x_x_x"";
x x x x x x x_x_x;


--
-- x: x x; x: x; x: x; x: x
--

x x x x x.x x ""x-x_x_x_x"";
x x,x,x,x x x x.x x ""x-x_x_x_x"";


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x x ""x-x_x_x_x"";


--
-- x: x x; x: x; x: x; x: x
--

x x x x x.x x ""x-x_x_x_x"";
x x,x,x,x x x x.x x ""x-x_x_x_x"";


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x x ""x-x_x_x_x"";


--
-- x: x x_x; x: x; x: x; x: x
--

x x,x,x,x x x x.x_x x ""x-x_x_x_x"";
x x x x x.x_x x ""x-x_x_x_x"";


--
-- x: x x_x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x_x x ""x-x_x_x_x"";


--
-- x: x x_x; x: x; x: x; x: x
--

x x,x,x,x x x x.x_x x ""x-x_x_x_x"";
x x x x x.x_x x ""x-x_x_x_x"";


--
-- x: x x_x_x_x_x_x; x: x; x: x; x: x
--

x x x x x.x_x_x_x_x_x x x_x_x;
x x x x x.x_x_x_x_x_x x ""x-x_x_x_x"";
x x,x,x,x,x x x x.x_x_x_x_x_x x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x,x x x  x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x x x  x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x x x  x ""x-x_x_x_x"";
x x x x x x x x,x,x,x x x  x ""x-x_x_x_x"";
"
database,notifications-prod,"--
-- x x x
--

-- x x x x 11.16
-- x x x_x x 15.1

x x_x = 0;
x x_x = 0;
x x_x_x_x_x = 0;
x x_x = 'x8';
x x_x_x = x;
x x_x.x_x('x_x', '', x);
x x_x_x = x;
x x = x;
x x_x_x = x;
x x_x = x;

--
--

x x ""x-x"" x x = x0 x = 'x8' ;



\x -x-x=x ""x='x-x'""

x x_x = 0;
x x_x = 0;
x x_x_x_x_x = 0;
x x_x = 'x8';
x x_x_x = x;
x x_x.x_x('x_x', '', x);
x x_x_x = x;
x x = x;
x x_x_x = x;
x x_x = x;

--
--

-- *x* x x, x x x x



--
-- x: x; x: x; x: -; x: -
--

x x x x x x x x x;


--
-- x: x x; x: x; x: -; x:
--

x x x x x 'x x';


--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x x x (
    'x',
    'x',
    'x',
    'x',
    'x',
    'x'
);


x x x.x_x x x x;

x x_x = '';

--
-- x: x_x_x; x: x; x: x; x: x
--

x x x.x_x_x (
    x x x x,
    x_x x x x,
    x_x x.x_x,
    x_x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);


x x x.x_x_x x x x;

--
-- x: x_x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x_x_x x x x;

--
-- x: x_x_x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x_x_x x x x.x_x_x.x;


--
-- x: x_x_x; x: x; x: x; x: x
--

x x x.x_x_x (
    x x x x,
    x_x x x x,
    x_x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);


x x x.x_x_x x x x;

--
-- x: x_x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x_x_x x x x;

--
-- x: x_x_x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x_x_x x x x.x_x_x.x;


--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x (
    x x x x,
    x_x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);


x x x.x_x x x x;

--
-- x: x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x_x x x x;

--
-- x: x_x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x_x x x x.x_x.x;


--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x (
    x x x x,
    x x x x
);


x x x.x_x x x x;

--
-- x: x_x_x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x_x_x x
 x x.x,
    x.x,
    x.x,
    x.""x""
   x ( x x.x,
            'x'::x x x,
            x.x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x (x_x x
             x x_x x x ((x.x = x.x)))
          x (x_x_x(x.x, x.x, 'x,x'::x) x (x.x x ( x x.x_x
                   x x_x.x
                  x (((x.x_x)::x <> 'x'::x) x ((x.x_x)::x = 'x'::x)))))
        x x
         x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x, x,x,x,x,x,x'::x) x x_x_x(x.x, x.x, 'x'::x))
        x x
         x x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x, x,x,x,x,x,x'::x) x x_x_x(x.x, x.x, 'x'::x) x (((x.x)::x)::x <> x (x['x'::x, 'x'::x, 'x'::x, 'x'::x, 'x'::x])))
        x x
         x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x,x'::x) x x_x_x(x.x, x.x, 'x'::x))
        x x
         x x_x.x x x,
            'x'::x x x,
            x_x.x_x x x_x,
            x_x.x_x
           x x_x.x_x
          x ((x_x.x_x)::x x ( x x.x_x
                   x x_x.x
                  x (((x.x_x)::x <> 'x'::x) x ((x.x_x)::x = 'x'::x))))) x
  x x x.x, x.x;


x x x.x_x_x_x_x_x x x x;

--
-- x: x_x_x x; x: x; x: x; x: x
--

x x x x.x_x_x x x x x x x('x.x_x_x_x_x'::x);


--
-- x: x_x_x x; x: x; x: x; x: x
--

x x x x.x_x_x x x x x x x('x.x_x_x_x_x'::x);


--
-- x: x_x x; x: x; x: x; x: x
--

x x x x.x_x x x x x x x('x.x_x_x_x'::x);


--
-- x: x_x_x x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x_x
    x x x_x_x_x x x (x);


--
-- x: x_x_x x_x_x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x_x
    x x x_x_x_x_x_x x (x_x);


--
-- x: x_x_x x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x_x
    x x x_x_x_x x x (x);


--
-- x: x_x x_x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x_x_x x (x_x);


--
-- x: x_x x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x x x (x);


--
-- x: x_x x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x x x (x);


--
-- x: x_x_x_x; x: x; x: x; x: x
--

x x x_x_x_x x x.x_x_x x x (x_x, x_x, x_x);


--
-- x: x_x_x_x; x: x; x: x; x: x
--

x x x_x_x_x x x.x_x_x x x (x_x);


--
-- x: x_x_x x_x_x_x_x_x; x: x x; x: x; x: x
--

x x x x.x_x_x
    x x x_x_x_x_x_x x x (x_x) x x.x_x(x);


--
-- x: x_x_x x_x_x_x_x_x; x: x x; x: x; x: x
--

x x x x.x_x_x
    x x x_x_x_x_x_x x x (x_x) x x.x_x(x);


--
--

x x,x x x ""x-x"" x ""x-x_x_x_x"";
x x,x x x ""x-x"" x ""x-x_x_x_x"";


--
--

x x x x x x x;
x x x x x x ""x-x_x_x_x"";
x x x x x x ""x-x_x_x_x"";
x x x x x x x_x_x;


--
-- x: x x(x); x: x; x: x; x: x
--

x x x x x.x(x) x ""x-x_x_x_x"";


--
-- x: x x(x, x[], x[]); x: x; x: x; x: x
--

x x x x x.x(x, x[], x[]) x ""x-x_x_x_x"";


--
-- x: x x(x, x); x: x; x: x; x: x
--

x x x x x.x(x, x) x ""x-x_x_x_x"";


--
-- x: x x(x); x: x; x: x; x: x
--

x x x x x.x(x) x ""x-x_x_x_x"";


--
-- x: x x(x, x, x); x: x; x: x; x: x
--

x x x x x.x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x(x, x, x, x); x: x; x: x; x: x
--

x x x x x.x_x(x, x, x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x); x: x; x: x; x: x
--

x x x x x.x(x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x); x: x; x: x; x: x
--

x x x x x.x(x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x, x); x: x; x: x; x: x
--

x x x x x.x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x(x, x, x, x); x: x; x: x; x: x
--

x x x x x.x_x(x, x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x); x: x; x: x; x: x
--

x x x x x.x_x_x(x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(); x: x; x: x; x: x
--

x x x x x.x_x_x() x ""x-x_x_x_x"";


--
-- x: x x_x(x); x: x; x: x; x: x
--

x x x x x.x_x(x) x ""x-x_x_x_x"";


--
-- x: x x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x, x); x: x; x: x; x: x
--

x x x x x.x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x, x); x: x; x: x; x: x
--

x x x x x.x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x x x, x x x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x x x, x x x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x); x: x; x: x; x: x
--

x x x x x.x_x_x(x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x,x,x,x x x x.x_x_x x ""x-x_x_x_x"";
x x x x x.x_x_x x ""x-x_x_x_x"";


--
-- x: x x_x_x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x_x_x x ""x-x_x_x_x"";


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x,x,x,x x x x.x_x_x x ""x-x_x_x_x"";
x x x x x.x_x_x x ""x-x_x_x_x"";


--
-- x: x x_x_x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x_x_x x ""x-x_x_x_x"";


--
-- x: x x_x; x: x; x: x; x: x
--

x x,x,x,x x x x.x_x x ""x-x_x_x_x"";
x x x x x.x_x x ""x-x_x_x_x"";


--
-- x: x x_x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x_x x ""x-x_x_x_x"";


--
-- x: x x_x; x: x; x: x; x: x
--

x x,x,x,x x x x.x_x x ""x-x_x_x_x"";
x x x x x.x_x x ""x-x_x_x_x"";


--
-- x: x x_x_x_x_x_x; x: x; x: x; x: x
--

x x x x x.x_x_x_x_x_x x x_x_x;
x x,x,x,x x x x.x_x_x_x_x_x x ""x-x_x_x_x"";
x x x x x.x_x_x_x_x_x x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x,x x x  x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x x x  x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x x x  x ""x-x_x_x_x"";
x x x x x x x x,x,x,x x x  x ""x-x_x_x_x"";
"
database,users-prod,"--
-- x x x
--

-- x x x x 11.16
-- x x x_x x 15.1

x x_x = 0;
x x_x = 0;
x x_x_x_x_x = 0;
x x_x = 'x8';
x x_x_x = x;
x x_x.x_x('x_x', '', x);
x x_x_x = x;
x x = x;
x x_x_x = x;
x x_x = x;

--
--

x x ""x-x"" x x = x0 x = 'x8' ;



\x -x-x=x ""x='x-x'""

x x_x = 0;
x x_x = 0;
x x_x_x_x_x = 0;
x x_x = 'x8';
x x_x_x = x;
x x_x.x_x('x_x', '', x);
x x_x_x = x;
x x = x;
x x_x_x = x;
x x_x = x;

--
--

-- *x* x x, x x x x



--
-- x: x; x: x; x: -; x: -
--

x x x x x x x x x;


--
-- x: x x; x: x; x: -; x:
--

x x x x x 'x x';


--
-- x: x_x_x(x, x); x: x; x: x; x: x
--

x x x.x_x_x(x x, x x) x x
    x x
    x $$
x
    x_x_x(
            x(x_x, x_x),
            x
                x x_x x x x_x
                x x_x x x x_x(x_x) = 'x' x x_x
                x x_x(x_x) <> 'x' x x_x(x_x) <> 'x'
                    x x_x
                x x_x_x(x_x, x_x) x
        )
x
    x_x(x) x1(x_x, x_x)
        x x x_x(x) x2(x_x, x_x) x x_x = x_x
    $$;


x x x.x_x_x(x x, x x) x x x;

x x_x = '';

--
-- x: x; x: x; x: x; x: x
--

x x x.x (
    x x x x.x_x_x() x x,
    x_x x x x,
    x_x x x x x x,
    x_x x x(255) x 'x x'::x x x x,
    x x x(255) x ''::x x x x,
    x_x_1_x x x x,
    x_x_1_x x x(255) x ''::x x x x,
    x_x_1_x2_x_x x x(255) x ''::x x x x,
    x_x_2_x x x x,
    x_x_2_x x x(255) x ''::x x x x,
    x_x_2_x2_x_x x x(255) x ''::x x x x,
    x_x_3_x x x x,
    x_x_3_x x x(255) x ''::x x x x,
    x_x_3_x2_x_x x x(255) x ''::x x x x,
    x_x_4_x x x x,
    x_x_4_x x x(255) x ''::x x x x,
    x_x_4_x2_x_x x x(255) x ''::x x x x,
    x_x x x(10) x ''::x x x x,
    x x x x((0)::x x, (0)::x x) x x,
    x_x2_x_x x x(255) x ''::x x x x,
    x x x(255) x ''::x x x x,
    x_x x x 0 x x,
    x_x x x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);

x x x x.x x x x;


x x x.x x x x;

--
-- x: x_x_x; x: x; x: x; x: x
--

x x x.x_x_x (
    x_x x x,
    x_x x x,
    x_x x x,
    x_x_x x x,
    x_x x x,
    x_x_x x x,
    x_x x x,
    x_x x x,
    x_x x x,
    x_x x x,
    x_x x x,
    x_x x x,
    x_x x x,
    x_x_x_x x x,
    x_x x x,
    x_x_x x x,
    x_x_x x x,
    x x x,
    x_x x x,
    x_x_x_x x x,
    x_x_x_x x x,
    x x,
    x_x x x,
    x_x x x,
    x_x_x x x,
    x_x_x_x x x,
    x_x_x_x x x,
    x_x_x x x,
    x_x_x x x
);

x x x x.x_x_x x x x;


x x x.x_x_x x x x;

--
-- x: x; x: x; x: x; x: x
--

x x x.x (
    x x x x,
    x_x x x x.x_x_x() x x,
    x x,
    x x x 0 x x,
    x x x 0 x x,
    x x x 0 x x,
    x_x x
);

x x x x.x x x x;


x x x.x x x x;

--
-- x: x_x_x; x: x; x: x; x: x
--

x x x.x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x x x x;

--
-- x: x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x x x x.x.x;


--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x (
    x x x x,
    x_x x,
    x_x x x x,
    x x x(255) x 'x'::x x x x,
    x x x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);

x x x x.x_x x x x;


x x x.x_x x x x;

--
-- x: x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x_x x x x;

--
-- x: x_x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x_x x x x.x_x.x;


--
-- x: x; x: x; x: x; x: x
--

x x x.x (
    x x x x,
    x x x(50) x x,
    x_x x,
    x x x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x,
    x_x x x(255)
);

x x x x.x x x x;


x x x.x x x x;

--
-- x: x_x_x; x: x; x: x; x: x
--

x x x.x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x x x x;

--
-- x: x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x x x x.x.x;


--
-- x: x; x: x; x: x; x: x
--

x x x.x (
    x_x x,
    x_x x
);

x x x x.x x x x;


x x x.x x x x;

--
-- x: x; x: x; x: x; x: x
--

x x x.x (
    x x x x,
    x_x x x x.x_x_x() x x,
    x_x x x(20) x x,
    x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);

x x x x.x x x x;


x x x.x x x x;

--
-- x: x x; x: x; x: x; x: x
--

x x x x.x x 'x x x x x, x, x x x_x_x_x - x x';


--
-- x: x_x_x; x: x; x: x; x: x
--

x x x.x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x x x x;

--
-- x: x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x x x x.x.x;


--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x (
    x x x x,
    x_x_x x x x,
    x_x_x x x x,
    x_x x x x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);

x x x x.x_x x x x;


x x x.x_x x x x;

--
-- x: x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x_x x x x;

--
-- x: x_x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x_x x x x.x_x.x;


--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x (
    x x x x,
    x x x x
);


x x x.x_x x x x;

--
-- x: x_x; x: x; x: x; x: x
--

x x x.x_x (
    x x x x,
    x_x x x(255) x x,
    x_x x x(255) x x,
    x_x_x x x '1970-01-01'::x x x,
    x_x x x x x x x() x x,
    x_x x x x x x x() x x
);

x x x x.x_x x x x;


x x x.x_x x x x;

--
-- x: x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x
    x x 1
    x x 1
    x x
    x x
    x 1;


x x x.x_x_x_x x x x;

--
-- x: x_x_x_x; x: x x x; x: x; x: x
--

x x x.x_x_x_x x x x.x_x.x;


--
-- x: x_x_x_x_x_x; x: x; x: x; x: x
--

x x x.x_x_x_x_x_x x
 x x.x,
    x.x,
    x.x,
    x.""x""
   x ( x x.x,
            'x'::x x x,
            x.x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x (x_x x
             x x_x x x ((x.x = x.x)))
          x (x_x_x(x.x, x.x, 'x,x'::x) x (x.x x ( x x.x_x
                   x x_x.x
                  x (((x.x_x)::x <> 'x'::x) x ((x.x_x)::x = 'x'::x)))))
        x x
         x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x, x,x,x,x,x,x'::x) x x_x_x(x.x, x.x, 'x'::x))
        x x
         x x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x, x,x,x,x,x,x'::x) x x_x_x(x.x, x.x, 'x'::x) x (((x.x)::x)::x <> x (x['x'::x, 'x'::x, 'x'::x, 'x'::x, 'x'::x])))
        x x
         x x.x,
            'x'::x x x,
            ((x.x)::x)::x x x,
            (x( x x.x
                   x x(x[
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x,
                        x
                            x x_x_x(x.x, x.x, 'x'::x) x 'x'::x
                            x x::x
                        x]) x(x)
                  x (x.x x x x)))::x x ""x""
           x ((x_x x
             x x_x x x ((x.x = x.x)))
             x x_x x x ((x.x = x.x)))
          x ((x.x <> x (x['x_x'::x, 'x_x'::x, 'x'::x])) x (x.x = 'x'::""x"") x x_x_x(x.x, x.x, 'x,x'::x) x x_x_x(x.x, x.x, 'x'::x))
        x x
         x x_x.x x x,
            'x'::x x x,
            x_x.x_x x x_x,
            x_x.x_x
           x x_x.x_x
          x ((x_x.x_x)::x x ( x x.x_x
                   x x_x.x
                  x (((x.x_x)::x <> 'x'::x) x ((x.x_x)::x = 'x'::x))))) x
  x x x.x, x.x;


x x x.x_x_x_x_x_x x x x;

--
-- x: x x; x: x; x: x; x: x
--

x x x x.x x x x x x x('x.x_x_x'::x);


--
-- x: x_x x; x: x; x: x; x: x
--

x x x x.x_x x x x x x x('x.x_x_x_x'::x);


--
-- x: x x; x: x; x: x; x: x
--

x x x x.x x x x x x x('x.x_x_x'::x);


--
-- x: x x; x: x; x: x; x: x
--

x x x x.x x x x x x x('x.x_x_x'::x);


--
-- x: x_x x; x: x; x: x; x: x
--

x x x x.x_x x x x x x x('x.x_x_x_x'::x);


--
-- x: x_x x; x: x; x: x; x: x
--

x x x x.x_x x x x x x x('x.x_x_x_x'::x);


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x x x (x);


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x x x (x);


--
-- x: x_x x_x_x_x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x_x_x_x_x x (x_x, x_x);


--
-- x: x_x x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x x x (x);


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x_x x (x);


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x x x (x);


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x x x (x);


--
-- x: x_x x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x x x (x);


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x x x.x
    x x x_x_x x (x_x);


--
-- x: x_x x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x x x (x);


--
-- x: x_x x_x_x_x_x_x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x_x_x_x_x_x_x x (x_x_x, x_x_x);


--
-- x: x_x x_x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x_x_x x (x_x);


--
-- x: x_x x_x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x_x x (x_x);


--
-- x: x_x x_x_x; x: x; x: x; x: x
--

x x x x.x_x
    x x x_x_x x x (x);


--
-- x: x_x_x_x; x: x; x: x; x: x
--

x x x_x_x_x x x.x x x (((x ->> 'x'::x)), (((x -> 'x'::x))::x));


--
-- x: x_x_x; x: x; x: x; x: x
--

x x x_x_x x x.x x x (x_x);


--
-- x: x_x_x_x_x_x; x: x; x: x; x: x
--

x x x_x_x_x_x_x x x.x_x x x (x_x_x);


--
-- x: x_x_x_x_x_x_x_x_x; x: x; x: x; x: x
--

x x x x_x_x_x_x_x_x_x_x x x.x_x x x (x_x_x, x_x_x);


--
-- x: x_x_x_x_x; x: x; x: x; x: x
--

x x x_x_x_x_x x x.x x x (((((x -> 'x'::x) ->> 'x_x'::x))::x));


--
-- x: x_x_x; x: x; x: x; x: x
--

x x x_x_x x x.x x x (x_x);


--
-- x: x_x x_x_x_x_x; x: x x; x: x; x: x
--

x x x x.x_x
    x x x_x_x_x_x x x (x_x) x x.x(x);


--
-- x: x_x; x: x; x: -; x: x_x_x
--

x x x_x x x x x (x = 'x, x, x, x');


x x x_x x x x_x_x;

--
--

x x,x x x ""x-x"" x ""x-x_x_x_x"";
x x,x x x ""x-x"" x ""x-x_x_x_x"";
x x x x ""x-x"" x ""x-x_x_x_x_x"";
x x,x x x ""x-x"" x ""x-x_x_x_x_x"";
x x,x x x ""x-x"" x ""x-x_x_x_x_x_x"";
x x,x x x ""x-x"" x ""x-x_x_x_x_x"";
x x x x ""x-x"" x x_x_x;


--
--

x x x x x x x;
x x x x x x ""x-x_x_x_x"";
x x x x x x ""x-x_x_x_x"";
x x x x x x x_x_x;
x x x x x x ""x-x_x_x_x_x"";
x x x x x x ""x-x_x_x_x_x"";
x x x x x x ""x-x_x_x_x_x_x"";
x x x x x x ""x-x_x_x_x_x"";


--
-- x: x x(x); x: x; x: x; x: x
--

x x x x x.x(x) x ""x-x_x_x_x"";


--
-- x: x x(x, x[], x[]); x: x; x: x; x: x
--

x x x x x.x(x, x[], x[]) x ""x-x_x_x_x"";


--
-- x: x x(x, x); x: x; x: x; x: x
--

x x x x x.x(x, x) x ""x-x_x_x_x"";


--
-- x: x x(x); x: x; x: x; x: x
--

x x x x x.x(x) x ""x-x_x_x_x"";


--
-- x: x x(x, x, x); x: x; x: x; x: x
--

x x x x x.x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x(x, x, x, x); x: x; x: x; x: x
--

x x x x x.x_x(x, x, x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x); x: x; x: x; x: x
--

x x x x x.x(x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x); x: x; x: x; x: x
--

x x x x x.x(x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x, x); x: x; x: x; x: x
--

x x x x x.x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x(x, x, x, x); x: x; x: x; x: x
--

x x x x x.x_x(x, x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x); x: x; x: x; x: x
--

x x x x x.x_x_x(x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(); x: x; x: x; x: x
--

x x x x x.x_x_x() x ""x-x_x_x_x"";


--
-- x: x x_x(x); x: x; x: x; x: x
--

x x x x x.x_x(x) x ""x-x_x_x_x"";


--
-- x: x x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x, x); x: x; x: x; x: x
--

x x x x x.x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x(x, x, x); x: x; x: x; x: x
--

x x x x x.x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x x, x x); x: x; x: x; x: x
--

x x x x x.x_x_x(x x, x x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x x x, x x x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x x x, x x x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x); x: x; x: x; x: x
--

x x x x x.x_x_x(x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x) x ""x-x_x_x_x"";


--
-- x: x x_x_x_x(x, x, x); x: x; x: x; x: x
--

x x x x x.x_x_x_x(x, x, x) x ""x-x_x_x_x"";


--
-- x: x x; x: x; x: x; x: x
--

x x x x x.x x ""x-x_x_x_x"";
x x,x,x,x x x x.x x ""x-x_x_x_x"";


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x,x,x,x x x x.x_x_x x ""x-x_x_x_x"";
x x x x x.x_x_x x ""x-x_x_x_x"";
x x,x,x,x,x x x x.x_x_x x ""x-x_x_x_x_x"";


--
-- x: x x; x: x; x: x; x: x
--

x x,x,x,x x x x.x x ""x-x_x_x_x"";
x x x x x.x x ""x-x_x_x_x"";
x x,x,x x x x.x x ""x-x_x_x_x_x_x"";


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x x ""x-x_x_x_x"";
x x,x x x x.x_x_x x ""x-x_x_x_x_x_x"";


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x x.x_x x ""x-x_x_x_x"";
x x,x,x,x x x x.x_x x ""x-x_x_x_x"";
x x,x,x,x,x x x x.x_x x ""x-x_x_x_x_x"";


--
-- x: x x_x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x_x x ""x-x_x_x_x"";
x x,x x x x.x_x_x_x x ""x-x_x_x_x_x"";


--
-- x: x x; x: x; x: x; x: x
--

x x x x x.x x ""x-x_x_x_x"";
x x,x,x,x x x x.x x ""x-x_x_x_x"";


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x x ""x-x_x_x_x"";


--
-- x: x x; x: x; x: x; x: x
--

x x x x x.x x ""x-x_x_x_x"";
x x,x,x,x x x x.x x ""x-x_x_x_x"";


--
-- x: x x; x: x; x: x; x: x
--

x x,x,x,x x x x.x x ""x-x_x_x_x"";
x x x x x.x x ""x-x_x_x_x"";
x x,x,x x x x.x x x_x_x_x_x;
x x,x,x x x x.x x ""x-x_x_x_x_x"";
x x,x,x x x x.x x ""x-x_x_x_x_x_x"";


--
-- x: x x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x x ""x-x_x_x_x"";
x x,x x x x.x_x_x x ""x-x_x_x_x_x"";
x x,x x x x.x_x_x x ""x-x_x_x_x_x"";
x x,x x x x.x_x_x x ""x-x_x_x_x_x_x"";


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x x.x_x x ""x-x_x_x_x"";
x x,x,x,x x x x.x_x x ""x-x_x_x_x"";


--
-- x: x x_x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x_x x ""x-x_x_x_x"";


--
-- x: x x_x; x: x; x: x; x: x
--

x x,x,x,x x x x.x_x x ""x-x_x_x_x"";
x x x x x.x_x x ""x-x_x_x_x"";


--
-- x: x x_x; x: x; x: x; x: x
--

x x x x x.x_x x ""x-x_x_x_x"";
x x,x,x,x x x x.x_x x ""x-x_x_x_x"";
x x,x,x,x,x x x x.x_x x ""x-x_x_x_x_x"";


--
-- x: x x_x_x_x; x: x; x: x; x: x
--

x x,x x x x.x_x_x_x x ""x-x_x_x_x"";
x x,x x x x.x_x_x_x x ""x-x_x_x_x_x"";


--
-- x: x x_x_x_x_x_x; x: x; x: x; x: x
--

x x x x x.x_x_x_x_x_x x x_x_x;
x x,x,x,x x x x.x_x_x_x_x_x x ""x-x_x_x_x"";
x x x x x.x_x_x_x_x_x x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x,x x x  x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x x x  x ""x-x_x_x_x"";


--
-- x: x x x x; x: x x; x: -; x: x
--

x x x x x x x x x x  x ""x-x_x_x_x"";
x x x x x x x x,x,x,x x x  x ""x-x_x_x_x"";
""#);
    let df= CsvReader::new(csv).finish().unwrap();
    dbg!(df);
}