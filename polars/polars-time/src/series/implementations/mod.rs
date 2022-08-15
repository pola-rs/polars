mod boolean;
#[cfg(feature = "dtype-categorical")]
mod categoricals;
#[cfg(feature = "dtype-date")]
mod date;
#[cfg(feature = "dtype-datetime")]
mod datetime;
#[cfg(feature = "dtype-duration")]
mod duration;
mod floats;
mod integers;
mod list;
#[cfg(feature = "object")]
mod object;
#[cfg(feature = "dtype-struct")]
mod struct_;
#[cfg(feature = "dtype-time")]
mod time;
mod utf8;

use polars_core::prelude::*;
use polars_core::utils::Wrap;

use crate::prelude::*;
use crate::series::*;
