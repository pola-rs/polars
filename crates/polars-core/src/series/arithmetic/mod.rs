mod bitops;
mod borrowed;
mod list_borrowed;
mod owned;

use std::borrow::Cow;
use std::ops::{Add, Div, Mul, Rem, Sub};

pub use borrowed::*;
#[cfg(feature = "dtype-array")]
pub use fixed_size_list::NumericFixedSizeListOp;
pub use list_borrowed::NumericListOp;
use num_traits::{Num, NumCast};
#[cfg(feature = "dtype-array")]
mod fixed_size_list;
mod list_utils;

use crate::prelude::*;
use crate::utils::{get_time_units, try_get_supertype};
