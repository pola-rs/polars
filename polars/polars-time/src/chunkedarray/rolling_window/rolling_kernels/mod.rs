pub(super) mod no_nulls;
use std::fmt::Debug;
use std::ops::{AddAssign, Div, Mul, Sub, SubAssign};

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use polars_arrow::data_types::IsFloat;
use polars_arrow::export::arrow;
use polars_arrow::index::IdxSize;
use polars_arrow::trusted_len::TrustedLen;
use polars_core::export::num::{Bounded, Float, NumCast, One};
use polars_core::prelude::*;

use crate::prelude::*;
