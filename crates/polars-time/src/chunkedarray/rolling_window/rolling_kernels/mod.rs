pub(super) mod no_nulls;
use std::fmt::Debug;
use std::ops::{AddAssign, Mul, SubAssign};

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::legacy::index::IdxSize;
use arrow::legacy::trusted_len::TrustedLen;
use arrow::types::NativeType;
use polars_core::export::num::{Bounded, Float, NumCast};
use polars_core::prelude::*;
use polars_utils::float::IsFloat;

use crate::prelude::*;
