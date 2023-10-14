pub(super) mod no_nulls;
use std::fmt::Debug;
use std::ops::{AddAssign, Mul, SubAssign};

use arrow::array::PrimitiveArray;
use arrow::types::NativeType;
use arrow::legacy::data_types::IsFloat;
use arrow::legacy::index::IdxSize;
use arrow::legacy::trusted_len::TrustedLen;
use polars_core::export::num::{Bounded, Float, NumCast};
use polars_core::prelude::*;

use crate::prelude::*;
