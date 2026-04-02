pub mod lazy;
pub mod series;

use polars::prelude::*;
pub trait PyPolarsNumericType: PolarsNumericType {}

impl PyPolarsNumericType for UInt8Type {}
impl PyPolarsNumericType for UInt16Type {}
impl PyPolarsNumericType for UInt32Type {}
impl PyPolarsNumericType for UInt64Type {}
impl PyPolarsNumericType for UInt128Type {}
impl PyPolarsNumericType for Int8Type {}
impl PyPolarsNumericType for Int16Type {}
impl PyPolarsNumericType for Int32Type {}
impl PyPolarsNumericType for Int64Type {}
impl PyPolarsNumericType for Int128Type {}
impl PyPolarsNumericType for Float16Type {}
impl PyPolarsNumericType for Float32Type {}
impl PyPolarsNumericType for Float64Type {}
