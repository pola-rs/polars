use polars::prelude::*;

pub trait JSPolarsNumericType: PolarsNumericType {}
impl JSPolarsNumericType for UInt8Type {}
impl JSPolarsNumericType for UInt16Type {}
impl JSPolarsNumericType for UInt32Type {}
impl JSPolarsNumericType for UInt64Type {}
impl JSPolarsNumericType for Int8Type {}
impl JSPolarsNumericType for Int16Type {}
impl JSPolarsNumericType for Int32Type {}
impl JSPolarsNumericType for Int64Type {}
impl JSPolarsNumericType for Float32Type {}
impl JSPolarsNumericType for Float64Type {}
