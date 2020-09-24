use crate::series::PySeries;
use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::PyTuple;

pub trait PyPolarsNumericType: PolarsNumericType {}

impl PyPolarsNumericType for UInt8Type {}
impl PyPolarsNumericType for UInt16Type {}
impl PyPolarsNumericType for UInt32Type {}
impl PyPolarsNumericType for UInt64Type {}
impl PyPolarsNumericType for Int8Type {}
impl PyPolarsNumericType for Int16Type {}
impl PyPolarsNumericType for Int32Type {}
impl PyPolarsNumericType for Int64Type {}
impl PyPolarsNumericType for Float32Type {}
impl PyPolarsNumericType for Float64Type {}
impl PyPolarsNumericType for Date32Type {}
impl PyPolarsNumericType for Date64Type {}
impl PyPolarsNumericType for Time64NanosecondType {}
impl PyPolarsNumericType for Time64MicrosecondType {}
impl PyPolarsNumericType for Time32MillisecondType {}
impl PyPolarsNumericType for Time32SecondType {}
impl PyPolarsNumericType for DurationNanosecondType {}
impl PyPolarsNumericType for DurationMicrosecondType {}
impl PyPolarsNumericType for DurationMillisecondType {}
impl PyPolarsNumericType for DurationSecondType {}
impl PyPolarsNumericType for IntervalYearMonthType {}
impl PyPolarsNumericType for IntervalDayTimeType {}
impl PyPolarsNumericType for TimestampNanosecondType {}
impl PyPolarsNumericType for TimestampMicrosecondType {}
impl PyPolarsNumericType for TimestampMillisecondType {}
impl PyPolarsNumericType for TimestampSecondType {}

pub trait ApplyLambda<'a> {
    fn apply_lambda(&self, _py: Python, _lambda: &'a PyAny) -> PyResult<PySeries> {
        unimplemented!()
    }
}

impl<'a, T> ApplyLambda<'a> for ChunkedArray<T>
where
    T: PyPolarsNumericType,
    T::Native: ToPyObject + FromPyObject<'a>,
{
    fn apply_lambda(&self, py: Python, lambda: &'a PyAny) -> PyResult<PySeries> {
        let ca = if self.null_count() == 0 {
            let mut it = self.into_no_null_iter();
            let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());

            while let Some(v) = it.next() {
                let arg = PyTuple::new(py, &[v]);
                let out = lambda.call1(arg)?;
                builder.append_value(out.extract()?)
            }
            builder.finish()
        } else {
            let mut it = self.into_iter();
            let mut builder = PrimitiveChunkedBuilder::<T>::new(self.name(), self.len());
            while let Some(opt_v) = it.next() {
                if let Some(v) = opt_v {
                    let arg = PyTuple::new(py, &[v]);
                    let out = lambda.call1(arg)?;
                    builder.append_value(out.extract()?)
                } else {
                    builder.append_null()
                }
            }
            builder.finish()
        };
        Ok(PySeries::new(ca.into_series()))
    }
}

impl<'a> ApplyLambda<'a> for LargeListChunked {}
impl<'a> ApplyLambda<'a> for BooleanChunked {}
impl<'a> ApplyLambda<'a> for Utf8Chunked {}
