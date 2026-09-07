use polars::prelude::*;
use pyo3::prelude::*;
use pyo3::types::{PyNone, PyTuple};

use super::*;
use crate::error::PyPolarsErr;
use crate::prelude::ObjectValue;
use crate::{PySeries, Wrap};

pub trait ApplyLambdaGeneric<'py> {
    fn apply_generic(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        skip_nulls: bool,
    ) -> PyResult<Series>;

    fn apply_generic_with_dtype(
        &self,
        py: Python<'py>,
        lambda: &Bound<'py, PyAny>,
        datatype: &DataType,
        skip_nulls: bool,
    ) -> PyResult<Series>;
}

fn call_and_collect_anyvalues<'py, T, I>(
    py: Python<'py>,
    lambda: &Bound<'py, PyAny>,
    len: usize,
    iter: I,
    skip_nulls: bool,
) -> PyResult<Vec<AnyValue<'static>>>
where
    T: IntoPyObject<'py>,
    I: Iterator<Item = Option<T>>,
{
    let mut avs = Vec::with_capacity(len);
    for opt_val in iter {
        let arg = match opt_val {
            None if skip_nulls => {
                avs.push(AnyValue::Null);
                continue;
            },
            None => PyTuple::new(py, [PyNone::get(py)])?,
            Some(val) => PyTuple::new(py, [val])?,
        };
        let out = lambda.call1(arg)?;
        let av: Option<Wrap<AnyValue>> = if out.is_none() {
            Ok(None)
        } else {
            out.extract().map(Some)
        }?;
        avs.push(av.map(|w| w.0).unwrap_or(AnyValue::Null));
    }
    Ok(avs)
}

/// Implement [`ApplyLambdaGeneric`]. The impls differ only in how a row becomes the
/// lambda's argument; `$rows` is that iterator, with `$ca` bound to `self`.
macro_rules! impl_apply_lambda_generic {
    ({$($impl_header:tt)*}, |$ca:ident| $rows:expr) => {
        $($impl_header)* {
            fn apply_generic(
                &self,
                py: Python<'py>,
                lambda: &Bound<'py, PyAny>,
                skip_nulls: bool,
            ) -> PyResult<Series> {
                let $ca = self;
                let avs = call_and_collect_anyvalues(py, lambda, self.len(), $rows, skip_nulls)?;
                Ok(Series::from_any_values(self.name().clone(), &avs, true)
                    .map_err(PyPolarsErr::from)?)
            }

            fn apply_generic_with_dtype(
                &self,
                py: Python<'py>,
                lambda: &Bound<'py, PyAny>,
                datatype: &DataType,
                skip_nulls: bool,
            ) -> PyResult<Series> {
                let $ca = self;
                let avs = call_and_collect_anyvalues(py, lambda, self.len(), $rows, skip_nulls)?;
                Ok(
                    Series::from_any_values_and_dtype(self.name().clone(), &avs, datatype, true)
                        .map_err(PyPolarsErr::from)?,
                )
            }
        }
    };
}

impl_apply_lambda_generic!(
    { impl<'py> ApplyLambdaGeneric<'py> for BooleanChunked },
    |ca| ca.iter()
);

impl_apply_lambda_generic!(
    {
        impl<'py, T> ApplyLambdaGeneric<'py> for ChunkedArray<T>
        where
            T: PyPolarsNumericType,
            T::Native: IntoPyObject<'py> + for<'a> FromPyObject<'a, 'py>,
    },
    |ca| ca.iter()
);

impl_apply_lambda_generic!(
    { impl<'py> ApplyLambdaGeneric<'py> for StringChunked },
    |ca| ca.iter()
);

impl_apply_lambda_generic!(
    { impl<'py> ApplyLambdaGeneric<'py> for BinaryChunked },
    |ca| ca.iter()
);

impl_apply_lambda_generic!(
    { impl<'py> ApplyLambdaGeneric<'py> for ListChunked },
    |ca| ca.series_iter().map(|opt_s| opt_s.map(Wrap))
);

impl_apply_lambda_generic!(
    {
        #[cfg(feature = "dtype-array")]
        impl<'py> ApplyLambdaGeneric<'py> for ArrayChunked
    },
    |ca| ca.series_iter().map(|opt_s| Some(PySeries::new(opt_s?)))
);

impl_apply_lambda_generic!(
    {
        #[cfg(feature = "object")]
        impl<'py> ApplyLambdaGeneric<'py> for ObjectChunked<ObjectValue>
    },
    |ca| ca.iter()
);

impl_apply_lambda_generic!(
    { impl<'py> ApplyLambdaGeneric<'py> for NullChunked },
    |ca| (0..ca.len()).map(|_| None::<Wrap<AnyValue<'static>>>)
);

// The remaining three have no cheap per-row representation, so they materialize an
// `AnyValue`.
impl_apply_lambda_generic!(
    { impl<'py> ApplyLambdaGeneric<'py> for MapChunked },
    |ca| ca.any_value_iter().map(|av| av.null_to_none().map(Wrap))
);

impl_apply_lambda_generic!(
    { impl<'py> ApplyLambdaGeneric<'py> for StructChunked },
    |ca| (0..ca.len()).map(|i| unsafe { ca.get_any_value_unchecked(i).null_to_none().map(Wrap) })
);

impl_apply_lambda_generic!(
    {
        impl<'py, L, P> ApplyLambdaGeneric<'py> for Logical<L, P>
        where
            L: PolarsDataType,
            P: PolarsDataType,
            Logical<L, P>: LogicalType,
    },
    |ca| (0..ca.len()).map(|i| unsafe { ca.get_any_value_unchecked(i).null_to_none().map(Wrap) })
);

impl<'py> ApplyLambdaGeneric<'py> for ExtensionChunked {
    fn apply_generic(
        &self,
        _py: Python<'py>,
        _lambda: &Bound<'py, PyAny>,
        _skip_nulls: bool,
    ) -> PyResult<Series> {
        unreachable!()
    }

    fn apply_generic_with_dtype(
        &self,
        _py: Python<'py>,
        _lambda: &Bound<'py, PyAny>,
        _datatype: &DataType,
        _skip_nulls: bool,
    ) -> PyResult<Series> {
        unreachable!()
    }
}
