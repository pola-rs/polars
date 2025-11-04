use std::ptr::NonNull;
use std::sync::Arc;

use polars::prelude::v2::{DataPtr, StatefulUdfTrait, UdfV2Flags, new_udf_vtable};
use polars::prelude::{DataType, Expr, Field, FunctionExpr, Schema, SpecialEq};
use polars::series::Series;
use polars_error::PolarsResult;
use pyo3::{Py, PyAny, PyResult, Python};

use crate::prelude::Wrap;
use crate::{PyExpr, PySeries};

#[pyo3::pyfunction]
pub fn plugin_v2_generate(
    inputs: Vec<PyExpr>,
    data: Py<PyAny>,
    initialize: Py<PyAny>,
    insert: Py<PyAny>,
    finalize: Option<Py<PyAny>>,
    combine: Option<Py<PyAny>>,
    new_empty: Py<PyAny>,

    to_field: Py<PyAny>,
    format: String,

    length_preserving: bool,
    row_separable: bool,
    returns_scalar: bool,
    zippable_inputs: bool,
    insert_has_output: bool,
    selection_expansion: bool,
) -> PyResult<PyExpr> {
    struct Data {
        data: Py<PyAny>,
        initialize: Py<PyAny>,
        insert: Py<PyAny>,
        finalize: Option<Py<PyAny>>,
        combine: Option<Py<PyAny>>,
        new_empty: Py<PyAny>,
        to_field: Py<PyAny>,
    }
    struct State(Py<PyAny>);

    impl StatefulUdfTrait for Data {
        type State = State;

        fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
            let (name, dtype) = Python::attach(|py| {
                PolarsResult::Ok(
                    self.to_field
                        .call1(py, (self.data.clone_ref(py), Wrap(fields.clone())))?
                        .extract::<(String, Wrap<DataType>)>(py)?,
                )
            })?;
            Ok(Field::new(name.into(), dtype.0))
        }

        fn initialize(&self, fields: &Schema) -> PolarsResult<Self::State> {
            Python::attach(|py| {
                Ok(State(self.initialize.call1(
                    py,
                    (self.data.clone_ref(py), Wrap(fields.clone())),
                )?))
            })
        }
        fn insert(
            &self,
            state: &mut Self::State,
            inputs: &[Series],
        ) -> PolarsResult<Option<Series>> {
            let inputs: Vec<PySeries> = inputs.iter().map(|i| PySeries::from(i.clone())).collect();
            Python::attach(|py| {
                Ok(self
                    .insert
                    .call1(py, (self.data.clone_ref(py), state.0.clone_ref(py), inputs))?
                    .extract::<Option<PySeries>>(py)?
                    .map(|s| s.series.read().clone()))
            })
        }
        fn finalize(&self, state: &mut Self::State) -> PolarsResult<Option<Series>> {
            match &self.finalize {
                None => unreachable!(),
                Some(finalize) => Python::attach(|py| {
                    Ok(finalize
                        .call1(py, (self.data.clone_ref(py), state.0.clone_ref(py)))?
                        .extract::<Option<PySeries>>(py)?
                        .map(|s| s.series.read().clone()))
                }),
            }
        }
        fn combine(&self, state: &mut Self::State, other: &Self::State) -> PolarsResult<()> {
            match &self.combine {
                None => unreachable!(),
                Some(combine) => Python::attach(|py| {
                    combine.call1(
                        py,
                        (
                            self.data.clone_ref(py),
                            state.0.clone_ref(py),
                            other.0.clone_ref(py),
                        ),
                    )?;
                    PolarsResult::Ok(())
                }),
            }
        }
        fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State> {
            Python::attach(|py| {
                Ok(State(self.new_empty.call1(
                    py,
                    (self.data.clone_ref(py), state.0.clone_ref(py)),
                )?))
            })
        }
        fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
            *state = self.new_empty(state)?;
            Ok(())
        }
    }

    let vtable = new_udf_vtable::<Data>();

    let mut flags = UdfV2Flags::empty();
    flags.set(UdfV2Flags::LENGTH_PRESERVING, length_preserving);
    flags.set(UdfV2Flags::ROW_SEPARABLE, row_separable);
    flags.set(UdfV2Flags::RETURNS_SCALAR, returns_scalar);
    flags.set(UdfV2Flags::ZIPPABLE_INPUTS, zippable_inputs);
    flags.set(UdfV2Flags::INSERT_HAS_OUTPUT, insert_has_output);
    flags.set(UdfV2Flags::NEEDS_FINALIZE, finalize.is_some());
    flags.set(UdfV2Flags::STATES_COMBINABLE, combine.is_some());
    flags.set(UdfV2Flags::SELECTOR_EXPANSION, selection_expansion);

    let data = Data {
        data,
        initialize,
        insert,
        finalize,
        combine,
        new_empty,
        to_field,
    };

    let udf = unsafe {
        vtable.new_udf(
            DataPtr::_new(NonNull::new(Box::into_raw(Box::new(data)) as *mut u8).unwrap()),
            flags,
            format.into(),
        )
    };
    let udf = Arc::new(udf);

    Ok(Expr::Function {
        input: inputs.into_iter().map(|e| e.inner).collect(),
        function: FunctionExpr::PluginV2(SpecialEq::new(udf)),
    }
    .into())
}
