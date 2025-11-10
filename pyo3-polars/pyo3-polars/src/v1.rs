use std::borrow::Cow;

#[cfg(feature = "serde")]
use polars::prelude::{Field, Schema};
#[cfg(feature = "serde")]
use polars::series::Series;
use polars_error::polars_err;
#[cfg(feature = "serde")]
use polars_error::PolarsResult;
use polars_ffi::version_1::GroupPositions;

#[macro_export]
macro_rules! polars_plugin_expr_info {
    (
        $name:literal, $data:expr, $data_ty:ty
    ) => {{
        #[unsafe(export_name = concat!("_PL_PLUGIN_V2::", $name))]
        pub static VTABLE: $crate::export::polars_ffi::version_1::PluginSymbol =
            $crate::export::polars_ffi::version_1::VTable::new::<$data_ty>().into_symbol();

        let data = ::std::boxed::Box::new($data);
        let data = ::std::boxed::Box::into_raw(data);
        $crate::v1::PolarsPluginExprInfo::_new($name, data as *const u8)
    }};
}

pub struct PolarsPluginExprInfo {
    symbol: &'static str,
    data_ptr: *const u8,
}

impl PolarsPluginExprInfo {
    #[doc(hidden)]
    pub fn _new(symbol: &'static str, data_ptr: *const u8) -> Self {
        Self { symbol, data_ptr }
    }
}

impl<'py> pyo3::IntoPyObject<'py> for PolarsPluginExprInfo {
    type Target = pyo3::types::PyTuple;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        use pyo3::IntoPyObjectExt;
        pyo3::types::PyTuple::new(
            py,
            [
                self.symbol.into_py_any(py)?,
                (self.data_ptr as usize).into_py_any(py)?,
            ],
        )
    }
}

pub mod elementwise {
    use super::*;

    #[cfg(feature = "serde")]
    pub trait PolarsElementwisePlugin:
        Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static
    {
        fn to_field(&self, fields: &Schema) -> PolarsResult<Field>;
        fn evaluate(&self, inputs: &[Series]) -> PolarsResult<Series>;
    }

    pub struct Plugin<T>(pub T);

    impl<T: PolarsElementwisePlugin> polars_ffi::version_1::PolarsPlugin for Plugin<T> {
        type State = ();

        fn serialize(&self) -> PolarsResult<Box<[u8]>> {
            Ok(
                bincode::serde::encode_to_vec(&self.0, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to serialize: {err}"))?
                    .into(),
            )
        }

        fn deserialize(buff: &[u8]) -> PolarsResult<Self> {
            let (data, num_bytes) =
                bincode::serde::decode_from_slice(buff, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to deserialize: {err}"))?;
            assert_eq!(num_bytes, buff.len());
            Ok(Plugin(data))
        }

        fn serialize_state(&self, _state: &Self::State) -> PolarsResult<Box<[u8]>> {
            Ok(Box::default())
        }

        fn deserialize_state(&self, _buffer: &[u8]) -> PolarsResult<Self::State> {
            Ok(())
        }

        fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
            self.0.to_field(fields)
        }

        fn new_state(&self, _fields: &Schema) -> PolarsResult<Self::State> {
            Ok(())
        }

        fn new_empty(&self, _state: &Self::State) -> PolarsResult<Self::State> {
            Ok(())
        }

        fn reset(&self, _state: &mut Self::State) -> PolarsResult<()> {
            Ok(())
        }

        fn combine(&self, _state: &mut Self::State, _other: &Self::State) -> PolarsResult<()> {
            unreachable!()
        }

        fn step(
            &self,
            _state: &mut Self::State,
            inputs: &[Series],
        ) -> PolarsResult<Option<Series>> {
            self.0.evaluate(inputs).map(Some)
        }

        fn finalize(&self, _state: &mut Self::State) -> PolarsResult<Option<Series>> {
            unreachable!()
        }

        unsafe fn evaluate_on_groups<'a>(
            &self,
            _inputs: &[(Series, &'a GroupPositions)],
        ) -> PolarsResult<(Series, Cow<'a, GroupPositions>)> {
            unreachable!()
        }
    }
}

pub mod map_reduce {
    use super::*;

    #[cfg(feature = "serde")]
    pub trait PolarsMapReducePlugin:
        Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static
    {
        type State: Default
            + Send
            + Sync
            + serde::Serialize
            + for<'de> serde::Deserialize<'de>
            + 'static;

        fn to_field(&self, fields: &Schema) -> PolarsResult<Field>;
        fn map(&self, inputs: &[Series]) -> PolarsResult<Self::State>;
        fn reduce(&self, left: &Self::State, right: &Self::State) -> PolarsResult<Self::State>;
        fn finalize(&self, state: Self::State) -> PolarsResult<Series>;
    }

    pub struct Plugin<T>(pub T);

    impl<T: PolarsMapReducePlugin> polars_ffi::version_1::PolarsPlugin for Plugin<T> {
        type State = T::State;

        fn serialize(&self) -> PolarsResult<Box<[u8]>> {
            Ok(
                bincode::serde::encode_to_vec(&self.0, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to serialize: {err}"))?
                    .into(),
            )
        }

        fn deserialize(buff: &[u8]) -> PolarsResult<Self> {
            let (data, num_bytes) =
                bincode::serde::decode_from_slice(buff, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to deserialize: {err}"))?;
            assert_eq!(num_bytes, buff.len());
            Ok(Plugin(data))
        }

        fn serialize_state(&self, state: &Self::State) -> PolarsResult<Box<[u8]>> {
            Ok(
                bincode::serde::encode_to_vec(&state, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to serialize: {err}"))?
                    .into(),
            )
        }

        fn deserialize_state(&self, buffer: &[u8]) -> PolarsResult<Self::State> {
            let (data, num_bytes) =
                bincode::serde::decode_from_slice(buffer, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to deserialize: {err}"))?;
            assert_eq!(num_bytes, buffer.len());
            Ok(data)
        }

        fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
            self.0.to_field(fields)
        }

        fn new_state(&self, _fields: &Schema) -> PolarsResult<Self::State> {
            Ok(Self::State::default())
        }

        fn new_empty(&self, _state: &Self::State) -> PolarsResult<Self::State> {
            Ok(Self::State::default())
        }

        fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
            *state = Self::State::default();
            Ok(())
        }

        fn combine(&self, state: &mut Self::State, other: &Self::State) -> PolarsResult<()> {
            *state = self.0.reduce(state, &other)?;
            Ok(())
        }

        fn step(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
            let other = self.0.map(inputs)?;
            *state = self.0.reduce(state, &other)?;
            Ok(None)
        }

        fn finalize(&self, state: &mut Self::State) -> PolarsResult<Option<Series>> {
            Ok(Some(self.0.finalize(std::mem::take(state))?))
        }

        unsafe fn evaluate_on_groups<'a>(
            &self,
            _inputs: &[(Series, &'a GroupPositions)],
        ) -> PolarsResult<(Series, Cow<'a, GroupPositions>)> {
            unreachable!()
        }
    }
}

pub mod scan {
    use super::*;

    #[cfg(feature = "serde")]
    pub trait PolarsScanPlugin:
        Send + Sync + serde::Serialize + for<'de> serde::Deserialize<'de> + 'static
    {
        type State: Clone
            + Send
            + Sync
            + serde::Serialize
            + for<'de> serde::Deserialize<'de>
            + 'static;

        fn to_field(&self, fields: &Schema) -> PolarsResult<Field>;
        fn new_state(&self, fields: &Schema) -> PolarsResult<Self::State>;
        fn reset(&self, state: &mut Self::State) -> PolarsResult<()>;
        fn step(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Series>;
    }

    pub struct Plugin<T>(pub T);

    impl<T: PolarsScanPlugin> polars_ffi::version_1::PolarsPlugin for Plugin<T> {
        type State = T::State;

        fn serialize(&self) -> PolarsResult<Box<[u8]>> {
            Ok(
                bincode::serde::encode_to_vec(&self.0, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to serialize: {err}"))?
                    .into(),
            )
        }

        fn deserialize(buff: &[u8]) -> PolarsResult<Self> {
            let (data, num_bytes) =
                bincode::serde::decode_from_slice(buff, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to deserialize: {err}"))?;
            assert_eq!(num_bytes, buff.len());
            Ok(Plugin(data))
        }

        fn serialize_state(&self, state: &Self::State) -> PolarsResult<Box<[u8]>> {
            Ok(
                bincode::serde::encode_to_vec(&state, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to serialize: {err}"))?
                    .into(),
            )
        }

        fn deserialize_state(&self, buffer: &[u8]) -> PolarsResult<Self::State> {
            let (data, num_bytes) =
                bincode::serde::decode_from_slice(buffer, bincode::config::standard())
                    .map_err(|err| polars_err!(InvalidOperation: "failed to deserialize: {err}"))?;
            assert_eq!(num_bytes, buffer.len());
            Ok(data)
        }

        fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
            self.0.to_field(fields)
        }

        fn new_state(&self, fields: &Schema) -> PolarsResult<Self::State> {
            self.0.new_state(fields)
        }

        fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State> {
            let mut state = state.clone();
            self.0.reset(&mut state)?;
            Ok(state)
        }

        fn reset(&self, state: &mut Self::State) -> PolarsResult<()> {
            self.0.reset(state)
        }

        fn combine(&self, _state: &mut Self::State, _other: &Self::State) -> PolarsResult<()> {
            unreachable!()
        }

        fn step(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>> {
            self.0.step(state, inputs).map(Some)
        }

        fn finalize(&self, _state: &mut Self::State) -> PolarsResult<Option<Series>> {
            unreachable!()
        }

        unsafe fn evaluate_on_groups<'a>(
            &self,
            _inputs: &[(Series, &'a GroupPositions)],
        ) -> PolarsResult<(Series, Cow<'a, GroupPositions>)> {
            unreachable!()
        }
    }
}
