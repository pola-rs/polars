use std::borrow::Cow;
use std::sync::Arc;

use arrow::datatypes::Field as ArrowField;
use arrow::io::ipc::IpcField;
use polars_core::prelude::{
    CategoricalMapping, CategoricalPhysicalDtypeExt, CompatLevel, DataType,
};
use polars_core::schema::Schema;
use polars_core::series::ToArrowConverter;
use polars_core::series::categorical_to_arrow::CategoricalToArrowConverter;
use polars_core::utils::arrow;
use polars_utils::pl_str::PlSmallStr;

#[expect(clippy::type_complexity)]
pub fn build_ipc_write_components(
    file_schema: &Schema,
    compat_level: CompatLevel,
) -> (
    Vec<(ToArrowConverter, ArrowField)>,
    Vec<IpcField>,
    Arc<[usize]>,
) {
    let arrow_converters: Vec<(ToArrowConverter, ArrowField)> = file_schema
        .iter_values()
        .map(|dtype| {
            let mut categorical_converter = CategoricalToArrowConverter {
                converters: Default::default(),
                persist_remap: true,
            };
            categorical_converter.initialize(dtype);
            let converter = ToArrowConverter {
                skip_attach_pl_metadata: false,
                categorical_converter,
            };

            let arrow_field = DataType::try_mutate_with(Cow::Borrowed(dtype), |dtype| {
                Ok(dtype
                    .cat_physical()
                    .ok()
                    .map_or(dtype, |physical| Cow::Owned(physical.dtype())))
            })
            .unwrap()
            .to_arrow_field(PlSmallStr::EMPTY, compat_level);

            (converter, arrow_field)
        })
        .collect();

    let dictionary_id_offsets: Arc<[usize]> =
        dictionary_id_offsets_iter(&arrow_converters).collect();

    let ipc_fields: Vec<IpcField> = file_schema
        .iter_values()
        .zip(&arrow_converters)
        .zip(dictionary_id_offsets.iter().copied())
        .map(|((dtype, (arrow_converter, _)), dictionary_id_offset)| {
            IpcFieldConverter {
                get_dictionary_id: |mapping: &Arc<CategoricalMapping>| {
                    let converter_key: usize = Arc::as_ptr(mapping) as *const () as _;
                    let converter_index: usize = arrow_converter
                        .categorical_converter
                        .converters
                        .get_index_of(&converter_key)
                        .unwrap();

                    i64::try_from(dictionary_id_offset + converter_index).unwrap()
                },
            }
            .dtype_to_ipc_field(dtype)
        })
        .collect();

    (arrow_converters, ipc_fields, dictionary_id_offsets)
}

/// Cumulative sum, excluding the current element.
///
/// Indicates total number of dictionaries in the columns to the left of the current one.
fn dictionary_id_offsets_iter<T>(
    arrow_converters: &[(ToArrowConverter, T)],
) -> impl Iterator<Item = usize> {
    arrow_converters
        .iter()
        .scan(0, |acc: &mut usize, (arrow_converter, _)| {
            let out = *acc;
            *acc += arrow_converter.categorical_converter.converters.len();
            Some(out)
        })
}

struct IpcFieldConverter<F>
where
    F: Fn(&Arc<CategoricalMapping>) -> i64,
{
    get_dictionary_id: F,
}

impl<F> IpcFieldConverter<F>
where
    F: Fn(&Arc<CategoricalMapping>) -> i64,
{
    fn dtype_to_ipc_field(&self, dtype: &DataType) -> IpcField {
        use DataType::*;

        match dtype {
            #[cfg(feature = "dtype-categorical")]
            Categorical(_, mapping) | Enum(_, mapping) => IpcField {
                fields: vec![self.dtype_to_ipc_field(&DataType::String)],
                dictionary_id: Some((self.get_dictionary_id)(mapping)),
            },
            List(inner) => IpcField {
                fields: vec![self.dtype_to_ipc_field(inner)],
                dictionary_id: None,
            },
            #[cfg(feature = "dtype-array")]
            Array(inner, _width) => IpcField {
                fields: vec![self.dtype_to_ipc_field(inner)],
                dictionary_id: None,
            },
            Struct(fields) => IpcField {
                fields: fields
                    .iter()
                    .map(|x| self.dtype_to_ipc_field(x.dtype()))
                    .collect(),
                dictionary_id: None,
            },
            #[cfg(feature = "dtype-extension")]
            Extension(_, storage) => self.dtype_to_ipc_field(storage.as_ref()),
            _ => {
                assert!(!dtype.is_nested());
                IpcField {
                    fields: vec![],
                    dictionary_id: None,
                }
            },
        }
    }
}
