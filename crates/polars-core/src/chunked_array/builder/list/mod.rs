mod anonymous;
mod binary;
mod boolean;
#[cfg(feature = "dtype-categorical")]
mod categorical;
mod dtypes;
mod null;
mod primitive;

pub use anonymous::*;
use arrow::legacy::array::list::AnonymousBuilder;
use arrow::legacy::array::null::MutableNullArray;
pub use binary::*;
pub use boolean::*;
#[cfg(feature = "dtype-categorical")]
use categorical::*;
use dtypes::*;
pub use null::*;
pub use primitive::*;

use super::*;

pub trait ListBuilderTrait {
    fn append_opt_series(&mut self, opt_s: Option<&Series>) -> PolarsResult<()> {
        match opt_s {
            Some(s) => return self.append_series(s),
            None => self.append_null(),
        }
        Ok(())
    }
    fn append_series(&mut self, s: &Series) -> PolarsResult<()>;
    fn append_null(&mut self);

    fn field(&self) -> &Field {
        unimplemented!()
    }

    fn inner_array(&mut self) -> ArrayRef {
        unimplemented!()
    }

    fn fast_explode(&self) -> bool {
        unimplemented!()
    }

    fn finish(&mut self) -> ListChunked {
        let arr = self.inner_array();

        let mut ca = ListChunked::new_with_compute_len(Arc::new(self.field().clone()), vec![arr]);
        if self.fast_explode() {
            ca.set_fast_explode()
        }
        ca
    }
}

impl<S: ?Sized> ListBuilderTrait for Box<S>
where
    S: ListBuilderTrait,
{
    fn append_opt_series(&mut self, opt_s: Option<&Series>) -> PolarsResult<()> {
        (**self).append_opt_series(opt_s)
    }

    fn append_series(&mut self, s: &Series) -> PolarsResult<()> {
        (**self).append_series(s)
    }

    fn append_null(&mut self) {
        (**self).append_null()
    }

    fn finish(&mut self) -> ListChunked {
        (**self).finish()
    }
}

type LargePrimitiveBuilder<T> = MutableListArray<i64, MutablePrimitiveArray<T>>;
type LargeListBinViewBuilder<T> = MutableListArray<i64, MutableBinaryViewArray<T>>;
type LargeListBooleanBuilder = MutableListArray<i64, MutableBooleanArray>;
type LargeListNullBuilder = MutableListArray<i64, MutableNullArray>;

pub fn get_list_builder(
    inner_type_logical: &DataType,
    value_capacity: usize,
    list_capacity: usize,
    name: &str,
) -> PolarsResult<Box<dyn ListBuilderTrait>> {
    match inner_type_logical {
        #[cfg(feature = "dtype-categorical")]
        DataType::Categorical(Some(rev_map), ordering) => {
            return Ok(create_categorical_chunked_listbuilder(
                name,
                *ordering,
                list_capacity,
                value_capacity,
                rev_map.clone(),
            ))
        },
        #[cfg(feature = "dtype-categorical")]
        DataType::Enum(Some(rev_map), ordering) => {
            let list_builder = ListEnumCategoricalChunkedBuilder::new(
                name,
                *ordering,
                list_capacity,
                value_capacity,
                (**rev_map).clone(),
            );
            return Ok(Box::new(list_builder));
        },
        _ => {},
    }

    let physical_type = inner_type_logical.to_physical();

    match &physical_type {
        #[cfg(feature = "object")]
        DataType::Object(_, _) => polars_bail!(opq = list_builder, &physical_type),
        #[cfg(feature = "dtype-struct")]
        DataType::Struct(_) => Ok(Box::new(AnonymousOwnedListBuilder::new(
            name,
            list_capacity,
            Some(inner_type_logical.clone()),
        ))),
        DataType::Null => Ok(Box::new(ListNullChunkedBuilder::new(name, list_capacity))),
        DataType::List(_) => Ok(Box::new(AnonymousOwnedListBuilder::new(
            name,
            list_capacity,
            Some(inner_type_logical.clone()),
        ))),
        #[cfg(feature = "dtype-array")]
        DataType::Array(..) => Ok(Box::new(AnonymousOwnedListBuilder::new(
            name,
            list_capacity,
            Some(inner_type_logical.clone()),
        ))),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => Ok(Box::new(
            ListPrimitiveChunkedBuilder::<Int128Type>::new_with_values_type(
                name,
                list_capacity,
                value_capacity,
                physical_type,
                inner_type_logical.clone(),
            ),
        )),
        _ => {
            macro_rules! get_primitive_builder {
                ($type:ty) => {{
                    let builder = ListPrimitiveChunkedBuilder::<$type>::new(
                        name,
                        list_capacity,
                        value_capacity,
                        inner_type_logical.clone(),
                    );
                    Box::new(builder)
                }};
            }
            macro_rules! get_bool_builder {
                () => {{
                    let builder =
                        ListBooleanChunkedBuilder::new(&name, list_capacity, value_capacity);
                    Box::new(builder)
                }};
            }
            macro_rules! get_string_builder {
                () => {{
                    let builder =
                        ListStringChunkedBuilder::new(&name, list_capacity, 5 * value_capacity);
                    Box::new(builder)
                }};
            }
            macro_rules! get_binary_builder {
                () => {{
                    let builder =
                        ListBinaryChunkedBuilder::new(&name, list_capacity, 5 * value_capacity);
                    Box::new(builder)
                }};
            }
            Ok(match_dtype_to_logical_apply_macro!(
                physical_type,
                get_primitive_builder,
                get_string_builder,
                get_binary_builder,
                get_bool_builder
            ))
        },
    }
}
