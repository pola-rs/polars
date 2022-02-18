// use crate::chunked_array::categorical::*;
// use super::*;
// use crate::prelude::*;
//
// pub struct CategoricalChunked {
//     logical: Logical<CategoricalType, UInt32Type>,
//     /// maps categorical u32 indexes to String values
//     rev_map: Arc<RevMapping>
// }
//
// impl Deref for CategoricalChunked {
//     type Target = UInt32Chunked;
//
//     fn deref(&self) -> &Self::Target {
//         &self.logical.0
//     }
// }
//
// impl LogicalType for CategoricalChunked {
//     fn dtype(&self) -> &DataType {
//         &DataType::Date
//     }
//
//     fn get_any_value(&self, i: usize) -> AnyValue<'_> {
//         match self.logical.0.get(i) {
//             Some(i) => {
//                 AnyValue::Categorical(i, &self.rev_map)
//             },
//             None => AnyValue::Null
//         }
//     }
//
//     fn cast(&self, dtype: &DataType) -> Result<Series> {
//         use DataType::*;
//         match (self.dtype(), dtype) {
//             #[cfg(feature = "dtype-datetime")]
//             (Date, Datetime(tu, tz)) => {
//                 let casted = self.0.cast(dtype)?;
//                 let casted = casted.datetime().unwrap();
//                 let conversion = match tu {
//                     TimeUnit::Nanoseconds => NS_IN_DAY,
//                     TimeUnit::Microseconds => US_IN_DAY,
//                     TimeUnit::Milliseconds => MS_IN_DAY,
//                 };
//                 Ok((casted.deref() * conversion)
//                     .into_datetime(*tu, tz.clone())
//                     .into_series())
//             }
//             _ => self.0.cast(dtype),
//         }
//     }
// }
