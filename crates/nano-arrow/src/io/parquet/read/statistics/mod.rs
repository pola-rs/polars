//! APIs exposing `parquet2`'s statistics as arrow's statistics.
use ethnum::I256;
use std::collections::VecDeque;
use std::sync::Arc;

use parquet2::metadata::RowGroupMetaData;
use parquet2::schema::types::{
    PhysicalType as ParquetPhysicalType, PrimitiveType as ParquetPrimitiveType,
};
use parquet2::statistics::{
    BinaryStatistics, BooleanStatistics, FixedLenStatistics, PrimitiveStatistics,
    Statistics as ParquetStatistics,
};
use parquet2::types::int96_to_i64_ns;

use crate::array::*;
use crate::datatypes::IntervalUnit;
use crate::datatypes::{DataType, Field, PhysicalType};
use crate::error::Error;
use crate::error::Result;
use crate::types::i256;

mod binary;
mod boolean;
mod dictionary;
mod fixlen;
mod list;
mod map;
mod null;
mod primitive;
mod struct_;
mod utf8;

use self::list::DynMutableListArray;

use super::get_field_columns;

/// Arrow-deserialized parquet Statistics of a file
#[derive(Debug, PartialEq)]
pub struct Statistics {
    /// number of nulls. This is a [`UInt64Array`] for non-nested types
    pub null_count: Box<dyn Array>,
    /// number of dictinct values. This is a [`UInt64Array`] for non-nested types
    pub distinct_count: Box<dyn Array>,
    /// Minimum
    pub min_value: Box<dyn Array>,
    /// Maximum
    pub max_value: Box<dyn Array>,
}

/// Arrow-deserialized parquet Statistics of a file
#[derive(Debug)]
struct MutableStatistics {
    /// number of nulls
    pub null_count: Box<dyn MutableArray>,
    /// number of dictinct values
    pub distinct_count: Box<dyn MutableArray>,
    /// Minimum
    pub min_value: Box<dyn MutableArray>,
    /// Maximum
    pub max_value: Box<dyn MutableArray>,
}

impl From<MutableStatistics> for Statistics {
    fn from(mut s: MutableStatistics) -> Self {
        let null_count = if let PhysicalType::Struct = s.null_count.data_type().to_physical_type() {
            s.null_count
                .as_box()
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone()
                .boxed()
        } else if let PhysicalType::Map = s.null_count.data_type().to_physical_type() {
            s.null_count
                .as_box()
                .as_any()
                .downcast_ref::<MapArray>()
                .unwrap()
                .clone()
                .boxed()
        } else if let PhysicalType::List = s.null_count.data_type().to_physical_type() {
            s.null_count
                .as_box()
                .as_any()
                .downcast_ref::<ListArray<i32>>()
                .unwrap()
                .clone()
                .boxed()
        } else if let PhysicalType::LargeList = s.null_count.data_type().to_physical_type() {
            s.null_count
                .as_box()
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap()
                .clone()
                .boxed()
        } else {
            s.null_count
                .as_box()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .clone()
                .boxed()
        };
        let distinct_count = if let PhysicalType::Struct =
            s.distinct_count.data_type().to_physical_type()
        {
            s.distinct_count
                .as_box()
                .as_any()
                .downcast_ref::<StructArray>()
                .unwrap()
                .clone()
                .boxed()
        } else if let PhysicalType::Map = s.distinct_count.data_type().to_physical_type() {
            s.distinct_count
                .as_box()
                .as_any()
                .downcast_ref::<MapArray>()
                .unwrap()
                .clone()
                .boxed()
        } else if let PhysicalType::List = s.distinct_count.data_type().to_physical_type() {
            s.distinct_count
                .as_box()
                .as_any()
                .downcast_ref::<ListArray<i32>>()
                .unwrap()
                .clone()
                .boxed()
        } else if let PhysicalType::LargeList = s.distinct_count.data_type().to_physical_type() {
            s.distinct_count
                .as_box()
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap()
                .clone()
                .boxed()
        } else {
            s.distinct_count
                .as_box()
                .as_any()
                .downcast_ref::<UInt64Array>()
                .unwrap()
                .clone()
                .boxed()
        };
        Self {
            null_count,
            distinct_count,
            min_value: s.min_value.as_box(),
            max_value: s.max_value.as_box(),
        }
    }
}

fn make_mutable(data_type: &DataType, capacity: usize) -> Result<Box<dyn MutableArray>> {
    Ok(match data_type.to_physical_type() {
        PhysicalType::Boolean => {
            Box::new(MutableBooleanArray::with_capacity(capacity)) as Box<dyn MutableArray>
        }
        PhysicalType::Primitive(primitive) => with_match_primitive_type!(primitive, |$T| {
            Box::new(MutablePrimitiveArray::<$T>::with_capacity(capacity).to(data_type.clone()))
                as Box<dyn MutableArray>
        }),
        PhysicalType::Binary => {
            Box::new(MutableBinaryArray::<i32>::with_capacity(capacity)) as Box<dyn MutableArray>
        }
        PhysicalType::LargeBinary => {
            Box::new(MutableBinaryArray::<i64>::with_capacity(capacity)) as Box<dyn MutableArray>
        }
        PhysicalType::Utf8 => {
            Box::new(MutableUtf8Array::<i32>::with_capacity(capacity)) as Box<dyn MutableArray>
        }
        PhysicalType::LargeUtf8 => {
            Box::new(MutableUtf8Array::<i64>::with_capacity(capacity)) as Box<dyn MutableArray>
        }
        PhysicalType::FixedSizeBinary => {
            Box::new(MutableFixedSizeBinaryArray::try_new(data_type.clone(), vec![], None).unwrap())
                as _
        }
        PhysicalType::LargeList | PhysicalType::List => Box::new(
            DynMutableListArray::try_with_capacity(data_type.clone(), capacity)?,
        ) as Box<dyn MutableArray>,
        PhysicalType::Dictionary(_) => Box::new(
            dictionary::DynMutableDictionary::try_with_capacity(data_type.clone(), capacity)?,
        ),
        PhysicalType::Struct => Box::new(struct_::DynMutableStructArray::try_with_capacity(
            data_type.clone(),
            capacity,
        )?),
        PhysicalType::Map => Box::new(map::DynMutableMapArray::try_with_capacity(
            data_type.clone(),
            capacity,
        )?),
        PhysicalType::Null => {
            Box::new(MutableNullArray::new(DataType::Null, 0)) as Box<dyn MutableArray>
        }
        other => {
            return Err(Error::NotYetImplemented(format!(
                "Deserializing parquet stats from {other:?} is still not implemented"
            )))
        }
    })
}

fn create_dt(data_type: &DataType) -> DataType {
    if let DataType::Struct(fields) = data_type.to_logical_type() {
        DataType::Struct(
            fields
                .iter()
                .map(|f| Field::new(&f.name, create_dt(&f.data_type), f.is_nullable))
                .collect(),
        )
    } else if let DataType::Map(f, ordered) = data_type.to_logical_type() {
        DataType::Map(
            Box::new(Field::new(&f.name, create_dt(&f.data_type), f.is_nullable)),
            *ordered,
        )
    } else if let DataType::List(f) = data_type.to_logical_type() {
        DataType::List(Box::new(Field::new(
            &f.name,
            create_dt(&f.data_type),
            f.is_nullable,
        )))
    } else if let DataType::LargeList(f) = data_type.to_logical_type() {
        DataType::LargeList(Box::new(Field::new(
            &f.name,
            create_dt(&f.data_type),
            f.is_nullable,
        )))
    } else {
        DataType::UInt64
    }
}

impl MutableStatistics {
    fn try_new(field: &Field) -> Result<Self> {
        let min_value = make_mutable(&field.data_type, 0)?;
        let max_value = make_mutable(&field.data_type, 0)?;

        let dt = create_dt(&field.data_type);
        Ok(Self {
            null_count: make_mutable(&dt, 0)?,
            distinct_count: make_mutable(&dt, 0)?,
            min_value,
            max_value,
        })
    }
}

fn push_others(
    from: Option<&dyn ParquetStatistics>,
    distinct_count: &mut UInt64Vec,
    null_count: &mut UInt64Vec,
) {
    let from = if let Some(from) = from {
        from
    } else {
        distinct_count.push(None);
        null_count.push(None);
        return;
    };
    let (distinct, null_count1) = match from.physical_type() {
        ParquetPhysicalType::Boolean => {
            let from = from.as_any().downcast_ref::<BooleanStatistics>().unwrap();
            (from.distinct_count, from.null_count)
        }
        ParquetPhysicalType::Int32 => {
            let from = from
                .as_any()
                .downcast_ref::<PrimitiveStatistics<i32>>()
                .unwrap();
            (from.distinct_count, from.null_count)
        }
        ParquetPhysicalType::Int64 => {
            let from = from
                .as_any()
                .downcast_ref::<PrimitiveStatistics<i64>>()
                .unwrap();
            (from.distinct_count, from.null_count)
        }
        ParquetPhysicalType::Int96 => {
            let from = from
                .as_any()
                .downcast_ref::<PrimitiveStatistics<[u32; 3]>>()
                .unwrap();
            (from.distinct_count, from.null_count)
        }
        ParquetPhysicalType::Float => {
            let from = from
                .as_any()
                .downcast_ref::<PrimitiveStatistics<f32>>()
                .unwrap();
            (from.distinct_count, from.null_count)
        }
        ParquetPhysicalType::Double => {
            let from = from
                .as_any()
                .downcast_ref::<PrimitiveStatistics<f64>>()
                .unwrap();
            (from.distinct_count, from.null_count)
        }
        ParquetPhysicalType::ByteArray => {
            let from = from.as_any().downcast_ref::<BinaryStatistics>().unwrap();
            (from.distinct_count, from.null_count)
        }
        ParquetPhysicalType::FixedLenByteArray(_) => {
            let from = from.as_any().downcast_ref::<FixedLenStatistics>().unwrap();
            (from.distinct_count, from.null_count)
        }
    };

    distinct_count.push(distinct.map(|x| x as u64));
    null_count.push(null_count1.map(|x| x as u64));
}

fn push(
    stats: &mut VecDeque<(Option<Arc<dyn ParquetStatistics>>, ParquetPrimitiveType)>,
    min: &mut dyn MutableArray,
    max: &mut dyn MutableArray,
    distinct_count: &mut dyn MutableArray,
    null_count: &mut dyn MutableArray,
) -> Result<()> {
    match min.data_type().to_logical_type() {
        List(_) | LargeList(_) => {
            let min = min
                .as_mut_any()
                .downcast_mut::<list::DynMutableListArray>()
                .unwrap();
            let max = max
                .as_mut_any()
                .downcast_mut::<list::DynMutableListArray>()
                .unwrap();
            let distinct_count = distinct_count
                .as_mut_any()
                .downcast_mut::<list::DynMutableListArray>()
                .unwrap();
            let null_count = null_count
                .as_mut_any()
                .downcast_mut::<list::DynMutableListArray>()
                .unwrap();
            return push(
                stats,
                min.inner.as_mut(),
                max.inner.as_mut(),
                distinct_count.inner.as_mut(),
                null_count.inner.as_mut(),
            );
        }
        Dictionary(_, _, _) => {
            let min = min
                .as_mut_any()
                .downcast_mut::<dictionary::DynMutableDictionary>()
                .unwrap();
            let max = max
                .as_mut_any()
                .downcast_mut::<dictionary::DynMutableDictionary>()
                .unwrap();
            return push(
                stats,
                min.inner.as_mut(),
                max.inner.as_mut(),
                distinct_count,
                null_count,
            );
        }
        Struct(_) => {
            let min = min
                .as_mut_any()
                .downcast_mut::<struct_::DynMutableStructArray>()
                .unwrap();
            let max = max
                .as_mut_any()
                .downcast_mut::<struct_::DynMutableStructArray>()
                .unwrap();
            let distinct_count = distinct_count
                .as_mut_any()
                .downcast_mut::<struct_::DynMutableStructArray>()
                .unwrap();
            let null_count = null_count
                .as_mut_any()
                .downcast_mut::<struct_::DynMutableStructArray>()
                .unwrap();
            return min
                .inner
                .iter_mut()
                .zip(max.inner.iter_mut())
                .zip(distinct_count.inner.iter_mut())
                .zip(null_count.inner.iter_mut())
                .try_for_each(|(((min, max), distinct_count), null_count)| {
                    push(
                        stats,
                        min.as_mut(),
                        max.as_mut(),
                        distinct_count.as_mut(),
                        null_count.as_mut(),
                    )
                });
        }
        Map(_, _) => {
            let min = min
                .as_mut_any()
                .downcast_mut::<map::DynMutableMapArray>()
                .unwrap();
            let max = max
                .as_mut_any()
                .downcast_mut::<map::DynMutableMapArray>()
                .unwrap();
            let distinct_count = distinct_count
                .as_mut_any()
                .downcast_mut::<map::DynMutableMapArray>()
                .unwrap();
            let null_count = null_count
                .as_mut_any()
                .downcast_mut::<map::DynMutableMapArray>()
                .unwrap();
            return push(
                stats,
                min.inner.as_mut(),
                max.inner.as_mut(),
                distinct_count.inner.as_mut(),
                null_count.inner.as_mut(),
            );
        }
        _ => {}
    }

    let (from, type_) = stats.pop_front().unwrap();
    let from = from.as_deref();

    let distinct_count = distinct_count
        .as_mut_any()
        .downcast_mut::<UInt64Vec>()
        .unwrap();
    let null_count = null_count.as_mut_any().downcast_mut::<UInt64Vec>().unwrap();

    push_others(from, distinct_count, null_count);

    let physical_type = &type_.physical_type;

    use DataType::*;
    match min.data_type().to_logical_type() {
        Boolean => boolean::push(from, min, max),
        Int8 => primitive::push(from, min, max, |x: i32| Ok(x as i8)),
        Int16 => primitive::push(from, min, max, |x: i32| Ok(x as i16)),
        Date32 | Time32(_) => primitive::push::<i32, i32, _>(from, min, max, Ok),
        Interval(IntervalUnit::YearMonth) => fixlen::push_year_month(from, min, max),
        Interval(IntervalUnit::DayTime) => fixlen::push_days_ms(from, min, max),
        UInt8 => primitive::push(from, min, max, |x: i32| Ok(x as u8)),
        UInt16 => primitive::push(from, min, max, |x: i32| Ok(x as u16)),
        UInt32 => match physical_type {
            // some implementations of parquet write arrow's u32 into i64.
            ParquetPhysicalType::Int64 => primitive::push(from, min, max, |x: i64| Ok(x as u32)),
            ParquetPhysicalType::Int32 => primitive::push(from, min, max, |x: i32| Ok(x as u32)),
            other => Err(Error::NotYetImplemented(format!(
                "Can't decode UInt32 type from parquet type {other:?}"
            ))),
        },
        Int32 => primitive::push::<i32, i32, _>(from, min, max, Ok),
        Date64 => match physical_type {
            ParquetPhysicalType::Int64 => primitive::push::<i64, i64, _>(from, min, max, Ok),
            // some implementations of parquet write arrow's date64 into i32.
            ParquetPhysicalType::Int32 => {
                primitive::push(from, min, max, |x: i32| Ok(x as i64 * 86400000))
            }
            other => Err(Error::NotYetImplemented(format!(
                "Can't decode Date64 type from parquet type {other:?}"
            ))),
        },
        Int64 | Time64(_) | Duration(_) => primitive::push::<i64, i64, _>(from, min, max, Ok),
        UInt64 => primitive::push(from, min, max, |x: i64| Ok(x as u64)),
        Timestamp(time_unit, _) => {
            let time_unit = *time_unit;
            if physical_type == &ParquetPhysicalType::Int96 {
                let from = from.map(|from| {
                    let from = from
                        .as_any()
                        .downcast_ref::<PrimitiveStatistics<[u32; 3]>>()
                        .unwrap();
                    PrimitiveStatistics::<i64> {
                        primitive_type: from.primitive_type.clone(),
                        null_count: from.null_count,
                        distinct_count: from.distinct_count,
                        min_value: from.min_value.map(int96_to_i64_ns),
                        max_value: from.max_value.map(int96_to_i64_ns),
                    }
                });
                primitive::push(
                    from.as_ref().map(|x| x as &dyn ParquetStatistics),
                    min,
                    max,
                    |x: i64| {
                        Ok(primitive::timestamp(
                            type_.logical_type.as_ref(),
                            time_unit,
                            x,
                        ))
                    },
                )
            } else {
                primitive::push(from, min, max, |x: i64| {
                    Ok(primitive::timestamp(
                        type_.logical_type.as_ref(),
                        time_unit,
                        x,
                    ))
                })
            }
        }
        Float32 => primitive::push::<f32, f32, _>(from, min, max, Ok),
        Float64 => primitive::push::<f64, f64, _>(from, min, max, Ok),
        Decimal(_, _) => match physical_type {
            ParquetPhysicalType::Int32 => primitive::push(from, min, max, |x: i32| Ok(x as i128)),
            ParquetPhysicalType::Int64 => primitive::push(from, min, max, |x: i64| Ok(x as i128)),
            ParquetPhysicalType::FixedLenByteArray(n) if *n > 16 => Err(Error::NotYetImplemented(
                format!("Can't decode Decimal128 type from Fixed Size Byte Array of len {n:?}"),
            )),
            ParquetPhysicalType::FixedLenByteArray(n) => fixlen::push_i128(from, *n, min, max),
            _ => unreachable!(),
        },
        Decimal256(_, _) => match physical_type {
            ParquetPhysicalType::Int32 => {
                primitive::push(from, min, max, |x: i32| Ok(i256(I256::new(x.into()))))
            }
            ParquetPhysicalType::Int64 => {
                primitive::push(from, min, max, |x: i64| Ok(i256(I256::new(x.into()))))
            }
            ParquetPhysicalType::FixedLenByteArray(n) if *n <= 16 => {
                fixlen::push_i256_with_i128(from, *n, min, max)
            }
            ParquetPhysicalType::FixedLenByteArray(n) if *n > 32 => Err(Error::NotYetImplemented(
                format!("Can't decode Decimal256 type from Fixed Size Byte Array of len {n:?}"),
            )),
            ParquetPhysicalType::FixedLenByteArray(_) => fixlen::push_i256(from, min, max),
            _ => unreachable!(),
        },
        Binary => binary::push::<i32>(from, min, max),
        LargeBinary => binary::push::<i64>(from, min, max),
        Utf8 => utf8::push::<i32>(from, min, max),
        LargeUtf8 => utf8::push::<i64>(from, min, max),
        FixedSizeBinary(_) => fixlen::push(from, min, max),
        Null => null::push(min, max),
        other => todo!("{:?}", other),
    }
}

/// Deserializes the statistics in the column chunks from all `row_groups`
/// into [`Statistics`] associated from `field`'s name.
///
/// # Errors
/// This function errors if the deserialization of the statistics fails (e.g. invalid utf8)
pub fn deserialize(field: &Field, row_groups: &[RowGroupMetaData]) -> Result<Statistics> {
    let mut statistics = MutableStatistics::try_new(field)?;

    // transpose
    row_groups.iter().try_for_each(|group| {
        let columns = get_field_columns(group.columns(), field.name.as_ref());
        let mut stats = columns
            .into_iter()
            .map(|column| {
                Ok((
                    column.statistics().transpose()?,
                    column.descriptor().descriptor.primitive_type.clone(),
                ))
            })
            .collect::<Result<VecDeque<(Option<_>, ParquetPrimitiveType)>>>()?;
        push(
            &mut stats,
            statistics.min_value.as_mut(),
            statistics.max_value.as_mut(),
            statistics.distinct_count.as_mut(),
            statistics.null_count.as_mut(),
        )
    })?;

    Ok(statistics.into())
}
