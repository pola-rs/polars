use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;
use std::hash::Hasher;

use hash_hasher::HashedMap;
use indexmap::map::IndexMap as HashMap;
use json_deserializer::{Number, Value};

use crate::{
    array::*,
    bitmap::MutableBitmap,
    chunk::Chunk,
    datatypes::{DataType, Field, IntervalUnit, Schema},
    error::Error,
    offset::{Offset, Offsets},
    temporal_conversions,
    types::{f16, NativeType},
};

/// A function that converts a &Value into an optional tuple of a byte slice and a Value.
/// This is used to create a dictionary, where the hashing depends on the DataType of the child object.
type Extract<'a> = Box<dyn Fn(&'a Value<'a>) -> Option<(u64, &'a Value<'a>)>>;

fn build_extract(data_type: &DataType) -> Extract {
    match data_type {
        DataType::Utf8 | DataType::LargeUtf8 => Box::new(|value| match &value {
            Value::String(v) => {
                let mut hasher = DefaultHasher::new();
                hasher.write(v.as_bytes());
                Some((hasher.finish(), value))
            }
            Value::Number(v) => match v {
                Number::Float(_, _) => todo!(),
                Number::Integer(_, _) => todo!(),
            },
            Value::Bool(v) => {
                let mut hasher = DefaultHasher::new();
                hasher.write(&[*v as u8]);
                Some((hasher.finish(), value))
            }
            _ => None,
        }),
        DataType::Int32 | DataType::Int64 | DataType::Int16 | DataType::Int8 => {
            Box::new(move |value| {
                let integer = match value {
                    Value::Number(number) => Some(deserialize_int_single::<i64>(*number)),
                    Value::Bool(number) => Some(i64::from(*number)),
                    _ => None,
                };
                integer.map(|integer| {
                    let mut hasher = DefaultHasher::new();
                    hasher.write(&integer.to_le_bytes());
                    (hasher.finish(), value)
                })
            })
        }
        _ => Box::new(|_| None),
    }
}

fn deserialize_boolean_into<'a, A: Borrow<Value<'a>>>(
    target: &mut MutableBooleanArray,
    rows: &[A],
) {
    let iter = rows.iter().map(|row| match row.borrow() {
        Value::Bool(v) => Some(v),
        _ => None,
    });
    target.extend_trusted_len(iter);
}

fn deserialize_int_single<T>(number: Number) -> T
where
    T: NativeType + lexical_core::FromLexical + Pow10,
{
    match number {
        Number::Float(fraction, exponent) => {
            let integer = fraction.split(|x| *x == b'.').next().unwrap();
            let mut integer: T = lexical_core::parse(integer).unwrap();
            if !exponent.is_empty() {
                let exponent: u32 = lexical_core::parse(exponent).unwrap();
                integer = integer.pow10(exponent);
            }
            integer
        }
        Number::Integer(integer, exponent) => {
            let mut integer: T = lexical_core::parse(integer).unwrap();
            if !exponent.is_empty() {
                let exponent: u32 = lexical_core::parse(exponent).unwrap();
                integer = integer.pow10(exponent);
            }
            integer
        }
    }
}

trait Powi10: NativeType + num_traits::One + std::ops::Add {
    fn powi10(self, exp: i32) -> Self;
}

impl Powi10 for f32 {
    #[inline]
    fn powi10(self, exp: i32) -> Self {
        self * 10.0f32.powi(exp)
    }
}

impl Powi10 for f64 {
    #[inline]
    fn powi10(self, exp: i32) -> Self {
        self * 10.0f64.powi(exp)
    }
}

trait Pow10: NativeType + num_traits::One + std::ops::Add {
    fn pow10(self, exp: u32) -> Self;
}

macro_rules! impl_pow10 {
    ($ty:ty) => {
        impl Pow10 for $ty {
            #[inline]
            fn pow10(self, exp: u32) -> Self {
                self * (10 as $ty).pow(exp)
            }
        }
    };
}
impl_pow10!(u8);
impl_pow10!(u16);
impl_pow10!(u32);
impl_pow10!(u64);
impl_pow10!(i8);
impl_pow10!(i16);
impl_pow10!(i32);
impl_pow10!(i64);

fn deserialize_float_single<T>(number: &Number) -> T
where
    T: NativeType + lexical_core::FromLexical + Powi10,
{
    match number {
        Number::Float(float, exponent) => {
            let mut float: T = lexical_core::parse(float).unwrap();
            if !exponent.is_empty() {
                let exponent: i32 = lexical_core::parse(exponent).unwrap();
                float = float.powi10(exponent);
            }
            float
        }
        Number::Integer(integer, exponent) => {
            let mut float: T = lexical_core::parse(integer).unwrap();
            if !exponent.is_empty() {
                let exponent: i32 = lexical_core::parse(exponent).unwrap();
                float = float.powi10(exponent);
            }
            float
        }
    }
}

fn deserialize_int_into<
    'a,
    T: NativeType + lexical_core::FromLexical + Pow10,
    A: Borrow<Value<'a>>,
>(
    target: &mut MutablePrimitiveArray<T>,
    rows: &[A],
) {
    let iter = rows.iter().map(|row| match row.borrow() {
        Value::Number(number) => Some(deserialize_int_single(*number)),
        Value::Bool(number) => Some(if *number { T::one() } else { T::default() }),
        _ => None,
    });
    target.extend_trusted_len(iter);
}

fn deserialize_float_into<
    'a,
    T: NativeType + lexical_core::FromLexical + Powi10,
    A: Borrow<Value<'a>>,
>(
    target: &mut MutablePrimitiveArray<T>,
    rows: &[A],
) {
    let iter = rows.iter().map(|row| match row.borrow() {
        Value::Number(number) => Some(deserialize_float_single(number)),
        Value::Bool(number) => Some(if *number { T::one() } else { T::default() }),
        _ => None,
    });
    target.extend_trusted_len(iter);
}

fn deserialize_binary<'a, O: Offset, A: Borrow<Value<'a>>>(rows: &[A]) -> BinaryArray<O> {
    let iter = rows.iter().map(|row| match row.borrow() {
        Value::String(v) => Some(v.as_bytes()),
        _ => None,
    });
    BinaryArray::from_trusted_len_iter(iter)
}

fn deserialize_utf8_into<'a, O: Offset, A: Borrow<Value<'a>>>(
    target: &mut MutableUtf8Array<O>,
    rows: &[A],
) {
    let mut scratch = vec![];
    for row in rows {
        match row.borrow() {
            Value::String(v) => target.push(Some(v.as_ref())),
            Value::Number(number) => match number {
                Number::Integer(number, exponent) | Number::Float(number, exponent) => {
                    scratch.clear();
                    scratch.extend_from_slice(number);
                    scratch.push(b'e');
                    scratch.extend_from_slice(exponent);
                }
            },
            Value::Bool(v) => target.push(Some(if *v { "true" } else { "false" })),
            _ => target.push_null(),
        }
    }
}

fn deserialize_list<'a, O: Offset, A: Borrow<Value<'a>>>(
    rows: &[A],
    data_type: DataType,
) -> ListArray<O> {
    let child = ListArray::<O>::get_child_type(&data_type);

    let mut validity = MutableBitmap::with_capacity(rows.len());
    let mut offsets = Offsets::<O>::with_capacity(rows.len());
    let mut inner = vec![];
    rows.iter().for_each(|row| match row.borrow() {
        Value::Array(value) => {
            inner.extend(value.iter());
            validity.push(true);
            offsets
                .try_push_usize(value.len())
                .expect("List offset is too large :/");
        }
        _ => {
            validity.push(false);
            offsets.extend_constant(1);
        }
    });

    let values = _deserialize(&inner, child.clone());

    ListArray::<O>::new(data_type, offsets.into(), values, validity.into())
}

// TODO: due to nesting, deduplicating this from the above is trickier than
// other `deserialize_xxx_into` functions. Punting on that for now.
fn deserialize_list_into<'a, O: Offset, A: Borrow<Value<'a>>>(
    target: &mut MutableListArray<O, Box<dyn MutableArray>>,
    rows: &[A],
) {
    let empty = vec![];
    let inner: Vec<_> = rows
        .iter()
        .flat_map(|row| match row.borrow() {
            Value::Array(value) => value.iter(),
            _ => empty.iter(),
        })
        .collect();

    deserialize_into(target.mut_values(), &inner);

    let lengths = rows.iter().map(|row| match row.borrow() {
        Value::Array(value) => Some(value.len()),
        _ => None,
    });

    target
        .try_extend_from_lengths(lengths)
        .expect("Offsets overflow");
}

fn deserialize_fixed_size_list_into<'a, A: Borrow<Value<'a>>>(
    target: &mut MutableFixedSizeListArray<Box<dyn MutableArray>>,
    rows: &[A],
) {
    for row in rows {
        match row.borrow() {
            Value::Array(value) => {
                if value.len() == target.size() {
                    deserialize_into(target.mut_values(), value);
                    // unless alignment is already off, the if above should
                    // prevent this from ever happening.
                    target.try_push_valid().expect("unaligned backing array");
                } else {
                    target.push_null();
                }
            }
            _ => target.push_null(),
        }
    }
}

fn deserialize_primitive_into<'a, A: Borrow<Value<'a>>, T: NativeType>(
    target: &mut Box<dyn MutableArray>,
    rows: &[A],
    deserialize_into: fn(&mut MutablePrimitiveArray<T>, &[A]) -> (),
) {
    generic_deserialize_into(target, rows, deserialize_into)
}

fn generic_deserialize_into<'a, A: Borrow<Value<'a>>, M: 'static>(
    target: &mut Box<dyn MutableArray>,
    rows: &[A],
    deserialize_into: fn(&mut M, &[A]) -> (),
) {
    deserialize_into(target.as_mut_any().downcast_mut::<M>().unwrap(), rows);
}

/// Deserialize `rows` by extending them into the given `target`
fn deserialize_into<'a, A: Borrow<Value<'a>>>(target: &mut Box<dyn MutableArray>, rows: &[A]) {
    match target.data_type() {
        DataType::Boolean => generic_deserialize_into(target, rows, deserialize_boolean_into),
        DataType::Float32 => {
            deserialize_primitive_into::<_, f32>(target, rows, deserialize_float_into)
        }
        DataType::Float64 => {
            deserialize_primitive_into::<_, f64>(target, rows, deserialize_float_into)
        }
        DataType::Int8 => deserialize_primitive_into::<_, i8>(target, rows, deserialize_int_into),
        DataType::Int16 => deserialize_primitive_into::<_, i16>(target, rows, deserialize_int_into),
        DataType::Int32 => deserialize_primitive_into::<_, i32>(target, rows, deserialize_int_into),
        DataType::Int64 => deserialize_primitive_into::<_, i64>(target, rows, deserialize_int_into),
        DataType::UInt8 => deserialize_primitive_into::<_, u8>(target, rows, deserialize_int_into),
        DataType::UInt16 => {
            deserialize_primitive_into::<_, u16>(target, rows, deserialize_int_into)
        }
        DataType::UInt32 => {
            deserialize_primitive_into::<_, u32>(target, rows, deserialize_int_into)
        }
        DataType::UInt64 => {
            deserialize_primitive_into::<_, u64>(target, rows, deserialize_int_into)
        }
        DataType::Utf8 => generic_deserialize_into::<_, MutableUtf8Array<i32>>(
            target,
            rows,
            deserialize_utf8_into,
        ),
        DataType::LargeUtf8 => generic_deserialize_into::<_, MutableUtf8Array<i64>>(
            target,
            rows,
            deserialize_utf8_into,
        ),
        DataType::FixedSizeList(_, _) => {
            generic_deserialize_into(target, rows, deserialize_fixed_size_list_into)
        }
        DataType::List(_) => deserialize_list_into(
            target
                .as_mut_any()
                .downcast_mut::<MutableListArray<i32, Box<dyn MutableArray>>>()
                .unwrap(),
            rows,
        ),
        _ => {
            todo!()
        }
    }
}

fn deserialize_struct<'a, A: Borrow<Value<'a>>>(rows: &[A], data_type: DataType) -> StructArray {
    let fields = StructArray::get_fields(&data_type);

    let mut values = fields
        .iter()
        .map(|f| (&f.name, (f.data_type(), vec![])))
        .collect::<HashMap<_, _>>();

    let mut validity = MutableBitmap::with_capacity(rows.len());

    rows.iter().for_each(|row| {
        match row.borrow() {
            Value::Object(value) => {
                values
                    .iter_mut()
                    .for_each(|(s, (_, inner))| inner.push(value.get(*s).unwrap_or(&Value::Null)));
                validity.push(true);
            }
            _ => {
                values
                    .iter_mut()
                    .for_each(|(_, (_, inner))| inner.push(&Value::Null));
                validity.push(false);
            }
        };
    });

    let values = values
        .into_iter()
        .map(|(_, (data_type, values))| _deserialize(&values, data_type.clone()))
        .collect::<Vec<_>>();

    StructArray::new(data_type, values, validity.into())
}

fn deserialize_dictionary<'a, K: DictionaryKey, A: Borrow<Value<'a>>>(
    rows: &[A],
    data_type: DataType,
) -> DictionaryArray<K> {
    let child = DictionaryArray::<K>::try_get_child(&data_type).unwrap();

    let mut map = HashedMap::<u64, K>::default();

    let extractor = build_extract(child);

    let mut inner = vec![];
    let keys = rows
        .iter()
        .map(|x| extractor(x.borrow()))
        .map(|item| match item {
            Some((hash, v)) => match map.get(&hash) {
                Some(key) => Some(*key),
                None => {
                    let key = match map.len().try_into() {
                        Ok(key) => key,
                        // todo: convert this to an error.
                        Err(_) => panic!("The maximum key is too small for this json struct"),
                    };
                    inner.push(v);
                    map.insert(hash, key);
                    Some(key)
                }
            },
            None => None,
        })
        .collect::<PrimitiveArray<K>>();

    drop(extractor);
    let values = _deserialize(&inner, child.clone());
    DictionaryArray::<K>::try_new(data_type, keys, values).unwrap()
}

fn fill_array_from<B, T, A>(
    f: fn(&mut MutablePrimitiveArray<T>, &[B]),
    data_type: DataType,
    rows: &[B],
) -> Box<dyn Array>
where
    T: NativeType,
    A: From<MutablePrimitiveArray<T>> + Array,
{
    let mut array = MutablePrimitiveArray::<T>::with_capacity(rows.len()).to(data_type);
    f(&mut array, rows);
    Box::new(A::from(array))
}

/// A trait describing an array with a backing store that can be preallocated to
/// a given size.
pub(crate) trait Container {
    /// Create this array with a given capacity.
    fn with_capacity(capacity: usize) -> Self
    where
        Self: Sized;
}

impl<O: Offset> Container for MutableBinaryArray<O> {
    fn with_capacity(capacity: usize) -> Self {
        MutableBinaryArray::with_capacity(capacity)
    }
}

impl Container for MutableBooleanArray {
    fn with_capacity(capacity: usize) -> Self {
        MutableBooleanArray::with_capacity(capacity)
    }
}

impl Container for MutableFixedSizeBinaryArray {
    fn with_capacity(capacity: usize) -> Self {
        MutableFixedSizeBinaryArray::with_capacity(capacity, 0)
    }
}

impl<O: Offset, M: MutableArray + Default + 'static> Container for MutableListArray<O, M> {
    fn with_capacity(capacity: usize) -> Self {
        MutableListArray::with_capacity(capacity)
    }
}

impl<T: NativeType> Container for MutablePrimitiveArray<T> {
    fn with_capacity(capacity: usize) -> Self {
        MutablePrimitiveArray::with_capacity(capacity)
    }
}

impl<O: Offset> Container for MutableUtf8Array<O> {
    fn with_capacity(capacity: usize) -> Self {
        MutableUtf8Array::with_capacity(capacity)
    }
}

fn fill_generic_array_from<B, M, A>(f: fn(&mut M, &[B]), rows: &[B]) -> Box<dyn Array>
where
    M: Container,
    A: From<M> + Array,
{
    let mut array = M::with_capacity(rows.len());
    f(&mut array, rows);
    Box::new(A::from(array))
}

pub(crate) fn _deserialize<'a, A: Borrow<Value<'a>>>(
    rows: &[A],
    data_type: DataType,
) -> Box<dyn Array> {
    match &data_type {
        DataType::Null => Box::new(NullArray::new(data_type, rows.len())),
        DataType::Boolean => {
            fill_generic_array_from::<_, _, BooleanArray>(deserialize_boolean_into, rows)
        }
        DataType::Int8 => {
            fill_array_from::<_, _, PrimitiveArray<i8>>(deserialize_int_into, data_type, rows)
        }
        DataType::Int16 => {
            fill_array_from::<_, _, PrimitiveArray<i16>>(deserialize_int_into, data_type, rows)
        }
        DataType::Int32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Interval(IntervalUnit::YearMonth) => {
            fill_array_from::<_, _, PrimitiveArray<i32>>(deserialize_int_into, data_type, rows)
        }
        DataType::Interval(IntervalUnit::DayTime) => {
            unimplemented!("There is no natural representation of DayTime in JSON.")
        }
        DataType::Int64 | DataType::Date64 | DataType::Time64(_) | DataType::Duration(_) => {
            fill_array_from::<_, _, PrimitiveArray<i64>>(deserialize_int_into, data_type, rows)
        }
        DataType::Timestamp(tu, tz) => {
            let iter = rows.iter().map(|row| match row.borrow() {
                Value::Number(v) => Some(deserialize_int_single(*v)),
                Value::String(v) => match (tu, tz) {
                    (_, None) => temporal_conversions::utf8_to_naive_timestamp_scalar(v, "%+", tu),
                    (_, Some(ref tz)) => {
                        let tz = temporal_conversions::parse_offset(tz).unwrap();
                        temporal_conversions::utf8_to_timestamp_scalar(v, "%+", &tz, tu)
                    }
                },
                _ => None,
            });
            Box::new(Int64Array::from_iter(iter).to(data_type))
        }
        DataType::UInt8 => {
            fill_array_from::<_, _, PrimitiveArray<u8>>(deserialize_int_into, data_type, rows)
        }
        DataType::UInt16 => {
            fill_array_from::<_, _, PrimitiveArray<u16>>(deserialize_int_into, data_type, rows)
        }
        DataType::UInt32 => {
            fill_array_from::<_, _, PrimitiveArray<u32>>(deserialize_int_into, data_type, rows)
        }
        DataType::UInt64 => {
            fill_array_from::<_, _, PrimitiveArray<u64>>(deserialize_int_into, data_type, rows)
        }
        DataType::Float16 => unreachable!(),
        DataType::Float32 => {
            fill_array_from::<_, _, PrimitiveArray<f32>>(deserialize_float_into, data_type, rows)
        }
        DataType::Float64 => {
            fill_array_from::<_, _, PrimitiveArray<f64>>(deserialize_float_into, data_type, rows)
        }
        DataType::Utf8 => {
            fill_generic_array_from::<_, _, Utf8Array<i32>>(deserialize_utf8_into, rows)
        }
        DataType::LargeUtf8 => {
            fill_generic_array_from::<_, _, Utf8Array<i64>>(deserialize_utf8_into, rows)
        }
        DataType::List(_) => Box::new(deserialize_list::<i32, _>(rows, data_type)),
        DataType::LargeList(_) => Box::new(deserialize_list::<i64, _>(rows, data_type)),
        DataType::Binary => Box::new(deserialize_binary::<i32, _>(rows)),
        DataType::LargeBinary => Box::new(deserialize_binary::<i64, _>(rows)),
        DataType::Struct(_) => Box::new(deserialize_struct(rows, data_type)),
        DataType::Dictionary(key_type, _, _) => {
            match_integer_type!(key_type, |$T| {
                Box::new(deserialize_dictionary::<$T, _>(rows, data_type))
            })
        }
        _ => todo!(),
        /*
        DataType::FixedSizeBinary(_) => Box::new(FixedSizeBinaryArray::new_empty(data_type)),
        DataType::FixedSizeList(_, _) => Box::new(FixedSizeListArray::new_empty(data_type)),
        DataType::Decimal(_, _) => Box::new(PrimitiveArray::<i128>::new_empty(data_type)),
        */
    }
}

/// Deserializes a `json` [`Value`] into an [`Array`] of [`DataType`]
/// This is CPU-bounded.
/// # Error
/// This function errors iff either:
/// * `json` is not a [`Value::Array`]
/// * `data_type` is neither [`DataType::List`] nor [`DataType::LargeList`]
pub fn deserialize(json: &Value, data_type: DataType) -> Result<Box<dyn Array>, Error> {
    match json {
        Value::Array(rows) => match data_type {
            DataType::List(inner) | DataType::LargeList(inner) => {
                Ok(_deserialize(rows, inner.data_type))
            }
            _ => Err(Error::nyi("read an Array from a non-Array data type")),
        },
        _ => Err(Error::nyi("read an Array from a non-Array JSON")),
    }
}

fn allocate_array(f: &Field) -> Box<dyn MutableArray> {
    match f.data_type() {
        DataType::Int8 => Box::new(MutablePrimitiveArray::<i8>::new()),
        DataType::Int16 => Box::new(MutablePrimitiveArray::<i16>::new()),
        DataType::Int32 => Box::new(MutablePrimitiveArray::<i32>::new()),
        DataType::Int64 => Box::new(MutablePrimitiveArray::<i64>::new()),
        DataType::UInt8 => Box::new(MutablePrimitiveArray::<u8>::new()),
        DataType::UInt16 => Box::new(MutablePrimitiveArray::<u16>::new()),
        DataType::UInt32 => Box::new(MutablePrimitiveArray::<u32>::new()),
        DataType::UInt64 => Box::new(MutablePrimitiveArray::<u64>::new()),
        DataType::Float16 => Box::new(MutablePrimitiveArray::<f16>::new()),
        DataType::Float32 => Box::new(MutablePrimitiveArray::<f32>::new()),
        DataType::Float64 => Box::new(MutablePrimitiveArray::<f64>::new()),
        DataType::Boolean => Box::new(MutableBooleanArray::new()),
        DataType::Utf8 => Box::new(MutableUtf8Array::<i32>::new()),
        DataType::LargeUtf8 => Box::new(MutableUtf8Array::<i64>::new()),
        DataType::FixedSizeList(inner, size) => Box::new(MutableFixedSizeListArray::<_>::new_from(
            allocate_array(inner),
            f.data_type().clone(),
            *size,
        )),
        DataType::List(inner) => Box::new(MutableListArray::<i32, _>::new_from(
            allocate_array(inner),
            f.data_type().clone(),
            0,
        )),
        _ => todo!(),
    }
}

/// Deserializes a `json` [`Value`] serialized in Pandas record format into
/// a [`Chunk`].
///
/// Uses the `Schema` provided, which can be inferred from arbitrary JSON with
/// [`infer_records_schema`].
///
/// This is CPU-bounded.
///
/// # Errors
///
/// This function errors iff either:
///
/// * `json` is not a [`Value::Array`]
/// * `data_type` contains any incompatible types:
///   * [`DataType::Struct`]
///   * [`DataType::Dictionary`]
///   * [`DataType::LargeList`]
pub fn deserialize_records(json: &Value, schema: &Schema) -> Result<Chunk<Box<dyn Array>>, Error> {
    let mut results = schema
        .fields
        .iter()
        .map(|f| (&f.name, allocate_array(f)))
        .collect::<HashMap<_, _>>();

    match json {
        Value::Array(rows) => {
            for row in rows.iter() {
                match row {
                    Value::Object(record) => {
                        for (key, value) in record.iter() {
                            let arr = results.get_mut(key).ok_or_else(|| {
                                Error::ExternalFormat(format!("unexpected key: '{key}'"))
                            })?;
                            deserialize_into(arr, &[value]);
                        }
                    }
                    _ => {
                        return Err(Error::ExternalFormat(
                            "each row must be an Object".to_string(),
                        ))
                    }
                }
            }
        }
        _ => {
            return Err(Error::ExternalFormat(
                "outer type must be an Array".to_string(),
            ))
        }
    }

    Ok(Chunk::new(
        results.into_values().map(|mut ma| ma.as_box()).collect(),
    ))
}
