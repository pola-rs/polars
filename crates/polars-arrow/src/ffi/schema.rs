use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::ptr;

use polars_error::{polars_bail, polars_err, PolarsResult};

use super::ArrowSchema;
use crate::datatypes::{
    ArrowDataType, Extension, Field, IntegerType, IntervalUnit, Metadata, TimeUnit, UnionMode,
};

#[allow(dead_code)]
struct SchemaPrivateData {
    name: CString,
    format: CString,
    metadata: Option<Vec<u8>>,
    children_ptr: Box<[*mut ArrowSchema]>,
    dictionary: Option<*mut ArrowSchema>,
}

// callback used to drop [ArrowSchema] when it is exported.
unsafe extern "C" fn c_release_schema(schema: *mut ArrowSchema) {
    if schema.is_null() {
        return;
    }
    let schema = &mut *schema;

    let private = Box::from_raw(schema.private_data as *mut SchemaPrivateData);
    for child in private.children_ptr.iter() {
        let _ = Box::from_raw(*child);
    }

    if let Some(ptr) = private.dictionary {
        let _ = Box::from_raw(ptr);
    }

    schema.release = None;
}

/// allocate (and hold) the children
fn schema_children(data_type: &ArrowDataType, flags: &mut i64) -> Box<[*mut ArrowSchema]> {
    match data_type {
        ArrowDataType::List(field)
        | ArrowDataType::FixedSizeList(field, _)
        | ArrowDataType::LargeList(field) => {
            Box::new([Box::into_raw(Box::new(ArrowSchema::new(field.as_ref())))])
        },
        ArrowDataType::Map(field, is_sorted) => {
            *flags += (*is_sorted as i64) * 4;
            Box::new([Box::into_raw(Box::new(ArrowSchema::new(field.as_ref())))])
        },
        ArrowDataType::Struct(fields) | ArrowDataType::Union(fields, _, _) => fields
            .iter()
            .map(|field| Box::into_raw(Box::new(ArrowSchema::new(field))))
            .collect::<Box<[_]>>(),
        ArrowDataType::Extension(_, inner, _) => schema_children(inner, flags),
        _ => Box::new([]),
    }
}

impl ArrowSchema {
    /// creates a new [ArrowSchema]
    pub(crate) fn new(field: &Field) -> Self {
        let format = to_format(field.data_type());
        let name = field.name.clone();

        let mut flags = field.is_nullable as i64 * 2;

        // note: this cannot be done along with the above because the above is fallible and this op leaks.
        let children_ptr = schema_children(field.data_type(), &mut flags);
        let n_children = children_ptr.len() as i64;

        let dictionary = if let ArrowDataType::Dictionary(_, values, is_ordered) = field.data_type()
        {
            flags += *is_ordered as i64;
            // we do not store field info in the dict values, so can't recover it all :(
            let field = Field::new("", values.as_ref().clone(), true);
            Some(Box::new(ArrowSchema::new(&field)))
        } else {
            None
        };

        let metadata = &field.metadata;

        let metadata =
            if let ArrowDataType::Extension(name, _, extension_metadata) = field.data_type() {
                // append extension information.
                let mut metadata = metadata.clone();

                // metadata
                if let Some(extension_metadata) = extension_metadata {
                    metadata.insert(
                        "ARROW:extension:metadata".to_string(),
                        extension_metadata.clone(),
                    );
                }

                metadata.insert("ARROW:extension:name".to_string(), name.clone());

                Some(metadata_to_bytes(&metadata))
            } else if !metadata.is_empty() {
                Some(metadata_to_bytes(metadata))
            } else {
                None
            };

        let name = CString::new(name).unwrap();
        let format = CString::new(format).unwrap();

        let mut private = Box::new(SchemaPrivateData {
            name,
            format,
            metadata,
            children_ptr,
            dictionary: dictionary.map(Box::into_raw),
        });

        // <https://arrow.apache.org/docs/format/CDataInterface.html#c.ArrowSchema>
        Self {
            format: private.format.as_ptr(),
            name: private.name.as_ptr(),
            metadata: private
                .metadata
                .as_ref()
                .map(|x| x.as_ptr())
                .unwrap_or(std::ptr::null()) as *const ::std::os::raw::c_char,
            flags,
            n_children,
            children: private.children_ptr.as_mut_ptr(),
            dictionary: private.dictionary.unwrap_or(std::ptr::null_mut()),
            release: Some(c_release_schema),
            private_data: Box::into_raw(private) as *mut ::std::os::raw::c_void,
        }
    }

    /// create an empty [ArrowSchema]
    pub fn empty() -> Self {
        Self {
            format: std::ptr::null_mut(),
            name: std::ptr::null_mut(),
            metadata: std::ptr::null_mut(),
            flags: 0,
            n_children: 0,
            children: ptr::null_mut(),
            dictionary: std::ptr::null_mut(),
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.private_data.is_null()
    }

    /// returns the format of this schema.
    pub(crate) fn format(&self) -> &str {
        assert!(!self.format.is_null());
        // safe because the lifetime of `self.format` equals `self`
        unsafe { CStr::from_ptr(self.format) }
            .to_str()
            .expect("The external API has a non-utf8 as format")
    }

    /// returns the name of this schema.
    ///
    /// Since this field is optional, `""` is returned if it is not set (as per the spec).
    pub(crate) fn name(&self) -> &str {
        if self.name.is_null() {
            return "";
        }
        // safe because the lifetime of `self.name` equals `self`
        unsafe { CStr::from_ptr(self.name) }.to_str().unwrap()
    }

    pub(crate) fn child(&self, index: usize) -> &'static Self {
        assert!(index < self.n_children as usize);
        unsafe { self.children.add(index).as_ref().unwrap().as_ref().unwrap() }
    }

    pub(crate) fn dictionary(&self) -> Option<&'static Self> {
        if self.dictionary.is_null() {
            return None;
        };
        Some(unsafe { self.dictionary.as_ref().unwrap() })
    }

    pub(crate) fn nullable(&self) -> bool {
        (self.flags / 2) & 1 == 1
    }
}

impl Drop for ArrowSchema {
    fn drop(&mut self) {
        match self.release {
            None => (),
            Some(release) => unsafe { release(self) },
        };
    }
}

pub(crate) unsafe fn to_field(schema: &ArrowSchema) -> PolarsResult<Field> {
    let dictionary = schema.dictionary();
    let data_type = if let Some(dictionary) = dictionary {
        let indices = to_integer_type(schema.format())?;
        let values = to_field(dictionary)?;
        let is_ordered = schema.flags & 1 == 1;
        ArrowDataType::Dictionary(indices, Box::new(values.data_type().clone()), is_ordered)
    } else {
        to_data_type(schema)?
    };
    let (metadata, extension) = unsafe { metadata_from_bytes(schema.metadata) };

    let data_type = if let Some((name, extension_metadata)) = extension {
        ArrowDataType::Extension(name, Box::new(data_type), extension_metadata)
    } else {
        data_type
    };

    Ok(Field::new(schema.name(), data_type, schema.nullable()).with_metadata(metadata))
}

fn to_integer_type(format: &str) -> PolarsResult<IntegerType> {
    use IntegerType::*;
    Ok(match format {
        "c" => Int8,
        "C" => UInt8,
        "s" => Int16,
        "S" => UInt16,
        "i" => Int32,
        "I" => UInt32,
        "l" => Int64,
        "L" => UInt64,
        _ => {
            polars_bail!(
                ComputeError:
                "dictionary indices can only be integers"
            )
        },
    })
}

unsafe fn to_data_type(schema: &ArrowSchema) -> PolarsResult<ArrowDataType> {
    Ok(match schema.format() {
        "n" => ArrowDataType::Null,
        "b" => ArrowDataType::Boolean,
        "c" => ArrowDataType::Int8,
        "C" => ArrowDataType::UInt8,
        "s" => ArrowDataType::Int16,
        "S" => ArrowDataType::UInt16,
        "i" => ArrowDataType::Int32,
        "I" => ArrowDataType::UInt32,
        "l" => ArrowDataType::Int64,
        "L" => ArrowDataType::UInt64,
        "e" => ArrowDataType::Float16,
        "f" => ArrowDataType::Float32,
        "g" => ArrowDataType::Float64,
        "z" => ArrowDataType::Binary,
        "Z" => ArrowDataType::LargeBinary,
        "u" => ArrowDataType::Utf8,
        "U" => ArrowDataType::LargeUtf8,
        "tdD" => ArrowDataType::Date32,
        "tdm" => ArrowDataType::Date64,
        "tts" => ArrowDataType::Time32(TimeUnit::Second),
        "ttm" => ArrowDataType::Time32(TimeUnit::Millisecond),
        "ttu" => ArrowDataType::Time64(TimeUnit::Microsecond),
        "ttn" => ArrowDataType::Time64(TimeUnit::Nanosecond),
        "tDs" => ArrowDataType::Duration(TimeUnit::Second),
        "tDm" => ArrowDataType::Duration(TimeUnit::Millisecond),
        "tDu" => ArrowDataType::Duration(TimeUnit::Microsecond),
        "tDn" => ArrowDataType::Duration(TimeUnit::Nanosecond),
        "tiM" => ArrowDataType::Interval(IntervalUnit::YearMonth),
        "tiD" => ArrowDataType::Interval(IntervalUnit::DayTime),
        "vu" => ArrowDataType::Utf8View,
        "vz" => ArrowDataType::BinaryView,
        "+l" => {
            let child = schema.child(0);
            ArrowDataType::List(Box::new(to_field(child)?))
        },
        "+L" => {
            let child = schema.child(0);
            ArrowDataType::LargeList(Box::new(to_field(child)?))
        },
        "+m" => {
            let child = schema.child(0);

            let is_sorted = (schema.flags & 4) != 0;
            ArrowDataType::Map(Box::new(to_field(child)?), is_sorted)
        },
        "+s" => {
            let children = (0..schema.n_children as usize)
                .map(|x| to_field(schema.child(x)))
                .collect::<PolarsResult<Vec<_>>>()?;
            ArrowDataType::Struct(children)
        },
        other => {
            match other.splitn(2, ':').collect::<Vec<_>>()[..] {
                // Timestamps with no timezone
                ["tss", ""] => ArrowDataType::Timestamp(TimeUnit::Second, None),
                ["tsm", ""] => ArrowDataType::Timestamp(TimeUnit::Millisecond, None),
                ["tsu", ""] => ArrowDataType::Timestamp(TimeUnit::Microsecond, None),
                ["tsn", ""] => ArrowDataType::Timestamp(TimeUnit::Nanosecond, None),

                // Timestamps with timezone
                ["tss", tz] => ArrowDataType::Timestamp(TimeUnit::Second, Some(tz.to_string())),
                ["tsm", tz] => {
                    ArrowDataType::Timestamp(TimeUnit::Millisecond, Some(tz.to_string()))
                },
                ["tsu", tz] => {
                    ArrowDataType::Timestamp(TimeUnit::Microsecond, Some(tz.to_string()))
                },
                ["tsn", tz] => ArrowDataType::Timestamp(TimeUnit::Nanosecond, Some(tz.to_string())),

                ["w", size_raw] => {
                    // Example: "w:42" fixed-width binary [42 bytes]
                    let size = size_raw
                        .parse::<usize>()
                        .map_err(|_| polars_err!(ComputeError: "size is not a valid integer"))?;
                    ArrowDataType::FixedSizeBinary(size)
                },
                ["+w", size_raw] => {
                    // Example: "+w:123" fixed-sized list [123 items]
                    let size = size_raw
                        .parse::<usize>()
                        .map_err(|_| polars_err!(ComputeError: "size is not a valid integer"))?;
                    let child = to_field(schema.child(0))?;
                    ArrowDataType::FixedSizeList(Box::new(child), size)
                },
                ["d", raw] => {
                    // Decimal
                    let (precision, scale) = match raw.split(',').collect::<Vec<_>>()[..] {
                        [precision_raw, scale_raw] => {
                            // Example: "d:19,10" decimal128 [precision 19, scale 10]
                            (precision_raw, scale_raw)
                        },
                        [precision_raw, scale_raw, width_raw] => {
                            // Example: "d:19,10,NNN" decimal bitwidth = NNN [precision 19, scale 10]
                            // Only bitwdth of 128 currently supported
                            let bit_width = width_raw.parse::<usize>().map_err(|_| {
                                polars_err!(ComputeError: "Decimal bit width is not a valid integer")
                            })?;
                            if bit_width == 256 {
                                return Ok(ArrowDataType::Decimal256(
                                    precision_raw.parse::<usize>().map_err(|_| {
                                        polars_err!(ComputeError: "Decimal precision is not a valid integer")
                                    })?,
                                    scale_raw.parse::<usize>().map_err(|_| {
                                        polars_err!(ComputeError: "Decimal scale is not a valid integer")
                                    })?,
                                ));
                            }
                            (precision_raw, scale_raw)
                        },
                        _ => {
                            polars_bail!(ComputeError:
                                "Decimal must contain 2 or 3 comma-separated values"
                            )
                        },
                    };

                    ArrowDataType::Decimal(
                        precision.parse::<usize>().map_err(|_| {
                            polars_err!(ComputeError:
                            "Decimal precision is not a valid integer"
                            )
                        })?,
                        scale.parse::<usize>().map_err(|_| {
                            polars_err!(ComputeError:
                            "Decimal scale is not a valid integer"
                            )
                        })?,
                    )
                },
                [union_type @ "+us", union_parts] | [union_type @ "+ud", union_parts] => {
                    // union, sparse
                    // Example "+us:I,J,..." sparse union with type ids I,J...
                    // Example: "+ud:I,J,..." dense union with type ids I,J...
                    let mode = UnionMode::sparse(union_type == "+us");
                    let type_ids = union_parts
                        .split(',')
                        .map(|x| {
                            x.parse::<i32>().map_err(|_| {
                                polars_err!(ComputeError:
                                "Union type id is not a valid integer"
                                )
                            })
                        })
                        .collect::<PolarsResult<Vec<_>>>()?;
                    let fields = (0..schema.n_children as usize)
                        .map(|x| to_field(schema.child(x)))
                        .collect::<PolarsResult<Vec<_>>>()?;
                    ArrowDataType::Union(fields, Some(type_ids), mode)
                },
                _ => {
                    polars_bail!(ComputeError:
                    "The datatype \"{other}\" is still not supported in Rust implementation",
                        )
                },
            }
        },
    })
}

/// the inverse of [to_field]
fn to_format(data_type: &ArrowDataType) -> String {
    match data_type {
        ArrowDataType::Null => "n".to_string(),
        ArrowDataType::Boolean => "b".to_string(),
        ArrowDataType::Int8 => "c".to_string(),
        ArrowDataType::UInt8 => "C".to_string(),
        ArrowDataType::Int16 => "s".to_string(),
        ArrowDataType::UInt16 => "S".to_string(),
        ArrowDataType::Int32 => "i".to_string(),
        ArrowDataType::UInt32 => "I".to_string(),
        ArrowDataType::Int64 => "l".to_string(),
        ArrowDataType::UInt64 => "L".to_string(),
        ArrowDataType::Float16 => "e".to_string(),
        ArrowDataType::Float32 => "f".to_string(),
        ArrowDataType::Float64 => "g".to_string(),
        ArrowDataType::Binary => "z".to_string(),
        ArrowDataType::LargeBinary => "Z".to_string(),
        ArrowDataType::Utf8 => "u".to_string(),
        ArrowDataType::LargeUtf8 => "U".to_string(),
        ArrowDataType::Date32 => "tdD".to_string(),
        ArrowDataType::Date64 => "tdm".to_string(),
        ArrowDataType::Time32(TimeUnit::Second) => "tts".to_string(),
        ArrowDataType::Time32(TimeUnit::Millisecond) => "ttm".to_string(),
        ArrowDataType::Time32(_) => {
            unreachable!("Time32 is only supported for seconds and milliseconds")
        },
        ArrowDataType::Time64(TimeUnit::Microsecond) => "ttu".to_string(),
        ArrowDataType::Time64(TimeUnit::Nanosecond) => "ttn".to_string(),
        ArrowDataType::Time64(_) => {
            unreachable!("Time64 is only supported for micro and nanoseconds")
        },
        ArrowDataType::Duration(TimeUnit::Second) => "tDs".to_string(),
        ArrowDataType::Duration(TimeUnit::Millisecond) => "tDm".to_string(),
        ArrowDataType::Duration(TimeUnit::Microsecond) => "tDu".to_string(),
        ArrowDataType::Duration(TimeUnit::Nanosecond) => "tDn".to_string(),
        ArrowDataType::Interval(IntervalUnit::YearMonth) => "tiM".to_string(),
        ArrowDataType::Interval(IntervalUnit::DayTime) => "tiD".to_string(),
        ArrowDataType::Interval(IntervalUnit::MonthDayNano) => {
            todo!("Spec for FFI for MonthDayNano still not defined.")
        },
        ArrowDataType::Timestamp(unit, tz) => {
            let unit = match unit {
                TimeUnit::Second => "s",
                TimeUnit::Millisecond => "m",
                TimeUnit::Microsecond => "u",
                TimeUnit::Nanosecond => "n",
            };
            format!(
                "ts{}:{}",
                unit,
                tz.as_ref().map(|x| x.as_ref()).unwrap_or("")
            )
        },
        ArrowDataType::Utf8View => "vu".to_string(),
        ArrowDataType::BinaryView => "vz".to_string(),
        ArrowDataType::Decimal(precision, scale) => format!("d:{precision},{scale}"),
        ArrowDataType::Decimal256(precision, scale) => format!("d:{precision},{scale},256"),
        ArrowDataType::List(_) => "+l".to_string(),
        ArrowDataType::LargeList(_) => "+L".to_string(),
        ArrowDataType::Struct(_) => "+s".to_string(),
        ArrowDataType::FixedSizeBinary(size) => format!("w:{size}"),
        ArrowDataType::FixedSizeList(_, size) => format!("+w:{size}"),
        ArrowDataType::Union(f, ids, mode) => {
            let sparsness = if mode.is_sparse() { 's' } else { 'd' };
            let mut r = format!("+u{sparsness}:");
            let ids = if let Some(ids) = ids {
                ids.iter()
                    .fold(String::new(), |a, b| a + &b.to_string() + ",")
            } else {
                (0..f.len()).fold(String::new(), |a, b| a + &b.to_string() + ",")
            };
            let ids = &ids[..ids.len() - 1]; // take away last ","
            r.push_str(ids);
            r
        },
        ArrowDataType::Map(_, _) => "+m".to_string(),
        ArrowDataType::Dictionary(index, _, _) => to_format(&(*index).into()),
        ArrowDataType::Extension(_, inner, _) => to_format(inner.as_ref()),
        ArrowDataType::Unknown => unimplemented!(),
    }
}

pub(super) fn get_child(data_type: &ArrowDataType, index: usize) -> PolarsResult<ArrowDataType> {
    match (index, data_type) {
        (0, ArrowDataType::List(field)) => Ok(field.data_type().clone()),
        (0, ArrowDataType::FixedSizeList(field, _)) => Ok(field.data_type().clone()),
        (0, ArrowDataType::LargeList(field)) => Ok(field.data_type().clone()),
        (0, ArrowDataType::Map(field, _)) => Ok(field.data_type().clone()),
        (index, ArrowDataType::Struct(fields)) => Ok(fields[index].data_type().clone()),
        (index, ArrowDataType::Union(fields, _, _)) => Ok(fields[index].data_type().clone()),
        (index, ArrowDataType::Extension(_, subtype, _)) => get_child(subtype, index),
        (child, data_type) => polars_bail!(ComputeError:
            "Requested child {child} to type {data_type:?} that has no such child",
        ),
    }
}

fn metadata_to_bytes(metadata: &BTreeMap<String, String>) -> Vec<u8> {
    let a = (metadata.len() as i32).to_ne_bytes().to_vec();
    metadata.iter().fold(a, |mut acc, (key, value)| {
        acc.extend((key.len() as i32).to_ne_bytes());
        acc.extend(key.as_bytes());
        acc.extend((value.len() as i32).to_ne_bytes());
        acc.extend(value.as_bytes());
        acc
    })
}

unsafe fn read_ne_i32(ptr: *const u8) -> i32 {
    let slice = std::slice::from_raw_parts(ptr, 4);
    i32::from_ne_bytes(slice.try_into().unwrap())
}

unsafe fn read_bytes(ptr: *const u8, len: usize) -> &'static str {
    let slice = std::slice::from_raw_parts(ptr, len);
    simdutf8::basic::from_utf8(slice).unwrap()
}

unsafe fn metadata_from_bytes(data: *const ::std::os::raw::c_char) -> (Metadata, Extension) {
    let mut data = data as *const u8; // u8 = i8
    if data.is_null() {
        return (Metadata::default(), None);
    };
    let len = read_ne_i32(data);
    data = data.add(4);

    let mut result = BTreeMap::new();
    let mut extension_name = None;
    let mut extension_metadata = None;
    for _ in 0..len {
        let key_len = read_ne_i32(data) as usize;
        data = data.add(4);
        let key = read_bytes(data, key_len);
        data = data.add(key_len);
        let value_len = read_ne_i32(data) as usize;
        data = data.add(4);
        let value = read_bytes(data, value_len);
        data = data.add(value_len);
        match key {
            "ARROW:extension:name" => {
                extension_name = Some(value.to_string());
            },
            "ARROW:extension:metadata" => {
                extension_metadata = Some(value.to_string());
            },
            _ => {
                result.insert(key.to_string(), value.to_string());
            },
        };
    }
    let extension = extension_name.map(|name| (name, extension_metadata));
    (result, extension)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all() {
        let mut dts = vec![
            ArrowDataType::Null,
            ArrowDataType::Boolean,
            ArrowDataType::UInt8,
            ArrowDataType::UInt16,
            ArrowDataType::UInt32,
            ArrowDataType::UInt64,
            ArrowDataType::Int8,
            ArrowDataType::Int16,
            ArrowDataType::Int32,
            ArrowDataType::Int64,
            ArrowDataType::Float32,
            ArrowDataType::Float64,
            ArrowDataType::Date32,
            ArrowDataType::Date64,
            ArrowDataType::Time32(TimeUnit::Second),
            ArrowDataType::Time32(TimeUnit::Millisecond),
            ArrowDataType::Time64(TimeUnit::Microsecond),
            ArrowDataType::Time64(TimeUnit::Nanosecond),
            ArrowDataType::Decimal(5, 5),
            ArrowDataType::Utf8,
            ArrowDataType::LargeUtf8,
            ArrowDataType::Binary,
            ArrowDataType::LargeBinary,
            ArrowDataType::FixedSizeBinary(2),
            ArrowDataType::List(Box::new(Field::new(
                "example",
                ArrowDataType::Boolean,
                false,
            ))),
            ArrowDataType::FixedSizeList(
                Box::new(Field::new("example", ArrowDataType::Boolean, false)),
                2,
            ),
            ArrowDataType::LargeList(Box::new(Field::new(
                "example",
                ArrowDataType::Boolean,
                false,
            ))),
            ArrowDataType::Struct(vec![
                Field::new("a", ArrowDataType::Int64, true),
                Field::new(
                    "b",
                    ArrowDataType::List(Box::new(Field::new("item", ArrowDataType::Int32, true))),
                    true,
                ),
            ]),
            ArrowDataType::Map(Box::new(Field::new("a", ArrowDataType::Int64, true)), true),
            ArrowDataType::Union(
                vec![
                    Field::new("a", ArrowDataType::Int64, true),
                    Field::new(
                        "b",
                        ArrowDataType::List(Box::new(Field::new(
                            "item",
                            ArrowDataType::Int32,
                            true,
                        ))),
                        true,
                    ),
                ],
                Some(vec![1, 2]),
                UnionMode::Dense,
            ),
            ArrowDataType::Union(
                vec![
                    Field::new("a", ArrowDataType::Int64, true),
                    Field::new(
                        "b",
                        ArrowDataType::List(Box::new(Field::new(
                            "item",
                            ArrowDataType::Int32,
                            true,
                        ))),
                        true,
                    ),
                ],
                Some(vec![0, 1]),
                UnionMode::Sparse,
            ),
        ];
        for time_unit in [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ] {
            dts.push(ArrowDataType::Timestamp(time_unit, None));
            dts.push(ArrowDataType::Timestamp(
                time_unit,
                Some("00:00".to_string()),
            ));
            dts.push(ArrowDataType::Duration(time_unit));
        }
        for interval_type in [
            IntervalUnit::DayTime,
            IntervalUnit::YearMonth,
            //IntervalUnit::MonthDayNano, // not yet defined on the C data interface
        ] {
            dts.push(ArrowDataType::Interval(interval_type));
        }

        for expected in dts {
            let field = Field::new("a", expected.clone(), true);
            let schema = ArrowSchema::new(&field);
            let result = unsafe { super::to_data_type(&schema).unwrap() };
            assert_eq!(result, expected);
        }
    }
}
