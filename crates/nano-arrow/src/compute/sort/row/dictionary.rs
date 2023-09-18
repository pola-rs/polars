// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::{
    array::{Array, BinaryArray, DictionaryArray, DictionaryKey, PrimitiveArray, Utf8Array},
    compute::sort::SortOptions,
    datatypes::PhysicalType,
    error::*,
    with_match_primitive_without_interval_type,
};

use super::{
    fixed::FixedLengthEncoding,
    interner::{Interned, OrderPreservingInterner},
    null_sentinel, Rows,
};

/// Computes the dictionary mapping for the given dictionary values
pub fn compute_dictionary_mapping(
    interner: &mut OrderPreservingInterner,
    values: &Box<dyn Array>,
) -> Result<Vec<Option<Interned>>> {
    Ok(match values.data_type().to_physical_type() {
        PhysicalType::Primitive(primitive) => {
            with_match_primitive_without_interval_type!(primitive, |$T| {
                let values = values
                    .as_any()
                    .downcast_ref::<PrimitiveArray<$T>>()
                    .unwrap();
                interner.intern(values.iter().map(|x| x.map(|x| x.encode())))
            })
        }
        PhysicalType::Binary => {
            let iter = values
                .as_any()
                .downcast_ref::<BinaryArray<i32>>()
                .unwrap()
                .iter();
            interner.intern(iter)
        }
        PhysicalType::LargeBinary => {
            let iter = values
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .unwrap()
                .iter();
            interner.intern(iter)
        }
        PhysicalType::Utf8 => {
            let iter = values
                .as_any()
                .downcast_ref::<Utf8Array<i32>>()
                .unwrap()
                .iter()
                .map(|x| x.map(|x| x.as_bytes()));
            interner.intern(iter)
        }
        PhysicalType::LargeUtf8 => {
            let iter = values
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .unwrap()
                .iter()
                .map(|x| x.map(|x| x.as_bytes()));
            interner.intern(iter)
        }
        t => {
            return Err(Error::NotYetImplemented(format!(
                "dictionary value {t:?} is not supported"
            )))
        }
    })
}

/// Dictionary types are encoded as
///
/// - single `0_u8` if null
/// - the bytes of the corresponding normalized key including the null terminator
pub fn encode_dictionary<K: DictionaryKey>(
    out: &mut Rows,
    column: &DictionaryArray<K>,
    normalized_keys: &[Option<&[u8]>],
    opts: SortOptions,
) {
    for (offset, k) in out.offsets.iter_mut().skip(1).zip(column.keys()) {
        match k.and_then(|k| normalized_keys[unsafe { k.as_usize() }]) {
            Some(normalized_key) => {
                let end_offset = *offset + 1 + normalized_key.len();
                out.buffer[*offset] = 1;
                out.buffer[*offset + 1..end_offset].copy_from_slice(normalized_key);
                // Negate if descending
                if opts.descending {
                    out.buffer[*offset..end_offset]
                        .iter_mut()
                        .for_each(|v| *v = !*v)
                }
                *offset = end_offset;
            }
            None => {
                out.buffer[*offset] = null_sentinel(opts);
                *offset += 1;
            }
        }
    }
}
