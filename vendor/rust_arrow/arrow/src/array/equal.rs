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

use super::*;
use crate::datatypes::*;
use crate::util::bit_util;
use hex::FromHex;
use serde_json::value::Value::{Null as JNull, Object, String as JString};
use serde_json::Value;

/// Trait for `Array` equality.
pub trait ArrayEqual {
    /// Returns true if this array is equal to the `other` array
    fn equals(&self, other: &dyn Array) -> bool;

    /// Returns true if the range [start_idx, end_idx) is equal to
    /// [other_start_idx, other_start_idx + end_idx - start_idx) in the `other` array
    fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool;
}

impl<T: ArrowPrimitiveType> ArrayEqual for PrimitiveArray<T> {
    default fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let value_buf = self.data_ref().buffers()[0].clone();
        let other_value_buf = other.data_ref().buffers()[0].clone();
        let byte_width = T::get_bit_width() / 8;

        if self.null_count() > 0 {
            let values = value_buf.data();
            let other_values = other_value_buf.data();

            for i in 0..self.len() {
                if self.is_valid(i) {
                    let start = (i + self.offset()) * byte_width;
                    let data = &values[start..(start + byte_width)];
                    let other_start = (i + other.offset()) * byte_width;
                    let other_data =
                        &other_values[other_start..(other_start + byte_width)];
                    if data != other_data {
                        return false;
                    }
                }
            }
        } else {
            let start = self.offset() * byte_width;
            let other_start = other.offset() * byte_width;
            let len = self.len() * byte_width;
            let data = &value_buf.data()[start..(start + len)];
            let other_data = &other_value_buf.data()[other_start..(other_start + len)];
            if data != other_data {
                return false;
            }
        }

        true
    }

    default fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        assert!(other_start_idx + (end_idx - start_idx) <= other.len());
        let other = other.as_any().downcast_ref::<PrimitiveArray<T>>().unwrap();

        let mut j = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(j);
            if is_null != other_is_null || (!is_null && self.value(i) != other.value(j)) {
                return false;
            }
            j += 1;
        }

        true
    }
}

impl ArrayEqual for BooleanArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let values = self.data_ref().buffers()[0].data();
        let other_values = other.data_ref().buffers()[0].data();

        // TODO: we can do this more efficiently if all values are not-null
        for i in 0..self.len() {
            if self.is_valid(i)
                && bit_util::get_bit(values, i + self.offset())
                    != bit_util::get_bit(other_values, i + other.offset())
            {
                return false;
            }
        }

        true
    }
}

impl<T: ArrowNumericType> PartialEq for PrimitiveArray<T> {
    fn eq(&self, other: &PrimitiveArray<T>) -> bool {
        self.equals(other)
    }
}

impl PartialEq for BooleanArray {
    fn eq(&self, other: &BooleanArray) -> bool {
        self.equals(other)
    }
}

impl PartialEq for StringArray {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl PartialEq for FixedSizeBinaryArray {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl PartialEq for BinaryArray {
    fn eq(&self, other: &Self) -> bool {
        self.equals(other)
    }
}

impl ArrayEqual for ListArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other.as_any().downcast_ref::<ListArray>().unwrap();

        if !value_offset_equal(self, other) {
            return false;
        }

        if !self.values().range_equals(
            &*other.values(),
            self.value_offset(0) as usize,
            self.value_offset(self.len()) as usize,
            other.value_offset(0) as usize,
        ) {
            return false;
        }

        true
    }

    fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        assert!(other_start_idx + (end_idx - start_idx) <= other.len());
        let other = other.as_any().downcast_ref::<ListArray>().unwrap();

        let mut j = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(j);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }

            let start_offset = self.value_offset(i) as usize;
            let end_offset = self.value_offset(i + 1) as usize;
            let other_start_offset = other.value_offset(j) as usize;
            let other_end_offset = other.value_offset(j + 1) as usize;

            if end_offset - start_offset != other_end_offset - other_start_offset {
                return false;
            }

            if !self.values().range_equals(
                &*other.values(),
                start_offset,
                end_offset,
                other_start_offset,
            ) {
                return false;
            }

            j += 1;
        }

        true
    }
}

impl<T: ArrowPrimitiveType> ArrayEqual for DictionaryArray<T> {
    fn equals(&self, other: &dyn Array) -> bool {
        self.range_equals(other, 0, self.len(), 0)
    }

    default fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        assert!(other_start_idx + (end_idx - start_idx) <= other.len());
        let other = other.as_any().downcast_ref::<DictionaryArray<T>>().unwrap();

        let iter_a = self.keys().take(end_idx).skip(start_idx);
        let iter_b = other.keys().skip(other_start_idx);

        // For now, all the values must be the same
        iter_a.eq(iter_b)
            && self
                .values()
                .range_equals(&*other.values(), 0, other.values().len(), 0)
    }
}

impl ArrayEqual for FixedSizeListArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        if !self.values().range_equals(
            &*other.values(),
            self.value_offset(0) as usize,
            self.value_offset(self.len()) as usize,
            other.value_offset(0) as usize,
        ) {
            return false;
        }

        true
    }

    fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        assert!(other_start_idx + (end_idx - start_idx) <= other.len());
        let other = other.as_any().downcast_ref::<FixedSizeListArray>().unwrap();

        let mut j = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(j);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }

            let start_offset = self.value_offset(i) as usize;
            let end_offset = self.value_offset(i + 1) as usize;
            let other_start_offset = other.value_offset(j) as usize;
            let other_end_offset = other.value_offset(j + 1) as usize;

            if end_offset - start_offset != other_end_offset - other_start_offset {
                return false;
            }

            if !self.values().range_equals(
                &*other.values(),
                start_offset,
                end_offset,
                other_start_offset,
            ) {
                return false;
            }

            j += 1;
        }

        true
    }
}

impl ArrayEqual for BinaryArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other.as_any().downcast_ref::<BinaryArray>().unwrap();

        if !value_offset_equal(self, other) {
            return false;
        }

        // TODO: handle null & length == 0 case?

        let value_buf = self.value_data();
        let other_value_buf = other.value_data();
        let value_data = value_buf.data();
        let other_value_data = other_value_buf.data();

        if self.null_count() == 0 {
            // No offset in both - just do memcmp
            if self.offset() == 0 && other.offset() == 0 {
                let len = self.value_offset(self.len()) as usize;
                return value_data[..len] == other_value_data[..len];
            } else {
                let start = self.value_offset(0) as usize;
                let other_start = other.value_offset(0) as usize;
                let len = (self.value_offset(self.len()) - self.value_offset(0)) as usize;
                return value_data[start..(start + len)]
                    == other_value_data[other_start..(other_start + len)];
            }
        } else {
            for i in 0..self.len() {
                if self.is_null(i) {
                    continue;
                }

                let start = self.value_offset(i) as usize;
                let other_start = other.value_offset(i) as usize;
                let len = self.value_length(i) as usize;
                if value_data[start..(start + len)]
                    != other_value_data[other_start..(other_start + len)]
                {
                    return false;
                }
            }
        }

        true
    }

    fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        assert!(other_start_idx + (end_idx - start_idx) <= other.len());
        let other = other.as_any().downcast_ref::<BinaryArray>().unwrap();

        let mut j = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(j);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }

            let start_offset = self.value_offset(i) as usize;
            let end_offset = self.value_offset(i + 1) as usize;
            let other_start_offset = other.value_offset(j) as usize;
            let other_end_offset = other.value_offset(j + 1) as usize;

            if end_offset - start_offset != other_end_offset - other_start_offset {
                return false;
            }

            let value_buf = self.value_data();
            let other_value_buf = other.value_data();
            let value_data = value_buf.data();
            let other_value_data = other_value_buf.data();

            if end_offset - start_offset > 0 {
                let len = end_offset - start_offset;
                if value_data[start_offset..(start_offset + len)]
                    != other_value_data[other_start_offset..(other_start_offset + len)]
                {
                    return false;
                }
            }

            j += 1;
        }

        true
    }
}

impl ArrayEqual for StringArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other.as_any().downcast_ref::<StringArray>().unwrap();

        if !value_offset_equal(self, other) {
            return false;
        }

        // TODO: handle null & length == 0 case?

        let value_buf = self.value_data();
        let other_value_buf = other.value_data();
        let value_data = value_buf.data();
        let other_value_data = other_value_buf.data();

        if self.null_count() == 0 {
            // No offset in both - just do memcmp
            if self.offset() == 0 && other.offset() == 0 {
                let len = self.value_offset(self.len()) as usize;
                return value_data[..len] == other_value_data[..len];
            } else {
                let start = self.value_offset(0) as usize;
                let other_start = other.value_offset(0) as usize;
                let len = (self.value_offset(self.len()) - self.value_offset(0)) as usize;
                return value_data[start..(start + len)]
                    == other_value_data[other_start..(other_start + len)];
            }
        } else {
            for i in 0..self.len() {
                if self.is_null(i) {
                    continue;
                }

                let start = self.value_offset(i) as usize;
                let other_start = other.value_offset(i) as usize;
                let len = self.value_length(i) as usize;
                if value_data[start..(start + len)]
                    != other_value_data[other_start..(other_start + len)]
                {
                    return false;
                }
            }
        }

        true
    }

    fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        assert!(other_start_idx + (end_idx - start_idx) <= other.len());
        let other = other.as_any().downcast_ref::<StringArray>().unwrap();

        let mut j = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(j);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }

            let start_offset = self.value_offset(i) as usize;
            let end_offset = self.value_offset(i + 1) as usize;
            let other_start_offset = other.value_offset(j) as usize;
            let other_end_offset = other.value_offset(j + 1) as usize;

            if end_offset - start_offset != other_end_offset - other_start_offset {
                return false;
            }

            let value_buf = self.value_data();
            let other_value_buf = other.value_data();
            let value_data = value_buf.data();
            let other_value_data = other_value_buf.data();

            if end_offset - start_offset > 0 {
                let len = end_offset - start_offset;
                if value_data[start_offset..(start_offset + len)]
                    != other_value_data[other_start_offset..(other_start_offset + len)]
                {
                    return false;
                }
            }

            j += 1;
        }

        true
    }
}

impl ArrayEqual for FixedSizeBinaryArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();

        if !value_offset_equal(self, other) {
            return false;
        }

        // TODO: handle null & length == 0 case?

        let value_buf = self.value_data();
        let other_value_buf = other.value_data();
        let value_data = value_buf.data();
        let other_value_data = other_value_buf.data();

        if self.null_count() == 0 {
            // No offset in both - just do memcmp
            if self.offset() == 0 && other.offset() == 0 {
                let len = self.value_offset(self.len()) as usize;
                return value_data[..len] == other_value_data[..len];
            } else {
                let start = self.value_offset(0) as usize;
                let other_start = other.value_offset(0) as usize;
                let len = (self.value_offset(self.len()) - self.value_offset(0)) as usize;
                return value_data[start..(start + len)]
                    == other_value_data[other_start..(other_start + len)];
            }
        } else {
            for i in 0..self.len() {
                if self.is_null(i) {
                    continue;
                }

                let start = self.value_offset(i) as usize;
                let other_start = other.value_offset(i) as usize;
                let len = self.value_length() as usize;
                if value_data[start..(start + len)]
                    != other_value_data[other_start..(other_start + len)]
                {
                    return false;
                }
            }
        }

        true
    }

    fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        assert!(other_start_idx + (end_idx - start_idx) <= other.len());
        let other = other
            .as_any()
            .downcast_ref::<FixedSizeBinaryArray>()
            .unwrap();

        let mut j = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(j);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }

            let start_offset = self.value_offset(i) as usize;
            let end_offset = self.value_offset(i + 1) as usize;
            let other_start_offset = other.value_offset(j) as usize;
            let other_end_offset = other.value_offset(j + 1) as usize;

            if end_offset - start_offset != other_end_offset - other_start_offset {
                return false;
            }

            let value_buf = self.value_data();
            let other_value_buf = other.value_data();
            let value_data = value_buf.data();
            let other_value_data = other_value_buf.data();

            if end_offset - start_offset > 0 {
                let len = end_offset - start_offset;
                if value_data[start_offset..(start_offset + len)]
                    != other_value_data[other_start_offset..(other_start_offset + len)]
                {
                    return false;
                }
            }

            j += 1;
        }

        true
    }
}

impl ArrayEqual for StructArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if !base_equal(&self.data(), &other.data()) {
            return false;
        }

        let other = other.as_any().downcast_ref::<StructArray>().unwrap();

        for i in 0..self.len() {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(i);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }
            for j in 0..self.num_columns() {
                if !self.column(j).range_equals(&**other.column(j), i, i + 1, i) {
                    return false;
                }
            }
        }

        true
    }

    fn range_equals(
        &self,
        other: &dyn Array,
        start_idx: usize,
        end_idx: usize,
        other_start_idx: usize,
    ) -> bool {
        assert!(other_start_idx + (end_idx - start_idx) <= other.len());
        let other = other.as_any().downcast_ref::<StructArray>().unwrap();

        let mut j = other_start_idx;
        for i in start_idx..end_idx {
            let is_null = self.is_null(i);
            let other_is_null = other.is_null(i);

            if is_null != other_is_null {
                return false;
            }

            if is_null {
                continue;
            }
            for k in 0..self.num_columns() {
                if !self.column(k).range_equals(&**other.column(k), i, i + 1, j) {
                    return false;
                }
            }

            j += 1;
        }

        true
    }
}

impl ArrayEqual for UnionArray {
    fn equals(&self, _other: &dyn Array) -> bool {
        unimplemented!(
            "Added to allow UnionArray to implement the Array trait: see ARROW-8576"
        )
    }

    fn range_equals(
        &self,
        _other: &dyn Array,
        _start_idx: usize,
        _end_idx: usize,
        _other_start_idx: usize,
    ) -> bool {
        unimplemented!(
            "Added to allow UnionArray to implement the Array trait: see ARROW-8576"
        )
    }
}

impl ArrayEqual for NullArray {
    fn equals(&self, other: &dyn Array) -> bool {
        if other.data_type() != &DataType::Null {
            return false;
        }

        if self.len() != other.len() {
            return false;
        }
        if self.null_count() != other.null_count() {
            return false;
        }

        true
    }

    fn range_equals(
        &self,
        _other: &dyn Array,
        _start_idx: usize,
        _end_idx: usize,
        _other_start_idx: usize,
    ) -> bool {
        unimplemented!("Range comparison for null array not yet supported")
    }
}

// Compare if the common basic fields between the two arrays are equal
fn base_equal(this: &ArrayDataRef, other: &ArrayDataRef) -> bool {
    if this.data_type() != other.data_type() {
        return false;
    }
    if this.len != other.len {
        return false;
    }
    if this.null_count != other.null_count {
        return false;
    }
    if this.null_count > 0 {
        let null_bitmap = this.null_bitmap().as_ref().unwrap();
        let other_null_bitmap = other.null_bitmap().as_ref().unwrap();
        let null_buf = null_bitmap.bits.data();
        let other_null_buf = other_null_bitmap.bits.data();
        for i in 0..this.len() {
            if bit_util::get_bit(null_buf, i + this.offset())
                != bit_util::get_bit(other_null_buf, i + other.offset())
            {
                return false;
            }
        }
    }
    true
}

// Compare if the value offsets are equal between the two list arrays
fn value_offset_equal<T: Array + ListArrayOps>(this: &T, other: &T) -> bool {
    // Check if offsets differ
    if this.offset() == 0 && other.offset() == 0 {
        let offset_data = &this.data_ref().buffers()[0];
        let other_offset_data = &other.data_ref().buffers()[0];
        return offset_data.data()[0..((this.len() + 1) * 4)]
            == other_offset_data.data()[0..((other.len() + 1) * 4)];
    }

    // The expensive case
    for i in 0..=this.len() {
        if this.value_offset_at(i) - this.value_offset_at(0)
            != other.value_offset_at(i) - other.value_offset_at(0)
        {
            return false;
        }
    }

    true
}

/// Trait for comparing some array with json array
pub trait JsonEqual {
    /// Checks whether some array equals to json array.
    fn equals_json(&self, json: &[&Value]) -> bool;

    /// Checks whether some array equals to json array.
    fn equals_json_values(&self, json: &[Value]) -> bool {
        let refs = json.iter().collect::<Vec<&Value>>();

        self.equals_json(&refs)
    }
}

/// Implement array equals for numeric type
impl<T: ArrowPrimitiveType> JsonEqual for PrimitiveArray<T> {
    fn equals_json(&self, json: &[&Value]) -> bool {
        if self.len() != json.len() {
            return false;
        }

        (0..self.len()).all(|i| match json[i] {
            Value::Null => self.is_null(i),
            v => self.is_valid(i) && Some(v) == self.value(i).into_json_value().as_ref(),
        })
    }
}

impl<T: ArrowPrimitiveType> PartialEq<Value> for PrimitiveArray<T> {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(array) => self.equals_json_values(&array),
            _ => false,
        }
    }
}

impl<T: ArrowPrimitiveType> PartialEq<PrimitiveArray<T>> for Value {
    fn eq(&self, arrow: &PrimitiveArray<T>) -> bool {
        match self {
            Value::Array(array) => arrow.equals_json_values(&array),
            _ => false,
        }
    }
}

impl JsonEqual for ListArray {
    fn equals_json(&self, json: &[&Value]) -> bool {
        if self.len() != json.len() {
            return false;
        }

        (0..self.len()).all(|i| match json[i] {
            Value::Array(v) => self.is_valid(i) && self.value(i).equals_json_values(v),
            Value::Null => self.is_null(i) || self.value_length(i) == 0,
            _ => false,
        })
    }
}

impl PartialEq<Value> for ListArray {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(json_array) => self.equals_json_values(json_array),
            _ => false,
        }
    }
}

impl PartialEq<ListArray> for Value {
    fn eq(&self, arrow: &ListArray) -> bool {
        match self {
            Value::Array(json_array) => arrow.equals_json_values(json_array),
            _ => false,
        }
    }
}

impl<T: ArrowPrimitiveType> JsonEqual for DictionaryArray<T> {
    fn equals_json(&self, json: &[&Value]) -> bool {
        self.keys().zip(json.iter()).all(|aj| match aj {
            (None, Value::Null) => true,
            (Some(a), Value::Number(j)) => {
                a.to_usize().unwrap() as u64 == j.as_u64().unwrap()
            }
            _ => false,
        })
    }
}

impl<T: ArrowPrimitiveType> PartialEq<Value> for DictionaryArray<T> {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(json_array) => self.equals_json_values(json_array),
            _ => false,
        }
    }
}

impl<T: ArrowPrimitiveType> PartialEq<DictionaryArray<T>> for Value {
    fn eq(&self, arrow: &DictionaryArray<T>) -> bool {
        match self {
            Value::Array(json_array) => arrow.equals_json_values(json_array),
            _ => false,
        }
    }
}

impl JsonEqual for FixedSizeListArray {
    fn equals_json(&self, json: &[&Value]) -> bool {
        if self.len() != json.len() {
            return false;
        }

        (0..self.len()).all(|i| match json[i] {
            Value::Array(v) => self.is_valid(i) && self.value(i).equals_json_values(v),
            Value::Null => self.is_null(i) || self.value_length() == 0,
            _ => false,
        })
    }
}

impl PartialEq<Value> for FixedSizeListArray {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(json_array) => self.equals_json_values(json_array),
            _ => false,
        }
    }
}

impl PartialEq<FixedSizeListArray> for Value {
    fn eq(&self, arrow: &FixedSizeListArray) -> bool {
        match self {
            Value::Array(json_array) => arrow.equals_json_values(json_array),
            _ => false,
        }
    }
}

impl JsonEqual for StructArray {
    fn equals_json(&self, json: &[&Value]) -> bool {
        if self.len() != json.len() {
            return false;
        }

        let all_object = json.iter().all(|v| match v {
            Object(_) | JNull => true,
            _ => false,
        });

        if !all_object {
            return false;
        }

        for column_name in self.column_names() {
            let json_values = json
                .iter()
                .map(|obj| obj.get(column_name).unwrap_or(&Value::Null))
                .collect::<Vec<&Value>>();

            if !self
                .column_by_name(column_name)
                .map(|arr| arr.equals_json(&json_values))
                .unwrap_or(false)
            {
                return false;
            }
        }

        true
    }
}

impl PartialEq<Value> for StructArray {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(json_array) => self.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl PartialEq<StructArray> for Value {
    fn eq(&self, arrow: &StructArray) -> bool {
        match self {
            Value::Array(json_array) => arrow.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl JsonEqual for BinaryArray {
    fn equals_json(&self, json: &[&Value]) -> bool {
        if self.len() != json.len() {
            return false;
        }

        (0..self.len()).all(|i| match json[i] {
            JString(s) => {
                // binary data is sometimes hex encoded, this checks if bytes are equal,
                // and if not converting to hex is attempted
                self.is_valid(i)
                    && (s.as_str().as_bytes() == self.value(i)
                        || Vec::from_hex(s.as_str()) == Ok(self.value(i).to_vec()))
            }
            JNull => self.is_null(i),
            _ => false,
        })
    }
}

impl PartialEq<Value> for BinaryArray {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(json_array) => self.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl PartialEq<BinaryArray> for Value {
    fn eq(&self, arrow: &BinaryArray) -> bool {
        match self {
            Value::Array(json_array) => arrow.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl JsonEqual for StringArray {
    fn equals_json(&self, json: &[&Value]) -> bool {
        if self.len() != json.len() {
            return false;
        }

        (0..self.len()).all(|i| match json[i] {
            JString(s) => self.is_valid(i) && s.as_str() == self.value(i),
            JNull => self.is_null(i),
            _ => false,
        })
    }
}

impl PartialEq<Value> for StringArray {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(json_array) => self.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl PartialEq<StringArray> for Value {
    fn eq(&self, arrow: &StringArray) -> bool {
        match self {
            Value::Array(json_array) => arrow.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl JsonEqual for FixedSizeBinaryArray {
    fn equals_json(&self, json: &[&Value]) -> bool {
        if self.len() != json.len() {
            return false;
        }

        (0..self.len()).all(|i| match json[i] {
            JString(s) => {
                // binary data is sometimes hex encoded, this checks if bytes are equal,
                // and if not converting to hex is attempted
                self.is_valid(i)
                    && (s.as_str().as_bytes() == self.value(i)
                        || Vec::from_hex(s.as_str()) == Ok(self.value(i).to_vec()))
            }
            JNull => self.is_null(i),
            _ => false,
        })
    }
}

impl PartialEq<Value> for FixedSizeBinaryArray {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(json_array) => self.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl PartialEq<FixedSizeBinaryArray> for Value {
    fn eq(&self, arrow: &FixedSizeBinaryArray) -> bool {
        match self {
            Value::Array(json_array) => arrow.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl JsonEqual for UnionArray {
    fn equals_json(&self, _json: &[&Value]) -> bool {
        unimplemented!(
            "Added to allow UnionArray to implement the Array trait: see ARROW-8547"
        )
    }
}

impl JsonEqual for NullArray {
    fn equals_json(&self, json: &[&Value]) -> bool {
        if self.len() != json.len() {
            return false;
        }

        // all JSON values must be nulls
        json.iter().all(|&v| v == &JNull)
    }
}

impl PartialEq<NullArray> for Value {
    fn eq(&self, arrow: &NullArray) -> bool {
        match self {
            Value::Array(json_array) => arrow.equals_json_values(&json_array),
            _ => false,
        }
    }
}

impl PartialEq<Value> for NullArray {
    fn eq(&self, json: &Value) -> bool {
        match json {
            Value::Array(json_array) => self.equals_json_values(&json_array),
            _ => false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::convert::TryFrom;

    use crate::error::Result;

    #[test]
    fn test_primitive_equal() {
        let a = Int32Array::from(vec![1, 2, 3]);
        let b = Int32Array::from(vec![1, 2, 3]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = Int32Array::from(vec![1, 2, 4]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where null_count > 0

        let a = Int32Array::from(vec![Some(1), None, Some(2), Some(3)]);
        let b = Int32Array::from(vec![Some(1), None, Some(2), Some(3)]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = Int32Array::from(vec![Some(1), None, None, Some(3)]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let b = Int32Array::from(vec![Some(1), None, Some(2), Some(4)]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where offset != 0

        let a_slice = a.slice(1, 2);
        let b_slice = b.slice(1, 2);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));
    }

    #[test]
    fn test_boolean_equal() {
        let a = BooleanArray::from(vec![false, false, true]);
        let b = BooleanArray::from(vec![false, false, true]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = BooleanArray::from(vec![false, false, false]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where null_count > 0

        let a = BooleanArray::from(vec![Some(false), None, None, Some(true)]);
        let b = BooleanArray::from(vec![Some(false), None, None, Some(true)]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = BooleanArray::from(vec![None, None, None, Some(true)]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let b = BooleanArray::from(vec![Some(true), None, None, Some(true)]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where offset != 0

        let a = BooleanArray::from(vec![false, true, false, true, false, false, true]);
        let b = BooleanArray::from(vec![false, false, false, true, false, true, true]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let a_slice = a.slice(2, 3);
        let b_slice = b.slice(2, 3);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));

        let a_slice = a.slice(3, 4);
        let b_slice = b.slice(3, 4);
        assert!(!a_slice.equals(&*b_slice));
        assert!(!b_slice.equals(&*a_slice));
    }

    #[test]
    fn test_list_equal() {
        let mut a_builder = ListBuilder::new(Int32Builder::new(10));
        let mut b_builder = ListBuilder::new(Int32Builder::new(10));

        let a = create_list_array(&mut a_builder, &[Some(&[1, 2, 3]), Some(&[4, 5, 6])])
            .unwrap();
        let b = create_list_array(&mut b_builder, &[Some(&[1, 2, 3]), Some(&[4, 5, 6])])
            .unwrap();

        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = create_list_array(&mut a_builder, &[Some(&[1, 2, 3]), Some(&[4, 5, 7])])
            .unwrap();
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where null_count > 0

        let a = create_list_array(
            &mut a_builder,
            &[Some(&[1, 2]), None, None, Some(&[3, 4]), None, None],
        )
        .unwrap();
        let b = create_list_array(
            &mut a_builder,
            &[Some(&[1, 2]), None, None, Some(&[3, 4]), None, None],
        )
        .unwrap();
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = create_list_array(
            &mut a_builder,
            &[
                Some(&[1, 2]),
                None,
                Some(&[5, 6]),
                Some(&[3, 4]),
                None,
                None,
            ],
        )
        .unwrap();
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let b = create_list_array(
            &mut a_builder,
            &[Some(&[1, 2]), None, None, Some(&[3, 5]), None, None],
        )
        .unwrap();
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where offset != 0

        let a_slice = a.slice(0, 3);
        let b_slice = b.slice(0, 3);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));

        let a_slice = a.slice(0, 5);
        let b_slice = b.slice(0, 5);
        assert!(!a_slice.equals(&*b_slice));
        assert!(!b_slice.equals(&*a_slice));

        let a_slice = a.slice(4, 1);
        let b_slice = b.slice(4, 1);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));
    }

    #[test]
    fn test_fixed_size_list_equal() {
        let mut a_builder = FixedSizeListBuilder::new(Int32Builder::new(10), 3);
        let mut b_builder = FixedSizeListBuilder::new(Int32Builder::new(10), 3);

        let a = create_fixed_size_list_array(
            &mut a_builder,
            &[Some(&[1, 2, 3]), Some(&[4, 5, 6])],
        )
        .unwrap();
        let b = create_fixed_size_list_array(
            &mut b_builder,
            &[Some(&[1, 2, 3]), Some(&[4, 5, 6])],
        )
        .unwrap();

        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = create_fixed_size_list_array(
            &mut a_builder,
            &[Some(&[1, 2, 3]), Some(&[4, 5, 7])],
        )
        .unwrap();
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where null_count > 0

        let a = create_fixed_size_list_array(
            &mut a_builder,
            &[Some(&[1, 2, 3]), None, None, Some(&[4, 5, 6]), None, None],
        )
        .unwrap();
        let b = create_fixed_size_list_array(
            &mut a_builder,
            &[Some(&[1, 2, 3]), None, None, Some(&[4, 5, 6]), None, None],
        )
        .unwrap();
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = create_fixed_size_list_array(
            &mut a_builder,
            &[
                Some(&[1, 2, 3]),
                None,
                Some(&[7, 8, 9]),
                Some(&[4, 5, 6]),
                None,
                None,
            ],
        )
        .unwrap();
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let b = create_fixed_size_list_array(
            &mut a_builder,
            &[Some(&[1, 2, 3]), None, None, Some(&[3, 6, 9]), None, None],
        )
        .unwrap();
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where offset != 0

        let a_slice = a.slice(0, 3);
        let b_slice = b.slice(0, 3);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));

        // let a_slice = a.slice(0, 5);
        // let b_slice = b.slice(0, 5);
        // assert!(!a_slice.equals(&*b_slice));
        // assert!(!b_slice.equals(&*a_slice));

        // let a_slice = a.slice(4, 1);
        // let b_slice = b.slice(4, 1);
        // assert!(a_slice.equals(&*b_slice));
        // assert!(b_slice.equals(&*a_slice));
    }

    #[test]
    fn test_string_equal() {
        let a = StringArray::from(vec!["hello", "world"]);
        let b = StringArray::from(vec!["hello", "world"]);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = StringArray::from(vec!["hello", "some"]);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where null_count > 0

        let a = StringArray::try_from(vec![
            Some("hello"),
            None,
            None,
            Some("world"),
            None,
            None,
        ])
        .unwrap();

        let b = StringArray::try_from(vec![
            Some("hello"),
            None,
            None,
            Some("world"),
            None,
            None,
        ])
        .unwrap();
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = StringArray::try_from(vec![
            Some("hello"),
            Some("foo"),
            None,
            Some("world"),
            None,
            None,
        ])
        .unwrap();
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        let b = StringArray::try_from(vec![
            Some("hello"),
            None,
            None,
            Some("some"),
            None,
            None,
        ])
        .unwrap();
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where offset != 0

        let a_slice = a.slice(0, 3);
        let b_slice = b.slice(0, 3);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));

        let a_slice = a.slice(0, 5);
        let b_slice = b.slice(0, 5);
        assert!(!a_slice.equals(&*b_slice));
        assert!(!b_slice.equals(&*a_slice));

        let a_slice = a.slice(4, 1);
        let b_slice = b.slice(4, 1);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));
    }

    #[test]
    fn test_struct_equal() {
        let string_builder = StringBuilder::new(5);
        let int_builder = Int32Builder::new(5);

        let mut fields = Vec::new();
        let mut field_builders = Vec::new();
        fields.push(Field::new("f1", DataType::Utf8, false));
        field_builders.push(Box::new(string_builder) as Box<ArrayBuilder>);
        fields.push(Field::new("f2", DataType::Int32, false));
        field_builders.push(Box::new(int_builder) as Box<ArrayBuilder>);

        let mut builder = StructBuilder::new(fields, field_builders);

        let a = create_struct_array(
            &mut builder,
            &[Some("joe"), None, None, Some("mark"), Some("doe")],
            &[Some(1), Some(2), None, Some(4), Some(5)],
            &[true, true, false, true, true],
        )
        .unwrap();
        let b = create_struct_array(
            &mut builder,
            &[Some("joe"), None, None, Some("mark"), Some("doe")],
            &[Some(1), Some(2), None, Some(4), Some(5)],
            &[true, true, false, true, true],
        )
        .unwrap();

        assert!(a.equals(&b));
        assert!(b.equals(&a));
    }

    #[test]
    fn test_null_equal() {
        let a = NullArray::new(12);
        let b = NullArray::new(12);
        assert!(a.equals(&b));
        assert!(b.equals(&a));

        let b = NullArray::new(10);
        assert!(!a.equals(&b));
        assert!(!b.equals(&a));

        // Test the case where offset != 0

        let a_slice = a.slice(2, 3);
        let b_slice = b.slice(1, 3);
        assert!(a_slice.equals(&*b_slice));
        assert!(b_slice.equals(&*a_slice));

        let a_slice = a.slice(5, 4);
        let b_slice = b.slice(3, 3);
        assert!(!a_slice.equals(&*b_slice));
        assert!(!b_slice.equals(&*a_slice));
    }

    fn create_list_array<'a, U: AsRef<[i32]>, T: AsRef<[Option<U>]>>(
        builder: &'a mut ListBuilder<Int32Builder>,
        data: T,
    ) -> Result<ListArray> {
        for d in data.as_ref() {
            if let Some(v) = d {
                builder.values().append_slice(v.as_ref())?;
                builder.append(true)?
            } else {
                builder.append(false)?
            }
        }
        Ok(builder.finish())
    }

    /// Create a fixed size list of 2 value lengths
    fn create_fixed_size_list_array<'a, U: AsRef<[i32]>, T: AsRef<[Option<U>]>>(
        builder: &'a mut FixedSizeListBuilder<Int32Builder>,
        data: T,
    ) -> Result<FixedSizeListArray> {
        for d in data.as_ref() {
            if let Some(v) = d {
                builder.values().append_slice(v.as_ref())?;
                builder.append(true)?
            } else {
                for _ in 0..builder.value_length() {
                    builder.values().append_null()?;
                }
                builder.append(false)?
            }
        }
        Ok(builder.finish())
    }

    #[test]
    fn test_primitive_json_equal() {
        // Test equaled array
        let arrow_array = Int32Array::from(vec![Some(1), None, Some(2), Some(3)]);
        let json_array: Value = serde_json::from_str(
            r#"
            [
                1, null, 2, 3
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.eq(&json_array));
        assert!(json_array.eq(&arrow_array));

        // Test unequaled array
        let arrow_array = Int32Array::from(vec![Some(1), None, Some(2), Some(3)]);
        let json_array: Value = serde_json::from_str(
            r#"
            [
                1, 1, 2, 3
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test unequal length case
        let arrow_array = Int32Array::from(vec![Some(1), None, Some(2), Some(3)]);
        let json_array: Value = serde_json::from_str(
            r#"
            [
                1, 1
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test not json array type case
        let arrow_array = Int32Array::from(vec![Some(1), None, Some(2), Some(3)]);
        let json_array: Value = serde_json::from_str(
            r#"
            {
               "a": 1
            }
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));
    }

    #[test]
    fn test_list_json_equal() {
        // Test equal case
        let arrow_array = create_list_array(
            &mut ListBuilder::new(Int32Builder::new(10)),
            &[Some(&[1, 2, 3]), None, Some(&[4, 5, 6])],
        )
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                [1, 2, 3],
                null,
                [4, 5, 6]
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.eq(&json_array));
        assert!(json_array.eq(&arrow_array));

        // Test unequal case
        let arrow_array = create_list_array(
            &mut ListBuilder::new(Int32Builder::new(10)),
            &[Some(&[1, 2, 3]), None, Some(&[4, 5, 6])],
        )
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                [1, 2, 3],
                [7, 8],
                [4, 5, 6]
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect type case
        let arrow_array = create_list_array(
            &mut ListBuilder::new(Int32Builder::new(10)),
            &[Some(&[1, 2, 3]), None, Some(&[4, 5, 6])],
        )
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            {
               "a": 1
            }
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));
    }

    #[test]
    fn test_fixed_size_list_json_equal() {
        // Test equal case
        let arrow_array = create_fixed_size_list_array(
            &mut FixedSizeListBuilder::new(Int32Builder::new(10), 3),
            &[Some(&[1, 2, 3]), None, Some(&[4, 5, 6])],
        )
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                [1, 2, 3],
                null,
                [4, 5, 6]
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.eq(&json_array));
        assert!(json_array.eq(&arrow_array));

        // Test unequal case
        let arrow_array = create_fixed_size_list_array(
            &mut FixedSizeListBuilder::new(Int32Builder::new(10), 3),
            &[Some(&[1, 2, 3]), None, Some(&[4, 5, 6])],
        )
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                [1, 2, 3],
                [7, 8, 9],
                [4, 5, 6]
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect type case
        let arrow_array = create_fixed_size_list_array(
            &mut FixedSizeListBuilder::new(Int32Builder::new(10), 3),
            &[Some(&[1, 2, 3]), None, Some(&[4, 5, 6])],
        )
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            {
               "a": 1
            }
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));
    }

    #[test]
    fn test_string_json_equal() {
        // Test the equal case
        let arrow_array = StringArray::try_from(vec![
            Some("hello"),
            None,
            None,
            Some("world"),
            None,
            None,
        ])
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                "world",
                null,
                null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.eq(&json_array));
        assert!(json_array.eq(&arrow_array));

        // Test unequal case
        let arrow_array = StringArray::try_from(vec![
            Some("hello"),
            None,
            None,
            Some("world"),
            None,
            None,
        ])
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                "arrow",
                null,
                null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test unequal length case
        let arrow_array =
            StringArray::try_from(vec![Some("hello"), None, None, Some("world"), None])
                .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                "arrow",
                null,
                null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect type case
        let arrow_array =
            StringArray::try_from(vec![Some("hello"), None, None, Some("world"), None])
                .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            {
                "a": 1
            }
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect value type case
        let arrow_array =
            StringArray::try_from(vec![Some("hello"), None, None, Some("world"), None])
                .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                1,
                null,
                null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));
    }

    #[test]
    fn test_binary_json_equal() {
        // Test the equal case
        let arrow_array = StringArray::try_from(vec![
            Some("hello"),
            None,
            None,
            Some("world"),
            None,
            None,
        ])
        .unwrap();
        let arrow_array = BinaryArray::from(arrow_array.data());
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                "world",
                null,
                null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.eq(&json_array));
        assert!(json_array.eq(&arrow_array));

        // Test unequal case
        let arrow_array = StringArray::try_from(vec![
            Some("hello"),
            None,
            None,
            Some("world"),
            None,
            None,
        ])
        .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                "arrow",
                null,
                null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test unequal length case
        let arrow_array =
            StringArray::try_from(vec![Some("hello"), None, None, Some("world"), None])
                .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                "arrow",
                null,
                null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect type case
        let arrow_array =
            StringArray::try_from(vec![Some("hello"), None, None, Some("world"), None])
                .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            {
                "a": 1
            }
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect value type case
        let arrow_array =
            StringArray::try_from(vec![Some("hello"), None, None, Some("world"), None])
                .unwrap();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                1,
                null,
                null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));
    }

    #[test]
    fn test_fixed_size_binary_json_equal() {
        // Test the equal case
        let mut builder = FixedSizeBinaryBuilder::new(15, 5);
        builder.append_value(b"hello").unwrap();
        builder.append_null().unwrap();
        builder.append_value(b"world").unwrap();
        let arrow_array: FixedSizeBinaryArray = builder.finish();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                "world"
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.eq(&json_array));
        assert!(json_array.eq(&arrow_array));

        // Test unequal case
        builder.append_value(b"hello").unwrap();
        builder.append_null().unwrap();
        builder.append_value(b"world").unwrap();
        let arrow_array: FixedSizeBinaryArray = builder.finish();
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                "arrow"
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test unequal length case
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                null,
                "world"
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect type case
        let json_array: Value = serde_json::from_str(
            r#"
            {
                "a": 1
            }
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect value type case
        let json_array: Value = serde_json::from_str(
            r#"
            [
                "hello",
                null,
                1
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));
    }

    #[test]
    fn test_struct_json_equal() {
        // Test equal case
        let string_builder = StringBuilder::new(5);
        let int_builder = Int32Builder::new(5);

        let mut fields = Vec::new();
        let mut field_builders = Vec::new();
        fields.push(Field::new("f1", DataType::Utf8, false));
        field_builders.push(Box::new(string_builder) as Box<ArrayBuilder>);
        fields.push(Field::new("f2", DataType::Int32, false));
        field_builders.push(Box::new(int_builder) as Box<ArrayBuilder>);

        let mut builder = StructBuilder::new(fields, field_builders);

        let arrow_array = create_struct_array(
            &mut builder,
            &[Some("joe"), None, None, Some("mark"), Some("doe")],
            &[Some(1), Some(2), None, Some(4), Some(5)],
            &[true, true, false, true, true],
        )
        .unwrap();

        let json_array: Value = serde_json::from_str(
            r#"
            [
              {
                "f1": "joe",
                "f2": 1
              },
              {
                "f2": 2
              },
              null,
              {
                "f1": "mark",
                "f2": 4
              },
              {
                "f1": "doe",
                "f2": 5
              }
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.eq(&json_array));
        assert!(json_array.eq(&arrow_array));

        // Test unequal length case
        let json_array: Value = serde_json::from_str(
            r#"
            [
              {
                "f1": "joe",
                "f2": 1
              },
              {
                "f2": 2
              },
              null,
              {
                "f1": "mark",
                "f2": 4
              }
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test incorrect type case
        let json_array: Value = serde_json::from_str(
            r#"
              {
                "f1": "joe",
                "f2": 1
              }
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));

        // Test not all object case
        let json_array: Value = serde_json::from_str(
            r#"
            [
              {
                "f1": "joe",
                "f2": 1
              },
              2,
              null,
              {
                "f1": "mark",
                "f2": 4
              }
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));
    }

    fn create_struct_array<
        'a,
        T: AsRef<[Option<&'a str>]>,
        U: AsRef<[Option<i32>]>,
        V: AsRef<[bool]>,
    >(
        builder: &'a mut StructBuilder,
        first: T,
        second: U,
        is_valid: V,
    ) -> Result<StructArray> {
        let string_builder = builder.field_builder::<StringBuilder>(0).unwrap();
        for v in first.as_ref() {
            if let Some(s) = v {
                string_builder.append_value(s)?;
            } else {
                string_builder.append_null()?;
            }
        }

        let int_builder = builder.field_builder::<Int32Builder>(1).unwrap();
        for v in second.as_ref() {
            if let Some(i) = v {
                int_builder.append_value(*i)?;
            } else {
                int_builder.append_null()?;
            }
        }

        for v in is_valid.as_ref() {
            builder.append(*v)?
        }

        Ok(builder.finish())
    }

    #[test]
    fn test_null_json_equal() {
        // Test equaled array
        let arrow_array = NullArray::new(4);
        let json_array: Value = serde_json::from_str(
            r#"
            [
                null, null, null, null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.eq(&json_array));
        assert!(json_array.eq(&arrow_array));

        // Test unequaled array
        let arrow_array = NullArray::new(2);
        let json_array: Value = serde_json::from_str(
            r#"
            [
                null, null, null
            ]
        "#,
        )
        .unwrap();
        assert!(arrow_array.ne(&json_array));
        assert!(json_array.ne(&arrow_array));
    }
}
