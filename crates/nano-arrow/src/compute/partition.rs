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

//! Defines partition kernel for [`crate::array::Array`]

use crate::array::ord::DynComparator;
use crate::compute::sort::{build_compare, SortColumn};
use crate::error::{Error, Result};
use std::cmp::Ordering;
use std::iter::Iterator;
use std::ops::Range;

/// Given a list of already sorted columns, find partition ranges that would partition
/// lexicographically equal values across columns.
///
/// Here LexicographicalComparator is used in conjunction with binary
/// search so the columns *MUST* be pre-sorted already.
///
/// The returned vec would be of size k where k is cardinality of the sorted values; Consecutive
/// values will be connected: (a, b) and (b, c), where start = 0 and end = n for the first and last
/// range.
pub fn lexicographical_partition_ranges(
    columns: &[SortColumn],
) -> Result<impl Iterator<Item = Range<usize>>> {
    LexicographicalPartitionIterator::try_new(columns)
}

struct LexicographicalPartitionIterator {
    comparator: DynComparator,
    num_rows: usize,
    previous_partition_point: usize,
    partition_point: usize,
    value_indices: Vec<usize>,
}

impl LexicographicalPartitionIterator {
    fn try_new(columns: &[SortColumn]) -> Result<Self> {
        if columns.is_empty() {
            return Err(Error::InvalidArgumentError(
                "Sort requires at least one column".to_string(),
            ));
        }
        let num_rows = columns[0].values.len();
        if columns.iter().any(|item| item.values.len() != num_rows) {
            return Err(Error::InvalidArgumentError(
                "Lexical sort columns have different row counts".to_string(),
            ));
        };

        let comparators = columns
            .iter()
            .map(|x| build_compare(x.values, x.options.unwrap_or_default()))
            .collect::<Result<Vec<_>>>()?;

        let comparator = Box::new(move |a_idx: usize, b_idx: usize| -> Ordering {
            for comparator in comparators.iter() {
                match comparator(a_idx, b_idx) {
                    Ordering::Equal => continue,
                    other => return other,
                }
            }

            Ordering::Equal
        });

        let value_indices = (0..num_rows).collect::<Vec<usize>>();
        Ok(Self {
            comparator,
            num_rows,
            previous_partition_point: 0,
            partition_point: 0,
            value_indices,
        })
    }
}

impl Iterator for LexicographicalPartitionIterator {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.partition_point < self.num_rows {
            // invariant:
            // value_indices[0..previous_partition_point] all are values <= value_indices[previous_partition_point]
            // so in order to save time we can do binary search on the value_indices[previous_partition_point..]
            // and find when any value is greater than value_indices[previous_partition_point]; because we are using
            // new indices, the new offset is _added_ to the previous_partition_point.
            //
            // be careful that idx is of type &usize which points to the actual value within value_indices, which itself
            // contains usize (0..row_count), providing access to lexicographical_comparator as pointers into the
            // original columnar data.
            self.partition_point +=
                self.value_indices[self.partition_point..].partition_point(|idx| {
                    (self.comparator)(*idx, self.partition_point) != Ordering::Greater
                });
            let start = self.previous_partition_point;
            let end = self.partition_point;
            self.previous_partition_point = self.partition_point;
            Some(Range { start, end })
        } else {
            None
        }
    }
}
