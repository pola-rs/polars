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

use crate::basic::Encoding;
use crate::column::page::PageReader;
use crate::column::page::{Page, PageIterator};
use crate::data_type::DataType;
use crate::encodings::encoding::{get_encoder, DictEncoder, Encoder};
use crate::encodings::levels::max_buffer_size;
use crate::encodings::levels::LevelEncoder;
use crate::errors::Result;
use crate::schema::types::{ColumnDescPtr, SchemaDescPtr};
use crate::util::memory::ByteBufferPtr;
use crate::util::memory::MemTracker;
use crate::util::memory::MemTrackerPtr;
use crate::util::test_common::random_numbers_range;
use rand::distributions::uniform::SampleUniform;
use std::collections::VecDeque;
use std::mem;
use std::rc::Rc;
use std::vec::IntoIter;

pub trait DataPageBuilder {
    fn add_rep_levels(&mut self, max_level: i16, rep_levels: &[i16]);
    fn add_def_levels(&mut self, max_level: i16, def_levels: &[i16]);
    fn add_values<T: DataType>(&mut self, encoding: Encoding, values: &[T::T]);
    fn add_indices(&mut self, indices: ByteBufferPtr);
    fn consume(self) -> Page;
}

/// A utility struct for building data pages (v1 or v2). Callers must call:
///   - add_rep_levels()
///   - add_def_levels()
///   - add_values() for normal data page / add_indices() for dictionary data page
///   - consume()
/// in order to populate and obtain a data page.
pub struct DataPageBuilderImpl {
    desc: ColumnDescPtr,
    encoding: Option<Encoding>,
    mem_tracker: MemTrackerPtr,
    num_values: u32,
    buffer: Vec<u8>,
    rep_levels_byte_len: u32,
    def_levels_byte_len: u32,
    datapage_v2: bool,
}

impl DataPageBuilderImpl {
    // `num_values` is the number of non-null values to put in the data page.
    // `datapage_v2` flag is used to indicate if the generated data page should use V2
    // format or not.
    pub fn new(desc: ColumnDescPtr, num_values: u32, datapage_v2: bool) -> Self {
        DataPageBuilderImpl {
            desc,
            encoding: None,
            mem_tracker: Rc::new(MemTracker::new()),
            num_values,
            buffer: vec![],
            rep_levels_byte_len: 0,
            def_levels_byte_len: 0,
            datapage_v2,
        }
    }

    // Adds levels to the buffer and return number of encoded bytes
    fn add_levels(&mut self, max_level: i16, levels: &[i16]) -> u32 {
        let size = max_buffer_size(Encoding::RLE, max_level, levels.len());
        let mut level_encoder = LevelEncoder::v1(Encoding::RLE, max_level, vec![0; size]);
        level_encoder.put(levels).expect("put() should be OK");
        let encoded_levels = level_encoder.consume().expect("consume() should be OK");
        // Actual encoded bytes (without length offset)
        let encoded_bytes = &encoded_levels[mem::size_of::<i32>()..];
        if self.datapage_v2 {
            // Level encoder always initializes with offset of i32, where it stores
            // length of encoded data; for data page v2 we explicitly
            // store length, therefore we should skip i32 bytes.
            self.buffer.extend_from_slice(encoded_bytes);
        } else {
            self.buffer.extend_from_slice(encoded_levels.as_slice());
        }
        encoded_bytes.len() as u32
    }
}

impl DataPageBuilder for DataPageBuilderImpl {
    fn add_rep_levels(&mut self, max_levels: i16, rep_levels: &[i16]) {
        self.num_values = rep_levels.len() as u32;
        self.rep_levels_byte_len = self.add_levels(max_levels, rep_levels);
    }

    fn add_def_levels(&mut self, max_levels: i16, def_levels: &[i16]) {
        assert!(
            self.num_values == def_levels.len() as u32,
            "Must call `add_rep_levels() first!`"
        );

        self.def_levels_byte_len = self.add_levels(max_levels, def_levels);
    }

    fn add_values<T: DataType>(&mut self, encoding: Encoding, values: &[T::T]) {
        assert!(
            self.num_values >= values.len() as u32,
            "num_values: {}, values.len(): {}",
            self.num_values,
            values.len()
        );
        self.encoding = Some(encoding);
        let mut encoder: Box<Encoder<T>> =
            get_encoder::<T>(self.desc.clone(), encoding, self.mem_tracker.clone())
                .expect("get_encoder() should be OK");
        encoder.put(values).expect("put() should be OK");
        let encoded_values = encoder
            .flush_buffer()
            .expect("consume_buffer() should be OK");
        self.buffer.extend_from_slice(encoded_values.data());
    }

    fn add_indices(&mut self, indices: ByteBufferPtr) {
        self.encoding = Some(Encoding::RLE_DICTIONARY);
        self.buffer.extend_from_slice(indices.data());
    }

    fn consume(self) -> Page {
        if self.datapage_v2 {
            Page::DataPageV2 {
                buf: ByteBufferPtr::new(self.buffer),
                num_values: self.num_values,
                encoding: self.encoding.unwrap(),
                num_nulls: 0, /* set to dummy value - don't need this when reading
                               * data page */
                num_rows: self.num_values, /* also don't need this when reading
                                            * data page */
                def_levels_byte_len: self.def_levels_byte_len,
                rep_levels_byte_len: self.rep_levels_byte_len,
                is_compressed: false,
                statistics: None, // set to None, we do not need statistics for tests
            }
        } else {
            Page::DataPage {
                buf: ByteBufferPtr::new(self.buffer),
                num_values: self.num_values,
                encoding: self.encoding.unwrap(),
                def_level_encoding: Encoding::RLE,
                rep_level_encoding: Encoding::RLE,
                statistics: None, // set to None, we do not need statistics for tests
            }
        }
    }
}

/// A utility page reader which stores pages in memory.
pub struct InMemoryPageReader {
    pages: Box<Iterator<Item = Page>>,
}

impl InMemoryPageReader {
    pub fn new(pages: Vec<Page>) -> Self {
        Self {
            pages: Box::new(pages.into_iter()),
        }
    }
}

impl PageReader for InMemoryPageReader {
    fn get_next_page(&mut self) -> Result<Option<Page>> {
        Ok(self.pages.next())
    }
}

/// A utility page iterator which stores page readers in memory, used for tests.
pub struct InMemoryPageIterator {
    schema: SchemaDescPtr,
    column_desc: ColumnDescPtr,
    page_readers: IntoIter<Box<dyn PageReader>>,
}

impl InMemoryPageIterator {
    pub fn new(
        schema: SchemaDescPtr,
        column_desc: ColumnDescPtr,
        pages: Vec<Vec<Page>>,
    ) -> Self {
        let page_readers = pages
            .into_iter()
            .map(|pages| Box::new(InMemoryPageReader::new(pages)) as Box<dyn PageReader>)
            .collect::<Vec<Box<dyn PageReader>>>()
            .into_iter();

        Self {
            schema,
            column_desc,
            page_readers,
        }
    }
}

impl Iterator for InMemoryPageIterator {
    type Item = Result<Box<dyn PageReader>>;

    fn next(&mut self) -> Option<Self::Item> {
        self.page_readers.next().map(|page_reader| Ok(page_reader))
    }
}

impl PageIterator for InMemoryPageIterator {
    fn schema(&mut self) -> Result<SchemaDescPtr> {
        Ok(self.schema.clone())
    }

    fn column_schema(&mut self) -> Result<ColumnDescPtr> {
        Ok(self.column_desc.clone())
    }
}

pub fn make_pages<T: DataType>(
    desc: ColumnDescPtr,
    encoding: Encoding,
    num_pages: usize,
    levels_per_page: usize,
    min: T::T,
    max: T::T,
    def_levels: &mut Vec<i16>,
    rep_levels: &mut Vec<i16>,
    values: &mut Vec<T::T>,
    pages: &mut VecDeque<Page>,
    use_v2: bool,
) where
    T::T: PartialOrd + SampleUniform + Copy,
{
    let mut num_values = 0;
    let max_def_level = desc.max_def_level();
    let max_rep_level = desc.max_rep_level();

    let mem_tracker = Rc::new(MemTracker::new());
    let mut dict_encoder = DictEncoder::<T>::new(desc.clone(), mem_tracker);

    for i in 0..num_pages {
        let mut num_values_cur_page = 0;
        let level_range = i * levels_per_page..(i + 1) * levels_per_page;

        if max_def_level > 0 {
            random_numbers_range(levels_per_page, 0, max_def_level + 1, def_levels);
            for dl in &def_levels[level_range.clone()] {
                if *dl == max_def_level {
                    num_values_cur_page += 1;
                }
            }
        } else {
            num_values_cur_page = levels_per_page;
        }
        if max_rep_level > 0 {
            random_numbers_range(levels_per_page, 0, max_rep_level + 1, rep_levels);
        }
        random_numbers_range(num_values_cur_page, min, max, values);

        // Generate the current page

        let mut pb =
            DataPageBuilderImpl::new(desc.clone(), num_values_cur_page as u32, use_v2);
        if max_rep_level > 0 {
            pb.add_rep_levels(max_rep_level, &rep_levels[level_range.clone()]);
        }
        if max_def_level > 0 {
            pb.add_def_levels(max_def_level, &def_levels[level_range]);
        }

        let value_range = num_values..num_values + num_values_cur_page;
        match encoding {
            Encoding::PLAIN_DICTIONARY | Encoding::RLE_DICTIONARY => {
                let _ = dict_encoder.put(&values[value_range.clone()]);
                let indices = dict_encoder
                    .write_indices()
                    .expect("write_indices() should be OK");
                pb.add_indices(indices);
            }
            Encoding::PLAIN => {
                pb.add_values::<T>(encoding, &values[value_range]);
            }
            enc @ _ => panic!("Unexpected encoding {}", enc),
        }

        let data_page = pb.consume();
        pages.push_back(data_page);
        num_values += num_values_cur_page;
    }

    if encoding == Encoding::PLAIN_DICTIONARY || encoding == Encoding::RLE_DICTIONARY {
        let dict = dict_encoder
            .write_dict()
            .expect("write_dict() should be OK");
        let dict_page = Page::DictionaryPage {
            buf: dict,
            num_values: dict_encoder.num_entries() as u32,
            encoding: Encoding::RLE_DICTIONARY,
            is_sorted: false,
        };
        pages.push_front(dict_page);
    }
}
