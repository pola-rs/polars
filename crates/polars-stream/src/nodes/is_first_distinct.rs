use std::cmp::Reverse;

use arrow::array::{Array, PrimitiveArray};
use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::types::NativeType;
use hashbrown::HashTable;
use polars_core::prelude::row_encode::_get_rows_encoded_ca_unordered;
use polars_core::prelude::{
    BooleanChunked, Column, DataType, IntoColumn, PlRandomState, PlSeedableRandomStateQuality,
};
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::priority::Priority;
use polars_utils::total_ord::{TotalHash, TotalOrdWrap};

use super::ComputeNode;
use crate::DEFAULT_LINEARIZER_BUFFER_SIZE;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::async_primitives::linearizer::Linearizer;
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::morsel::{Morsel, MorselSeq, SourceToken};
use crate::pipe::{RecvPort, SendPort};

#[derive(Clone)]
pub struct FixedHashTable<K> {
    mask: u64,
    table: Box<[Option<K>]>,
}

impl<K: Default> FixedHashTable<K> {
    pub fn new(num_slots: usize) -> Self {
        assert!(num_slots > 0);
        assert!(num_slots.is_power_of_two());

        let mask = (1 << num_slots.ilog2()) - 1;
        Self {
            mask,
            table: (0..num_slots).map(|_| None).collect(),
        }
    }
}

impl<K: PartialEq + Clone> FixedHashTable<K> {
    pub fn insert(&mut self, hash: u64, key: K) -> bool {
        let tag = hash & self.mask;
        debug_assert!(tag < self.table.len() as u64);
        let current_key = unsafe { self.table.get_unchecked_mut(tag as usize) };
        let is_same = current_key.as_ref().is_some_and(|v| v == &key);
        *current_key = Some(key);
        !is_same
    }
}

pub trait FixedTable: Send + Sync {
    fn clear(&mut self);
    fn boxed_clone(&self) -> Box<dyn FixedTable>;
    fn insert_values(&mut self, hashes: &[u64], array: &dyn Array, builder: &mut BitmapBuilder);
}

pub trait Table: Send + Sync {
    fn clear(&mut self);
    fn insert_values(
        &mut self,
        hashes: &[u64],
        pre_pass: &Bitmap,
        array: &dyn Array,
        builder: &mut BitmapBuilder,
    );
}

struct NullTable {
    seen_one: bool,
}

impl NullTable {
    fn new() -> Self {
        Self { seen_one: false }
    }

    fn insert_values(&mut self, array: &dyn Array, builder: &mut BitmapBuilder) {
        if array.is_empty() {
            return;
        }

        if self.seen_one {
            builder.extend_constant(array.len(), false);
        } else {
            builder.push(true);
            builder.extend_constant(array.len() - 1, false);
            self.seen_one = true;
        }
    }
}

impl FixedTable for NullTable {
    fn clear(&mut self) {}

    fn boxed_clone(&self) -> Box<dyn FixedTable> {
        Box::new(Self {
            seen_one: self.seen_one,
        })
    }

    fn insert_values(&mut self, _hashes: &[u64], array: &dyn Array, builder: &mut BitmapBuilder) {
        self.insert_values(array, builder)
    }
}

impl Table for NullTable {
    fn clear(&mut self) {}

    fn insert_values(
        &mut self,
        _hashes: &[u64],
        _may_be_unique: &Bitmap,
        array: &dyn Array,
        builder: &mut BitmapBuilder,
    ) {
        self.insert_values(array, builder)
    }
}

struct SingleKeyTable<K> {
    has_none: bool,
    random_state: PlRandomState,
    table: HashTable<TotalOrdWrap<K>>,
}

struct SingleKeyFixedTable<K> {
    has_none: bool,
    table: FixedHashTable<TotalOrdWrap<K>>,
}

impl<K> SingleKeyTable<K> {
    fn new() -> Self {
        Self {
            has_none: false,
            random_state: PlRandomState::default(),
            table: HashTable::new(),
        }
    }
}

impl<K: NativeType + TotalHash> SingleKeyTable<K> {
    fn insert(&mut self, hash: u64, key: K) -> bool {
        use std::hash::BuildHasher;
        let hasher = |val: &_| self.random_state.hash_one(val);
        let entry = self.table.entry(hash, |k| k == &TotalOrdWrap(key), hasher);
        let is_unique = matches!(entry, hashbrown::hash_table::Entry::Vacant(..));
        entry.insert(TotalOrdWrap(key));
        is_unique
    }

    fn insert_null(&mut self) -> bool {
        std::mem::replace(&mut self.has_none, true)
    }
}

impl<K: Default> SingleKeyFixedTable<K> {
    fn new() -> Self {
        Self {
            has_none: false,
            table: FixedHashTable::new(16),
        }
    }
}

impl<K: NativeType + TotalHash> Table for SingleKeyTable<K> {
    fn clear(&mut self) {
        self.table.clear();
    }

    fn insert_values(
        &mut self,
        hashes: &[u64],
        may_be_unique: &Bitmap,
        array: &dyn Array,
        builder: &mut BitmapBuilder,
    ) {
        let num_may_be_unique_items = may_be_unique.set_bits();

        if num_may_be_unique_items == 0 {
            return builder.extend_constant(array.len(), false);
        }

        // @TODO: Deal with null chunks

        let array = array.as_any().downcast_ref::<PrimitiveArray<K>>().unwrap();

        assert_eq!(hashes.len(), array.len());

        if num_may_be_unique_items == array.len() {
            if array.has_nulls() {
                for (hash, key) in hashes.iter().zip(array.iter()) {
                    let is_unique = match key {
                        None => self.insert_null(),
                        Some(key) => self.insert(*hash, *key),
                    };
                    unsafe { builder.push_unchecked(is_unique) };
                }
            } else {
                for (hash, key) in hashes.iter().zip(array.values_iter()) {
                    let is_unique = self.insert(*hash, *key);
                    unsafe { builder.push_unchecked(is_unique) };
                }
            }
        } else {
            if array.has_nulls() {
                for (may_be_unique, (hash, key)) in
                    may_be_unique.iter().zip(hashes.iter().zip(array.iter()))
                {
                    if !may_be_unique {
                        unsafe { builder.push_unchecked(false) };
                        continue;
                    }

                    let is_unique = match key {
                        None => self.insert_null(),
                        Some(key) => self.insert(*hash, *key),
                    };
                    unsafe { builder.push_unchecked(is_unique) };
                }
            } else {
                for (may_be_unique, (hash, key)) in may_be_unique
                    .iter()
                    .zip(hashes.iter().zip(array.values_iter()))
                {
                    if !may_be_unique {
                        unsafe { builder.push_unchecked(false) };
                        continue;
                    }

                    let is_unique = self.insert(*hash, *key);
                    unsafe { builder.push_unchecked(is_unique) };
                }
            }
        }
    }
}

impl<K: NativeType> FixedTable for SingleKeyFixedTable<K> {
    fn clear(&mut self) {
        self.table.table = Box::default();
    }

    fn boxed_clone(&self) -> Box<dyn FixedTable> {
        Box::new(Self {
            has_none: self.has_none,
            table: self.table.clone(),
        })
    }

    fn insert_values(&mut self, hashes: &[u64], array: &dyn Array, builder: &mut BitmapBuilder) {
        // @TODO: Deal with null chunks
        //
        let array = array.as_any().downcast_ref::<PrimitiveArray<K>>().unwrap();

        assert_eq!(hashes.len(), array.len());

        if array.has_nulls() {
            for (hash, key) in hashes.iter().zip(array.iter()) {
                let may_be_unique = match key {
                    None => std::mem::replace(&mut self.has_none, true),
                    Some(key) => self.table.insert(*hash, TotalOrdWrap(*key)),
                };
                unsafe { builder.push_unchecked(may_be_unique) };
            }
        } else {
            for (hash, key) in hashes.iter().zip(array.values_iter()) {
                let may_be_unique = self.table.insert(*hash, TotalOrdWrap(*key));
                unsafe { builder.push_unchecked(may_be_unique) };
            }
        }
    }
}

pub struct IsFirstDistinctNode {
    output_name: PlSmallStr,
    dtype: DataType,

    row_encode: bool,
    table: Box<dyn Table>,
    fixed_tables: Vec<Box<dyn FixedTable>>,
}

impl IsFirstDistinctNode {
    pub fn new(output_name: PlSmallStr, dtype: DataType) -> Self {
        fn single_key<T>() -> (Box<dyn Table>, Box<dyn FixedTable>) {
            (
                Box::new(SingleKeyTable::<i8>::new()) as Box<dyn Table>,
                Box::new(SingleKeyFixedTable::<i8>::new()) as Box<dyn FixedTable>,
            )
        }
        fn binview_key() -> (Box<dyn Table>, Box<dyn FixedTable>) {
            todo!()
        }
        fn row_encoded_key() -> (Box<dyn Table>, Box<dyn FixedTable>) {
            todo!()
        }

        let mut row_encode = false;

        use DataType as D;
        let (table, fixed_table) = match dtype.to_physical() {
            D::Null => (
                Box::new(NullTable::new()) as _,
                Box::new(NullTable::new()) as _,
            ),
            D::Boolean => todo!(),
            D::Int8 => single_key::<i8>(),
            D::Int16 => single_key::<i16>(),
            D::Int32 => single_key::<i32>(),
            D::Int64 => single_key::<i64>(),
            D::Int128 => single_key::<i128>(),
            D::UInt8 => single_key::<u8>(),
            D::UInt16 => single_key::<u16>(),
            D::UInt32 => single_key::<u32>(),
            D::UInt64 => single_key::<u64>(),
            D::Float32 => single_key::<f32>(),
            D::Float64 => single_key::<f64>(),
            D::BinaryOffset => row_encoded_key(),
            D::String | D::Binary => binview_key(),
            D::Struct(_) | D::Array(_, _) | D::List(_) => {
                row_encode = true;
                row_encoded_key()
            },

            D::Object(_) => todo!(),
            D::Unknown(_) => unreachable!(),

            // Logical types
            D::Decimal(..) | D::Date | D::Datetime(..) | D::Duration(..) | D::Time => {
                unreachable!()
            },
            #[cfg(feature = "dtype-categorical")]
            D::Categorical(..) | D::Enum(..) => unreachable!(),
        };

        Self {
            output_name,
            dtype,

            row_encode,
            table,
            fixed_tables: vec![fixed_table],
        }
    }
}

impl ComputeNode for IsFirstDistinctNode {
    fn name(&self) -> &str {
        "is_first_distinct"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert_eq!(recv.len(), 1);
        assert_eq!(send.len(), 1);

        recv.swap_with_slice(send);

        if recv[0] == PortState::Done {
            self.table.clear();
            for ft in self.fixed_tables.iter_mut() {
                ft.clear();
            }
        }

        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert_eq!(send_ports.len(), 1);

        let recv = recv_ports[0].take().unwrap().parallel();
        let mut send = send_ports[0].take().unwrap().serial();

        if self.fixed_tables.len() == 1 && recv.len() != 1 {
            let fixed_table = self.fixed_tables.pop().unwrap();
            self.fixed_tables
                .extend((0..recv.len()).map(|_| fixed_table.boxed_clone()));
        }

        let (mut lin_rx, lin_txs) = Linearizer::<
            Priority<Reverse<MorselSeq>, (Column, Vec<u64>, Bitmap, SourceToken)>,
        >::new(recv.len(), *DEFAULT_LINEARIZER_BUFFER_SIZE);

        let dtype = &self.dtype;
        let row_encode = self.row_encode;
        join_handles.extend(
            recv.into_iter()
                .zip(lin_txs)
                .zip(self.fixed_tables.iter_mut())
                .map(|((mut rx, mut tx), fixed_table)| {
                    scope.spawn_task(TaskPriority::High, async move {
                        while let Ok(morsel) = rx.recv().await {
                            let (frame, seq, source_token, wait_token) = morsel.into_inner();
                            assert_eq!(frame.width(), 1);

                            let mut column = frame.take_columns().pop().unwrap();
                            column = column.to_physical_repr();
                            if column.dtype().matches_schema_type(dtype)? {
                                column = column.cast(dtype)?;
                            }
                            if row_encode {
                                column =
                                    _get_rows_encoded_ca_unordered(PlSmallStr::EMPTY, &[column])?
                                        .into_column();
                            }

                            let mut hashes = Vec::with_capacity(column.len());
                            let rs = PlSeedableRandomStateQuality::fixed();
                            column.vec_hash(rs, &mut hashes)?;

                            let mut may_be_unique = BitmapBuilder::with_capacity(column.len());
                            {
                                let mut hashes = hashes.as_slice();
                                for array in column.as_materialized_series().chunks() {
                                    let chunk_hashes;
                                    (chunk_hashes, hashes) = hashes.split_at(array.len());
                                    fixed_table.insert_values(
                                        chunk_hashes,
                                        array.as_ref(),
                                        &mut may_be_unique,
                                    );
                                }
                            }
                            let may_be_unique = may_be_unique.freeze();

                            let value = Priority(
                                Reverse(seq),
                                (column, hashes, may_be_unique, source_token),
                            );
                            if tx.insert(value).await.is_err() {
                                break;
                            }
                            drop(wait_token);
                        }

                        Ok(())
                    })
                }),
        );

        let output_name = &self.output_name;
        let table = self.table.as_mut();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Some(value) = lin_rx.get().await {
                let Priority(Reverse(seq), value) = value;
                let (column, hashes, may_be_unique, source_token) = value;

                let mut hashes = hashes.as_slice();
                let mut result = BitmapBuilder::with_capacity(column.len());
                for array in column.as_materialized_series().chunks() {
                    let chunk_hashes;
                    (chunk_hashes, hashes) = hashes.split_at(array.len());
                    table.insert_values(chunk_hashes, &may_be_unique, array.as_ref(), &mut result);
                }
                let result = result.freeze();

                let df = BooleanChunked::from_bitmap(output_name.clone(), result)
                    .into_column()
                    .into_frame();
                let morsel = Morsel::new(df, seq, source_token);

                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
