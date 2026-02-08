use std::borrow::Cow;

use polars_core::frame::DataFrame;
use polars_core::prelude::row_encode::_get_rows_encoded_ca_unordered;
use polars_core::prelude::{BinaryOffsetChunked, Column, IntoGroupsType};
use polars_error::PolarsResult;
use polars_expr::hash_keys::{HashKeysVariant, hash_keys_variant_for_dtype};
use polars_expr::state::ExecutionState;
use polars_utils::pl_str::PlSmallStr;

use crate::async_primitives::wait_group::WaitToken;
use crate::expression::StreamExpr;
use crate::morsel::Morsel;
use crate::nodes::io_sinks::components::exclude_keys_projection::ExcludeKeysProjection;
use crate::nodes::io_sinks::components::partition_key::{PartitionKey, PreComputedKeys};
use crate::nodes::io_sinks::components::size::RowCountAndSize;

pub struct PartitionedDataFrames {
    pub partitions_vec: Vec<Partition>,
    pub input_size: RowCountAndSize,
    pub input_wait_token: Option<WaitToken>,
}

pub struct Partition {
    pub key: PartitionKey,
    /// 1-row df with keys.
    pub keys_df: DataFrame,
    /// Does not include columns in `keys_df`
    pub df: DataFrame,
}

pub enum Partitioner {
    /// All rows to a single partition
    FileSize,
    Keyed(KeyedPartitioner),
}

impl Partitioner {
    pub async fn partition_morsel(
        &self,
        morsel: Morsel,
        in_memory_exec_state: &ExecutionState,
    ) -> PolarsResult<PartitionedDataFrames> {
        let (df, _, _, input_wait_token) = morsel.into_inner();
        let input_size = RowCountAndSize::new_from_df(&df);
        let partitions_vec = match self {
            Self::FileSize => vec![Partition {
                key: PartitionKey::NULL,
                keys_df: DataFrame::empty_with_height(1),
                df,
            }],
            Self::Keyed(v) => v.partition_df(df, in_memory_exec_state).await?,
        };

        let out = PartitionedDataFrames {
            partitions_vec,
            input_size,
            input_wait_token,
        };

        Ok(out)
    }

    pub fn verbose_display(&self) -> impl std::fmt::Display + '_ {
        struct DisplayWrap<'a>(&'a Partitioner);

        impl std::fmt::Display for DisplayWrap<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self.0 {
                    Partitioner::FileSize => f.write_str("FileSize"),
                    Partitioner::Keyed(kp) => write!(
                        f,
                        "Keyed({} key{})",
                        kp.key_exprs.len(),
                        if kp.key_exprs.len() == 1 { "" } else { "s" }
                    ),
                }
            }
        }

        DisplayWrap(self)
    }
}

pub struct KeyedPartitioner {
    /// Must be non-empty
    pub key_exprs: Vec<StreamExpr>,
    /// Exclude key columns from full gather. Can be `None` if all key exprs output
    /// names do not overlap with existing names.
    pub exclude_keys_projection: Option<ExcludeKeysProjection>,
}

impl KeyedPartitioner {
    async fn partition_df(
        &self,
        df: DataFrame,
        in_memory_exec_state: &ExecutionState,
    ) -> PolarsResult<Vec<Partition>> {
        assert!(!self.key_exprs.is_empty());

        let mut key_columns = Vec::with_capacity(self.key_exprs.len());

        for e in self.key_exprs.as_slice() {
            key_columns.push(
                e.evaluate_preserve_len_broadcast(&df, in_memory_exec_state)
                    .await?,
            );
        }

        let mut pre_computed_keys: Option<PreComputedKeys> = None;
        let single_non_encode = match key_columns.as_slice() {
            [c] => {
                pre_computed_keys = PreComputedKeys::opt_new_non_encoded(c);
                hash_keys_variant_for_dtype(c.dtype()) != HashKeysVariant::RowEncoded
            },
            _ => false,
        };

        let row_encode = |columns: &[Column]| -> BinaryOffsetChunked {
            _get_rows_encoded_ca_unordered(PlSmallStr::EMPTY, columns)
                .unwrap()
                .rechunk()
                .into_owned()
        };

        let mut keys_encoded_ca: Option<BinaryOffsetChunked> =
            (!single_non_encode).then(|| row_encode(&key_columns));

        let groups = if single_non_encode {
            key_columns[0]
                .as_materialized_series()
                .group_tuples(false, false)
        } else {
            keys_encoded_ca.as_ref().unwrap().group_tuples(false, false)
        }
        .unwrap();

        if pre_computed_keys.is_none() {
            if keys_encoded_ca.is_none() && groups.len() > (df.height() / 4) {
                keys_encoded_ca = Some(row_encode(&key_columns));
            }

            pre_computed_keys = keys_encoded_ca
                .as_ref()
                .map(|x| PreComputedKeys::RowEncoded(x.downcast_as_array().clone()));
        }

        let gather_source_df: Cow<DataFrame> =
            if let Some(projection) = self.exclude_keys_projection.as_ref() {
                let columns = df.columns();

                Cow::Owned(unsafe {
                    DataFrame::new_unchecked(
                        df.height(),
                        projection
                            .iter_indices()
                            .map(|i| columns[i].clone())
                            .collect(),
                    )
                })
            } else {
                Cow::Borrowed(&df)
            };

        let partitions_vec: Vec<Partition> = groups
            .iter()
            .map(|groups_indicator| {
                let first_idx = groups_indicator.first();
                let df = unsafe { gather_source_df.gather_group_unchecked(&groups_indicator) };

                // Ensure 0-width is handled properly.
                assert_eq!(df.height(), groups_indicator.len());

                let keys_df: DataFrame = unsafe {
                    DataFrame::new_unchecked(
                        1,
                        key_columns
                            .iter()
                            .map(|c| c.take_slice_unchecked(&[first_idx]))
                            .collect(),
                    )
                };

                let key: PartitionKey = pre_computed_keys.as_ref().map_or_else(
                    || PartitionKey::from_slice(row_encode(keys_df.columns()).get(0).unwrap()),
                    |keys| keys.get_key(first_idx.try_into().unwrap()),
                );

                Partition { key, keys_df, df }
            })
            .collect();

        Ok(partitions_vec)
    }
}
