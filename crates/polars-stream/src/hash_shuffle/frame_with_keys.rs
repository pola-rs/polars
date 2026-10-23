use futures::future::join;
use polars_arrow::array::builder::ShareStrategy;
use polars_arrow::bitmap::Bitmap;
use polars_core::datatypes::BooleanChunked;
use polars_core::frame::DataFrame;
use polars_core::frame::builder::DataFrameBuilder;
use polars_core::prelude::IntoColumn;
use polars_core::schema::SchemaRef;
use polars_expr::hash_keys::{HashKeys, HashKeysBuilder, SpilledHashKeys};
use polars_ooc::{ParameterFreeSpillContext, SpillToken, Spillable};
use polars_utils::IdxSize;
use polars_utils::pl_str::PlSmallStr;

use crate::unique_column_name;

/// Rows with their hashed keys, and an optional mask of the rows that belong to it.
pub struct FrameWithKeys {
    pub frame: DataFrame,
    pub keys: HashKeys,
    pub mask: Option<Bitmap>,
}

impl FrameWithKeys {
    /// The number of rows that belong to this frame.
    fn num_rows(&self) -> usize {
        self.mask.as_ref().map_or(self.keys.len(), |m| m.set_bits())
    }

    fn estimated_size(&self) -> usize {
        self.frame.estimated_size(false)
            + self.keys.estimated_size()
            + self.mask.as_ref().map_or(0, |m| m.len().div_ceil(8))
    }
}

/// A spilled `FrameWithKeys`: the frame, with the mask as one more column if there is one, and
/// the keys.
pub struct SpilledFrameWithKeys {
    frame: <DataFrame as Spillable>::Spilled,
    keys: SpilledHashKeys,
    mask_column: Option<PlSmallStr>,
}

impl Spillable for FrameWithKeys {
    type Spilled = SpilledFrameWithKeys;

    fn estimate_byte_size(&self) -> usize {
        self.estimated_size()
    }

    async fn spill(&self, context_id: &str) -> SpilledFrameWithKeys {
        let mut frame = self.frame.clone();
        let mask_column = self.mask.as_ref().map(|mask| {
            let name = unique_column_name();
            let column = BooleanChunked::from_bitmap(name.clone(), mask.clone()).into_column();
            frame.with_column(column).unwrap();
            name
        });
        let (frame, keys) = join(frame.spill(context_id), self.keys.spill(context_id)).await;
        SpilledFrameWithKeys {
            frame,
            keys,
            mask_column,
        }
    }

    async fn unspill(spilled: &SpilledFrameWithKeys) -> Self {
        let (mut frame, keys) = join(
            DataFrame::unspill(&spilled.frame),
            HashKeys::unspill(&spilled.keys),
        )
        .await;
        let mask = spilled.mask_column.as_ref().map(|name| {
            let column = frame.drop_in_place(name).unwrap();
            let mask = column.bool().unwrap().rechunk();
            mask.downcast_as_array().values().clone()
        });
        if frame.width() == 0 {
            frame = DataFrame::empty_with_height(keys.len());
        }
        Self { frame, keys, mask }
    }
}

/// A `FrameWithKeys` that can spill, like `SpillFrame`: one `SpillToken`, its size known while
/// spilled.
pub struct SpillFrameWithKeys {
    token: SpillToken<FrameWithKeys>,
    num_rows: usize,
    estimated_size: usize,
}

impl AsRef<SpillToken<FrameWithKeys>> for SpillFrameWithKeys {
    fn as_ref(&self) -> &SpillToken<FrameWithKeys> {
        &self.token
    }
}

impl SpillFrameWithKeys {
    pub fn new_unregistered(frame: FrameWithKeys) -> Self {
        Self {
            num_rows: frame.num_rows(),
            estimated_size: frame.estimated_size(),
            token: SpillToken::new(frame),
        }
    }

    pub async fn new<C: ParameterFreeSpillContext>(frame: FrameWithKeys, ctx: &C) -> Self {
        let slf = Self::new_unregistered(frame);
        ctx.register(&slf).await;
        slf
    }

    /// The number of rows that belong to the frame: those set in its mask, or all rows.
    pub fn num_rows(&self) -> usize {
        self.num_rows
    }

    /// The estimated memory of the frame, in bytes.
    pub fn estimated_size(&self) -> usize {
        self.estimated_size
    }

    /// Whether the frame is in memory and not being spilled.
    pub fn is_in_memory(&self) -> bool {
        self.token.try_get().is_some()
    }

    /// Takes the frame out of its token, unspilling it if needed, with every column in one
    /// chunk.
    pub async fn load(self) -> FrameWithKeys {
        let mut frame = self.token.into_inner().await;
        frame.frame.rechunk_mut();
        frame
    }
}

/// Appends row subsets of frames with keys of one input; strings are copied
/// (`ShareStrategy::Never`).
pub struct FrameWithKeysBuilder {
    frame: DataFrameBuilder,
    keys: HashKeysBuilder,
}

impl FrameWithKeysBuilder {
    /// A builder for frames of `schema` with keys of the kind and key schema of `like`.
    pub fn new(schema: SchemaRef, like: &HashKeys) -> Self {
        Self {
            frame: DataFrameBuilder::new(schema),
            keys: HashKeysBuilder::new(like),
        }
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }

    pub fn reserve(&mut self, additional: usize) {
        self.frame.reserve(additional);
        self.keys.reserve(additional);
    }

    /// Appends the rows `rows` of `frame` and `keys`. Panics unless the columns of `frame` are
    /// single chunks.
    ///
    /// # Safety
    /// The rows must be in-bounds of `frame` and `keys`, which have the same number of rows.
    pub unsafe fn gather_extend(&mut self, frame: &DataFrame, keys: &HashKeys, rows: &[IdxSize]) {
        unsafe {
            self.frame.gather_extend(frame, rows, ShareStrategy::Never);
            self.keys.gather_extend(keys, rows);
        }
    }

    /// Takes the appended rows, leaving the builder empty.
    pub fn freeze_reset(&mut self) -> FrameWithKeys {
        FrameWithKeys {
            frame: self.frame.freeze_reset(),
            keys: self.keys.freeze_reset(),
            mask: None,
        }
    }
}
