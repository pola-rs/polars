use arrow::bitmap::utils::BitmapIter;
use arrow::bitmap::{Bitmap, MutableBitmap};
use arrow::datatypes::ArrowDataType;

use super::utils;
use super::{BasicDecompressor, Filter};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage};
use crate::parquet::read::levels::get_bit_width;

#[derive(Debug)]
pub struct Nested {
    validity: Option<MutableBitmap>,
    length: usize,
    content: NestedContent,

    // We batch the collection of valids and invalids to amortize the costs. This only really works
    // when valids and invalids are grouped or there is a disbalance in the amount of valids vs.
    // invalids. This, however, is a very common situation.
    num_valids: usize,
    num_invalids: usize,
}

#[derive(Debug)]
pub enum NestedContent {
    Primitive,
    List { offsets: Vec<i64> },
    FixedSizeList { width: usize },
    Struct,
}

impl Nested {
    fn primitive(is_nullable: bool) -> Self {
        // @NOTE: We allocate with `0` capacity here since we will not be pushing to this bitmap.
        // This is because primitive does not keep track of the validity here. It keeps track in
        // the decoder. We do still want to put something so that we can check for nullability by
        // looking at the option.
        let validity = is_nullable.then(|| MutableBitmap::with_capacity(0));

        Self {
            validity,
            length: 0,
            content: NestedContent::Primitive,

            num_valids: 0,
            num_invalids: 0,
        }
    }

    fn list_with_capacity(is_nullable: bool, capacity: usize) -> Self {
        let offsets = Vec::with_capacity(capacity);
        let validity = is_nullable.then(|| MutableBitmap::with_capacity(capacity));
        Self {
            validity,
            length: 0,
            content: NestedContent::List { offsets },

            num_valids: 0,
            num_invalids: 0,
        }
    }

    fn fixedlist_with_capacity(is_nullable: bool, width: usize, capacity: usize) -> Self {
        let validity = is_nullable.then(|| MutableBitmap::with_capacity(capacity));
        Self {
            validity,
            length: 0,
            content: NestedContent::FixedSizeList { width },

            num_valids: 0,
            num_invalids: 0,
        }
    }

    fn struct_with_capacity(is_nullable: bool, capacity: usize) -> Self {
        let validity = is_nullable.then(|| MutableBitmap::with_capacity(capacity));
        Self {
            validity,
            length: 0,
            content: NestedContent::Struct,

            num_valids: 0,
            num_invalids: 0,
        }
    }

    fn take(mut self) -> (usize, Vec<i64>, Option<MutableBitmap>) {
        if !matches!(self.content, NestedContent::Primitive) {
            if let Some(validity) = self.validity.as_mut() {
                validity.extend_constant(self.num_valids, true);
                validity.extend_constant(self.num_invalids, false);
            }

            debug_assert!(self
                .validity
                .as_ref()
                .map_or(true, |v| v.len() == self.length));
        }

        self.num_valids = 0;
        self.num_invalids = 0;

        match self.content {
            NestedContent::Primitive => {
                debug_assert!(self.validity.map_or(true, |validity| validity.is_empty()));
                (self.length, Vec::new(), None)
            },
            NestedContent::List { offsets } => (self.length, offsets, self.validity),
            NestedContent::FixedSizeList { .. } => (self.length, Vec::new(), self.validity),
            NestedContent::Struct => (self.length, Vec::new(), self.validity),
        }
    }

    fn is_nullable(&self) -> bool {
        self.validity.is_some()
    }

    fn is_repeated(&self) -> bool {
        match self.content {
            NestedContent::Primitive => false,
            NestedContent::List { .. } => true,
            NestedContent::FixedSizeList { .. } => true,
            NestedContent::Struct => false,
        }
    }

    fn is_required(&self) -> bool {
        match self.content {
            NestedContent::Primitive => false,
            NestedContent::List { .. } => false,
            NestedContent::FixedSizeList { .. } => false,
            NestedContent::Struct => true,
        }
    }

    /// number of rows
    fn len(&self) -> usize {
        self.length
    }

    fn invalid_num_values(&self) -> usize {
        match &self.content {
            NestedContent::Primitive => 1,
            NestedContent::List { .. } => 0,
            NestedContent::FixedSizeList { width } => *width,
            NestedContent::Struct => 1,
        }
    }

    fn push(&mut self, value: i64, is_valid: bool) {
        let is_primitive = matches!(self.content, NestedContent::Primitive);

        if is_valid && self.num_invalids != 0 {
            debug_assert!(!is_primitive);

            // @NOTE: Having invalid items might not necessarily mean that we have a validity mask.
            //
            // For instance, if we have a optional struct with a required list in it, that struct
            // will have a validity mask and the list will not. In the arrow representation of this
            // array, however, the list will still have invalid items where the struct is null.
            //
            // Array:
            // [
            //     { 'x': [1] },
            //     None,
            //     { 'x': [1, 2] },
            // ]
            //
            // Arrow:
            // struct = [ list[0] None list[2] ]
            // list   = {
            //     values  = [ 1, 1, 2 ],
            //     offsets = [ 0, 1, 1, 3 ],
            // }
            //
            // Parquet:
            // [ 1, 1, 2 ] + definition + repetition levels
            //
            // As you can see we need to insert an invalid item into the list even though it does
            // not have a validity mask.
            if let Some(validity) = self.validity.as_mut() {
                validity.extend_constant(self.num_valids, true);
                validity.extend_constant(self.num_invalids, false);
            }

            self.num_valids = 0;
            self.num_invalids = 0;
        }

        self.num_valids += usize::from(!is_primitive & is_valid);
        self.num_invalids += usize::from(!is_primitive & !is_valid);

        self.length += 1;
        if let NestedContent::List { offsets } = &mut self.content {
            offsets.push(value);
        }
    }

    fn push_default(&mut self, length: i64) {
        let is_primitive = matches!(self.content, NestedContent::Primitive);
        self.num_invalids += usize::from(!is_primitive);

        self.length += 1;
        if let NestedContent::List { offsets } = &mut self.content {
            offsets.push(length);
        }
    }
}

/// Utility structure to create a `Filter` and `Validity` mask for the leaf values.
///
/// This batches the extending.
pub struct BatchedNestedDecoder<'a> {
    pub(crate) num_waiting_valids: usize,
    pub(crate) num_waiting_invalids: usize,

    filter: &'a mut MutableBitmap,
    validity: &'a mut MutableBitmap,
}

impl<'a> BatchedNestedDecoder<'a> {
    fn push_valid(&mut self) -> ParquetResult<()> {
        self.push_n_valids(1)
    }

    fn push_invalid(&mut self) -> ParquetResult<()> {
        self.push_n_invalids(1)
    }

    fn push_n_valids(&mut self, n: usize) -> ParquetResult<()> {
        if self.num_waiting_invalids == 0 {
            self.num_waiting_valids += n;
            return Ok(());
        }

        self.filter.extend_constant(self.num_waiting_valids, true);
        self.validity.extend_constant(self.num_waiting_valids, true);

        self.filter.extend_constant(self.num_waiting_invalids, true);
        self.validity
            .extend_constant(self.num_waiting_invalids, false);

        self.num_waiting_valids = n;
        self.num_waiting_invalids = 0;

        Ok(())
    }

    fn push_n_invalids(&mut self, n: usize) -> ParquetResult<()> {
        self.num_waiting_invalids += n;
        Ok(())
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        if self.num_waiting_valids > 0 {
            self.filter.extend_constant(self.num_waiting_valids, true);
            self.validity.extend_constant(self.num_waiting_valids, true);
            self.num_waiting_valids = 0;
        }
        if self.num_waiting_invalids > 0 {
            self.filter.extend_constant(self.num_waiting_invalids, true);
            self.validity
                .extend_constant(self.num_waiting_invalids, false);
            self.num_waiting_invalids = 0;
        }

        self.filter.extend_constant(n, false);
        self.validity.extend_constant(n, true);

        Ok(())
    }

    fn finalize(self) -> ParquetResult<()> {
        self.filter.extend_constant(self.num_waiting_valids, true);
        self.validity.extend_constant(self.num_waiting_valids, true);

        self.filter.extend_constant(self.num_waiting_invalids, true);
        self.validity
            .extend_constant(self.num_waiting_invalids, false);

        Ok(())
    }
}

/// The initial info of nested data types.
/// The `bool` indicates if the type is nullable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InitNested {
    /// Primitive data types
    Primitive(bool),
    /// List data types
    List(bool),
    /// Fixed-Size List data types
    FixedSizeList(bool, usize),
    /// Struct data types
    Struct(bool),
}

/// Initialize [`NestedState`] from `&[InitNested]`.
pub fn init_nested(init: &[InitNested], capacity: usize) -> NestedState {
    use {InitNested as IN, Nested as N};

    let container = init
        .iter()
        .map(|init| match init {
            IN::Primitive(is_nullable) => N::primitive(*is_nullable),
            IN::List(is_nullable) => N::list_with_capacity(*is_nullable, capacity),
            IN::FixedSizeList(is_nullable, width) => {
                N::fixedlist_with_capacity(*is_nullable, *width, capacity)
            },
            IN::Struct(is_nullable) => N::struct_with_capacity(*is_nullable, capacity),
        })
        .collect();

    NestedState::new(container)
}

/// The state of nested data types.
#[derive(Debug, Default)]
pub struct NestedState {
    /// The nesteds composing `NestedState`.
    nested: Vec<Nested>,
}

impl NestedState {
    /// Creates a new [`NestedState`].
    fn new(nested: Vec<Nested>) -> Self {
        Self { nested }
    }

    pub fn pop(&mut self) -> Option<(usize, Vec<i64>, Option<MutableBitmap>)> {
        Some(self.nested.pop()?.take())
    }

    pub fn last(&self) -> Option<&NestedContent> {
        self.nested.last().map(|v| &v.content)
    }

    /// The number of rows in this state
    pub fn len(&self) -> usize {
        // outermost is the number of rows
        self.nested[0].len()
    }

    /// Returns the definition and repetition levels for each nesting level
    fn levels(&self) -> (Vec<u16>, Vec<u16>) {
        let depth = self.nested.len();

        let mut def_levels = Vec::with_capacity(depth + 1);
        let mut rep_levels = Vec::with_capacity(depth + 1);

        def_levels.push(0);
        rep_levels.push(0);

        for i in 0..depth {
            let nest = &self.nested[i];

            let def_delta = nest.is_nullable() as u16 + nest.is_repeated() as u16;
            let rep_delta = nest.is_repeated() as u16;

            def_levels.push(def_levels[i] + def_delta);
            rep_levels.push(rep_levels[i] + rep_delta);
        }

        (def_levels, rep_levels)
    }
}

fn collect_level_values(
    target: &mut Vec<u16>,
    hybrid_rle: HybridRleDecoder<'_>,
) -> ParquetResult<()> {
    target.reserve(hybrid_rle.len());

    for chunk in hybrid_rle.into_chunk_iter() {
        let chunk = chunk?;

        match chunk {
            HybridRleChunk::Rle(value, size) => {
                target.resize(target.len() + size, value as u16);
            },
            HybridRleChunk::Bitpacked(decoder) => {
                decoder.lower_element::<u16>()?.collect_into(target);
            },
        }
    }

    Ok(())
}

/// State to keep track of how many top-level values (i.e. rows) still need to be skipped and
/// collected.
///
/// This state should be kept between pages because a top-level value / row value may span several
/// pages.
///
/// - `num_skips = Some(n)` means that it will skip till the `n + 1`-th occurrence of the repetition
///   level of `0` (i.e. the start of a top-level value / row value). 
/// - `num_collects = Some(n)` means that it will collect values till the `n + 1`-th occurrence of
///   the repetition level of `0` (i.e. the start of a top-level value / row value). 
struct DecodingState {
    num_skips: Option<usize>,
    num_collects: Option<usize>,
}

#[allow(clippy::too_many_arguments)]
fn decode_nested(
    mut current_def_levels: &[u16],
    mut current_rep_levels: &[u16],

    batched_collector: &mut BatchedNestedDecoder<'_>,
    nested: &mut [Nested],

    state: &mut DecodingState,
    top_level_filter: &mut BitmapIter<'_>,

    // Amortized allocations
    def_levels: &[u16],
    rep_levels: &[u16],
) -> ParquetResult<()> {
    let max_depth = nested.len();
    let leaf_def_level = *def_levels.last().unwrap();

    while !current_def_levels.is_empty() {
        debug_assert_eq!(current_def_levels.len(), current_rep_levels.len());

        // Handle skips
        if let Some(ref mut num_skips) = state.num_skips {
            let mut i = 0;
            let mut num_skipped_values = 0;
            while i < current_def_levels.len() && (*num_skips > 0 || current_rep_levels[i] != 0) {
                let def = current_def_levels[i];
                let rep = current_rep_levels[i];

                *num_skips -= usize::from(rep == 0);
                i += 1;

                // @NOTE:
                // We don't need to account for higher def-levels that imply extra values, since we
                // don't have those higher levels either.
                num_skipped_values += usize::from(def == leaf_def_level);
            }
            batched_collector.skip_in_place(num_skipped_values)?;

            current_def_levels = &current_def_levels[i..];
            current_rep_levels = &current_rep_levels[i..];

            if current_def_levels.is_empty() {
                break;
            } else {
                state.num_skips = None;
            }
        }

        // Handle collects
        if let Some(ref mut num_collects) = state.num_collects {
            let mut i = 0;
            while i < current_def_levels.len() && (*num_collects > 0 || current_rep_levels[i] != 0)
            {
                let def = current_def_levels[i];
                let rep = current_rep_levels[i];

                *num_collects -= usize::from(rep == 0);
                i += 1;

                let mut is_required = false;

                for depth in 0..max_depth {
                    // Defines whether this element is defined at `depth`
                    //
                    // e.g. [ [ [ 1 ] ] ] is defined at [ ... ], [ [ ... ] ], [ [ [ ... ] ] ] and
                    // [ [ [ 1 ] ] ].
                    let is_defined_at_this_depth =
                        rep <= rep_levels[depth] && def >= def_levels[depth];

                    let length = nested
                        .get(depth + 1)
                        .map(|x| x.len() as i64)
                        // the last depth is the leaf, which is always increased by 1
                        .unwrap_or(1);

                    let nest = &mut nested[depth];

                    let is_valid = !nest.is_nullable() || def > def_levels[depth];

                    if is_defined_at_this_depth && !is_valid {
                        let mut num_elements = 1;

                        nest.push(length, is_valid);

                        for embed_depth in depth..max_depth {
                            let embed_length = nested
                                .get(embed_depth + 1)
                                .map(|x| x.len() as i64)
                                // the last depth is the leaf, which is always increased by 1
                                .unwrap_or(1);

                            let embed_nest = &mut nested[embed_depth];

                            if embed_depth > depth {
                                for _ in 0..num_elements {
                                    embed_nest.push_default(embed_length);
                                }
                            }

                            let embed_num_values = embed_nest.invalid_num_values();
                            num_elements *= embed_num_values;

                            if embed_num_values == 0 {
                                break;
                            }
                        }

                        batched_collector.push_n_invalids(num_elements)?;

                        break;
                    }

                    if is_required || is_defined_at_this_depth {
                        nest.push(length, is_valid);

                        if depth == max_depth - 1 {
                            // the leaf / primitive
                            let is_valid = (def != def_levels[depth]) || !nest.is_nullable();

                            if is_valid {
                                batched_collector.push_valid()?;
                            } else {
                                batched_collector.push_invalid()?;
                            }
                        }
                    }

                    is_required = (is_required || is_defined_at_this_depth)
                        && nest.is_required()
                        && !is_valid;
                }
            }

            current_def_levels = &current_def_levels[i..];
            current_rep_levels = &current_rep_levels[i..];

            if current_def_levels.is_empty() {
                break;
            } else {
                state.num_collects = None;
            }
        }

        if top_level_filter.num_remaining() == 0 {
            break;
        }

        state.num_skips = Some(top_level_filter.take_leading_zeros()).filter(|v| *v != 0);
        state.num_collects = Some(top_level_filter.take_leading_ones()).filter(|v| *v != 0);
    }

    Ok(())
}

pub struct PageNestedDecoder<D: utils::Decoder> {
    pub iter: BasicDecompressor,
    pub dtype: ArrowDataType,
    pub dict: Option<D::Dict>,
    pub decoder: D,
    pub init: Vec<InitNested>,
}

/// Return the definition and repetition level iterators for this page.
fn level_iters(page: &DataPage) -> ParquetResult<(HybridRleDecoder, HybridRleDecoder)> {
    let split = split_buffer(page)?;
    let def = split.def;
    let rep = split.rep;

    let max_def_level = page.descriptor.max_def_level;
    let max_rep_level = page.descriptor.max_rep_level;

    let def_iter = HybridRleDecoder::new(def, get_bit_width(max_def_level), page.num_values());
    let rep_iter = HybridRleDecoder::new(rep, get_bit_width(max_rep_level), page.num_values());

    Ok((def_iter, rep_iter))
}

impl<D: utils::Decoder> PageNestedDecoder<D> {
    pub fn new(
        mut iter: BasicDecompressor,
        dtype: ArrowDataType,
        mut decoder: D,
        init: Vec<InitNested>,
    ) -> ParquetResult<Self> {
        let dict_page = iter.read_dict_page()?;
        let dict = dict_page.map(|d| decoder.deserialize_dict(d)).transpose()?;

        Ok(Self {
            iter,
            dtype,
            dict,
            decoder,
            init,
        })
    }

    pub fn collect_n(mut self, filter: Option<Filter>) -> ParquetResult<(NestedState, D::Output)> {
        // @TODO: We should probably count the filter so that we don't overallocate
        let mut target = self.decoder.with_capacity(self.iter.total_num_values());
        // @TODO: Self capacity
        let mut nested_state = init_nested(&self.init, 0);

        if let Some(dict) = self.dict.as_ref() {
            self.decoder.apply_dictionary(&mut target, dict)?;
        }

        // Amortize the allocations.
        let (def_levels, rep_levels) = nested_state.levels();

        let mut current_def_levels = Vec::<u16>::new();
        let mut current_rep_levels = Vec::<u16>::new();

        let (mut decode_state, top_level_filter) = match filter {
            None => (
                DecodingState {
                    num_skips: None,
                    num_collects: Some(usize::MAX),
                },
                Bitmap::new(),
            ),
            Some(Filter::Range(range)) => (
                DecodingState {
                    num_skips: Some(range.start),
                    num_collects: Some(range.len()),
                },
                Bitmap::new(),
            ),
            Some(Filter::Mask(mask)) => (
                DecodingState {
                    num_skips: None,
                    num_collects: None,
                },
                mask,
            ),
        };

        let mut top_level_filter = top_level_filter.iter();

        loop {
            let Some(page) = self.iter.next() else {
                break;
            };
            let page = page?;
            let page = page.decompress(&mut self.iter)?;

            let (mut def_iter, mut rep_iter) = level_iters(&page)?;

            let num_levels = def_iter.len().min(rep_iter.len());
            def_iter.limit_to(num_levels);
            rep_iter.limit_to(num_levels);

            current_def_levels.clear();
            current_rep_levels.clear();

            collect_level_values(&mut current_def_levels, def_iter)?;
            collect_level_values(&mut current_rep_levels, rep_iter)?;

            let mut leaf_filter = MutableBitmap::new();
            let mut leaf_validity = MutableBitmap::new();

            // @TODO: move this to outside the loop.
            let mut batched_collector = BatchedNestedDecoder {
                num_waiting_valids: 0,
                num_waiting_invalids: 0,

                filter: &mut leaf_filter,
                validity: &mut leaf_validity,
            };

            decode_nested(
                &current_def_levels,
                &current_rep_levels,
                &mut batched_collector,
                &mut nested_state.nested,
                &mut decode_state,
                &mut top_level_filter,
                &def_levels,
                &rep_levels,
            )?;

            batched_collector.finalize()?;

            let state = utils::State::new_nested(
                &self.decoder,
                &page,
                self.dict.as_ref(),
                Some(leaf_validity.freeze()),
            )?;
            state.decode(
                &mut self.decoder,
                &mut target,
                Some(Filter::Mask(leaf_filter.freeze())),
            )?;

            self.iter.reuse_page_buffer(page);
        }

        // we pop the primitive off here.
        debug_assert!(matches!(
            nested_state.nested.last().unwrap().content,
            NestedContent::Primitive
        ));
        _ = nested_state.pop().unwrap();

        let array = self.decoder.finalize(self.dtype, self.dict, target)?;

        Ok((nested_state, array))
    }
}
