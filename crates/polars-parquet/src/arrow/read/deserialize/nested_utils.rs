use std::collections::VecDeque;
use std::iter::{Peekable, Zip};

use arrow::array::Array;
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::{polars_bail, PolarsResult};

use super::utils::{BatchableCollector, ExactSize, PageState};
use super::{BasicDecompressor, CompressedPagesIter};
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, DictPage, Page};
use crate::parquet::read::levels::get_bit_width;
use crate::read::deserialize::utils::BatchedCollector;

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

    fn take(mut self) -> (Vec<i64>, Option<MutableBitmap>) {
        if !matches!(self.content, NestedContent::Primitive) {
            if let Some(validity) = self.validity.as_mut() {
                validity.extend_constant(self.num_valids, true);
                validity.extend_constant(self.num_invalids, false);
            }
        }

        self.num_valids = 0;
        self.num_invalids = 0;

        match self.content {
            NestedContent::Primitive => {
                debug_assert!(self.validity.map_or(true, |validity| validity.is_empty()));
                (Vec::new(), None)
            },
            NestedContent::List { offsets } => (offsets, self.validity),
            NestedContent::FixedSizeList { .. } => (Vec::new(), self.validity),
            NestedContent::Struct => (Vec::new(), self.validity),
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
            NestedContent::Primitive => 0,
            NestedContent::List { .. } => 0,
            NestedContent::FixedSizeList { width } => *width,
            NestedContent::Struct => 1,
        }
    }

    fn push(&mut self, value: i64, is_valid: bool) {
        let is_primitive = matches!(self.content, NestedContent::Primitive);

        if is_valid && self.num_invalids != 0 {
            debug_assert!(!is_primitive);

            let validity = self.validity.as_mut().unwrap();
            validity.extend_constant(self.num_valids, true);
            validity.extend_constant(self.num_invalids, false);

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
        debug_assert!(self.validity.is_some());

        let is_primitive = matches!(self.content, NestedContent::Primitive);
        self.num_invalids += usize::from(!is_primitive);

        self.length += 1;
        if let NestedContent::List { offsets } = &mut self.content {
            offsets.push(length);
        }
    }
}

pub struct BatchedNestedDecoder<'a, 'b, 'c, D: NestedDecoder> {
    state: &'b mut D::State<'a>,
    decoder: &'c D,
}

impl<'a, 'b, 'c, D: NestedDecoder> BatchableCollector<(), D::DecodedState>
    for BatchedNestedDecoder<'a, 'b, 'c, D>
{
    fn reserve(_target: &mut D::DecodedState, _n: usize) {
        unreachable!()
    }

    fn push_n(&mut self, target: &mut D::DecodedState, n: usize) -> ParquetResult<()> {
        self.decoder.push_n_valid(self.state, target, n)
    }

    fn push_n_nulls(&mut self, target: &mut D::DecodedState, n: usize) -> ParquetResult<()> {
        self.decoder.push_n_nulls(target, n);
        Ok(())
    }
}

/// A decoder that knows how to map `State` -> Array
pub(super) trait NestedDecoder {
    type State<'a>: PageState<'a>;
    type Dict;
    type DecodedState: ExactSize;

    fn build_state<'a>(
        &self,
        page: &'a DataPage,
        dict: Option<&'a Self::Dict>,
    ) -> PolarsResult<Self::State<'a>>;

    /// Initializes a new state
    fn with_capacity(&self, capacity: usize) -> Self::DecodedState;

    fn push_n_valid(
        &self,
        state: &mut Self::State<'_>,
        decoded: &mut Self::DecodedState,
        n: usize,
    ) -> ParquetResult<()>;
    fn push_n_nulls(&self, decoded: &mut Self::DecodedState, n: usize);

    fn deserialize_dict(&self, page: DictPage) -> Self::Dict;

    fn finalize(
        &self,
        data_type: ArrowDataType,
        decoded: Self::DecodedState,
    ) -> ParquetResult<Box<dyn Array>>;
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

pub struct NestedPage<'a> {
    iter: Peekable<Zip<HybridRleDecoder<'a>, HybridRleDecoder<'a>>>,
}

impl<'a> NestedPage<'a> {
    pub fn try_new(page: &'a DataPage) -> PolarsResult<Self> {
        let split = split_buffer(page)?;
        let rep_levels = split.rep;
        let def_levels = split.def;

        let max_rep_level = page.descriptor.max_rep_level;
        let max_def_level = page.descriptor.max_def_level;

        let reps =
            HybridRleDecoder::new(rep_levels, get_bit_width(max_rep_level), page.num_values());
        let defs =
            HybridRleDecoder::new(def_levels, get_bit_width(max_def_level), page.num_values());

        let reps = reps.into_iter();
        let defs = defs.into_iter();

        let iter = reps.zip(defs).peekable();

        Ok(Self { iter })
    }

    // number of values (!= number of rows)
    pub fn len(&self) -> usize {
        self.iter.size_hint().0
    }
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

    pub fn pop(&mut self) -> Option<(Vec<i64>, Option<MutableBitmap>)> {
        Some(self.nested.pop()?.take())
    }

    /// The number of rows in this state
    pub fn len(&self) -> usize {
        // outermost is the number of rows
        self.nested[0].len()
    }
}

/// Extends `items` by consuming `page`, first trying to complete the last `item`
/// and extending it if more are needed.
///
/// Note that as the page iterator being passed does not guarantee it reads to
/// the end, this function cannot always determine whether it has finished
/// reading. It therefore returns a bool indicating:
/// * true  : the row is fully read
/// * false : the row may not be fully read
pub(super) fn extend<D: NestedDecoder>(
    page: &DataPage,
    init: &[InitNested],
    items: &mut VecDeque<(NestedState, D::DecodedState)>,
    dict: Option<&D::Dict>,
    remaining: &mut usize,
    decoder: &D,
) -> PolarsResult<bool> {
    let mut values_page = decoder.build_state(page, dict)?;
    let mut page = NestedPage::try_new(page)?;

    debug_assert!(
        items.len() < 2,
        "Should have yielded already completed item before reading more."
    );

    let additional = *remaining;
    let mut first_item_is_fully_read = false;

    // Amortize the allocations.
    let mut def_levels = vec![];
    let mut rep_levels = vec![];

    loop {
        if let Some((mut nested, mut decoded)) = items.pop_back() {
            let existing = nested.len();

            let mut batched_collector = BatchedCollector::new(
                BatchedNestedDecoder {
                    state: &mut values_page,
                    decoder,
                },
                &mut decoded,
            );

            let depth = nested.nested.len();

            def_levels.clear();
            rep_levels.clear();

            def_levels.push(0);
            rep_levels.push(0);

            for i in 0..depth {
                let nest = &nested.nested[i];

                let def_delta = nest.is_nullable() as u32 + nest.is_repeated() as u32;
                let rep_delta = nest.is_repeated() as u32;

                def_levels.push(def_levels[i] + def_delta);
                rep_levels.push(rep_levels[i] + rep_delta);
            }

            let is_fully_read = extend_offsets2(
                &mut page,
                &mut batched_collector,
                &mut nested.nested,
                additional,
                &def_levels,
                &rep_levels,
            )?;

            batched_collector.finalize()?;

            first_item_is_fully_read |= is_fully_read;
            *remaining -= nested.len() - existing;
            items.push_back((nested, decoded));

            if page.len() == 0 {
                break;
            }

            if is_fully_read && *remaining == 0 {
                break;
            };
        };

        // At this point:
        // * There are more pages.
        // * The remaining rows have not been fully read.
        // * The deque is empty, or the last item already holds completed data.
        let nested = init_nested(init, *remaining);
        let decoded = decoder.with_capacity(0);
        items.push_back((nested, decoded));
    }

    Ok(first_item_is_fully_read)
}

#[allow(clippy::too_many_arguments)]
fn extend_offsets2<'a, D: NestedDecoder>(
    page: &mut NestedPage<'a>,
    batched_collector: &mut BatchedCollector<
        '_,
        (),
        D::DecodedState,
        BatchedNestedDecoder<'a, '_, '_, D>,
    >,
    nested: &mut [Nested],
    additional: usize,
    // Amortized allocations
    def_levels: &[u32],
    rep_levels: &[u32],
) -> PolarsResult<bool> {
    let max_depth = nested.len();

    let mut rows = 0;
    loop {
        // SAFETY: page.iter is always non-empty on first loop.
        // The current function gets called multiple times with iterators that
        // yield batches of pages. This means e.g. it could be that the very
        // first page is a new row, and the existing nested state has already
        // contains all data from the additional rows.
        if page.iter.peek().unwrap().0 == 0 {
            if rows == additional {
                return Ok(true);
            }
            rows += 1;
        }

        // The errors of the FallibleIterators use in this zipped not checked yet.
        // If one of them errors, the iterator returns None, and this `unwrap` will panic.
        let Some((rep, def)) = page.iter.next() else {
            polars_bail!(ComputeError: "cannot read rep/def levels")
        };

        let mut is_required = false;

        for depth in 0..max_depth {
            // Defines whether this element is defined at `depth`
            //
            // e.g. [ [ [ 1 ] ] ] is defined at [ ... ], [ [ ... ] ], [ [ [ ... ] ] ] and
            // [ [ [ 1 ] ] ].
            let is_defined_at_this_depth = rep <= rep_levels[depth] && def >= def_levels[depth];

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

                    if embed_depth == max_depth - 1 {
                        for _ in 0..num_elements {
                            batched_collector.push_invalid();
                        }

                        break;
                    }

                    let embed_num_values = embed_nest.invalid_num_values();

                    if embed_num_values == 0 {
                        break;
                    }

                    num_elements *= embed_num_values;
                }

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
                        batched_collector.push_invalid();
                    }
                }
            }

            is_required =
                (is_required || is_defined_at_this_depth) && nest.is_required() && !is_valid;
        }

        if page.iter.len() == 0 {
            return Ok(false);
        }
    }
}

pub struct PageNestedDecoder<I: CompressedPagesIter, D: NestedDecoder> {
    pub iter: BasicDecompressor<I>,
    pub data_type: ArrowDataType,
    pub dict: Option<D::Dict>,
    pub decoder: D,
    pub init: Vec<InitNested>,
}

impl<I: CompressedPagesIter, D: NestedDecoder> PageNestedDecoder<I, D> {
    pub fn new(
        mut iter: BasicDecompressor<I>,
        data_type: ArrowDataType,
        decoder: D,
        init: Vec<InitNested>,
    ) -> ParquetResult<Self> {
        let dict_page = iter.read_dict_page()?;
        let dict = dict_page.map(|d| decoder.deserialize_dict(d));

        Ok(Self {
            iter,
            data_type,
            dict,
            decoder,
            init,
        })
    }

    pub fn collect_n(mut self, limit: usize) -> ParquetResult<(NestedState, Box<dyn Array>)> {
        use streaming_decompression::FallibleStreamingIterator;

        let mut target = self.decoder.with_capacity(limit);
        // @TODO: Self capacity
        let mut nested_state = init_nested(&self.init, 0);

        if limit == 0 {
            return Ok((nested_state, self.decoder.finalize(self.data_type, target)?));
        }

        let mut limit = limit;

        // Amortize the allocations.
        let depth = nested_state.nested.len();

        let mut def_levels = Vec::with_capacity(depth + 1);
        let mut rep_levels = Vec::with_capacity(depth + 1);

        def_levels.push(0);
        rep_levels.push(0);

        for i in 0..depth {
            let nest = &nested_state.nested[i];

            let def_delta = nest.is_nullable() as u32 + nest.is_repeated() as u32;
            let rep_delta = nest.is_repeated() as u32;

            def_levels.push(def_levels[i] + def_delta);
            rep_levels.push(rep_levels[i] + rep_delta);
        }

        loop {
            let Some(page) = self.iter.next()? else {
                break;
            };

            let Page::Data(page) = page else {
                // @TODO This should be removed
                unreachable!();
            };

            let mut values_page = self.decoder.build_state(page, self.dict.as_ref())?;
            let mut page = NestedPage::try_new(page)?;

            let start_length = nested_state.len();

            // @TODO: move this to outside the loop.
            let mut batched_collector = BatchedCollector::new(
                BatchedNestedDecoder {
                    state: &mut values_page,
                    decoder: &self.decoder,
                },
                &mut target,
            );

            let is_fully_read = extend_offsets2(
                &mut page,
                &mut batched_collector,
                &mut nested_state.nested,
                limit,
                &def_levels,
                &rep_levels,
            )?;

            batched_collector.finalize()?;

            let num_done = nested_state.len() - start_length;
            limit -= num_done;

            debug_assert!(values_page.len() == 0 || limit == 0);

            if is_fully_read {
                break;
            }
        }

        // we pop the primitive off here.
        debug_assert!(matches!(
            nested_state.nested.last().unwrap().content,
            NestedContent::Primitive
        ));
        _ = nested_state.pop().unwrap();

        let array = self.decoder.finalize(self.data_type, target)?;

        Ok((nested_state, array))
    }
}

/// Type def for a sharable, boxed dyn [`Iterator`] of NestedStates and arrays
pub type NestedArrayIter<'a> =
    Box<dyn Iterator<Item = PolarsResult<(NestedState, Box<dyn Array>)>> + Send + Sync + 'a>;
