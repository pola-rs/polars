use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::utils::{self, BatchableCollector};
use super::{BasicDecompressor, Filter};
use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage};
use crate::parquet::read::levels::get_bit_width;
use crate::read::deserialize::utils::{hybrid_rle_count_zeros, BatchedCollector};

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

pub struct BatchedNestedDecoder<'a, 'b, 'c, D: utils::NestedDecoder> {
    state: &'b mut utils::State<'a, D>,
    decoder: &'c mut D,
}

impl<'a, 'b, 'c, D: utils::NestedDecoder> BatchableCollector<(), D::DecodedState>
    for BatchedNestedDecoder<'a, 'b, 'c, D>
{
    fn reserve(_target: &mut D::DecodedState, _n: usize) {
        unreachable!()
    }

    fn push_n(&mut self, target: &mut D::DecodedState, n: usize) -> ParquetResult<()> {
        self.decoder.push_n_valids(self.state, target, n)?;
        Ok(())
    }

    fn push_n_nulls(&mut self, target: &mut D::DecodedState, n: usize) -> ParquetResult<()> {
        self.decoder.push_n_nulls(self.state, target, n);
        Ok(())
    }

    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.state.skip_in_place(n)
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

    pub fn pop(&mut self) -> Option<(Vec<i64>, Option<MutableBitmap>)> {
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

/// Calculate the number of leaf values that are covered by the first `limit` definition level
/// values.
fn limit_to_num_values(
    def_iter: &HybridRleDecoder<'_>,
    def_levels: &[u16],
    limit: usize,
) -> ParquetResult<usize> {
    struct NumValuesGatherer {
        leaf_def_level: u16,
    }
    struct NumValuesState {
        num_values: usize,
        length: usize,
    }

    impl HybridRleGatherer<u32> for NumValuesGatherer {
        type Target = NumValuesState;

        fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

        fn target_num_elements(&self, target: &Self::Target) -> usize {
            target.length
        }

        fn hybridrle_to_target(&self, value: u32) -> ParquetResult<u32> {
            Ok(value)
        }

        fn gather_one(&self, target: &mut Self::Target, value: u32) -> ParquetResult<()> {
            target.num_values += usize::from(value == self.leaf_def_level as u32);
            target.length += 1;
            Ok(())
        }

        fn gather_repeated(
            &self,
            target: &mut Self::Target,
            value: u32,
            n: usize,
        ) -> ParquetResult<()> {
            target.num_values += n * usize::from(value == self.leaf_def_level as u32);
            target.length += n;
            Ok(())
        }
    }

    let mut state = NumValuesState {
        num_values: 0,
        length: 0,
    };
    def_iter.clone().gather_n_into(
        &mut state,
        limit,
        &NumValuesGatherer {
            leaf_def_level: *def_levels.last().unwrap(),
        },
    )?;

    Ok(state.num_values)
}

fn idx_to_limit(rep_iter: &HybridRleDecoder<'_>, idx: usize) -> ParquetResult<usize> {
    struct RowIdxOffsetGatherer;
    struct RowIdxOffsetState {
        num_elements_seen: usize,
        top_level_limit: usize,
        found: Option<usize>,
    }

    impl HybridRleGatherer<bool> for RowIdxOffsetGatherer {
        type Target = RowIdxOffsetState;

        fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

        fn target_num_elements(&self, target: &Self::Target) -> usize {
            target.num_elements_seen
        }

        fn hybridrle_to_target(&self, value: u32) -> ParquetResult<bool> {
            Ok(value == 0)
        }

        fn gather_one(&self, target: &mut Self::Target, value: bool) -> ParquetResult<()> {
            let idx = target.num_elements_seen;
            target.num_elements_seen += 1;

            if !value || target.found.is_some() {
                return Ok(());
            }

            if target.top_level_limit > 0 {
                target.top_level_limit -= 1;
                return Ok(());
            }

            target.found = Some(idx);

            Ok(())
        }

        fn gather_repeated(
            &self,
            target: &mut Self::Target,
            value: bool,
            n: usize,
        ) -> ParquetResult<()> {
            let idx = target.num_elements_seen;
            target.num_elements_seen += n;

            if !value || target.found.is_some() {
                return Ok(());
            }

            if target.top_level_limit >= n {
                target.top_level_limit -= n;
                return Ok(());
            }

            target.found = Some(idx + target.top_level_limit);
            target.top_level_limit = 0;

            Ok(())
        }

        // @TODO: Add specialization for other methods
    }

    let mut state = RowIdxOffsetState {
        num_elements_seen: 0,
        top_level_limit: idx,
        found: None,
    };

    const ROW_IDX_BATCH_SIZE: usize = 1024;

    let mut row_idx_iter = rep_iter.clone();
    while row_idx_iter.len() > 0 && state.found.is_none() {
        row_idx_iter.gather_n_into(&mut state, ROW_IDX_BATCH_SIZE, &RowIdxOffsetGatherer)?;
    }

    Ok(state.found.unwrap_or(rep_iter.len()))
}

#[allow(clippy::too_many_arguments)]
fn extend_offsets2<'a, D: utils::NestedDecoder>(
    mut def_iter: HybridRleDecoder<'a>,
    mut rep_iter: HybridRleDecoder<'a>,
    batched_collector: &mut BatchedCollector<
        '_,
        (),
        D::DecodedState,
        BatchedNestedDecoder<'a, '_, '_, D>,
    >,
    nested: &mut [Nested],
    filter: Option<Filter>,

    def_levels: &[u16],
    rep_levels: &[u16],
) -> PolarsResult<()> {
    debug_assert_eq!(def_iter.len(), rep_iter.len());

    match filter {
        None => {
            let limit = def_iter.len();

            extend_offsets_limited(
                &mut def_iter,
                &mut rep_iter,
                batched_collector,
                nested,
                limit,
                def_levels,
                rep_levels,
            )?;

            debug_assert_eq!(def_iter.len(), rep_iter.len());
            debug_assert_eq!(def_iter.len(), 0);

            Ok(())
        },
        Some(Filter::Range(range)) => {
            let start = range.start;
            let end = range.end;

            if start > 0 {
                let start_cell = idx_to_limit(&rep_iter, start)?;

                let num_skipped_values = limit_to_num_values(&def_iter, def_levels, start_cell)?;
                batched_collector.skip_in_place(num_skipped_values)?;

                rep_iter.skip_in_place(start_cell)?;
                def_iter.skip_in_place(start_cell)?;
            }

            if end - start > 0 {
                let limit = idx_to_limit(&rep_iter, end - start)?;

                extend_offsets_limited(
                    &mut def_iter,
                    &mut rep_iter,
                    batched_collector,
                    nested,
                    limit,
                    def_levels,
                    rep_levels,
                )?;
            }

            // @NOTE: This is kind of unused
            let last_skip = def_iter.len();
            let num_skipped_values = limit_to_num_values(&def_iter, def_levels, last_skip)?;
            batched_collector.skip_in_place(num_skipped_values)?;
            rep_iter.skip_in_place(last_skip)?;
            def_iter.skip_in_place(last_skip)?;

            Ok(())
        },
        Some(Filter::Mask(bitmap)) => {
            let mut iter = bitmap.iter();
            while iter.num_remaining() > 0 {
                let num_zeros = iter.take_leading_zeros();
                if num_zeros > 0 {
                    let offset = idx_to_limit(&rep_iter, num_zeros)?;
                    let num_skipped_values = limit_to_num_values(&def_iter, def_levels, offset)?;
                    batched_collector.skip_in_place(num_skipped_values)?;
                    rep_iter.skip_in_place(offset)?;
                    def_iter.skip_in_place(offset)?;
                }

                let num_ones = iter.take_leading_ones();
                if num_ones > 0 {
                    let limit = idx_to_limit(&rep_iter, num_ones)?;
                    extend_offsets_limited(
                        &mut def_iter,
                        &mut rep_iter,
                        batched_collector,
                        nested,
                        limit,
                        def_levels,
                        rep_levels,
                    )?;
                }
            }

            Ok(())
        },
    }
}

fn extend_offsets_limited<'a, D: utils::NestedDecoder>(
    def_iter: &mut HybridRleDecoder<'a>,
    rep_iter: &mut HybridRleDecoder<'a>,
    batched_collector: &mut BatchedCollector<
        '_,
        (),
        D::DecodedState,
        BatchedNestedDecoder<'a, '_, '_, D>,
    >,
    nested: &mut [Nested],
    mut limit: usize,
    // Amortized allocations
    def_levels: &[u16],
    rep_levels: &[u16],
) -> PolarsResult<()> {
    #[derive(Default)]
    struct LevelGatherer<'a>(std::marker::PhantomData<&'a ()>);
    struct LevelGathererState<'a> {
        offset: usize,
        slice: &'a mut [u16],
    }

    impl<'a> HybridRleGatherer<u16> for LevelGatherer<'a> {
        type Target = LevelGathererState<'a>;

        fn target_reserve(&self, _target: &mut Self::Target, _n: usize) {}

        fn target_num_elements(&self, target: &Self::Target) -> usize {
            target.offset
        }

        fn hybridrle_to_target(&self, value: u32) -> ParquetResult<u16> {
            debug_assert!(value <= u16::MAX as u32);
            Ok(value as u16)
        }

        fn gather_one(&self, target: &mut Self::Target, value: u16) -> ParquetResult<()> {
            debug_assert!(target.offset < target.slice.len());

            target.slice[target.offset] = value;
            target.offset += 1;

            Ok(())
        }

        fn gather_repeated(
            &self,
            target: &mut Self::Target,
            value: u16,
            n: usize,
        ) -> ParquetResult<()> {
            debug_assert!(target.offset + n <= target.slice.len());

            for i in 0..n {
                target.slice[target.offset + i] = value;
            }
            target.offset += n;

            Ok(())
        }

        // @TODO: Add specialization for other methods
    }

    let mut def_values = [0u16; DECODE_BATCH_SIZE];
    let mut rep_values = [0u16; DECODE_BATCH_SIZE];

    let max_depth = nested.len();

    const DECODE_BATCH_SIZE: usize = 1024;
    while def_iter.len() > 0 && limit > 0 {
        let additional = usize::min(limit, DECODE_BATCH_SIZE);

        let mut def_state = LevelGathererState {
            offset: 0,
            slice: &mut def_values,
        };
        let mut rep_state = LevelGathererState {
            offset: 0,
            slice: &mut rep_values,
        };

        def_iter.gather_n_into(&mut def_state, additional, &LevelGatherer::default())?;
        rep_iter.gather_n_into(&mut rep_state, additional, &LevelGatherer::default())?;

        debug_assert_eq!(def_state.offset, rep_state.offset);
        debug_assert_eq!(def_state.offset, additional);

        for i in 0..additional {
            let def = def_values[i];
            let rep = rep_values[i];

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

                        let embed_num_values = embed_nest.invalid_num_values();
                        num_elements *= embed_num_values;

                        if embed_num_values == 0 {
                            break;
                        }
                    }

                    batched_collector.push_n_invalids(num_elements);

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
        }

        limit -= additional;
    }

    Ok(())
}

pub struct PageNestedDecoder<D: utils::NestedDecoder> {
    pub iter: BasicDecompressor,
    pub data_type: ArrowDataType,
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

impl<D: utils::NestedDecoder> PageNestedDecoder<D> {
    pub fn new(
        mut iter: BasicDecompressor,
        data_type: ArrowDataType,
        decoder: D,
        init: Vec<InitNested>,
    ) -> ParquetResult<Self> {
        let dict_page = iter.read_dict_page()?;
        let dict = dict_page.map(|d| decoder.deserialize_dict(d)).transpose()?;

        Ok(Self {
            iter,
            data_type,
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

        match filter {
            None => {
                loop {
                    let Some(page) = self.iter.next() else {
                        break;
                    };
                    let page = page?;
                    let page = page.decompress(&mut self.iter)?;

                    let mut state =
                        utils::State::new_nested(&self.decoder, &page, self.dict.as_ref())?;
                    let (def_iter, rep_iter) = level_iters(&page)?;

                    // @TODO: move this to outside the loop.
                    let mut batched_collector = BatchedCollector::new(
                        BatchedNestedDecoder {
                            state: &mut state,
                            decoder: &mut self.decoder,
                        },
                        &mut target,
                    );

                    extend_offsets2(
                        def_iter,
                        rep_iter,
                        &mut batched_collector,
                        &mut nested_state.nested,
                        None,
                        &def_levels,
                        &rep_levels,
                    )?;

                    batched_collector.finalize()?;

                    drop(state);
                    self.iter.reuse_page_buffer(page);
                }
            },
            Some(mut filter) => {
                enum PageStartAction {
                    Skip,
                    Collect,
                }

                // We may have an action (skip / collect) for one row value left over from the
                // previous page. Every page may state what the next page needs to do until the
                // first of its own row values (rep_lvl = 0).
                let mut last_row_value_action = PageStartAction::Skip;
                let mut num_rows_remaining = filter.num_rows();

                while num_rows_remaining > 0
                    || matches!(last_row_value_action, PageStartAction::Collect)
                {
                    let Some(page) = self.iter.next() else {
                        break;
                    };
                    let page = page?;
                    // We cannot lazily decompress because we don't have the number of row values
                    // at this point. We need repetition levels for that. *sign*. In general, lazy
                    // decompression is quite difficult with nested values.
                    //
                    // @TODO
                    // Lazy decompression is quite doable in the V2 specification since that does
                    // not compress the repetition and definition levels. However, not a lot of
                    // people use the V2 specification. So let us ignore that for now.
                    let page = page.decompress(&mut self.iter)?;

                    let (mut def_iter, mut rep_iter) = level_iters(&page)?;

                    let mut state;
                    let mut batched_collector;

                    let start_length = nested_state.len();

                    // rep lvl == 0 ==> row value
                    let num_row_values = hybrid_rle_count_zeros(&rep_iter)?;

                    let state_filter;
                    (state_filter, filter) = Filter::split_at(&filter, num_row_values);

                    match last_row_value_action {
                        PageStartAction::Skip => {
                            // Fast path: skip the whole page.
                            // No new row values or we don't care about any of the row values.
                            if num_row_values == 0 && state_filter.num_rows() == 0 {
                                self.iter.reuse_page_buffer(page);
                                continue;
                            }

                            let limit = idx_to_limit(&rep_iter, 0)?;

                            // We just saw that we had at least one row value.
                            debug_assert!(limit < rep_iter.len());

                            state =
                                utils::State::new_nested(&self.decoder, &page, self.dict.as_ref())?;
                            batched_collector = BatchedCollector::new(
                                BatchedNestedDecoder {
                                    state: &mut state,
                                    decoder: &mut self.decoder,
                                },
                                &mut target,
                            );

                            let num_leaf_values =
                                limit_to_num_values(&def_iter, &def_levels, limit)?;
                            batched_collector.skip_in_place(num_leaf_values)?;
                            rep_iter.skip_in_place(limit)?;
                            def_iter.skip_in_place(limit)?;
                        },
                        PageStartAction::Collect => {
                            let limit = if num_row_values == 0 {
                                rep_iter.len()
                            } else {
                                idx_to_limit(&rep_iter, 0)?
                            };

                            // Fast path: we are not interested in any of the row values in this
                            // page.
                            if limit == 0 && state_filter.num_rows() == 0 {
                                self.iter.reuse_page_buffer(page);
                                last_row_value_action = PageStartAction::Skip;
                                continue;
                            }

                            state =
                                utils::State::new_nested(&self.decoder, &page, self.dict.as_ref())?;
                            batched_collector = BatchedCollector::new(
                                BatchedNestedDecoder {
                                    state: &mut state,
                                    decoder: &mut self.decoder,
                                },
                                &mut target,
                            );

                            extend_offsets_limited(
                                &mut def_iter,
                                &mut rep_iter,
                                &mut batched_collector,
                                &mut nested_state.nested,
                                limit,
                                &def_levels,
                                &rep_levels,
                            )?;

                            // No new row values. Keep collecting.
                            if rep_iter.len() == 0 {
                                batched_collector.finalize()?;

                                let num_done = nested_state.len() - start_length;
                                debug_assert!(num_done <= num_rows_remaining);
                                debug_assert!(num_done <= num_row_values);
                                num_rows_remaining -= num_done;

                                drop(state);
                                self.iter.reuse_page_buffer(page);

                                continue;
                            }
                        },
                    }

                    // Two cases:
                    // 1. First page: Must always start with a row value.
                    // 2. Other pages: If they did not have a row value, they would have been
                    //    handled by the last_row_value_action.
                    debug_assert!(num_row_values > 0);

                    last_row_value_action = if state_filter.do_include_at(num_row_values - 1) {
                        PageStartAction::Collect
                    } else {
                        PageStartAction::Skip
                    };

                    extend_offsets2(
                        def_iter,
                        rep_iter,
                        &mut batched_collector,
                        &mut nested_state.nested,
                        Some(state_filter),
                        &def_levels,
                        &rep_levels,
                    )?;

                    batched_collector.finalize()?;

                    let num_done = nested_state.len() - start_length;
                    debug_assert!(num_done <= num_rows_remaining);
                    debug_assert!(num_done <= num_row_values);
                    num_rows_remaining -= num_done;

                    drop(state);
                    self.iter.reuse_page_buffer(page);
                }
            },
        }

        // we pop the primitive off here.
        debug_assert!(matches!(
            nested_state.nested.last().unwrap().content,
            NestedContent::Primitive
        ));
        _ = nested_state.pop().unwrap();

        let array = self.decoder.finalize(self.data_type, self.dict, target)?;

        Ok((nested_state, array))
    }
}
