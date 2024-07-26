use arrow::array::{Array, DictionaryArray, DictionaryKey};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::ArrowDataType;
use polars_error::PolarsResult;

use super::utils::{self, BatchableCollector};
use super::{BasicDecompressor, CompressedPagesIter, ParquetError};
use crate::parquet::encoding::hybrid_rle::gatherer::HybridRleGatherer;
use crate::parquet::encoding::hybrid_rle::HybridRleDecoder;
use crate::parquet::error::ParquetResult;
use crate::parquet::page::{split_buffer, DataPage, Page};
use crate::parquet::read::levels::get_bit_width;
use crate::read::deserialize::dictionary::DictionaryDecoder;
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

#[allow(clippy::too_many_arguments)]
fn extend_offsets2<'a, D: utils::NestedDecoder>(
    def_iter: HybridRleDecoder<'a>,
    rep_iter: HybridRleDecoder<'a>,
    batched_collector: &mut BatchedCollector<
        '_,
        (),
        D::DecodedState,
        BatchedNestedDecoder<'a, '_, '_, D>,
    >,
    nested: &mut [Nested],
    additional: usize,
    // Amortized allocations
    def_levels: &[u16],
    rep_levels: &[u16],
) -> PolarsResult<bool> {
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

    const ROW_IDX_BATCH_SIZE: usize = 1024;
    let mut state = RowIdxOffsetState {
        num_elements_seen: 0,
        top_level_limit: additional,
        found: None,
    };

    let mut row_idx_iter = rep_iter.clone();
    while row_idx_iter.len() > 0 && state.found.is_none() {
        row_idx_iter.gather_n_into(&mut state, ROW_IDX_BATCH_SIZE, &RowIdxOffsetGatherer)?;
    }

    debug_assert_eq!(def_iter.len(), rep_iter.len());

    let mut def_iter = def_iter;
    let mut rep_iter = rep_iter;
    let mut def_values = [0u16; DECODE_BATCH_SIZE];
    let mut rep_values = [0u16; DECODE_BATCH_SIZE];

    let max_depth = nested.len();

    const DECODE_BATCH_SIZE: usize = 1024;
    let mut limit = state.found.unwrap_or(def_iter.len());
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
        }

        limit -= additional;
    }

    Ok(def_iter.len() != 0)
}

pub struct PageNestedDecoder<I: CompressedPagesIter, D: utils::NestedDecoder> {
    pub iter: BasicDecompressor<I>,
    pub data_type: ArrowDataType,
    pub dict: Option<D::Dict>,
    pub decoder: D,
    pub init: Vec<InitNested>,
}

pub struct PageNestedDictArrayDecoder<
    I: CompressedPagesIter,
    K: DictionaryKey,
    D: utils::NestedDecoder,
> {
    pub iter: BasicDecompressor<I>,
    pub data_type: ArrowDataType,
    pub dict: D::Dict,
    pub decoder: D,
    pub init: Vec<InitNested>,
    _pd: std::marker::PhantomData<K>,
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

impl<I: CompressedPagesIter, D: utils::NestedDecoder> PageNestedDecoder<I, D> {
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
        let (def_levels, rep_levels) = nested_state.levels();

        loop {
            let Some(page) = self.iter.next()? else {
                break;
            };

            let Page::Data(page) = page else {
                // @TODO This should be removed
                unreachable!();
            };

            let mut values_page =
                utils::State::new_nested(&self.decoder, page, self.dict.as_ref())?;
            let (def_iter, rep_iter) = level_iters(page)?;

            let start_length = nested_state.len();

            // @TODO: move this to outside the loop.
            let mut batched_collector = BatchedCollector::new(
                BatchedNestedDecoder {
                    state: &mut values_page,
                    decoder: &mut self.decoder,
                },
                &mut target,
            );

            let is_fully_read = extend_offsets2(
                def_iter,
                rep_iter,
                &mut batched_collector,
                &mut nested_state.nested,
                limit,
                &def_levels,
                &rep_levels,
            )?;

            batched_collector.finalize()?;

            let num_done = nested_state.len() - start_length;
            debug_assert!(num_done <= limit);
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

impl<I: CompressedPagesIter, K: DictionaryKey, D: utils::NestedDecoder>
    PageNestedDictArrayDecoder<I, K, D>
{
    pub fn new(
        mut iter: BasicDecompressor<I>,
        data_type: ArrowDataType,
        decoder: D,
        init: Vec<InitNested>,
    ) -> ParquetResult<Self> {
        let dict_page = iter
            .read_dict_page()?
            .ok_or(ParquetError::FeatureNotSupported(
                "Dictionary array without a dictionary page".to_string(),
            ))?;
        let dict = decoder.deserialize_dict(dict_page);

        Ok(Self {
            iter,
            data_type,
            dict,
            decoder,
            init,
            _pd: std::marker::PhantomData,
        })
    }

    pub fn collect_n(mut self, limit: usize) -> ParquetResult<(NestedState, DictionaryArray<K>)> {
        use streaming_decompression::FallibleStreamingIterator;

        let mut target = (
            Vec::with_capacity(limit),
            MutableBitmap::with_capacity(limit),
        );
        // @TODO: Self capacity
        let mut nested_state = init_nested(&self.init, 0);

        if limit == 0 {
            let (values, validity) = target;
            let validity = if !validity.is_empty() {
                Some(validity.freeze())
            } else {
                None
            };
            return Ok((
                nested_state,
                self.decoder
                    .finalize_dict_array(self.data_type, self.dict, (values, validity))?,
            ));
        }

        let mut limit = limit;

        // Amortize the allocations.
        let (def_levels, rep_levels) = nested_state.levels();

        loop {
            let Some(page) = self.iter.next()? else {
                break;
            };

            let Page::Data(page) = page else {
                // @TODO This should be removed
                unreachable!();
            };

            use utils::ExactSize;
            let mut dictionary_decoder = DictionaryDecoder::new(self.dict.len());
            let mut values_page = utils::State::new_nested(&dictionary_decoder, page, Some(&()))?;
            let (def_iter, rep_iter) = level_iters(page)?;

            let start_length = nested_state.len();

            // @TODO: move this to outside the loop.
            let mut batched_collector = BatchedCollector::new(
                BatchedNestedDecoder {
                    state: &mut values_page,
                    decoder: &mut dictionary_decoder,
                },
                &mut target,
            );

            let is_fully_read = extend_offsets2(
                def_iter,
                rep_iter,
                &mut batched_collector,
                &mut nested_state.nested,
                limit,
                &def_levels,
                &rep_levels,
            )?;

            batched_collector.finalize()?;

            let num_done = nested_state.len() - start_length;
            debug_assert!(num_done <= limit);
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

        let (values, validity) = target;
        let validity = if !validity.is_empty() {
            Some(validity.freeze())
        } else {
            None
        };
        let array =
            self.decoder
                .finalize_dict_array(self.data_type, self.dict, (values, validity))?;

        Ok((nested_state, array))
    }
}

/// Type def for a sharable, boxed dyn [`Iterator`] of NestedStates and arrays
pub type NestedArrayIter<'a> =
    Box<dyn Iterator<Item = PolarsResult<(NestedState, Box<dyn Array>)>> + Send + Sync + 'a>;
