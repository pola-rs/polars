use std::ops::Range;

use polars_arrow::bitmap::utils::BitmapIter;
use polars_arrow::bitmap::{Bitmap, BitmapBuilder, MutableBitmap};
use polars_arrow::trusted_len::TrustedLen;

use super::utils::PageDecoder;
use super::{Filter, utils};
use crate::parquet::encoding::hybrid_rle::{HybridRleChunk, HybridRleDecoder};
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::page::{DataPage, split_buffer};
use crate::parquet::read::levels::get_bit_width;
use crate::read::deserialize::utils::Decoded;

pub struct Nested {
    validity: Option<BitmapBuilder>,
    length: usize,
    content: NestedContent,

    // We batch the collection of valids and invalids to amortize the costs. This only really works
    // when valids and invalids are grouped or there is a disbalance in the amount of valids vs.
    // invalids. This, however, is a very common situation.
    num_valids: usize,
    num_invalids: usize,

    // Whether a fixed-size list row was found with a number of children other than `width`.
    is_misaligned: bool,
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
        let validity = is_nullable.then(|| BitmapBuilder::with_capacity(0));

        Self {
            validity,
            length: 0,
            content: NestedContent::Primitive,

            num_valids: 0,
            num_invalids: 0,
            is_misaligned: false,
        }
    }

    fn list_with_capacity(is_nullable: bool, capacity: usize) -> Self {
        let offsets = Vec::with_capacity(capacity);
        let validity = is_nullable.then(|| BitmapBuilder::with_capacity(capacity));
        Self {
            validity,
            length: 0,
            content: NestedContent::List { offsets },

            num_valids: 0,
            num_invalids: 0,
            is_misaligned: false,
        }
    }

    fn fixedlist_with_capacity(is_nullable: bool, width: usize, capacity: usize) -> Self {
        let validity = is_nullable.then(|| BitmapBuilder::with_capacity(capacity));
        Self {
            validity,
            length: 0,
            content: NestedContent::FixedSizeList { width },

            num_valids: 0,
            num_invalids: 0,
            is_misaligned: false,
        }
    }

    fn struct_with_capacity(is_nullable: bool, capacity: usize) -> Self {
        let validity = is_nullable.then(|| BitmapBuilder::with_capacity(capacity));
        Self {
            validity,
            length: 0,
            content: NestedContent::Struct,

            num_valids: 0,
            num_invalids: 0,
            is_misaligned: false,
        }
    }

    fn take(mut self) -> (usize, Vec<i64>, Option<BitmapBuilder>) {
        if !matches!(self.content, NestedContent::Primitive) {
            self.flush_validity();

            debug_assert!(
                self.validity
                    .as_ref()
                    .is_none_or(|v| v.len() == self.length)
            );
        }

        match self.content {
            NestedContent::Primitive => {
                debug_assert!(self.validity.is_none_or(|validity| validity.is_empty()));
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

    // Always inlined, as it is on the hot path of every collect loop.
    #[inline(always)]
    fn push(&mut self, value: i64, is_valid: bool) {
        let is_primitive = matches!(self.content, NestedContent::Primitive);

        if is_valid && self.num_invalids != 0 {
            self.flush_validity();
        }
        self.num_valids += usize::from(!is_primitive & is_valid);
        self.num_invalids += usize::from(!is_primitive & !is_valid);
        self.length += 1;
        if let NestedContent::List { offsets } = &mut self.content {
            offsets.push(value);
        }
    }

    /// Pushes `n` valid items, the same as `n` calls to `push` apart from the offsets of a list.
    fn push_n_valids(&mut self, n: usize) {
        let is_primitive = matches!(self.content, NestedContent::Primitive);
        if self.num_invalids != 0 {
            self.flush_validity();
        }
        self.num_valids += if is_primitive { 0 } else { n };
        self.length += n;
    }

    /// Writes the batched validity out.
    fn flush_validity(&mut self) {
        debug_assert!(!matches!(self.content, NestedContent::Primitive));

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

    /// Pushes an item per element of `is_valid`, valid where it is `true`, the same as calls to
    /// `push` apart from the offsets of a list.
    fn push_validities(&mut self, is_valid: impl TrustedLen<Item = bool>) {
        self.flush_validity();
        self.length += is_valid.size_hint().0;
        if let Some(validity) = self.validity.as_mut() {
            validity.extend_trusted_len_iter(is_valid);
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

    /// Marks a fixed-size list as misaligned unless its rows before the one just pushed have
    /// `width` children each, i.e. `child_len` in total. Null rows are padded to `width` children.
    fn mark_if_misaligned(&mut self, child_len: usize) {
        if let NestedContent::FixedSizeList { width } = self.content {
            self.is_misaligned |= child_len != (self.length - 1) * width;
        }
    }
}

/// Bulk collection of runs of valid, non-empty rows of the innermost (fixed-size) list, with
/// the same result as `collect_value` for each value.
struct LeafRows<'a> {
    /// A value reaches `nested[d]` iff its definition level is at least `def_levels[d]`, and
    /// is valid (and, for a list, non-empty) there iff it is at least `def_levels[d + 1]`.
    def_levels: &'a [u16],
    /// A value starts a new item of `nested[d]` iff its repetition level is at most
    /// `rep_levels[d]`, and otherwise continues the current one.
    rep_levels: &'a [u16],
    /// The indices into `nested` of the levels that each row adds a valid item to: the
    /// structs right above the list, and the list itself.
    depths: Range<usize>,
    /// The repetition level of a value that starts a new row of the list;
    /// a lower one also starts a row of an outer level.
    row_rep: u16,
    /// The repetition level of a value that adds an item to the current row, i.e. `row_rep + 1`.
    item_rep: u16,
    /// The lowest definition level of a value in a valid, non-empty row, i.e. of an item, which
    /// can still be a null struct or leaf.
    item_def: u16,
    /// Whether the list is a fixed-size list.
    is_fixed_size_list: bool,
}

impl<'a> LeafRows<'a> {
    fn new(nested: &[Nested], def_levels: &'a [u16], rep_levels: &'a [u16]) -> Option<Self> {
        let leaf = nested.len() - 1;
        let list = nested.iter().rposition(Nested::is_repeated)?;
        let row_rep = rep_levels[list];
        Some(Self {
            def_levels,
            rep_levels,
            depths: rep_levels.partition_point(|&rep| rep < row_rep)..list + 1,
            row_rep,
            item_rep: rep_levels[leaf],
            item_def: def_levels[list + 1],
            is_fixed_size_list: matches!(nested[list].content, NestedContent::FixedSizeList { .. }),
        })
    }

    /// Collects the level values of a page, in bulk for runs of these rows and with
    /// `collect_value` for any other value, and returns how many. It is not inlined,
    /// to keep `decode_nested` lean.
    #[inline(never)]
    fn collect_values<const HAS_FIXED_SIZE_LIST: bool>(
        &self,
        page_def_levels: &[u16],
        page_rep_levels: &[u16],
        num_collects: &mut usize,
        batched_collector: &mut BatchedNestedDecoder<'_>,
        nested: &mut [Nested],
    ) -> ParquetResult<usize> {
        let collect_run = if self.is_fixed_size_list {
            Self::collect_run::<true>
        } else {
            Self::collect_run::<false>
        };

        let mut i = 0;
        while i < page_def_levels.len() && (*num_collects > 0 || page_rep_levels[i] != 0) {
            // Only an item of a row can start a run.
            let n = if page_def_levels[i] >= self.item_def {
                collect_run(
                    self,
                    &page_def_levels[i..],
                    &page_rep_levels[i..],
                    num_collects,
                    nested,
                )
            } else {
                0
            };
            if n > 0 {
                // Here rather than in `collect_run`, whose loop runs out of registers otherwise.
                push_items(
                    &page_def_levels[i..i + n],
                    &mut nested[self.depths.end..],
                    &self.def_levels[self.depths.end + 1..],
                    batched_collector,
                )?;
                i += n;
            } else {
                collect_value::<HAS_FIXED_SIZE_LIST>(
                    page_def_levels[i],
                    page_rep_levels[i],
                    num_collects,
                    batched_collector,
                    nested,
                    self.def_levels,
                    self.rep_levels,
                )?;
                i += 1;
            }
        }
        Ok(i)
    }

    /// Pushes a valid item to each outer level that a row starting with repetition level `rep`
    /// starts as well, where `num_row_items` is the length of the level below them. It is cold, so
    /// that the loop of `collect_run` keeps its registers for the rows of the list.
    #[cold]
    fn push_outer_rows(&self, rep: u16, outer: &mut [Nested], num_row_items: usize) {
        for depth in self.rep_levels.partition_point(|&r| r < rep)..outer.len() {
            let child_len = outer
                .get(depth + 1)
                .map_or(num_row_items, |child| child.len());
            outer[depth].push(child_len as i64, true);
            outer[depth].mark_if_misaligned(child_len);
        }
    }

    /// Collects the leading level values that continue the current row or start new rows, apart
    /// from their items, which `push_items` pushes, and returns how many.
    fn collect_run<const IS_FIXED_SIZE_LIST: bool>(
        &self,
        page_def_levels: &[u16],
        page_rep_levels: &[u16],
        num_collects: &mut usize,
        nested: &mut [Nested],
    ) -> usize {
        let page_rep_levels = &page_rep_levels[..page_def_levels.len()];
        let num_items = nested[self.depths.end].len();
        let (outer, nested) = nested.split_at_mut(self.depths.start);
        let rows = &mut nested[..self.depths.len()];
        let num_row_items = rows[0].len();
        let list = rows.last_mut().unwrap();
        let num_list_rows = list.len();
        // A run starts at most a row per value, so reserving that many offsets keeps reallocation,
        // and the spilling of registers around it, out of the loop.
        let (mut new_offsets, width) = match &mut list.content {
            NestedContent::List { offsets } => {
                offsets.reserve(page_def_levels.len());
                (Some(offsets.spare_capacity_mut()), 0)
            },
            NestedContent::FixedSizeList { width } => (None, *width),
            _ => unreachable!(),
        };

        // As in `collect_value`, each new top-level row takes a collect. These are the rows of the
        // list if it is the top level, else those of the outermost level. Counting them is cheaper
        // than checking the repetition level of each value.
        let (max_rows, max_outer_rows) = if self.row_rep == 0 {
            (*num_collects, 0)
        } else {
            (usize::MAX, *num_collects)
        };
        let mut num_rows = 0;
        let mut num_outer_rows = 0;
        let mut is_misaligned = false;
        let mut n = 0;
        loop {
            // Values that continue the current row.
            while n < page_def_levels.len()
                && page_rep_levels[n] == self.item_rep
                && page_def_levels[n] >= self.item_def
            {
                n += 1;
            }

            // A value that starts a row, and possibly rows of outer levels.
            if n == page_def_levels.len() || page_def_levels[n] < self.item_def {
                break;
            }
            let rep = page_rep_levels[n];
            if rep == self.row_rep {
                if num_rows == max_rows {
                    break;
                }
            } else if rep < self.row_rep && (rep > 0 || num_outer_rows < max_outer_rows) {
                num_outer_rows += usize::from(rep == 0);
                self.push_outer_rows(rep, outer, num_row_items + num_rows);
            } else {
                break;
            }

            let child_len = num_items + n;
            if IS_FIXED_SIZE_LIST {
                // As in `Nested::mark_if_misaligned`.
                is_misaligned |= child_len != (num_list_rows + num_rows) * width;
            } else if let Some(new_offsets) = &mut new_offsets {
                new_offsets[num_rows].write(child_len as i64);
            }
            num_rows += 1;
            n += 1;

            // The rest of a wide fixed-size list row is checked eight values at a time, which
            // vectorizes. The loop above checks what remains.
            if IS_FIXED_SIZE_LIST && width > 8 {
                let row_end = (n + width - 1).min(page_def_levels.len());
                while n + 8 <= row_end
                    && page_rep_levels[n..n + 8]
                        .iter()
                        .zip(&page_def_levels[n..n + 8])
                        .fold(true, |all, (&rep, &def)| {
                            all & (rep == self.item_rep) & (def >= self.item_def)
                        })
                {
                    n += 8;
                }
            }
        }
        if let NestedContent::List { offsets } = &mut list.content {
            // SAFETY: The loop wrote the first `num_rows` spare offsets, one for each row.
            unsafe { offsets.set_len(offsets.len() + num_rows) };
        }
        if n == 0 {
            return 0;
        }

        *num_collects -= if self.row_rep == 0 {
            num_rows
        } else {
            num_outer_rows
        };
        for nest in rows.iter_mut() {
            nest.push_n_valids(num_rows);
        }
        rows.last_mut().unwrap().is_misaligned |= is_misaligned;
        n
    }
}

/// Pushes an item for each of the values `defs` to each of `levels`, i.e. the structs right above
/// the leaf and the leaf, with the same result as `collect_value` for each. The item of
/// `levels[d]` is valid iff its value is defined past that level, i.e. its definition level is at
/// least `valid_defs[d]`.
///
/// Typically, all items are valid, which the lowest definition level tells. That case is inlined,
/// as a run can be as short as a value.
#[inline(always)]
fn push_items(
    defs: &[u16],
    levels: &mut [Nested],
    valid_defs: &[u16],
    batched_collector: &mut BatchedNestedDecoder<'_>,
) -> ParquetResult<()> {
    let min_def = defs.iter().copied().min().unwrap_or(u16::MAX);
    if min_def < *valid_defs.last().unwrap() {
        return push_items_with_nulls(defs, levels, valid_defs, min_def, batched_collector);
    }
    for nest in levels {
        nest.push_n_valids(defs.len());
    }
    batched_collector.push_n_valids(defs.len())
}

/// `push_items` for values `defs` whose lowest definition level, `min_def`, makes some item null.
#[inline(never)]
fn push_items_with_nulls(
    defs: &[u16],
    levels: &mut [Nested],
    valid_defs: &[u16],
    min_def: u16,
    batched_collector: &mut BatchedNestedDecoder<'_>,
) -> ParquetResult<()> {
    let (leaf, structs) = levels.split_last_mut().unwrap();
    for (nest, &valid_def) in structs.iter_mut().zip(valid_defs) {
        if min_def >= valid_def {
            nest.push_n_valids(defs.len());
        } else {
            nest.push_validities(defs.iter().map(|&def| def >= valid_def));
        }
    }
    leaf.push_n_valids(defs.len());

    let leaf_def = valid_defs[structs.len()];
    batched_collector.push_validities(defs.iter().map(|&def| def >= leaf_def))
}

/// Collects the leading values of a page of a column without a list, e.g. of structs, in bulk, and
/// returns how many. Each value is a row, with an item in each level.
#[inline(never)]
fn collect_rows_without_list(
    defs: &[u16],
    num_collects: &mut usize,
    batched_collector: &mut BatchedNestedDecoder<'_>,
    nested: &mut [Nested],
    def_levels: &[u16],
) -> ParquetResult<usize> {
    let n = defs.len().min(*num_collects);
    *num_collects -= n;
    push_items(&defs[..n], nested, &def_levels[1..], batched_collector)?;
    Ok(n)
}

/// Checks that each fixed-size list row so far has `width` children. With `is_partial`, the last
/// row may still be missing children.
#[inline(never)]
fn check_fixed_size_list_rows(nested: &[Nested], is_partial: bool) -> ParquetResult<()> {
    for pair in nested.windows(2) {
        let (nest, child_len) = (&pair[0], pair[1].len());
        if let NestedContent::FixedSizeList { width } = nest.content {
            let expected_len = nest.length * width;
            if nest.is_misaligned
                || child_len > expected_len
                || (!is_partial && child_len != expected_len)
            {
                return Err(fixed_size_list_row_len_err(width));
            }
        }
    }
    Ok(())
}

#[cold]
fn fixed_size_list_row_len_err(width: usize) -> ParquetError {
    let note = if width == 0 {
        " (pyarrow writes non-null zero-width fixed-size lists this way)"
    } else {
        ""
    };
    ParquetError::oos(format!(
        "definition and repetition levels encode a fixed-size list row whose number of elements \
         differs from its width ({width}){note}"
    ))
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

impl BatchedNestedDecoder<'_> {
    fn push_valid(&mut self) -> ParquetResult<()> {
        self.push_n_valids(1)
    }

    fn push_n_valids(&mut self, n: usize) -> ParquetResult<()> {
        if self.num_waiting_invalids == 0 {
            self.num_waiting_valids += n;
            return Ok(());
        }
        self.flush();
        self.num_waiting_valids = n;
        Ok(())
    }

    fn push_n_invalids(&mut self, n: usize) -> ParquetResult<()> {
        self.num_waiting_invalids += n;
        Ok(())
    }

    /// Pushes a value per element of `is_valid`, valid where it is `true`.
    fn push_validities(&mut self, is_valid: impl TrustedLen<Item = bool>) -> ParquetResult<()> {
        self.flush();
        self.filter.extend_constant(is_valid.size_hint().0, true);
        self.validity.extend_from_trusted_len_iter(is_valid);
        Ok(())
    }

    // Inlined into each instance of the generic `decode_nested`.
    #[inline]
    fn skip_in_place(&mut self, n: usize) -> ParquetResult<()> {
        self.flush();
        self.filter.extend_constant(n, false);
        self.validity.extend_constant(n, true);
        Ok(())
    }

    fn finalize(mut self) -> ParquetResult<()> {
        self.flush();
        Ok(())
    }

    /// Writes the waiting values out.
    fn flush(&mut self) {
        let num_waiting = self.num_waiting_valids + self.num_waiting_invalids;
        self.filter.extend_constant(num_waiting, true);
        self.validity.extend_constant(self.num_waiting_valids, true);
        self.validity
            .extend_constant(self.num_waiting_invalids, false);

        self.num_waiting_valids = 0;
        self.num_waiting_invalids = 0;
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
    use InitNested as IN;
    use Nested as N;

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
#[derive(Default)]
pub struct NestedState {
    /// The nesteds composing `NestedState`.
    nested: Vec<Nested>,
}

impl NestedState {
    /// Creates a new [`NestedState`].
    fn new(nested: Vec<Nested>) -> Self {
        Self { nested }
    }

    pub fn pop(&mut self) -> Option<(usize, Vec<i64>, Option<BitmapBuilder>)> {
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

/// Collects a single level value with the general per-level logic, for those that
/// `LeafRows::collect_values` does not collect in bulk, e.g. of null and empty rows.
#[inline(always)]
fn collect_value<const HAS_FIXED_SIZE_LIST: bool>(
    def: u16,
    rep: u16,
    num_collects: &mut usize,
    batched_collector: &mut BatchedNestedDecoder<'_>,
    nested: &mut [Nested],
    def_levels: &[u16],
    rep_levels: &[u16],
) -> ParquetResult<()> {
    let max_depth = nested.len();
    *num_collects -= usize::from(rep == 0);

    // The value is defined from the first level whose repetition level admits `rep`, down to the
    // last one that `def` reaches. E.g. [ [ [ 1 ] ] ] is defined at [ ... ], [ [ ... ] ],
    // [ [ [ ... ] ] ] and [ [ [ 1 ] ] ].
    let first_depth = rep_levels[..max_depth].partition_point(|&r| r < rep);
    for depth in first_depth..max_depth {
        if def < def_levels[depth] {
            break;
        }
        let length = nested
            .get(depth + 1)
            .map(|x| x.len() as i64)
            // the last depth is the leaf, which is always increased by 1
            .unwrap_or(1);

        let nest = &mut nested[depth];
        let is_valid = !nest.is_nullable() || def > def_levels[depth];
        if !is_valid {
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
        nest.push(length, is_valid);
        if depth == max_depth - 1 {
            // The leaf is valid, as an invalid one is pushed above.
            batched_collector.push_valid()?;
        } else if HAS_FIXED_SIZE_LIST {
            nest.mark_if_misaligned(length as usize);
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn decode_nested<const HAS_FIXED_SIZE_LIST: bool>(
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
    let leaf_def_level = *def_levels.last().unwrap();
    let leaf_rows = LeafRows::new(nested, def_levels, rep_levels);

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
            let i = match &leaf_rows {
                Some(leaf_rows) => leaf_rows.collect_values::<HAS_FIXED_SIZE_LIST>(
                    current_def_levels,
                    current_rep_levels,
                    num_collects,
                    batched_collector,
                    nested,
                )?,
                // No level is repeated, so each value is a row.
                None => collect_rows_without_list(
                    current_def_levels,
                    num_collects,
                    batched_collector,
                    nested,
                    def_levels,
                )?,
            };

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

/// Return the definition and repetition level iterators for this page.
fn level_iters(page: &DataPage) -> ParquetResult<(HybridRleDecoder<'_>, HybridRleDecoder<'_>)> {
    let split = split_buffer(page)?;
    let def = split.def;
    let rep = split.rep;

    let max_def_level = page.descriptor.max_def_level;
    let max_rep_level = page.descriptor.max_rep_level;

    let def_iter = HybridRleDecoder::new(def, get_bit_width(max_def_level), page.num_values());
    let rep_iter = HybridRleDecoder::new(rep, get_bit_width(max_rep_level), page.num_values());

    Ok((def_iter, rep_iter))
}

impl<D: utils::Decoder> PageDecoder<D> {
    pub fn collect_nested(
        mut self,
        filter: Option<Filter>,
    ) -> ParquetResult<(NestedState, Vec<D::Output>, Bitmap)> {
        let init = self.init_nested.as_mut().unwrap();

        // @TODO: We should probably count the filter so that we don't overallocate
        let mut target = self.decoder.with_capacity(self.iter.total_num_values());
        // @TODO: Self capacity
        let mut nested_state = init_nested(init, 0);

        if let Some(dict) = self.dict.as_ref() {
            self.decoder.apply_dictionary(&mut target, dict)?;
        }

        // Amortize the allocations.
        let (def_levels, rep_levels) = nested_state.levels();

        // The decoder uses Arrow-derived levels. Check their maxima against
        // the Parquet descriptor before decoding.
        let max_def_level = def_levels.last().copied().unwrap_or(0);
        let max_rep_level = rep_levels.last().copied().unwrap_or(0);
        let descriptor = self.iter.descriptor();
        if i32::from(max_def_level) != i32::from(descriptor.max_def_level)
            || i32::from(max_rep_level) != i32::from(descriptor.max_rep_level)
        {
            return Err(ParquetError::oos(format!(
                "Parquet nesting mismatch: the inferred nesting does not match the file's \
                 definition/repetition levels (inferred {max_def_level}/{max_rep_level}, file \
                 {}/{})",
                descriptor.max_def_level, descriptor.max_rep_level,
            )));
        }

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
            Some(Filter::Predicate(_)) => todo!(),
        };

        let mut top_level_filter = top_level_filter.iter();

        // Only columns with a fixed-size list pay for checking its rows.
        let has_fixed_size_list = nested_state
            .nested
            .iter()
            .any(|n| matches!(n.content, NestedContent::FixedSizeList { .. }));
        let decode_page = if has_fixed_size_list {
            decode_nested::<true>
        } else {
            decode_nested::<false>
        };

        let mut chunks = Vec::new();
        while let Some(page) = self.iter.next() {
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

            decode_page(
                &current_def_levels,
                &current_rep_levels,
                &mut batched_collector,
                &mut nested_state.nested,
                &mut decode_state,
                &mut top_level_filter,
                &def_levels,
                &rep_levels,
            )?;
            // A row may continue on the next page, so the last one is only checked for excess
            // children.
            if has_fixed_size_list {
                check_fixed_size_list_rows(&nested_state.nested, true)?;
            }

            batched_collector.finalize()?;

            let leaf_validity = leaf_validity.freeze();
            let leaf_filter = leaf_filter.freeze();

            let state = utils::State::new_nested(
                &self.decoder,
                &page,
                self.dict.as_ref(),
                Some(leaf_validity),
            )?;
            state.decode(
                &mut self.decoder,
                &mut target,
                Some(Filter::Mask(leaf_filter)),
                &mut chunks,
            )?;

            self.iter.reuse_page_buffer(page);
        }

        // Rows don't span column chunks, so the last row must be complete here.
        if has_fixed_size_list {
            check_fixed_size_list_rows(&nested_state.nested, false)?;
        }

        // we pop the primitive off here.
        debug_assert!(matches!(
            nested_state.nested.last().unwrap().content,
            NestedContent::Primitive
        ));
        _ = nested_state.pop().unwrap();

        if target.len() > 0 || chunks.is_empty() {
            chunks.push(self.decoder.finalize(self.dtype, self.dict, target)?);
        }

        Ok((nested_state, chunks, Bitmap::new()))
    }
}
