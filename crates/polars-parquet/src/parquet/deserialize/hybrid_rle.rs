use crate::error::Error;

use crate::encoding::hybrid_rle::{self, BitmapIter};

/// The decoding state of the hybrid-RLE decoder with a maximum definition level of 1
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HybridEncoded<'a> {
    /// a bitmap
    Bitmap(&'a [u8], usize),
    /// A repeated item. The first attribute corresponds to whether the value is set
    /// the second attribute corresponds to the number of repetitions.
    Repeated(bool, usize),
}

impl<'a> HybridEncoded<'a> {
    /// Returns the length of the run in number of items
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            HybridEncoded::Bitmap(_, length) => *length,
            HybridEncoded::Repeated(_, length) => *length,
        }
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

pub trait HybridRleRunsIterator<'a>: Iterator<Item = Result<HybridEncoded<'a>, Error>> {
    /// Number of elements remaining. This may not be the items of the iterator - an item
    /// of the iterator may contain more than one element.
    fn number_of_elements(&self) -> usize;
}

/// An iterator of [`HybridEncoded`], adapter over [`hybrid_rle::HybridEncoded`].
#[derive(Debug, Clone)]
pub struct HybridRleIter<'a, I>
where
    I: Iterator<Item = Result<hybrid_rle::HybridEncoded<'a>, Error>>,
{
    iter: I,
    length: usize,
    consumed: usize,
}

impl<'a, I> HybridRleIter<'a, I>
where
    I: Iterator<Item = Result<hybrid_rle::HybridEncoded<'a>, Error>>,
{
    /// Returns a new [`HybridRleIter`]
    #[inline]
    pub fn new(iter: I, length: usize) -> Self {
        Self {
            iter,
            length,
            consumed: 0,
        }
    }

    /// the number of elements in the iterator. Note that this _is not_ the number of runs.
    #[inline]
    pub fn len(&self) -> usize {
        self.length - self.consumed
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<'a, I> HybridRleRunsIterator<'a> for HybridRleIter<'a, I>
where
    I: Iterator<Item = Result<hybrid_rle::HybridEncoded<'a>, Error>>,
{
    fn number_of_elements(&self) -> usize {
        self.len()
    }
}

impl<'a, I> Iterator for HybridRleIter<'a, I>
where
    I: Iterator<Item = Result<hybrid_rle::HybridEncoded<'a>, Error>>,
{
    type Item = Result<HybridEncoded<'a>, Error>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if self.consumed == self.length {
            return None;
        };
        let run = self.iter.next()?;

        Some(run.map(|run| match run {
            hybrid_rle::HybridEncoded::Bitpacked(pack) => {
                // a pack has at most `pack.len() * 8` bits
                let pack_size = pack.len() * 8;

                let additional = pack_size.min(self.len());

                self.consumed += additional;
                HybridEncoded::Bitmap(pack, additional)
            }
            hybrid_rle::HybridEncoded::Rle(value, length) => {
                let is_set = value[0] == 1;

                let additional = length.min(self.len());

                self.consumed += additional;
                HybridEncoded::Repeated(is_set, additional)
            }
        }))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

/// Type definition for a [`HybridRleIter`] using [`hybrid_rle::Decoder`].
pub type HybridDecoderBitmapIter<'a> = HybridRleIter<'a, hybrid_rle::Decoder<'a>>;

#[derive(Debug)]
enum HybridBooleanState<'a> {
    /// a bitmap
    Bitmap(BitmapIter<'a>),
    /// A repeated item. The first attribute corresponds to whether the value is set
    /// the second attribute corresponds to the number of repetitions.
    Repeated(bool, usize),
}

/// An iterator adapter that maps an iterator of [`HybridEncoded`] into an iterator
/// over [`bool`].
#[derive(Debug)]
pub struct HybridRleBooleanIter<'a, I>
where
    I: Iterator<Item = Result<HybridEncoded<'a>, Error>>,
{
    iter: I,
    current_run: Option<HybridBooleanState<'a>>,
}

impl<'a, I> HybridRleBooleanIter<'a, I>
where
    I: Iterator<Item = Result<HybridEncoded<'a>, Error>>,
{
    pub fn new(iter: I) -> Self {
        Self {
            iter,
            current_run: None,
        }
    }
}

impl<'a, I> Iterator for HybridRleBooleanIter<'a, I>
where
    I: HybridRleRunsIterator<'a>,
{
    type Item = Result<bool, Error>;

    #[inline]
    fn next(&mut self) -> Option<Self::Item> {
        if let Some(run) = &mut self.current_run {
            match run {
                HybridBooleanState::Bitmap(bitmap) => bitmap.next().map(Ok),
                HybridBooleanState::Repeated(value, remaining) => if *remaining == 0 {
                    None
                } else {
                    *remaining -= 1;
                    Some(*value)
                }
                .map(Ok),
            }
        } else if let Some(run) = self.iter.next() {
            let run = run.map(|run| match run {
                HybridEncoded::Bitmap(bitmap, length) => {
                    HybridBooleanState::Bitmap(BitmapIter::new(bitmap, 0, length))
                }
                HybridEncoded::Repeated(value, length) => {
                    HybridBooleanState::Repeated(value, length)
                }
            });
            match run {
                Ok(run) => {
                    self.current_run = Some(run);
                    self.next()
                }
                Err(e) => Some(Err(e)),
            }
        } else {
            None
        }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        let exact = self.iter.number_of_elements();
        (exact, Some(exact))
    }
}

/// Type definition for a [`HybridRleBooleanIter`] using [`hybrid_rle::Decoder`].
pub type HybridRleDecoderIter<'a> = HybridRleBooleanIter<'a, HybridDecoderBitmapIter<'a>>;
