use std::cmp::Reverse;
use std::future::Future;
use std::sync::Arc;

use polars_async::primitives::linearizer::{Inserter, Linearizer};
use polars_async::primitives::wait_group::WaitToken;
use polars_core::frame::DataFrame;
use polars_ooc::{PinnedFrameMut, PinnedRef, SpillFrame};
use polars_utils::priority::Priority;
use polars_utils::relaxed_cell::RelaxedCell;

pub fn get_ideal_morsel_size() -> usize {
    polars_config::config().ideal_morsel_size() as usize
}

/// A token indicating the order of morsels in a stream.
///
/// The sequence tokens going through a pipe are monotonely non-decreasing and are allowed to be
/// discontinuous. Consequently, `1 -> 1 -> 2` and `1 -> 3 -> 5` are valid streams of sequence
/// tokens.
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Debug, Default)]
pub struct MorselSeq(u64);

impl MorselSeq {
    // TODO: use least significant bit to indicate 'last morsel with this
    // sequence number'.
    pub fn new(seq: u64) -> Self {
        Self(seq.checked_mul(2).unwrap())
    }

    // The morsel sequence id which comes after this morsel.
    pub fn successor(self) -> Self {
        // We increment by two because in the future we want to use the least
        // significant bit to indicate the final morsel with that sequence id.
        Self(self.0.checked_add(2).unwrap())
    }

    // Ensures this morsel sequence comes after the offset.
    pub fn offset_by(self, offset: Self) -> Self {
        Self(self.0 + offset.0)
    }

    pub fn offset_by_u64(self, offset: u64) -> Self {
        Self(self.0 + 2 * offset)
    }

    pub fn to_u64(self) -> u64 {
        self.0
    }
}

/// A token indicating which source this morsel originated from, and a way to
/// pass information/signals to it. Currently it's only used to request a source
/// to stop with passing new morsels this execution phase.
#[derive(Clone, Debug)]
pub struct SourceToken {
    stop: Arc<RelaxedCell<bool>>,
}

impl Default for SourceToken {
    fn default() -> Self {
        Self::new()
    }
}

impl SourceToken {
    pub fn new() -> Self {
        Self {
            stop: Arc::default(),
        }
    }

    pub fn stop(&self) {
        self.stop.store(true);
    }

    pub fn stop_requested(&self) -> bool {
        self.stop.load()
    }
}

#[derive(Debug)]
pub struct Morsel {
    /// The data contained in this morsel.
    sf: SpillFrame,

    /// The sequence number of this morsel. May only stay equal or increase
    /// within a pipeline.
    seq: MorselSeq,

    /// A token that indicates which source this morsel originates from.
    source_token: SourceToken,

    /// Used to notify someone when this morsel is consumed, to provide backpressure.
    consume_token: Option<WaitToken>,
}

impl Morsel {
    pub fn new(sf: SpillFrame, seq: MorselSeq, source_token: SourceToken) -> Self {
        Self {
            sf,
            seq,
            source_token,
            consume_token: None,
        }
    }

    pub fn new_unregistered(df: DataFrame, seq: MorselSeq, source_token: SourceToken) -> Self {
        Self {
            sf: SpillFrame::new_unregistered(df),
            seq,
            source_token,
            consume_token: None,
        }
    }

    #[expect(unused)]
    pub fn into_inner(self) -> (SpillFrame, MorselSeq, SourceToken, Option<WaitToken>) {
        (self.sf, self.seq, self.source_token, self.consume_token)
    }

    #[inline(always)]
    pub fn height(&self) -> usize {
        self.sf.height()
    }

    #[inline(always)]
    pub fn sf(&self) -> &SpillFrame {
        &self.sf
    }

    #[inline(always)]
    pub fn into_sf(self) -> SpillFrame {
        self.sf
    }

    #[inline(always)]
    pub async fn into_df2(self) -> DataFrame {
        self.sf.into_df().await
    }

    #[inline(always)]
    pub fn into_df_blocking(self) -> DataFrame {
        self.sf.into_df_blocking()
    }
    
    #[inline(always)]
    pub async fn get_df(&self) -> PinnedRef<'_, DataFrame> {
        self.sf.get().await
    }
    
    #[inline(always)]
    pub fn get_df_blocking(&self) -> PinnedRef<'_, DataFrame> {
        self.sf.get_blocking()
    }

    #[inline(always)]
    pub async fn get_df_mut(&mut self) -> PinnedFrameMut<'_> {
        self.sf.get_mut().await
    }
    
    #[inline(always)]
    pub fn get_df_mut_blocking(&mut self) -> PinnedFrameMut<'_> {
        self.sf.get_mut_blocking()
    }

    /*
    pub fn into_df(self) -> DataFrame {
        self.df
    }

    pub fn df(&self) -> &DataFrame {
        &self.df
    }

    pub fn df_mut(&mut self) -> &mut DataFrame {
        &mut self.df
    }
    */

    pub fn seq(&self) -> MorselSeq {
        self.seq
    }

    pub fn set_seq(&mut self, seq: MorselSeq) {
        self.seq = seq;
    }
    
    pub fn set_df(&mut self, df: DataFrame) {
        let old_registry = self.sf.unregister();
        let sf = SpillFrame::new_unregistered(df);
        self.sf = sf;
        if let Some((ctx, param)) = old_registry {
            ctx.register(&self.sf, param);
        }
    }

    #[expect(unused)]
    pub async fn map<F: FnOnce(DataFrame) -> DataFrame>(self, f: F) -> Self {
        let Self { mut sf, seq, source_token, consume_token } = self;
        let old_registry = sf.unregister();
        let df = f(sf.into_df().await);
        let sf = SpillFrame::new_unregistered(df);
        if let Some((ctx, param)) = old_registry {
            ctx.register(&sf, param);
        }
        Self { sf, seq, source_token, consume_token }
    }

    pub async fn try_map<E, F: FnOnce(DataFrame) -> Result<DataFrame, E>>(
        self,
        f: F,
    ) -> Result<Self, E> {
        let Self { mut sf, seq, source_token, consume_token } = self;
        let old_registry = sf.unregister();
        let df = f(sf.into_df().await)?;
        let sf = SpillFrame::new_unregistered(df);
        if let Some((ctx, param)) = old_registry {
            ctx.register(&sf, param);
        }
        Ok(Self { sf, seq, source_token, consume_token })
    }

    pub async fn async_try_map<E, M, F>(self, f: M) -> Result<Self, E>
    where
        M: FnOnce(DataFrame) -> F,
        F: Future<Output = Result<DataFrame, E>>,
    {
        let Self { mut sf, seq, source_token, consume_token } = self;
        let old_registry = sf.unregister();
        let df = f(sf.into_df().await).await?;
        let sf = SpillFrame::new_unregistered(df);
        if let Some((ctx, param)) = old_registry {
            ctx.register(&sf, param);
        }
        Ok(Self { sf, seq, source_token, consume_token })
    }

    pub fn set_consume_token(&mut self, token: WaitToken) {
        self.consume_token = Some(token);
    }

    pub fn take_consume_token(&mut self) -> Option<WaitToken> {
        self.consume_token.take()
    }

    pub fn source_token(&self) -> &SourceToken {
        &self.source_token
    }

    pub fn replace_source_token(&mut self, new_token: SourceToken) -> SourceToken {
        core::mem::replace(&mut self.source_token, new_token)
    }
}

pub struct MorselLinearizer(Linearizer<Priority<Reverse<MorselSeq>, Morsel>>);
pub struct MorselInserter(Inserter<Priority<Reverse<MorselSeq>, Morsel>>);

impl MorselLinearizer {
    pub fn new(num_inserters: usize, buffer_size: usize) -> (Self, Vec<MorselInserter>) {
        let (lin, inserters) = Linearizer::new(num_inserters, buffer_size);

        (
            MorselLinearizer(lin),
            inserters.into_iter().map(MorselInserter).collect(),
        )
    }

    pub async fn get(&mut self) -> Option<Morsel> {
        self.0.get().await.map(|x| x.1)
    }
}

impl MorselInserter {
    pub async fn insert(&mut self, morsel: Morsel) -> Result<(), Morsel> {
        self.0
            .insert(Priority(Reverse(morsel.seq()), morsel))
            .await
            .map_err(|Priority(_, v)| v)
    }
}
