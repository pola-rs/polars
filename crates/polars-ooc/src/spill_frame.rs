use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use polars_core::frame::DataFrame;

use crate::spill_context::ParameterFreeSpillContext;
use crate::{PinnedMut, PinnedRef, SpillToken, Spillable, memory_manager};

impl Spillable for DataFrame {}

pub struct SpillFrame {
    token: SpillToken<DataFrame>,
    height: usize,
}

impl AsRef<SpillToken<DataFrame>> for SpillFrame {
    fn as_ref(&self) -> &SpillToken<DataFrame> {
        &self.token
    }
}

impl SpillFrame {
    pub fn new_unregistered(df: DataFrame) -> Self {
        let height = df.height();
        let token = SpillToken::new(df);
        Self { token, height }
    }

    pub async fn new<C: ParameterFreeSpillContext>(df: DataFrame, ctx: &C) -> Self {
        let slf = Self::new_unregistered(df);
        ctx.register(&slf);
        memory_manager().spill().await;
        slf
    }

    pub fn new_blocking<C: ParameterFreeSpillContext>(df: DataFrame, ctx: &C) -> Self {
        let slf = Self::new_unregistered(df);
        ctx.register(&slf);
        memory_manager().spill_blocking();
        slf
    }

    /// The height of the contained DataFrame. Does not need to unspill DataFrame.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get a reference to the underlying DataFrame, returning None if it was spilled.
    pub fn try_get(&self) -> Option<PinnedRef<'_, DataFrame>> {
        self.token.try_get()
    }

    /// Get a reference to the underlying DataFrame, unspilling it if it
    /// was spilled.
    pub async fn get(&self) -> PinnedRef<'_, DataFrame> {
        self.token.get().await
    }

    /// Blocking version of get.
    pub fn get_blocking(&self) -> PinnedRef<'_, DataFrame> {
        self.token.get_blocking()
    }

    /// Get a mutable reference to the underlying DataFrame, unspilling it if it
    /// was spilled.
    pub async fn get_mut(&mut self) -> PinnedFrameMut<'_> {
        PinnedFrameMut {
            inner: self.token.get_mut().await,
            height: &mut self.height,
        }
    }

    /// Blocking version of get_mut.
    pub fn get_mut_blocking(&mut self) -> PinnedFrameMut<'_> {
        PinnedFrameMut {
            inner: self.token.get_mut_blocking(),
            height: &mut self.height,
        }
    }

    /// Consumes this SpillFrame, unspilling it if it were spilled.
    pub async fn into_df(self) -> DataFrame {
        self.token.into_inner().await
    }

    /// Blocking version of into_df.
    pub fn into_df_blocking(self) -> DataFrame {
        self.token.into_inner_blocking()
    }
}

impl Debug for SpillFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("SpillFrame");
        match self.token.try_get() {
            Some(df) => s.field("df", &*df),
            None => s.field("df", &"spilled"),
        };
        s.finish()
    }
}

pub struct PinnedFrameMut<'a> {
    height: &'a mut usize,
    inner: PinnedMut<'a, DataFrame>,
}

impl<'a> Deref for PinnedFrameMut<'a> {
    type Target = DataFrame;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> DerefMut for PinnedFrameMut<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a> Drop for PinnedFrameMut<'a> {
    fn drop(&mut self) {
        *self.height = self.inner.height();
    }
}
