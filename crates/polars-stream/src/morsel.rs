use std::sync::OnceLock;

use polars_core::frame::DataFrame;

use crate::async_primitives::wait_group::WaitToken;

static IDEAL_MORSEL_SIZE: OnceLock<usize> = OnceLock::new();

pub fn get_ideal_morsel_size() -> usize {
    *IDEAL_MORSEL_SIZE.get_or_init(|| {
        std::env::var("POLARS_IDEAL_MORSEL_SIZE")
            .map(|m| m.parse().unwrap())
            .unwrap_or(100_000)
    })
}

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

    pub fn to_u64(self) -> u64 {
        self.0
    }
}

pub struct Morsel {
    /// The data contained in this morsel.
    df: DataFrame,

    /// The sequence number of this morsel. May only stay equal or increase
    /// within a pipeline.
    seq: MorselSeq,

    /// Used to notify someone when this morsel is consumed, to provide backpressure.
    consume_token: Option<WaitToken>,
}

impl Morsel {
    pub fn new(df: DataFrame, seq: MorselSeq) -> Self {
        Self {
            df,
            seq,
            consume_token: None,
        }
    }

    #[allow(unused)]
    pub fn into_inner(self) -> (DataFrame, MorselSeq, Option<WaitToken>) {
        (self.df, self.seq, self.consume_token)
    }

    pub fn df(&self) -> &DataFrame {
        &self.df
    }

    pub fn seq(&self) -> MorselSeq {
        self.seq
    }

    pub fn set_seq(&mut self, seq: MorselSeq) {
        self.seq = seq;
    }

    #[allow(unused)]
    pub fn map<F: FnOnce(DataFrame) -> DataFrame>(mut self, f: F) -> Self {
        self.df = f(self.df);
        self
    }

    pub fn try_map<E, F: FnOnce(DataFrame) -> Result<DataFrame, E>>(
        mut self,
        f: F,
    ) -> Result<Self, E> {
        self.df = f(self.df)?;
        Ok(self)
    }

    pub fn set_consume_token(&mut self, token: WaitToken) {
        self.consume_token = Some(token);
    }

    pub fn take_consume_token(&mut self) -> Option<WaitToken> {
        self.consume_token.take()
    }
}
