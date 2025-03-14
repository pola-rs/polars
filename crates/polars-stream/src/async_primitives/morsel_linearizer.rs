//! Saves us from typing `Priority<Reverse<...` everywhere.
use std::cmp::Reverse;

use polars_utils::priority::Priority;

use super::linearizer::{Inserter, Linearizer};
use crate::morsel::{Morsel, MorselSeq};

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
