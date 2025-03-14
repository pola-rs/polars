use crate::async_primitives::connector;
use crate::async_primitives::morsel_linearizer::{MorselInserter, MorselLinearizer};
use crate::async_primitives::wait_group::WaitGroup;
use crate::morsel::Morsel;

/// This is just a generic enum that can receive from either single or multiple senders.
pub enum FileReaderOutputRecv {
    Connector(connector::Receiver<Morsel>),
    Linearized(MorselLinearizer),
}

/// Wraps either a connector or a linearizer inserter. Also attaches / waits for wait tokens
/// automatically.
pub enum FileReaderOutputSend {
    Connector(connector::Sender<Morsel>, WaitGroup),
    Linearized(MorselInserter, WaitGroup),
}

impl FileReaderOutputRecv {
    pub async fn recv(&mut self) -> Result<Morsel, ()> {
        use FileReaderOutputRecv::*;
        match self {
            Connector(v) => v.recv().await,
            Linearized(v) => v.get().await.ok_or(()),
        }
    }
}

impl FileReaderOutputSend {
    pub fn new_serial() -> (FileReaderOutputSend, FileReaderOutputRecv) {
        let (tx, rx) = connector::connector();
        (
            FileReaderOutputSend::Connector(tx, WaitGroup::default()),
            FileReaderOutputRecv::Connector(rx),
        )
    }

    pub fn new_parallel(num_pipelines: usize) -> (Vec<FileReaderOutputSend>, FileReaderOutputRecv) {
        let (lin, inserters) = MorselLinearizer::new(num_pipelines, 1);
        (
            inserters
                .into_iter()
                .map(|tx| FileReaderOutputSend::Linearized(tx, WaitGroup::default()))
                .collect(),
            FileReaderOutputRecv::Linearized(lin),
        )
    }

    pub async fn send_morsel(&mut self, morsel: Morsel) -> Result<(), Morsel> {
        use FileReaderOutputSend::*;

        // We order to wait first, then send. This is intended to allow the producer create the
        // next morsel while waiting for the current one to be consumed.

        match self {
            Connector(tx, wait_group) => {
                wait_group.wait().await;
                tx.send(morsel).await
            },
            Linearized(tx, wait_group) => {
                wait_group.wait().await;
                tx.insert(morsel).await
            },
        }
    }
}
