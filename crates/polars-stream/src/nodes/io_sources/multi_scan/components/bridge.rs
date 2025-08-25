use crate::async_primitives::morsel_linearizer::MorselLinearizer;
use crate::morsel::Morsel;
use crate::nodes::io_sources::multi_scan::reader_interface::output::FileReaderOutputRecv;

#[derive(Copy, Clone)]
pub enum BridgeState {
    NotYetStarted,
    Running,
    Stopped(StopReason),
}

#[derive(Copy, Clone)]
pub enum StopReason {
    /// Disconnected from the reader side. The reader pipeline handle should be joined on in this case to
    /// determine if the readers disconnected due to an error.
    ReadersDisconnected,
    /// Disconnected from the multi scan ComputeNode.
    ComputeNodeDisconnected,
}

/// Port for the reader side.
///
/// Note: `first_morsel` is a residual from post-op initialization.
pub enum BridgeRecvPort {
    Direct {
        rx: FileReaderOutputRecv,
        first_morsel: Option<Morsel>,
    },
    /// Parallel post-apply ops will connect through this.
    Linearized { rx: MorselLinearizer },
}

impl BridgeRecvPort {
    pub async fn recv(&mut self) -> Result<Morsel, ()> {
        use BridgeRecvPort::*;
        match self {
            Direct { rx, first_morsel } => {
                if let Some(v) = first_morsel.take() {
                    Ok(v)
                } else {
                    rx.recv().await
                }
            },
            Linearized { rx } => rx.get().await.ok_or(()),
        }
    }
}
