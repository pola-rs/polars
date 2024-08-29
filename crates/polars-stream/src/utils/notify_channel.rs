use tokio::sync::mpsc::error::TrySendError;
use tokio::sync::mpsc::{channel, Receiver, Sender};

/// Receiver that calls `notify()` before `recv()`
pub struct NotifyReceiver<T> {
    receiver: Receiver<T>,
    /// We use a channel for notify because it lets the sender know when the receiver has been
    /// dropped.
    notify: Sender<()>,
}

impl<T: Send> NotifyReceiver<T> {
    pub async fn recv(&mut self) -> Option<T> {
        match self.notify.try_send(()) {
            Err(TrySendError::Closed(_)) => None,
            Ok(_) => self.receiver.recv().await,
            v @ Err(TrySendError::Full(_)) => {
                v.unwrap();
                unreachable!();
            },
        }
    }
}

/// The notify allows us to make the producer only produce values when requested. Otherwise it would
/// produce a new value as soon as the previous value was consumed (as there would be channel
/// capacity).
pub fn notify_channel<T>() -> (Sender<T>, Receiver<()>, NotifyReceiver<T>) {
    let (tx, rx) = channel::<T>(1);
    let (notify_tx, notify_rx) = channel(1);

    (
        tx,
        notify_rx,
        NotifyReceiver {
            receiver: rx,
            notify: notify_tx,
        },
    )
}

mod tests {

    #[test]
    fn test_notify_channel() {
        use futures::FutureExt;

        use super::notify_channel;
        let (tx, mut notify, mut rx) = notify_channel();
        assert!(notify.recv().now_or_never().is_none());
        assert!(rx.recv().now_or_never().is_none());
        assert_eq!(notify.recv().now_or_never().unwrap(), Some(()));
        assert!(tx.try_send(()).is_ok());
        assert!(rx.recv().now_or_never().is_some());
    }
}
