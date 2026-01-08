use crate::async_primitives::connector;

pub fn channel<T>() -> (Sender<T>, Receiver<T>) {
    let (tx, rx) = connector::connector();
    (Sender { inner: tx }, Receiver { inner: rx })
}

pub struct Sender<T> {
    inner: connector::Sender<T>,
}

impl<T: Send> Sender<T> {
    pub fn send(mut self, value: T) -> Result<(), connector::SendError<T>> {
        self.inner.try_send(value)
    }
}

pub struct Receiver<T> {
    inner: connector::Receiver<T>,
}

impl<T: Send> Receiver<T> {
    pub async fn recv(mut self) -> Result<T, ()> {
        self.inner.recv().await
    }
}
