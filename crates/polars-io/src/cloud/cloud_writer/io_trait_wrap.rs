use std::pin::Pin;
use std::task::{Poll, ready};

use futures::FutureExt;

use crate::cloud::cloud_writer::CloudWriter;
use crate::pl_async;
use crate::utils::file::WriteableTrait;

/// Wrapper on [`CloudWriter`] that implements [`std::io::Write`] and [`tokio::io::AsyncWrite`].
pub struct CloudWriterIoTraitWrap {
    state: WriterState,
}

enum WriterState {
    Ready(Box<CloudWriter>),
    Poll(
        Pin<Box<dyn Future<Output = std::io::Result<WriterState>> + Send + 'static>>,
        PollOperation,
    ),
    Finished,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PollOperation {
    // (slice_addr, slice_len)
    Write { slice_ptr: usize, written: usize },
    Flush,
    Shutdown,
}

struct FinishActivePoll<'a>(Pin<&'a mut WriterState>);

impl<'a> Future for FinishActivePoll<'a> {
    type Output = std::io::Result<Option<PollOperation>>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Self::Output> {
        match &mut *self.0 {
            WriterState::Poll(fut, _) => match fut.poll_unpin(cx) {
                Poll::Ready(Ok(new_state)) => {
                    debug_assert!(!matches!(&new_state, WriterState::Poll(..)));

                    let WriterState::Poll(_, operation) =
                        std::mem::replace(&mut *self.0, new_state)
                    else {
                        unreachable!()
                    };

                    Poll::Ready(Ok(Some(operation)))
                },
                Poll::Ready(Err(e)) => {
                    *self.0 = WriterState::Finished;
                    Poll::Ready(Err(e))
                },
                Poll::Pending => Poll::Pending,
            },

            WriterState::Ready(_) | WriterState::Finished => Poll::Ready(Ok(None)),
        }
    }
}

impl CloudWriterIoTraitWrap {
    fn finish_active_poll(&mut self) -> FinishActivePoll<'_> {
        FinishActivePoll(Pin::new(&mut self.state))
    }

    fn take_writer_from_ready_state(&mut self) -> Option<Box<CloudWriter>> {
        if !matches!(&self.state, WriterState::Ready(_)) {
            return None;
        }

        let WriterState::Ready(writer) = std::mem::replace(&mut self.state, WriterState::Finished)
        else {
            unreachable!()
        };

        Some(writer)
    }

    fn get_writer_mut_from_ready_state(&mut self) -> Option<&mut CloudWriter> {
        if let WriterState::Ready(writer) = &mut self.state {
            Some(writer.as_mut())
        } else {
            None
        }
    }

    pub async fn into_cloud_writer(mut self) -> std::io::Result<CloudWriter> {
        self.finish_active_poll().await?;

        match self.state {
            WriterState::Ready(writer) => Ok(*writer),
            WriterState::Poll(..) => unreachable!(),
            WriterState::Finished => panic!(),
        }
    }

    pub fn as_cloud_writer(&mut self) -> std::io::Result<&mut CloudWriter> {
        if !matches!(self.state, WriterState::Ready(_)) {
            match &mut self.state {
                WriterState::Ready(_) => unreachable!(),
                WriterState::Poll(..) => {
                    pl_async::get_runtime().block_in_place_on(self.finish_active_poll())?
                },
                WriterState::Finished => panic!(),
            };
        }

        let WriterState::Ready(writer) = &mut self.state else {
            panic!()
        };

        Ok(writer)
    }
}

impl From<CloudWriter> for CloudWriterIoTraitWrap {
    fn from(writer: CloudWriter) -> Self {
        Self {
            state: WriterState::Ready(Box::new(writer)),
        }
    }
}

impl std::io::Write for CloudWriterIoTraitWrap {
    fn write(&mut self, mut buf: &[u8]) -> std::io::Result<usize> {
        let total_buf_len = buf.len();
        let buf: &mut &[u8] = &mut buf;

        if let Some(writer) = self.get_writer_mut_from_ready_state() {
            let should_poll = writer.fill_buffer_from_slice(buf);

            if !should_poll {
                assert!(buf.is_empty());
                return Ok(total_buf_len);
            }
        }

        pl_async::get_runtime().block_in_place_on(async {
            self.finish_active_poll().await?;

            let writer = self.get_writer_mut_from_ready_state().unwrap();

            loop {
                writer.flush_complete_chunk().await?;

                if !writer.fill_buffer_from_slice(buf) {
                    break;
                }
            }

            assert!(buf.is_empty());

            Ok(total_buf_len)
        })
    }

    fn flush(&mut self) -> std::io::Result<()> {
        if self
            .get_writer_mut_from_ready_state()
            .is_some_and(|w| !w.has_buffered_bytes())
        {
            return Ok(());
        }

        pl_async::get_runtime().block_in_place_on(async {
            self.finish_active_poll().await?;

            self.get_writer_mut_from_ready_state()
                .unwrap()
                .flush()
                .await?;

            Ok(())
        })
    }
}

impl WriteableTrait for CloudWriterIoTraitWrap {
    fn close(&mut self) -> std::io::Result<()> {
        pl_async::get_runtime().block_in_place_on(async {
            self.finish_active_poll().await?;

            if let Some(mut writer) = self.take_writer_from_ready_state() {
                writer.flush().await?;
                writer.finish().await?;
            }

            Ok(())
        })
    }

    fn sync_all(&self) -> std::io::Result<()> {
        Ok(())
    }

    fn sync_data(&self) -> std::io::Result<()> {
        Ok(())
    }
}

impl tokio::io::AsyncWrite for CloudWriterIoTraitWrap {
    fn poll_write(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
        buf: &[u8],
    ) -> std::task::Poll<std::io::Result<usize>> {
        loop {
            let offset = match ready!(self.finish_active_poll().poll_unpin(cx))? {
                Some(PollOperation::Write { slice_ptr, written })
                    if slice_ptr == buf.as_ptr() as usize =>
                {
                    written
                },
                Some(_) => panic!(),
                None => 0,
            };

            let writer = self.get_writer_mut_from_ready_state().unwrap();

            let offset_buf: &mut &[u8] = &mut &buf[offset..];

            let should_poll_flush = writer.fill_buffer_from_slice(offset_buf);

            if !should_poll_flush {
                assert!(offset_buf.is_empty());
                return Poll::Ready(Ok(buf.len()));
            };

            let new_offset = buf.len() - offset_buf.len();

            let mut writer = self.take_writer_from_ready_state().unwrap();

            let fut = async move {
                writer.flush_complete_chunk().await?;
                Ok(WriterState::Ready(writer))
            };

            self.state = WriterState::Poll(
                Box::pin(fut),
                PollOperation::Write {
                    slice_ptr: buf.as_ptr() as usize,
                    written: new_offset,
                },
            );
        }
    }

    fn poll_flush(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        loop {
            if let Some(operation) = ready!(self.finish_active_poll().poll_unpin(cx))? {
                debug_assert_eq!(operation, PollOperation::Flush);

                if operation == PollOperation::Flush {
                    return Poll::Ready(Ok(()));
                }
            }

            let mut writer = self.take_writer_from_ready_state().unwrap();

            let fut = async move {
                writer.flush().await?;
                Ok(WriterState::Ready(writer))
            };

            self.state = WriterState::Poll(Box::pin(fut), PollOperation::Flush);
        }
    }

    fn poll_shutdown(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<std::io::Result<()>> {
        loop {
            if let Some(operation) = ready!(self.finish_active_poll().poll_unpin(cx))? {
                debug_assert_eq!(operation, PollOperation::Shutdown);

                if operation == PollOperation::Shutdown {
                    return Poll::Ready(Ok(()));
                }
            }

            let Some(mut writer) = self.take_writer_from_ready_state() else {
                return Poll::Ready(Ok(()));
            };

            let fut = async move {
                writer.finish().await?;
                Ok(WriterState::Finished)
            };

            self.state = WriterState::Poll(Box::pin(fut), PollOperation::Shutdown);
        }
    }
}
