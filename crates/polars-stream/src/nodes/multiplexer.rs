use std::collections::VecDeque;

use tokio::sync::mpsc::{unbounded_channel, UnboundedSender};

use super::compute_node_prelude::*;
use crate::morsel::SourceToken;

// TODO: replace this with an out-of-core buffering solution.
enum BufferedStream {
    Open(VecDeque<Morsel>),
    Closed,
}

impl BufferedStream {
    fn new() -> Self {
        Self::Open(VecDeque::new())
    }
}

pub struct MultiplexerNode {
    buffers: Vec<BufferedStream>,
}

impl MultiplexerNode {
    pub fn new() -> Self {
        Self {
            buffers: Vec::default(),
        }
    }
}

impl ComputeNode for MultiplexerNode {
    fn name(&self) -> &str {
        "multiplexer"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) -> PolarsResult<()> {
        assert!(recv.len() == 1 && !send.is_empty());

        // Initialize buffered streams, and mark those for which the receiver
        // is no longer interested as closed.
        self.buffers.resize_with(send.len(), BufferedStream::new);
        for (s, b) in send.iter().zip(&mut self.buffers) {
            if *s == PortState::Done {
                *b = BufferedStream::Closed;
            }
        }

        // Check if either the input is done, or all outputs are done.
        let input_done = recv[0] == PortState::Done
            && self.buffers.iter().all(|b| match b {
                BufferedStream::Open(v) => v.is_empty(),
                BufferedStream::Closed => true,
            });
        let output_done = send.iter().all(|p| *p == PortState::Done);

        // If either side is done, everything is done.
        if input_done || output_done {
            recv[0] = PortState::Done;
            for s in send {
                *s = PortState::Done;
            }
            return Ok(());
        }

        let all_blocked = send.iter().all(|p| *p == PortState::Blocked);

        // Pass along the input state to the output.
        for (i, s) in send.iter_mut().enumerate() {
            let buffer_empty = match &self.buffers[i] {
                BufferedStream::Open(v) => v.is_empty(),
                BufferedStream::Closed => true,
            };
            *s = if buffer_empty && recv[0] == PortState::Done {
                PortState::Done
            } else if !buffer_empty || recv[0] == PortState::Ready {
                PortState::Ready
            } else {
                PortState::Blocked
            };
        }

        // We say we are ready to receive unless all outputs are blocked.
        recv[0] = if all_blocked {
            PortState::Blocked
        } else {
            PortState::Ready
        };
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        _state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(recv.len() == 1 && !send.is_empty());
        assert!(self.buffers.len() == send.len());

        enum Listener<'a> {
            Active(UnboundedSender<Morsel>),
            Buffering(&'a mut VecDeque<Morsel>),
            Inactive,
        }

        let buffered_source_token = SourceToken::new();

        let (mut buf_senders, buf_receivers): (Vec<_>, Vec<_>) = self
            .buffers
            .iter_mut()
            .enumerate()
            .map(|(port_idx, buffer)| {
                if let BufferedStream::Open(buf) = buffer {
                    if send[port_idx].is_some() {
                        // TODO: replace with a bounded channel and store data
                        // out-of-core beyond a certain size.
                        let (rx, tx) = unbounded_channel();
                        (Listener::Active(rx), Some((buf, tx)))
                    } else {
                        (Listener::Buffering(buf), None)
                    }
                } else {
                    (Listener::Inactive, None)
                }
            })
            .unzip();

        // TODO: parallel multiplexing.
        if let Some(mut receiver) = recv[0].take().map(|r| r.serial()) {
            let buffered_source_token = buffered_source_token.clone();
            join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                loop {
                    let Ok(morsel) = receiver.recv().await else {
                        break;
                    };

                    let mut anyone_interested = false;
                    let mut active_listener_interested = false;
                    for buf_sender in &mut buf_senders {
                        match buf_sender {
                            Listener::Active(s) => match s.send(morsel.clone()) {
                                Ok(_) => {
                                    anyone_interested = true;
                                    active_listener_interested = true;
                                },
                                Err(_) => *buf_sender = Listener::Inactive,
                            },
                            Listener::Buffering(b) => {
                                // Make sure to count buffered morsels as
                                // consumed to not block the source.
                                let mut m = morsel.clone();
                                m.take_consume_token();
                                b.push_front(m);
                                anyone_interested = true;
                            },
                            Listener::Inactive => {},
                        }
                    }

                    if !anyone_interested {
                        break;
                    }

                    // If only buffering inputs are left, or we got a stop
                    // request from an input reading from old buffered data,
                    // request a stop from the source.
                    if !active_listener_interested || buffered_source_token.stop_requested() {
                        morsel.source_token().stop();
                    }
                }

                Ok(())
            }));
        }

        for (send_port, opt_buf_recv) in send.iter_mut().zip(buf_receivers) {
            if let Some((buf, mut rx)) = opt_buf_recv {
                let mut sender = send_port.take().unwrap().serial();

                let buffered_source_token = buffered_source_token.clone();
                join_handles.push(scope.spawn_task(TaskPriority::High, async move {
                    // First we try to flush all the old buffered data.
                    while let Some(mut morsel) = buf.pop_back() {
                        morsel.replace_source_token(buffered_source_token.clone());
                        if sender.send(morsel).await.is_err()
                            || buffered_source_token.stop_requested()
                        {
                            break;
                        }
                    }

                    // Then send along data from the multiplexer.
                    while let Some(morsel) = rx.recv().await {
                        if sender.send(morsel).await.is_err() {
                            break;
                        }
                    }
                    Ok(())
                }));
            }
        }
    }
}
