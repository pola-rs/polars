use super::compute_node_prelude::*;

pub struct ZipNode {
    out_seq: MorselSeq,
    partials: Vec<Option<Morsel>>,
}

impl ZipNode {
    pub fn new() -> Self {
        Self {
            out_seq: MorselSeq::new(0),
            partials: Vec::new()
        }
    }
}

impl ComputeNode for ZipNode {
    fn name(&self) -> &str {
        "zip"
    }

    fn update_state(&mut self, recv: &mut [PortState], send: &mut [PortState]) {
        assert!(send.len() == 1);
        
        let any_input_blocked = recv.iter().any(|s| *s == PortState::Blocked);
        let all_input_done = !any_input_blocked && recv.iter().all(|s| *s == PortState::Done);
        
        let new_recv_state = if send[0] == PortState::Done || all_input_done {
            send[0] = PortState::Done;
            PortState::Done
        } else if send[0] == PortState::Blocked || any_input_blocked {
            send[0] = if any_input_blocked { PortState::Blocked } else { PortState::Ready };
            PortState::Blocked
        } else {
            send[0] = PortState::Ready;
            PortState::Ready
        };
        
        for r in recv {
            *r = new_recv_state;
        }
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv: &mut [Option<RecvPort<'_>>],
        send: &mut [Option<SendPort<'_>>],
        state: &'s ExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert!(send.len() == 1);
        let sender = send[0].take().unwrap().serial();
        let mut receivers: Vec<_> = recv.iter_mut().map(|r| Some(r.take()?.serial())).collect();
        
        self.partials.resize_with(receivers.len(), || None);
        
        // let mut out = Vec::new();
        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            loop {
                // Fill partials with non-empty morsels.
                for (recv_idx, opt_recv) in receivers.iter_mut().enumerate() {
                    if let Some(recv) = opt_recv {
                        if self.partials[recv_idx].is_none() {
                            while let Ok(morsel) = recv.recv().await {
                                if morsel.df().height() > 0 {
                                    self.partials[recv_idx] = Some(morsel);
                                    break;
                                }
                            }
                        }
                    }
                }
                
                if all_receivers_done {
                    break;
                }
            }
            
            // We might have partials left over, drop their consume tokens so
            // their senders unblock and we can stop this execution phase.
            for opt_partial in &mut self.partials {
                if let Some(partial) = opt_partial {
                    drop(partial.take_consume_token());
                }
            }
            
            Ok(())
        }));
        
        


        // for (mut recv, mut send) in receivers.into_iter().zip(senders) {
        //     let slf = &*self;
        //     join_handles.push(scope.spawn_task(TaskPriority::High, async move {
        //         while let Ok(morsel) = recv.recv().await {
        //             let morsel = morsel.try_map(|df| {
        //                 let mask = slf.predicate.evaluate(&df, state)?;
        //                 let mask = mask.bool().map_err(|_| {
        //                     polars_err!(
        //                         ComputeError: "filter predicate must be of type `Boolean`, got `{}`", mask.dtype()
        //                     )
        //                 })?;

        //                 // We already parallelize, call the sequential filter.
        //                 df._filter_seq(mask)
        //             })?;

        //             if morsel.df().is_empty() {
        //                 continue;
        //             }

        //             if send.send(morsel).await.is_err() {
        //                 break;
        //             }
        //         }

        //         Ok(())
        //     }));
        // }
    }
}
