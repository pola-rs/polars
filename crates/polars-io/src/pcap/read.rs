use std::io::{Read, Seek};

use pcap_file::pcap::PcapReader as LegacyPcapReader;
use polars_core::error::to_compute_err;
use polars_core::prelude::*;
use polars_core::utils::concat_df;

use crate::prelude::*;

pub struct PcapReader<R: Read + Seek> {
    pcap_reader: Option<LegacyPcapReader<R>>,
    reader: Option<R>,
    rechunk: bool,
    n_rows: Option<usize>,
    total_rows_read: usize,
}

impl<R: Read + Seek> SerReader<R> for PcapReader<R> {
    fn new(reader: R) -> Self {
        PcapReader {
            pcap_reader: None,
            reader: Some(reader),
            rechunk: true,
            n_rows: None,
            total_rows_read: 0,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(mut self) -> PolarsResult<DataFrame> {
        let mut dfs = Vec::new();
        while let Some(df) = self.next_batch(1024 * 1024)? {
            dfs.push(df);
        }
        if dfs.is_empty() {
            return self.empty_df();
        }
        let mut df = concat_df(&dfs)?;
        if self.rechunk {
            df.align_chunks();
        }
        Ok(df)
    }
}

impl<R: Read + Seek> PcapReader<R> {
    pub fn with_n_rows(mut self, n: Option<usize>) -> Self {
        self.n_rows = n;
        self
    }

    fn empty_df(&self) -> PolarsResult<DataFrame> {
        Ok(DataFrame::new(vec![
            Series::new_empty("time_s".into(), &DataType::Int64).into(),
            Series::new_empty("time_ns".into(), &DataType::UInt32).into(),
            Series::new_empty("incl_len".into(), &DataType::UInt32).into(),
            Series::new_empty("orig_len".into(), &DataType::UInt32).into(),
            Series::new_empty("data".into(), &DataType::Binary).into(),
        ])?)
    }

    pub fn next_batch(&mut self, batch_size: usize) -> PolarsResult<Option<DataFrame>> {
        if self.pcap_reader.is_none() {
            let reader = self
                .reader
                .take()
                .ok_or_else(|| polars_err!(ComputeError: "reader already consumed"))?;
            self.pcap_reader = Some(LegacyPcapReader::new(reader).map_err(to_compute_err)?);
        }
        let pcap_reader = self.pcap_reader.as_mut().unwrap();

        let mut ts_secs = Vec::new();
        let mut ts_nsecs = Vec::new();
        let mut incl_lens = Vec::new();
        let mut orig_lens = Vec::new();
        let mut packet_data = Vec::new();

        let mut count = 0;
        while count < batch_size {
            if let Some(n) = self.n_rows {
                if self.total_rows_read >= n {
                    break;
                }
            }

            match pcap_reader.next_packet() {
                Some(Ok(packet)) => {
                    ts_secs.push(packet.timestamp.as_secs() as i64);
                    ts_nsecs.push(packet.timestamp.subsec_nanos());
                    incl_lens.push(packet.data.len() as u32);
                    orig_lens.push(packet.orig_len);
                    packet_data.push(packet.data.to_vec());

                    count += 1;
                    self.total_rows_read += 1;
                },
                Some(Err(e)) => return Err(to_compute_err(e)),
                None => break,
            }
        }

        if count == 0 {
            return Ok(None);
        }

        let mut df = DataFrame::new(vec![
            Series::new("time_s".into(), ts_secs).into(),
            Series::new("time_ns".into(), ts_nsecs).into(),
            Series::new("incl_len".into(), incl_lens).into(),
            Series::new("orig_len".into(), orig_lens).into(),
            Series::new("data".into(), packet_data).into(),
        ])?;

        if self.rechunk {
            df.align_chunks();
        }

        Ok(Some(df))
    }
}
