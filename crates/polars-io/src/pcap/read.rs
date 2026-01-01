use std::io::{Read, Seek};

use pcap_file::pcap::PcapReader as LegacyPcapReader;
use polars_core::error::to_compute_err;
use polars_core::prelude::*;

use crate::prelude::*;

pub struct PcapReader<R: Read + Seek> {
    reader: R,
    rechunk: bool,
    n_rows: Option<usize>,
}

impl<R: Read + Seek> SerReader<R> for PcapReader<R> {
    fn new(reader: R) -> Self {
        PcapReader {
            reader,
            rechunk: true,
            n_rows: None,
        }
    }

    fn set_rechunk(mut self, rechunk: bool) -> Self {
        self.rechunk = rechunk;
        self
    }

    fn finish(self) -> PolarsResult<DataFrame> {
        let mut pcap_reader = LegacyPcapReader::new(self.reader).map_err(to_compute_err)?;

        let mut ts_secs = Vec::new();
        let mut ts_nsecs = Vec::new();
        let mut incl_lens = Vec::new();
        let mut orig_lens = Vec::new();
        let mut packet_data = Vec::new();

        let mut count = 0;
        while let Some(packet) = pcap_reader.next_packet() {
            let packet = packet.map_err(to_compute_err)?;

            ts_secs.push(packet.timestamp.as_secs() as i64);
            ts_nsecs.push(packet.timestamp.subsec_nanos());
            incl_lens.push(packet.data.len() as u32);
            orig_lens.push(packet.orig_len);
            packet_data.push(packet.data.to_vec());

            count += 1;
            if let Some(n) = self.n_rows {
                if count >= n {
                    break;
                }
            }
        }

        let ts_series = Series::new("time_s".into(), ts_secs);
        let ts_ns_series = Series::new("time_ns".into(), ts_nsecs);
        let incl_len_series = Series::new("incl_len".into(), incl_lens);
        let orig_len_series = Series::new("orig_len".into(), orig_lens);
        let data_series = Series::new("data".into(), packet_data);

        let mut df = DataFrame::new(vec![
            ts_series.into(),
            ts_ns_series.into(),
            incl_len_series.into(),
            orig_len_series.into(),
            data_series.into(),
        ])?;

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
}
