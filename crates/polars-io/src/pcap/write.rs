use std::io::Write;
use std::time::Duration;

use pcap_file::DataLink;
use pcap_file::pcap::{PcapHeader, PcapPacket, PcapWriter as LegacyPcapWriter};
use polars_core::error::to_compute_err;
use polars_core::prelude::*;

pub struct PcapWriter<W: Write> {
    writer: W,
}

impl<W: Write> PcapWriter<W> {
    pub fn new(writer: W) -> Self {
        PcapWriter { writer }
    }

    pub fn finish(self, df: &mut DataFrame) -> PolarsResult<()> {
        let ts_s = df.column("time_s")?.i64()?;
        let ts_ns = df.column("time_ns")?.u32()?;
        let orig_len = df.column("orig_len")?.u32()?;
        let data = df.column("data")?.binary()?;

        let header = PcapHeader {
            datalink: DataLink::ETHERNET,
            ..Default::default()
        };

        let mut pcap_writer =
            LegacyPcapWriter::with_header(self.writer, header).map_err(to_compute_err)?;

        for i in 0..df.height() {
            let s = ts_s
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "missing value"))?;
            let ns = ts_ns
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "missing value"))?;
            let olen = orig_len
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "missing value"))?;
            let d = data
                .get(i)
                .ok_or_else(|| polars_err!(ComputeError: "missing value"))?;

            let packet = PcapPacket {
                timestamp: Duration::new(s as u64, ns),
                orig_len: olen,
                data: d.into(),
            };

            pcap_writer.write_packet(&packet).map_err(to_compute_err)?;
        }

        Ok(())
    }
}
