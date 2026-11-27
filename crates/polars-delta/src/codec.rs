//! Delta deletion-vector validation and decoding, independent of the reader and Arrow.
//! Callers supply the row allocation bound and own transport, credentials and scheduling.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DecodeError(String);

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for DecodeError {}

type Result<T> = std::result::Result<T, DecodeError>;
macro_rules! error {
    ($($arg:tt)*) => { DecodeError(format!($($arg)*)) };
}
macro_rules! ensure {
    ($condition:expr, $($arg:tt)*) => {
        if !$condition { return Err(error!($($arg)*)); }
    };
}

/// Arrow-compatible packed keep bits: low bit first, with no validity buffer.
/// Rows after `len` are kept; `len` ends at the final deleted row.
#[derive(Debug)]
pub struct KeepMask {
    pub bytes: Vec<u8>,
    pub len: usize,
}

/// A frame follows the file's version byte, so it cannot start at offset 0.
pub fn validate_frame_offset(offset: u64) -> Result<()> {
    ensure!(
        offset >= 1,
        "Delta deletion vector offset must follow version byte"
    );
    Ok(())
}

/// Validate the first byte of a deletion vector file.
pub fn validate_file_version(header: &[u8]) -> Result<()> {
    ensure!(
        header == [1],
        "unsupported Delta deletion vector file version"
    );
    Ok(())
}

pub fn decode_inline(
    data: &[u8],
    size: u32,
    cardinality: u64,
    max_rows: usize,
) -> Result<KeepMask> {
    ensure!(
        data.len() == size as usize,
        "inline deletion vector size mismatch"
    );
    decode_bitmap(data, Some(cardinality), max_rows)
}

pub const MAX_SERIALIZED_BYTES: usize = 512 * 1024 * 1024;

const MAGIC: u32 = 1681511377;
const Z85: &[u8] =
    b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";

pub fn decode_z85(value: &str) -> Result<Vec<u8>> {
    ensure!(
        value.len().is_multiple_of(5),
        "invalid deletion vector Z85 length"
    );
    ensure!(
        value.len() / 5 * 4 <= MAX_SERIALIZED_BYTES,
        "deletion vector exceeds 512 MiB serialized size resource limit"
    );
    let mut out = Vec::with_capacity(value.len() / 5 * 4);
    for chunk in value.as_bytes().as_chunks::<5>().0 {
        let mut n = 0u32;
        for c in chunk {
            let digit = Z85
                .iter()
                .position(|v| v == c)
                .ok_or_else(|| error!("invalid deletion vector Z85 character"))?
                as u32;
            n = n
                .checked_mul(85)
                .and_then(|n| n.checked_add(digit))
                .ok_or_else(|| error!("deletion vector Z85 overflow"))?;
        }
        out.extend_from_slice(&n.to_be_bytes());
    }
    Ok(out)
}

pub fn decode_framed(
    data: &[u8],
    expected_size: Option<u32>,
    cardinality: Option<u64>,
    max_rows: usize,
) -> Result<KeepMask> {
    ensure!(data.len() >= 8, "truncated deletion vector frame");
    let size = u32::from_be_bytes(data[..4].try_into().unwrap());
    ensure!(
        expected_size.is_none_or(|n| n == size) && data.len() as u64 == u64::from(size) + 8,
        "deletion vector frame size mismatch"
    );
    let body = &data[4..data.len() - 4];
    let crc = u32::from_be_bytes(data[data.len() - 4..].try_into().unwrap());
    ensure!(crc32fast::hash(body) == crc, "deletion vector CRC mismatch");
    decode_bitmap(body, cardinality, max_rows)
}

pub fn decode_bitmap(data: &[u8], cardinality: Option<u64>, max_rows: usize) -> Result<KeepMask> {
    ensure!(
        data.len() <= MAX_SERIALIZED_BYTES,
        "deletion vector exceeds 512 MiB serialized size resource limit"
    );
    let mut max = None;
    let count = visit(data, |span| {
        max = Some(span.last());
        Ok(())
    })?;
    ensure!(
        cardinality.is_none_or(|n| n == count),
        "deletion vector cardinality mismatch"
    );
    let len = match max {
        None => 0,
        Some(n) => usize::try_from(n).ok().and_then(|n| n.checked_add(1))
            .filter(|n| *n <= max_rows)
            .ok_or_else(|| error!("deletion vector exceeds maximum supported row position / keep-mask resource limit ({max_rows} rows)"))?,
    };
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(len.div_ceil(8))
        .map_err(|_| error!("cannot allocate deletion vector keep mask"))?;
    bytes.resize(len.div_ceil(8), u8::MAX);
    visit(data, |span| {
        match span {
            DeletedSpan::Range(first, last) => {
                clear_range(&mut bytes, first as usize, last as usize)
            },
            DeletedSpan::Word { base, bits } => {
                let base = base as usize / 8;
                // Roaring and Arrow both number the low bit of each byte first.
                // The final nonzero word can extend past the mask's last byte.
                for (dst, src) in bytes[base..].iter_mut().zip(bits.to_le_bytes()) {
                    *dst &= !src;
                }
            },
        }
        Ok(())
    })?;
    Ok(KeepMask { bytes, len })
}

// Array/run spans and packed bitmap words avoid expanding compressed containers
// into individual row positions during either validation or mask materialization.
enum DeletedSpan {
    Range(u64, u64),
    Word { base: u64, bits: u64 },
}
impl DeletedSpan {
    fn last(&self) -> u64 {
        match *self {
            Self::Range(_, last) => last,
            Self::Word { base, bits } => base + (63 - bits.leading_zeros()) as u64,
        }
    }
}

fn clear_range(bytes: &mut [u8], first: usize, last: usize) {
    let first_byte = first / 8;
    let last_byte = last / 8;
    let start_mask = u8::MAX << (first % 8);
    let end_mask = u8::MAX >> (7 - last % 8);
    if first_byte == last_byte {
        bytes[first_byte] &= !(start_mask & end_mask);
    } else {
        bytes[first_byte] &= !start_mask;
        bytes[first_byte + 1..last_byte].fill(0);
        bytes[last_byte] &= !end_mask;
    }
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|n| *n <= self.data.len())
            .ok_or_else(|| error!("truncated deletion vector bitmap"))?;
        let out = &self.data[self.pos..end];
        self.pos = end;
        Ok(out)
    }
    fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }
    fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
}

fn visit(data: &[u8], mut emit: impl FnMut(DeletedSpan) -> Result<()>) -> Result<u64> {
    let mut r = Reader { data, pos: 0 };
    ensure!(r.u32()? == MAGIC, "invalid deletion vector magic");
    let buckets = r.u64()?;
    ensure!(
        buckets <= (data.len() / 12) as u64,
        "invalid deletion vector bucket count"
    );
    let mut previous_bucket = None;
    let mut total = 0u64;
    for _ in 0..buckets {
        let high = r.u32()?;
        ensure!(
            high < 0x80000000 && previous_bucket.is_none_or(|v| v < high),
            "invalid deletion vector bucket key order"
        );
        previous_bucket = Some(high);
        let start = r.pos;
        let cookie = r.u32()?;
        let has_run = cookie & 0xffff == 12347;
        let n = if has_run {
            (cookie >> 16) as usize + 1
        } else {
            ensure!(cookie == 12346, "invalid deletion vector roaring cookie");
            r.u32()? as usize
        };
        ensure!(n <= 65536, "invalid deletion vector container count");
        let runs = if has_run { r.take(n.div_ceil(8))? } else { &[] };
        let descriptions = r.take(n * 4)?;
        let offsets = if !has_run || n >= 4 {
            r.take(n * 4)?
        } else {
            &[]
        };
        let mut previous_key = None;
        for i in 0..n {
            let key = u16::from_le_bytes(descriptions[i * 4..i * 4 + 2].try_into().unwrap());
            let card = u16::from_le_bytes(descriptions[i * 4 + 2..i * 4 + 4].try_into().unwrap())
                as u32
                + 1;
            ensure!(
                previous_key.is_none_or(|v| v < key),
                "invalid deletion vector container key order"
            );
            previous_key = Some(key);
            if !offsets.is_empty() {
                let offset =
                    u32::from_le_bytes(offsets[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
                ensure!(
                    offset == r.pos - start,
                    "invalid deletion vector container offset"
                );
            }
            let base = ((high as u64) << 32) | ((key as u64) << 16);
            let mut actual = 0;
            if has_run && runs[i / 8] & (1 << (i % 8)) != 0 {
                let count = r.u16()?;
                let mut previous_end = None;
                for _ in 0..count {
                    let first = r.u16()? as u32;
                    let last = first + r.u16()? as u32;
                    ensure!(
                        last <= 65535 && previous_end.is_none_or(|v| v < first),
                        "invalid deletion vector run"
                    );
                    previous_end = Some(last);
                    actual += last - first + 1;
                    emit(DeletedSpan::Range(base | first as u64, base | last as u64))?;
                }
            } else if card <= 4096 {
                let mut previous = None;
                for _ in 0..card {
                    let low = r.u16()?;
                    ensure!(
                        previous.is_none_or(|v| v < low),
                        "invalid deletion vector array order"
                    );
                    previous = Some(low);
                    emit(DeletedSpan::Range(base | low as u64, base | low as u64))?;
                }
                actual = card;
            } else {
                for j in 0..1024u64 {
                    let word = r.u64()?;
                    actual += word.count_ones();
                    if word != 0 {
                        emit(DeletedSpan::Word {
                            base: base | (j * 64),
                            bits: word,
                        })?;
                    }
                }
            }
            ensure!(
                actual == card,
                "deletion vector container cardinality mismatch"
            );
            total += card as u64;
        }
    }
    ensure!(
        r.pos == data.len(),
        "trailing bytes in deletion vector bitmap"
    );
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    impl KeepMask {
        fn len(&self) -> usize {
            self.len
        }
        fn is_empty(&self) -> bool {
            self.len == 0
        }
        fn get_bit(&self, i: usize) -> bool {
            self.bytes[i / 8] & (1 << (i % 8)) != 0
        }
        fn unset_bits(&self) -> usize {
            (0..self.len).filter(|i| !self.get_bit(*i)).count()
        }
    }

    #[test]
    fn frame_offset_and_version() {
        assert!(validate_frame_offset(1).is_ok());
        assert!(validate_frame_offset(0).is_err());
        assert!(validate_file_version(&[1]).is_ok());
        for header in [vec![], vec![0], vec![2], vec![1, 1]] {
            assert!(validate_file_version(&header).is_err());
        }
    }

    #[test]
    fn inline_declared_size() {
        let data = example_bitmap();
        assert!(decode_inline(&data, data.len() as u32, 6, 30).is_ok());
        assert!(decode_inline(&data, data.len() as u32 + 1, 6, 30).is_err());
        assert!(decode_inline(&data, data.len() as u32 - 1, 6, 30).is_err());
    }

    fn example_bitmap() -> Vec<u8> {
        // The corrected portable example in delta-kernel's deletion_vector.rs.
        decode_z85("^Bg9^0rr910000000000iXQKl0rr91000f55c8Xg0@@D72lkbi5=-{L").unwrap()
    }
    #[test]
    fn protocol_literal_uses_unsupported_magic() {
        // PROTOCOL.md's literal predates the portable format. Keep it visible as an
        // explicit rejection, rather than treating its undocumented byte order as portable.
        let data = decode_z85("wi5b=000010000siXQKl0rr91000f55c8Xg0@@D72lkbi5=-{L").unwrap();
        assert_eq!(&data[..4], &[0x64, 0x39, 0xd3, 0xd0]);
        assert!(
            decode_bitmap(&data, Some(6), 30)
                .unwrap_err()
                .to_string()
                .contains("magic")
        );
    }
    #[test]
    fn portable_protocol_positions() {
        let data = example_bitmap();
        let mask = decode_bitmap(&data, Some(6), 30).unwrap();
        assert_eq!(
            (0..mask.len())
                .filter(|i| !mask.get_bit(*i))
                .collect::<Vec<_>>(),
            [3, 4, 7, 11, 18, 29]
        );
        assert!(decode_bitmap(&data, Some(5), 30).is_err());
        assert!(decode_bitmap(&data, Some(6), 29).is_err());
        for n in 0..data.len() {
            assert!(decode_bitmap(&data[..n], None, 30).is_err());
        }
    }
    #[test]
    fn empty_and_invalid_metadata() {
        let mut empty = MAGIC.to_le_bytes().to_vec();
        empty.extend_from_slice(&0u64.to_le_bytes());
        assert!(decode_bitmap(&empty, Some(0), 0).unwrap().is_empty());
        assert!(decode_bitmap(&empty, Some(u64::MAX), 0).is_err());
        let data = example_bitmap();
        let mut invalid = data.clone();
        invalid[12..16].copy_from_slice(&0x80000000u32.to_le_bytes());
        assert!(decode_bitmap(&invalid, None, 30).is_err());
        let mut invalid = data.clone();
        invalid[28..32].copy_from_slice(&999u32.to_le_bytes());
        assert!(decode_bitmap(&invalid, None, 30).is_err());
        let mut invalid = data.clone();
        invalid[34..36].copy_from_slice(&3u16.to_le_bytes());
        assert!(decode_bitmap(&invalid, None, 30).is_err());
        assert!(decode_framed(&[0; 8], Some(0), None, 0).is_err());
    }
    #[test]
    fn run_and_bitmap_containers() {
        let mut run = MAGIC.to_le_bytes().to_vec();
        run.extend_from_slice(&1u64.to_le_bytes());
        run.extend_from_slice(&0u32.to_le_bytes());
        run.extend_from_slice(&12347u32.to_le_bytes());
        run.push(1); // run-container flags, no offsets for <4 containers
        run.extend_from_slice(&0u16.to_le_bytes());
        run.extend_from_slice(&4u16.to_le_bytes());
        run.extend_from_slice(&1u16.to_le_bytes());
        run.extend_from_slice(&10u16.to_le_bytes());
        run.extend_from_slice(&4u16.to_le_bytes());
        let mask = decode_bitmap(&run, Some(5), 15).unwrap();
        assert_eq!(mask.unset_bits(), 5);
        // A 65k-row run is validated without expanding its rows on the first pass.
        assert!(decode_bitmap(&run, Some(5), 14).is_err());
        let mut bits = MAGIC.to_le_bytes().to_vec();
        bits.extend_from_slice(&1u64.to_le_bytes());
        bits.extend_from_slice(&0u32.to_le_bytes());
        bits.extend_from_slice(&12346u32.to_le_bytes());
        bits.extend_from_slice(&1u32.to_le_bytes());
        bits.extend_from_slice(&0u16.to_le_bytes());
        bits.extend_from_slice(&65535u16.to_le_bytes());
        bits.extend_from_slice(&16u32.to_le_bytes());
        bits.extend_from_slice(&[255; 8192]);
        assert_eq!(
            decode_bitmap(&bits, Some(65536), 65536)
                .unwrap()
                .unset_bits(),
            65536
        );
        bits[32] = 0;
        assert!(decode_bitmap(&bits, None, 65536).is_err());
    }
    #[test]
    fn z85_rejects_invalid_input() {
        for s in ["0000", "~~~~~", "#####"] {
            assert!(decode_z85(s).is_err());
        }
    }
    #[test]
    fn packed_bitmap_orientation_and_work_bound() {
        // Scattered values exercise low/high bits of bytes and 64-bit words, and
        // the truncated final word exercises a mask not divisible by eight.
        let mut words = [0x81010100000001ffu64; 1024];
        words[1023] = (1 << 60) | (1 << 5);
        let cardinality = words.iter().map(|w| w.count_ones() as u64).sum::<u64>();
        assert!(cardinality > 4096);
        let mut data = MAGIC.to_le_bytes().to_vec();
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&12346u32.to_le_bytes());
        data.extend_from_slice(&1u32.to_le_bytes());
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&((cardinality - 1) as u16).to_le_bytes());
        data.extend_from_slice(&16u32.to_le_bytes());
        for word in words {
            data.extend_from_slice(&word.to_le_bytes());
        }
        let mut spans = 0;
        assert_eq!(
            visit(&data, |_| {
                spans += 1;
                Ok(())
            })
            .unwrap(),
            cardinality
        );
        assert_eq!(spans, 1024); // one operation per packed word, not per deleted row
        let mask = decode_bitmap(&data, Some(cardinality), 65536).unwrap();
        assert_eq!(mask.len(), 1023 * 64 + 61);
        for i in 0..mask.len() {
            assert_eq!(
                mask.get_bit(i),
                words[i / 64] & (1 << (i % 64)) == 0,
                "bit {i}"
            );
        }
    }

    #[test]
    fn run_materialization_byte_and_word_boundaries() {
        let runs = [(0u16, 0u16), (2, 8), (15, 32), (63, 65), (65534, 65535)];
        let cardinality = runs
            .iter()
            .map(|(a, b)| *b as u64 - *a as u64 + 1)
            .sum::<u64>();
        let mut data = MAGIC.to_le_bytes().to_vec();
        data.extend_from_slice(&1u64.to_le_bytes());
        data.extend_from_slice(&0u32.to_le_bytes());
        data.extend_from_slice(&12347u32.to_le_bytes());
        data.push(1);
        data.extend_from_slice(&0u16.to_le_bytes());
        data.extend_from_slice(&((cardinality - 1) as u16).to_le_bytes());
        data.extend_from_slice(&(runs.len() as u16).to_le_bytes());
        for (first, last) in runs {
            data.extend_from_slice(&first.to_le_bytes());
            data.extend_from_slice(&(last - first).to_le_bytes());
        }
        let mut spans = 0;
        assert_eq!(
            visit(&data, |_| {
                spans += 1;
                Ok(())
            })
            .unwrap(),
            cardinality
        );
        assert_eq!(spans, runs.len());
        let mask = decode_bitmap(&data, Some(cardinality), 65536).unwrap();
        for i in 0..mask.len() {
            assert_eq!(
                mask.get_bit(i),
                !runs
                    .iter()
                    .any(|(a, b)| (*a as usize..=*b as usize).contains(&i)),
                "bit {i}"
            );
        }
    }

    #[test]
    fn frame_crc_and_length() {
        let body = example_bitmap();
        let mut data = (body.len() as u32).to_be_bytes().to_vec();
        data.extend_from_slice(&body);
        data.extend_from_slice(&crc32fast::hash(&body).to_be_bytes());
        assert!(decode_framed(&data, Some(body.len() as u32), Some(6), 30).is_ok());
        data[5] ^= 1;
        assert!(
            decode_framed(&data, None, None, 30)
                .unwrap_err()
                .to_string()
                .contains("CRC")
        );
    }
}
