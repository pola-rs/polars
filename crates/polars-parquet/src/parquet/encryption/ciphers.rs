// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0.

use ring::aead::{AES_128_GCM, AES_256_GCM, Aad, LessSafeKey, NonceSequence, UnboundKey};
use ring::rand::{SecureRandom, SystemRandom};

use crate::parquet::error::{ParquetError, ParquetResult};

const RIGHT_TWELVE: u128 = 0x0000_0000_ffff_ffff_ffff_ffff_ffff_ffff;
pub(crate) const NONCE_LEN: usize = 12;
pub(crate) const SIZE_LEN: usize = 4;
pub(crate) const TAG_LEN: usize = 16;

pub(crate) trait BlockDecryptor: std::fmt::Debug + Send + Sync {
    fn decrypt(&self, length_and_ciphertext: &[u8], aad: &[u8]) -> ParquetResult<Vec<u8>>;

    fn compute_plaintext_tag(&self, aad: &[u8], plaintext: &[u8]) -> ParquetResult<Vec<u8>>;
}

#[derive(Debug, Clone)]
pub(crate) struct RingGcmBlockDecryptor {
    key: LessSafeKey,
}

impl RingGcmBlockDecryptor {
    pub(crate) fn new(key_bytes: &[u8]) -> ParquetResult<Self> {
        let algorithm = if key_bytes.len() == AES_128_GCM.key_len() {
            &AES_128_GCM
        } else if key_bytes.len() == AES_256_GCM.key_len() {
            &AES_256_GCM
        } else {
            return Err(ParquetError::InvalidParameter(format!(
                "unsupported AES-GCM key length: {} bytes",
                key_bytes.len()
            )));
        };

        let key = UnboundKey::new(algorithm, key_bytes).map_err(|_| {
            ParquetError::InvalidParameter("failed to create AES-GCM key".to_string())
        })?;

        Ok(Self {
            key: LessSafeKey::new(key),
        })
    }
}

impl BlockDecryptor for RingGcmBlockDecryptor {
    fn decrypt(&self, length_and_ciphertext: &[u8], aad: &[u8]) -> ParquetResult<Vec<u8>> {
        if length_and_ciphertext.len() < SIZE_LEN + NONCE_LEN + TAG_LEN {
            return Err(ParquetError::oos("encrypted module is too short"));
        }

        let expected_ciphertext_len = u32::from_le_bytes(
            length_and_ciphertext[..SIZE_LEN]
                .try_into()
                .expect("slice length checked"),
        ) as usize;
        if expected_ciphertext_len + SIZE_LEN != length_and_ciphertext.len() {
            return Err(ParquetError::oos(format!(
                "encrypted module length mismatch: declared {}, actual {}",
                expected_ciphertext_len,
                length_and_ciphertext.len() - SIZE_LEN
            )));
        }

        let nonce = ring::aead::Nonce::try_assume_unique_for_key(
            &length_and_ciphertext[SIZE_LEN..SIZE_LEN + NONCE_LEN],
        )
        .map_err(|_| ParquetError::oos("invalid AES-GCM nonce"))?;

        let mut result = Vec::with_capacity(expected_ciphertext_len - NONCE_LEN);
        result.extend_from_slice(&length_and_ciphertext[SIZE_LEN + NONCE_LEN..]);
        self.key
            .open_in_place(nonce, Aad::from(aad), &mut result)
            .map_err(|_| ParquetError::oos("AES-GCM authentication failed"))?;

        result.truncate(result.len() - TAG_LEN);
        Ok(result)
    }

    fn compute_plaintext_tag(&self, aad: &[u8], plaintext: &[u8]) -> ParquetResult<Vec<u8>> {
        if plaintext.len() < NONCE_LEN + TAG_LEN {
            return Err(ParquetError::oos("signed plaintext module is too short"));
        }

        let nonce_start = plaintext.len() - NONCE_LEN - TAG_LEN;
        let tag_start = plaintext.len() - TAG_LEN;
        let nonce =
            ring::aead::Nonce::try_assume_unique_for_key(&plaintext[nonce_start..tag_start])
                .map_err(|_| ParquetError::oos("invalid AES-GCM nonce"))?;

        let mut signed_plaintext = plaintext.to_vec();
        let tag = self
            .key
            .seal_in_place_separate_tag(nonce, Aad::from(aad), &mut signed_plaintext[..nonce_start])
            .map_err(|_| ParquetError::oos("AES-GCM footer signing failed"))?;
        Ok(tag.as_ref().to_vec())
    }
}

pub(crate) trait BlockEncryptor: std::fmt::Debug + Send + Sync {
    fn encrypt(&mut self, plaintext: &[u8], aad: &[u8]) -> ParquetResult<Vec<u8>>;
}

#[derive(Debug, Clone)]
struct CounterNonce {
    start: u128,
    counter: u128,
}

impl CounterNonce {
    fn new(rng: &SystemRandom) -> ParquetResult<Self> {
        let mut buffer = [0; 16];
        rng.fill(&mut buffer)
            .map_err(|_| ParquetError::oos("failed to generate AES-GCM nonce"))?;

        let start = u128::from_ne_bytes(buffer) & RIGHT_TWELVE;
        let counter = start.wrapping_add(1);

        Ok(Self { start, counter })
    }

    fn get_bytes(&self) -> [u8; NONCE_LEN] {
        self.counter.to_le_bytes()[..NONCE_LEN]
            .try_into()
            .expect("nonce length is fixed")
    }
}

impl NonceSequence for CounterNonce {
    fn advance(&mut self) -> Result<ring::aead::Nonce, ring::error::Unspecified> {
        if (self.counter & RIGHT_TWELVE) == (self.start & RIGHT_TWELVE) {
            Err(ring::error::Unspecified)
        } else {
            let buffer = self.get_bytes();
            self.counter = self.counter.wrapping_add(1);
            Ok(ring::aead::Nonce::assume_unique_for_key(buffer))
        }
    }
}

#[derive(Debug, Clone)]
pub(crate) struct RingGcmBlockEncryptor {
    key: LessSafeKey,
    nonce_sequence: CounterNonce,
}

impl RingGcmBlockEncryptor {
    pub(crate) fn new(key_bytes: &[u8]) -> ParquetResult<Self> {
        let algorithm = if key_bytes.len() == AES_128_GCM.key_len() {
            &AES_128_GCM
        } else if key_bytes.len() == AES_256_GCM.key_len() {
            &AES_256_GCM
        } else {
            return Err(ParquetError::InvalidParameter(format!(
                "unsupported AES-GCM key length: {} bytes",
                key_bytes.len()
            )));
        };

        let key = UnboundKey::new(algorithm, key_bytes).map_err(|_| {
            ParquetError::InvalidParameter("failed to create AES-GCM key".to_string())
        })?;
        let nonce_sequence = CounterNonce::new(&SystemRandom::new())?;

        Ok(Self {
            key: LessSafeKey::new(key),
            nonce_sequence,
        })
    }
}

impl BlockEncryptor for RingGcmBlockEncryptor {
    fn encrypt(&mut self, plaintext: &[u8], aad: &[u8]) -> ParquetResult<Vec<u8>> {
        let ciphertext_length: u32 = (NONCE_LEN + plaintext.len() + TAG_LEN)
            .try_into()
            .map_err(|err| ParquetError::oos(format!("plaintext module is too long: {err}")))?;

        let mut ciphertext = Vec::with_capacity(SIZE_LEN + ciphertext_length as usize);
        ciphertext.extend(ciphertext_length.to_le_bytes());

        let nonce = self
            .nonce_sequence
            .advance()
            .map_err(|_| ParquetError::oos("AES-GCM nonce sequence exhausted"))?;
        ciphertext.extend(nonce.as_ref());
        ciphertext.extend(plaintext);

        let tag = self
            .key
            .seal_in_place_separate_tag(
                nonce,
                Aad::from(aad),
                &mut ciphertext[SIZE_LEN + NONCE_LEN..],
            )
            .map_err(|_| ParquetError::oos("AES-GCM encryption failed"))?;
        ciphertext.extend(tag.as_ref());

        Ok(ciphertext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn aes_gcm_round_trip() {
        let key = b"0123456789012345";
        let mut encryptor = RingGcmBlockEncryptor::new(key).unwrap();
        let decryptor = RingGcmBlockDecryptor::new(key).unwrap();
        let ciphertext = encryptor.encrypt(b"hello", b"aad").unwrap();
        assert_eq!(decryptor.decrypt(&ciphertext, b"aad").unwrap(), b"hello");
    }
}
