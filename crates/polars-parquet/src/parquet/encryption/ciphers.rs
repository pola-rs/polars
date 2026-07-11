// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0.

use std::fmt::Formatter;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use aes::cipher::{KeyIvInit, StreamCipher};
use aes::{Aes128, Aes192, Aes256};
use aes_gcm::aead::consts::U12;
use aes_gcm::aead::{AeadInPlace, KeyInit};
use aes_gcm::{AesGcm, Nonce, Tag};
use zeroize::Zeroizing;

use crate::parquet::error::{ParquetError, ParquetResult};

pub(crate) const NONCE_LEN: usize = 12;
pub(crate) const SIZE_LEN: usize = 4;
pub(crate) const TAG_LEN: usize = 16;
const MAX_INVOCATIONS_PER_KEY: u64 = 1 << 32;

type Aes128Gcm = AesGcm<Aes128, U12>;
type Aes192Gcm = AesGcm<Aes192, U12>;
type Aes256Gcm = AesGcm<Aes256, U12>;
type Aes128Ctr = ctr::Ctr32BE<Aes128>;
type Aes192Ctr = ctr::Ctr32BE<Aes192>;
type Aes256Ctr = ctr::Ctr32BE<Aes256>;

pub(crate) trait BlockDecryptor: std::fmt::Debug + Send + Sync {
    fn decrypt(&self, length_and_ciphertext: &[u8], aad: &[u8]) -> ParquetResult<Vec<u8>>;

    fn compute_plaintext_tag(&self, aad: &[u8], plaintext: &[u8]) -> ParquetResult<Vec<u8>>;
}

enum GcmCipher {
    Aes128(Aes128Gcm),
    Aes192(Aes192Gcm),
    Aes256(Aes256Gcm),
}

impl std::fmt::Debug for GcmCipher {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::Aes128(_) => "AES-128-GCM",
            Self::Aes192(_) => "AES-192-GCM",
            Self::Aes256(_) => "AES-256-GCM",
        })
    }
}

impl GcmCipher {
    fn new(key_bytes: &[u8]) -> ParquetResult<Self> {
        match key_bytes.len() {
            16 => Ok(Self::Aes128(Aes128Gcm::new_from_slice(key_bytes).unwrap())),
            24 => Ok(Self::Aes192(Aes192Gcm::new_from_slice(key_bytes).unwrap())),
            32 => Ok(Self::Aes256(Aes256Gcm::new_from_slice(key_bytes).unwrap())),
            length => Err(ParquetError::InvalidParameter(format!(
                "unsupported AES-GCM key length: {length} bytes"
            ))),
        }
    }

    fn encrypt(
        &self,
        nonce: &[u8; NONCE_LEN],
        plaintext: &mut [u8],
        aad: &[u8],
    ) -> ParquetResult<Vec<u8>> {
        let nonce = Nonce::from_slice(nonce);
        let tag = match self {
            Self::Aes128(cipher) => cipher.encrypt_in_place_detached(nonce, aad, plaintext),
            Self::Aes192(cipher) => cipher.encrypt_in_place_detached(nonce, aad, plaintext),
            Self::Aes256(cipher) => cipher.encrypt_in_place_detached(nonce, aad, plaintext),
        }
        .map_err(|_| ParquetError::oos("AES-GCM encryption failed"))?;
        Ok(tag.to_vec())
    }

    fn decrypt(
        &self,
        nonce: &[u8; NONCE_LEN],
        ciphertext: &mut [u8],
        tag: &[u8; TAG_LEN],
        aad: &[u8],
    ) -> ParquetResult<()> {
        let nonce = Nonce::from_slice(nonce);
        let tag = Tag::from_slice(tag);
        match self {
            Self::Aes128(cipher) => cipher.decrypt_in_place_detached(nonce, aad, ciphertext, tag),
            Self::Aes192(cipher) => cipher.decrypt_in_place_detached(nonce, aad, ciphertext, tag),
            Self::Aes256(cipher) => cipher.decrypt_in_place_detached(nonce, aad, ciphertext, tag),
        }
        .map_err(|_| ParquetError::oos("AES-GCM authentication failed"))
    }
}

#[derive(Debug, Default)]
pub(crate) struct CipherInvocationCounter {
    count: AtomicU64,
}

impl CipherInvocationCounter {
    fn next_nonce(&self) -> ParquetResult<[u8; NONCE_LEN]> {
        self.count
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |count| {
                (count < MAX_INVOCATIONS_PER_KEY).then_some(count + 1)
            })
            .map_err(|_| {
                ParquetError::InvalidParameter(
                    "AES cipher invocation limit reached for encryption key".to_string(),
                )
            })?;

        let mut nonce = [0; NONCE_LEN];
        getrandom::fill(&mut nonce)
            .map_err(|_| ParquetError::oos("failed to generate AES-GCM nonce"))?;
        Ok(nonce)
    }

    #[cfg(test)]
    fn with_count(count: u64) -> Self {
        Self {
            count: AtomicU64::new(count),
        }
    }
}

#[derive(Debug)]
pub(crate) struct AesGcmBlockDecryptor {
    cipher: GcmCipher,
}

impl AesGcmBlockDecryptor {
    pub(crate) fn new(key_bytes: &[u8]) -> ParquetResult<Self> {
        Ok(Self {
            cipher: GcmCipher::new(key_bytes)?,
        })
    }
}

impl BlockDecryptor for AesGcmBlockDecryptor {
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

        let nonce: &[u8; NONCE_LEN] = length_and_ciphertext[SIZE_LEN..SIZE_LEN + NONCE_LEN]
            .try_into()
            .expect("slice length checked");
        let tag: &[u8; TAG_LEN] = length_and_ciphertext[length_and_ciphertext.len() - TAG_LEN..]
            .try_into()
            .expect("slice length checked");
        let mut result = length_and_ciphertext
            [SIZE_LEN + NONCE_LEN..length_and_ciphertext.len() - TAG_LEN]
            .to_vec();
        self.cipher.decrypt(nonce, &mut result, tag, aad)?;
        Ok(result)
    }

    fn compute_plaintext_tag(&self, aad: &[u8], plaintext: &[u8]) -> ParquetResult<Vec<u8>> {
        if plaintext.len() < NONCE_LEN + TAG_LEN {
            return Err(ParquetError::oos("signed plaintext module is too short"));
        }

        let nonce_start = plaintext.len() - NONCE_LEN - TAG_LEN;
        let tag_start = plaintext.len() - TAG_LEN;
        let nonce: &[u8; NONCE_LEN] = plaintext[nonce_start..tag_start]
            .try_into()
            .expect("slice length checked");
        let mut signed_plaintext = plaintext[..nonce_start].to_vec();
        self.cipher.encrypt(nonce, &mut signed_plaintext, aad)
    }
}

pub(crate) trait BlockEncryptor: std::fmt::Debug + Send + Sync {
    fn encrypt(&mut self, plaintext: &[u8], aad: &[u8]) -> ParquetResult<Vec<u8>>;
}

#[derive(Debug)]
pub(crate) struct AesGcmBlockEncryptor {
    cipher: GcmCipher,
    invocation_counter: Arc<CipherInvocationCounter>,
}

impl AesGcmBlockEncryptor {
    #[cfg(test)]
    pub(crate) fn new(key_bytes: &[u8]) -> ParquetResult<Self> {
        Self::new_with_counter(key_bytes, Arc::default())
    }

    pub(crate) fn new_with_counter(
        key_bytes: &[u8],
        invocation_counter: Arc<CipherInvocationCounter>,
    ) -> ParquetResult<Self> {
        Ok(Self {
            cipher: GcmCipher::new(key_bytes)?,
            invocation_counter,
        })
    }
}

impl BlockEncryptor for AesGcmBlockEncryptor {
    fn encrypt(&mut self, plaintext: &[u8], aad: &[u8]) -> ParquetResult<Vec<u8>> {
        let ciphertext_length: u32 = (NONCE_LEN + plaintext.len() + TAG_LEN)
            .try_into()
            .map_err(|err| ParquetError::oos(format!("plaintext module is too long: {err}")))?;

        let nonce = self.invocation_counter.next_nonce()?;
        let mut encrypted_plaintext = plaintext.to_vec();
        let tag = self.cipher.encrypt(&nonce, &mut encrypted_plaintext, aad)?;

        let mut ciphertext = Vec::with_capacity(SIZE_LEN + ciphertext_length as usize);
        ciphertext.extend(ciphertext_length.to_le_bytes());
        ciphertext.extend(nonce);
        ciphertext.extend(encrypted_plaintext);
        ciphertext.extend(tag);
        Ok(ciphertext)
    }
}

pub(crate) fn encrypt_key_locally(
    plaintext_key: &[u8],
    wrapping_key: &[u8],
    aad: &[u8],
) -> ParquetResult<Vec<u8>> {
    let cipher = GcmCipher::new(wrapping_key)?;
    let nonce = CipherInvocationCounter::default().next_nonce()?;
    let mut encrypted_key = plaintext_key.to_vec();
    let tag = cipher.encrypt(&nonce, &mut encrypted_key, aad)?;
    let mut output = Vec::with_capacity(NONCE_LEN + encrypted_key.len() + TAG_LEN);
    output.extend_from_slice(&nonce);
    output.extend_from_slice(&encrypted_key);
    output.extend_from_slice(&tag);
    Ok(output)
}

pub(crate) fn decrypt_key_locally(
    encrypted_key: &[u8],
    wrapping_key: &[u8],
    aad: &[u8],
) -> ParquetResult<Vec<u8>> {
    if encrypted_key.len() < NONCE_LEN + TAG_LEN {
        return Err(ParquetError::oos("locally wrapped key is too short"));
    }
    let cipher = GcmCipher::new(wrapping_key)?;
    let nonce = encrypted_key[..NONCE_LEN]
        .try_into()
        .expect("slice length checked");
    let tag = encrypted_key[encrypted_key.len() - TAG_LEN..]
        .try_into()
        .expect("slice length checked");
    let mut plaintext = encrypted_key[NONCE_LEN..encrypted_key.len() - TAG_LEN].to_vec();
    cipher.decrypt(nonce, &mut plaintext, tag, aad)?;
    Ok(plaintext)
}

struct CtrCipher {
    key: Zeroizing<Vec<u8>>,
}

impl std::fmt::Debug for CtrCipher {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CtrCipher")
            .field("key", &"<redacted>")
            .field("key_length", &self.key.len())
            .finish()
    }
}

impl CtrCipher {
    fn new(key_bytes: &[u8]) -> ParquetResult<Self> {
        match key_bytes.len() {
            16 | 24 | 32 => Ok(Self {
                key: Zeroizing::new(key_bytes.to_vec()),
            }),
            length => Err(ParquetError::InvalidParameter(format!(
                "unsupported AES-CTR key length: {length} bytes"
            ))),
        }
    }

    fn apply(&self, nonce: &[u8; NONCE_LEN], buffer: &mut [u8]) -> ParquetResult<()> {
        let mut iv = [0; 16];
        iv[..NONCE_LEN].copy_from_slice(nonce);
        iv[15] = 1;

        match self.key.len() {
            16 => Aes128Ctr::new_from_slices(&self.key, &iv)
                .unwrap()
                .try_apply_keystream(buffer),
            24 => Aes192Ctr::new_from_slices(&self.key, &iv)
                .unwrap()
                .try_apply_keystream(buffer),
            32 => Aes256Ctr::new_from_slices(&self.key, &iv)
                .unwrap()
                .try_apply_keystream(buffer),
            _ => unreachable!("key length validated in constructor"),
        }
        .map_err(|_| ParquetError::oos("AES-CTR page is too large"))
    }
}

#[derive(Debug)]
pub(crate) struct AesCtrBlockDecryptor {
    cipher: CtrCipher,
}

impl AesCtrBlockDecryptor {
    pub(crate) fn new(key_bytes: &[u8]) -> ParquetResult<Self> {
        Ok(Self {
            cipher: CtrCipher::new(key_bytes)?,
        })
    }
}

impl BlockDecryptor for AesCtrBlockDecryptor {
    fn decrypt(&self, length_and_ciphertext: &[u8], _aad: &[u8]) -> ParquetResult<Vec<u8>> {
        if length_and_ciphertext.len() < SIZE_LEN + NONCE_LEN {
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

        let nonce: &[u8; NONCE_LEN] = length_and_ciphertext[SIZE_LEN..SIZE_LEN + NONCE_LEN]
            .try_into()
            .expect("slice length checked");
        let mut result = length_and_ciphertext[SIZE_LEN + NONCE_LEN..].to_vec();
        self.cipher.apply(nonce, &mut result)?;
        Ok(result)
    }

    fn compute_plaintext_tag(&self, _aad: &[u8], _plaintext: &[u8]) -> ParquetResult<Vec<u8>> {
        Err(ParquetError::InvalidParameter(
            "AES-CTR cannot authenticate a plaintext footer".to_string(),
        ))
    }
}

#[derive(Debug)]
pub(crate) struct AesCtrBlockEncryptor {
    cipher: CtrCipher,
    invocation_counter: Arc<CipherInvocationCounter>,
}

impl AesCtrBlockEncryptor {
    #[cfg(test)]
    pub(crate) fn new(key_bytes: &[u8]) -> ParquetResult<Self> {
        Self::new_with_counter(key_bytes, Arc::default())
    }

    pub(crate) fn new_with_counter(
        key_bytes: &[u8],
        invocation_counter: Arc<CipherInvocationCounter>,
    ) -> ParquetResult<Self> {
        Ok(Self {
            cipher: CtrCipher::new(key_bytes)?,
            invocation_counter,
        })
    }
}

impl BlockEncryptor for AesCtrBlockEncryptor {
    fn encrypt(&mut self, plaintext: &[u8], _aad: &[u8]) -> ParquetResult<Vec<u8>> {
        let ciphertext_length: u32 = (NONCE_LEN + plaintext.len())
            .try_into()
            .map_err(|err| ParquetError::oos(format!("plaintext module is too long: {err}")))?;
        let nonce = self.invocation_counter.next_nonce()?;
        let mut encrypted_plaintext = plaintext.to_vec();
        self.cipher.apply(&nonce, &mut encrypted_plaintext)?;

        let mut ciphertext = Vec::with_capacity(SIZE_LEN + ciphertext_length as usize);
        ciphertext.extend(ciphertext_length.to_le_bytes());
        ciphertext.extend(nonce);
        ciphertext.extend(encrypted_plaintext);
        Ok(ciphertext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn round_trip(key: &[u8]) {
        let mut encryptor = AesGcmBlockEncryptor::new(key).unwrap();
        let decryptor = AesGcmBlockDecryptor::new(key).unwrap();
        let ciphertext = encryptor.encrypt(b"hello", b"aad").unwrap();
        assert_eq!(decryptor.decrypt(&ciphertext, b"aad").unwrap(), b"hello");
    }

    #[test]
    fn aes_gcm_round_trip() {
        round_trip(b"0123456789012345");
    }

    #[test]
    fn aes_gcm_192_round_trip() {
        round_trip(b"012345678901234567890123");
    }

    #[test]
    fn aes_gcm_256_round_trip() {
        round_trip(b"01234567890123456789012345678901");
    }

    #[test]
    fn aes_gcm_rejects_modified_ciphertext() {
        let key = b"0123456789012345";
        let mut encryptor = AesGcmBlockEncryptor::new(key).unwrap();
        let decryptor = AesGcmBlockDecryptor::new(key).unwrap();
        let mut ciphertext = encryptor.encrypt(b"hello", b"aad").unwrap();
        ciphertext[SIZE_LEN + NONCE_LEN] ^= 1;
        assert!(decryptor.decrypt(&ciphertext, b"aad").is_err());
    }

    #[test]
    fn cipher_invocation_limit_is_enforced() {
        let counter = Arc::new(CipherInvocationCounter::with_count(
            MAX_INVOCATIONS_PER_KEY - 1,
        ));
        let mut encryptor =
            AesGcmBlockEncryptor::new_with_counter(b"0123456789012345", counter).unwrap();
        encryptor.encrypt(b"first", b"aad").unwrap();
        assert!(encryptor.encrypt(b"second", b"aad").is_err());
    }

    fn ctr_round_trip(key: &[u8]) {
        let mut encryptor = AesCtrBlockEncryptor::new(key).unwrap();
        let decryptor = AesCtrBlockDecryptor::new(key).unwrap();
        let ciphertext = encryptor.encrypt(b"hello", b"ignored").unwrap();
        assert_eq!(ciphertext.len(), SIZE_LEN + NONCE_LEN + 5);
        assert_eq!(
            decryptor.decrypt(&ciphertext, b"different-aad").unwrap(),
            b"hello"
        );
    }

    #[test]
    fn aes_ctr_round_trip_for_all_key_lengths() {
        ctr_round_trip(b"0123456789012345");
        ctr_round_trip(b"012345678901234567890123");
        ctr_round_trip(b"01234567890123456789012345678901");
    }

    #[test]
    fn locally_wrapped_key_omits_module_length_prefix() {
        let plaintext = b"0123456789abcdef";
        let wrapping_key = b"abcdef0123456789";
        let encrypted = encrypt_key_locally(plaintext, wrapping_key, b"kek-id").unwrap();
        assert_eq!(encrypted.len(), plaintext.len() + NONCE_LEN + TAG_LEN);
        assert_eq!(
            decrypt_key_locally(&encrypted, wrapping_key, b"kek-id").unwrap(),
            plaintext
        );
    }
}
