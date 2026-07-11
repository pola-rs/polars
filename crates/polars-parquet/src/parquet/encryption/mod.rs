// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this file
// to you under the Apache License, Version 2.0.

//! Parquet modular encryption support.

/// Cipher suite used by Parquet modular encryption.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
pub enum ParquetEncryptionAlgorithm {
    /// Encrypt every module with authenticated AES-GCM.
    #[default]
    AesGcmV1,
    /// Encrypt page bodies with AES-CTR and every other module with AES-GCM.
    AesGcmCtrV1,
}

pub(crate) mod ciphers;
pub mod decrypt;
pub mod encrypt;
pub(crate) mod modules;

pub use decrypt::{DecryptionPropertiesBuilder, FileDecryptionProperties, KeyRetriever};
pub use encrypt::{EncryptionPropertiesBuilder, FileEncryptionProperties};

pub fn encrypt_key_locally(
    plaintext_key: &[u8],
    wrapping_key: &[u8],
    aad: &[u8],
) -> crate::parquet::error::ParquetResult<Vec<u8>> {
    ciphers::encrypt_key_locally(plaintext_key, wrapping_key, aad)
}

pub fn decrypt_key_locally(
    encrypted_key: &[u8],
    wrapping_key: &[u8],
    aad: &[u8],
) -> crate::parquet::error::ParquetResult<Vec<u8>> {
    ciphers::decrypt_key_locally(encrypted_key, wrapping_key, aad)
}
