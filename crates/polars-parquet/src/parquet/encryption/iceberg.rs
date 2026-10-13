use super::AesGcmDecryptor;
use super::ciphers::{NONCE_LEN, TAG_LEN};
use crate::parquet::error::ParquetResult;

/// Decrypt an Iceberg AGS1 stream into its original Avro or Puffin bytes.
pub fn decrypt_stream(input: &[u8], key: &[u8], aad_prefix: &[u8]) -> ParquetResult<Vec<u8>> {
    const BLOCK_SIZE: usize = 1024 * 1024;
    const HEADER_SIZE: usize = 8;
    const CIPHER_BLOCK_SIZE: usize = BLOCK_SIZE + NONCE_LEN + TAG_LEN;

    if input.len() < HEADER_SIZE + NONCE_LEN + TAG_LEN
        || &input[..4] != b"AGS1"
        || input[4..8] != (BLOCK_SIZE as u32).to_le_bytes()
    {
        return Err(encryption_err!(
            "Invalid Iceberg AGS1 header or truncated stream"
        ));
    }
    let decryptor = AesGcmDecryptor::new(key)?;
    let mut output = Vec::with_capacity(input.len() - HEADER_SIZE);
    let mut aad = Vec::with_capacity(aad_prefix.len() + 4);
    aad.extend_from_slice(aad_prefix);
    aad.extend_from_slice(&[0; 4]);
    for (index, block) in input[HEADER_SIZE..].chunks(CIPHER_BLOCK_SIZE).enumerate() {
        let index = u32::try_from(index)
            .map_err(|_| encryption_err!("Iceberg AGS1 block index overflow"))?;
        aad[aad_prefix.len()..].copy_from_slice(&index.to_le_bytes());
        output.extend_from_slice(&decryptor.decrypt_bytes(block, &aad)?);
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use aws_lc_rs::aead::{
        AES_128_GCM, AES_192_GCM, AES_256_GCM, Aad, LessSafeKey, Nonce, UnboundKey,
    };

    use super::*;

    fn encrypt(plaintext: &[u8], key: &[u8], prefix: &[u8]) -> Vec<u8> {
        let algorithm = match key.len() {
            16 => &AES_128_GCM,
            24 => &AES_192_GCM,
            32 => &AES_256_GCM,
            _ => unreachable!(),
        };
        let key = LessSafeKey::new(UnboundKey::new(algorithm, key).unwrap());
        let mut output = b"AGS1".to_vec();
        output.extend_from_slice(&(1024_u32 * 1024).to_le_bytes());
        for (index, chunk) in plaintext.chunks(1024 * 1024).enumerate() {
            let index = (index as u32).to_le_bytes();
            let mut nonce = [0; 12];
            nonce[..4].copy_from_slice(&index);
            let aad = [prefix, &index].concat();
            let mut ciphertext = chunk.to_vec();
            key.seal_in_place_append_tag(
                Nonce::assume_unique_for_key(nonce),
                Aad::from(aad),
                &mut ciphertext,
            )
            .unwrap();
            output.extend_from_slice(&nonce);
            output.extend_from_slice(&ciphertext);
        }
        output
    }

    #[test]
    fn decrypt_multiple_blocks_and_key_sizes() {
        let plaintext: Vec<u8> = (0..1024 * 1024 + 17).map(|i| (i % 251) as u8).collect();
        for size in [16, 24, 32] {
            let key = vec![7; size];
            let encrypted = encrypt(&plaintext, &key, b"file");
            assert_eq!(
                decrypt_stream(&encrypted, &key, b"file").unwrap(),
                plaintext
            );
            assert!(decrypt_stream(&encrypted, &key, b"other").is_err());
            assert!(decrypt_stream(&encrypted, &vec![8; size], b"file").is_err());
            assert!(decrypt_stream(&encrypted[..encrypted.len() - 1], &key, b"file").is_err());
            let mut corrupt = encrypted.clone();
            corrupt[20] ^= 1;
            assert!(decrypt_stream(&corrupt, &key, b"file").is_err());
        }
    }

    #[test]
    fn reject_invalid_headers_and_truncated_blocks() {
        for input in [vec![], b"AGS1".to_vec(), vec![0; 36]] {
            assert!(decrypt_stream(&input, &[0; 16], b"").is_err());
        }
        let mut encrypted = encrypt(b"abc", &[0; 16], b"");
        encrypted[4] ^= 1;
        assert!(decrypt_stream(&encrypted, &[0; 16], b"").is_err());
    }
}
