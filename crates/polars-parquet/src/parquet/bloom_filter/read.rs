use std::io::{Cursor, Read, Seek, SeekFrom};

use polars_parquet_format::thrift::protocol::TCompactInputProtocol;
use polars_parquet_format::{
    BloomFilterAlgorithm, BloomFilterCompression, BloomFilterHeader, SplitBlockAlgorithm,
    Uncompressed,
};

use crate::parquet::encryption::ciphers::{NONCE_LEN, TAG_LEN};
use crate::parquet::encryption::decrypt::read_and_decrypt;
use crate::parquet::error::{ParquetError, ParquetResult};
use crate::parquet::metadata::ColumnChunkMetadata;

/// Reads the bloom filter associated to [`ColumnChunkMetadata`] into `bitset`.
/// Results in an empty `bitset` if there is no associated bloom filter or the algorithm is not supported.
/// # Error
/// Errors if the column contains no metadata or the filter can't be read or deserialized.
pub fn read<R: Read + Seek>(
    column_metadata: &ColumnChunkMetadata,
    mut reader: &mut R,
    bitset: &mut Vec<u8>,
) -> ParquetResult<()> {
    let offset = if let Some(offset) = column_metadata.bloom_filter_offset() {
        offset as u64
    } else {
        bitset.clear();
        return Ok(());
    };
    reader.seek(SeekFrom::Start(offset))?;

    let crypto_context = column_metadata.crypto_context();
    let max_filter_length = column_metadata
        .bloom_filter_length()
        .map(usize::try_from)
        .transpose()?
        .unwrap_or(128 * 1024 * 1024);

    let header = if let Some(crypto_context) = crypto_context {
        let aad = crypto_context.create_bloom_filter_header_aad()?;
        let encrypted_header = read_and_decrypt(
            crypto_context.metadata_decryptor(),
            &mut reader,
            &aad,
            max_filter_length.min(1024 * 1024 + NONCE_LEN + TAG_LEN),
        )?;
        let mut cursor = Cursor::new(encrypted_header);
        let mut prot = TCompactInputProtocol::new(&mut cursor, 1024 * 1024);
        BloomFilterHeader::read_from_in_protocol(&mut prot)?
    } else {
        let mut prot = TCompactInputProtocol::new(&mut reader, 1024 * 1024);
        BloomFilterHeader::read_from_in_protocol(&mut prot)?
    };

    if header.algorithm != BloomFilterAlgorithm::BLOCK(SplitBlockAlgorithm {}) {
        bitset.clear();
        return Ok(());
    }
    if header.compression != BloomFilterCompression::UNCOMPRESSED(Uncompressed {}) {
        bitset.clear();
        return Ok(());
    }

    let length: usize = header.num_bytes.try_into()?;
    if length > max_filter_length {
        return Err(ParquetError::WouldOverAllocate);
    }

    let decrypted_bitset = if let Some(crypto_context) = crypto_context {
        let consumed: usize = (reader.stream_position()? - offset).try_into()?;
        let max_remaining = max_filter_length.saturating_sub(consumed);
        let aad = crypto_context.create_bloom_filter_bitset_aad()?;
        Some(read_and_decrypt(
            crypto_context.metadata_decryptor(),
            &mut reader,
            &aad,
            max_remaining.min(length.saturating_add(NONCE_LEN).saturating_add(TAG_LEN)),
        )?)
    } else {
        None
    };

    bitset.clear();
    bitset.try_reserve(length)?;
    if let Some(decrypted_bitset) = decrypted_bitset {
        if decrypted_bitset.len() != length {
            return Err(ParquetError::oos(format!(
                "bloom filter bitset length mismatch: declared {length}, actual {}",
                decrypted_bitset.len()
            )));
        }
        bitset.extend_from_slice(&decrypted_bitset);
    } else {
        bitset.resize(length, 0);
        reader.read_exact(bitset)?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use polars_parquet_format::thrift::protocol::TCompactOutputProtocol;
    use polars_parquet_format::{
        BloomFilterHash, ColumnCryptoMetaData, EncryptionWithFooterKey, XxHash,
    };

    use super::*;
    use crate::parquet::compression::Compression;
    use crate::parquet::encryption::ParquetEncryptionAlgorithm;
    use crate::parquet::encryption::decrypt::{
        CryptoContext, FileDecryptionProperties, FileDecryptor,
    };
    use crate::parquet::encryption::encrypt::{
        FileEncryptionProperties, FileEncryptor, encrypt_bytes, encrypted_thrift_object_to_vec,
    };
    use crate::parquet::metadata::{
        ColumnChunkMetadata, ColumnDescriptorRef, CompactColumnChunk, CompactColumnMetaData,
        SchemaDescriptor,
    };

    fn encrypted_bloom_filter() -> (Vec<u8>, ColumnChunkMetadata, Vec<u8>) {
        let key = b"0123456789abcdef".to_vec();
        let encryption_properties = FileEncryptionProperties::builder(key.clone())
            .build()
            .unwrap();
        let file_encryptor = FileEncryptor::new(encryption_properties).unwrap();
        let decryption_properties = FileDecryptionProperties::builder(key).build().unwrap();
        let file_decryptor = FileDecryptor::new(
            &decryption_properties,
            None,
            file_encryptor.aad_file_unique().clone(),
            Vec::new(),
            ParquetEncryptionAlgorithm::AesGcmV1,
        )
        .unwrap();
        let crypto_context = CryptoContext::for_column(
            &file_decryptor,
            &ColumnCryptoMetaData::ENCRYPTIONWITHFOOTERKEY(EncryptionWithFooterKey {}),
            0,
            0,
        )
        .unwrap();
        let mut module_encryptor = file_encryptor.get_column_metadata_encryptor("a").unwrap();
        let header = BloomFilterHeader {
            num_bytes: 32,
            algorithm: BloomFilterAlgorithm::BLOCK(SplitBlockAlgorithm {}),
            hash: BloomFilterHash::XXHASH(XxHash {}),
            compression: BloomFilterCompression::UNCOMPRESSED(Uncompressed {}),
        };
        let encrypted_header = encrypted_thrift_object_to_vec(
            &mut *module_encryptor,
            &crypto_context.create_bloom_filter_header_aad().unwrap(),
            |protocol: &mut TCompactOutputProtocol<&mut Vec<u8>>| {
                header.write_to_out_protocol(protocol)
            },
        )
        .unwrap();
        let expected_bitset = (0..32).collect::<Vec<u8>>();
        let encrypted_bitset = encrypt_bytes(
            &expected_bitset,
            &mut *module_encryptor,
            &crypto_context.create_bloom_filter_bitset_aad().unwrap(),
        )
        .unwrap();
        let mut bytes = encrypted_header;
        bytes.extend(encrypted_bitset);

        let schema =
            SchemaDescriptor::try_from_message("message schema { REQUIRED INT32 a; }").unwrap();
        let column_chunk = CompactColumnChunk {
            meta_data: CompactColumnMetaData {
                codec: Compression::Uncompressed,
                num_values: 0,
                total_uncompressed_size: 0,
                total_compressed_size: 0,
                data_page_offset: 0,
                index_page_offset: None,
                dictionary_page_offset: None,
                statistics: None,
                bloom_filter_offset: Some(0),
                bloom_filter_length: Some(bytes.len() as i32),
            },
            offset_index_offset: None,
            offset_index_length: None,
            column_index_offset: None,
            column_index_length: None,
            crypto_metadata: None,
        };
        let column = ColumnChunkMetadata::from_compact_with_crypto(
            ColumnDescriptorRef::new(Arc::clone(schema.columns_arc()), 0),
            column_chunk,
            Some(crypto_context),
        );
        (bytes, column, expected_bitset)
    }

    #[test]
    fn encrypted_bloom_filter_round_trip() {
        let (bytes, column, expected_bitset) = encrypted_bloom_filter();
        let mut actual = Vec::new();
        read(&column, &mut Cursor::new(bytes), &mut actual).unwrap();
        assert_eq!(actual, expected_bitset);
    }

    #[test]
    fn encrypted_bloom_filter_rejects_tampering() {
        let (mut bytes, column, _) = encrypted_bloom_filter();
        let last = bytes.len() - 1;
        bytes[last] ^= 1;
        assert!(read(&column, &mut Cursor::new(bytes), &mut Vec::new()).is_err());
    }
}
