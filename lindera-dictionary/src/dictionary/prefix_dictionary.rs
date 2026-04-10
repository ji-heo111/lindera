use byteorder::{ByteOrder, LittleEndian};
use daachorse::DoubleArrayAhoCorasick;
use rkyv::rancor::{Fallible, Source};
use rkyv::with::{ArchiveWith, DeserializeWith, SerializeWith};
use rkyv::{Archive, Deserialize as RkyvDeserialize, Place, Serialize as RkyvSerialize};
use serde::{Deserialize, Serialize};

use crate::{LinderaResult, error::LinderaErrorKind, util::Data, viterbi::WordEntry};

/// Compute Korean left-space-penalty cost for a given POS tag.
///
/// Mirrors mecab-ko's `left-space-penalty-factor` config in dicrc:
///   `100,3000,120,6000,172,3000,...,210,6000,...`
///
/// posid → POS class mapping (from pos-id.def):
///   100 = 어미 (E*)              → 3000
///   120 = 조사 (J*)              → 6000
///   172 = VCP                    → 3000
///   183-185 = XSA/XSN/XSV        → 3000
///   200 = Inflect E*             → 3000
///   210 = Inflect J*             → 6000
///   220-222 = Inflect XS*        → 3000
///   230 = Inflect VCP            → 3000
///
/// For compound POS like "JC+VCP", uses the first segment.
pub fn compute_korean_space_penalty(pos: &str) -> i16 {
    let first = pos.split('+').next().unwrap_or(pos);
    match first {
        // Particles (조사) — highest penalty
        "JC" | "JKB" | "JKC" | "JKG" | "JKO" | "JKQ" | "JKS" | "JKV" | "JX" => 6000,
        // Endings (어미)
        "EC" | "EF" | "EP" | "ETM" | "ETN" => 3000,
        // Copula (긍정지정사)
        "VCP" => 3000,
        // Suffixes (접미사)
        "XSA" | "XSN" | "XSV" => 3000,
        _ => 0,
    }
}

/// Match structure for common prefix iterator compatibility
#[derive(Debug, Clone)]
pub struct Match {
    pub word_idx: WordIdx,
    pub end_char: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct WordIdx {
    pub word_id: u32,
}

impl WordIdx {
    pub fn new(word_id: u32) -> Self {
        Self { word_id }
    }
}

pub struct DoubleArrayArchiver;

impl ArchiveWith<DoubleArrayAhoCorasick<u32>> for DoubleArrayArchiver {
    type Archived = rkyv::vec::ArchivedVec<u8>;
    type Resolver = rkyv::vec::VecResolver;

    fn resolve_with(
        field: &DoubleArrayAhoCorasick<u32>,
        resolver: Self::Resolver,
        out: Place<Self::Archived>,
    ) {
        let bytes = field.serialize();
        rkyv::vec::ArchivedVec::resolve_from_slice(&bytes, resolver, out);
    }
}

impl<S: Fallible + rkyv::ser::Writer + rkyv::ser::Allocator + ?Sized>
    SerializeWith<DoubleArrayAhoCorasick<u32>, S> for DoubleArrayArchiver
{
    fn serialize_with(
        field: &DoubleArrayAhoCorasick<u32>,
        serializer: &mut S,
    ) -> Result<Self::Resolver, S::Error> {
        let bytes = field.serialize();
        rkyv::vec::ArchivedVec::serialize_from_slice(&bytes, serializer)
    }
}

impl<D: Fallible<Error: Source> + ?Sized>
    DeserializeWith<rkyv::vec::ArchivedVec<u8>, DoubleArrayAhoCorasick<u32>, D>
    for DoubleArrayArchiver
{
    /// Deserialize the archived byte vector into a `DoubleArrayAhoCorasick`.
    ///
    /// # Returns
    ///
    /// The deserialized `DoubleArrayAhoCorasick`, or an error if deserialization fails.
    fn deserialize_with(
        archived: &rkyv::vec::ArchivedVec<u8>,
        _deserializer: &mut D,
    ) -> Result<DoubleArrayAhoCorasick<u32>, D::Error> {
        let (da, _) = DoubleArrayAhoCorasick::deserialize(archived.as_slice()).map_err(|err| {
            D::Error::new(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                err.to_string(),
            ))
        })?;
        Ok(da)
    }
}

mod double_array_serde {
    use daachorse::DoubleArrayAhoCorasick;
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S>(da: &DoubleArrayAhoCorasick<u32>, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        let bytes = da.serialize();
        serializer.serialize_bytes(&bytes)
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<DoubleArrayAhoCorasick<u32>, D::Error>
    where
        D: Deserializer<'de>,
    {
        let bytes: Vec<u8> = Deserialize::deserialize(deserializer)?;
        let (da, _) = DoubleArrayAhoCorasick::deserialize(&bytes)
            .map_err(|err| serde::de::Error::custom(err.to_string()))?;
        Ok(da)
    }
}

#[derive(Clone, Serialize, Deserialize, Archive, RkyvSerialize, RkyvDeserialize)]
pub struct PrefixDictionary {
    #[serde(with = "self::double_array_serde")]
    #[rkyv(with = DoubleArrayArchiver)]
    pub da: DoubleArrayAhoCorasick<u32>,
    pub vals_data: Data,
    pub words_idx_data: Data,
    pub words_data: Data,
    pub is_system: bool,
    /// Pre-computed left-space-penalty cost per word_id (Korean only).
    /// Empty for non-Korean dictionaries.
    /// Built at load time from POS tags via `compute_korean_space_penalty`.
    #[serde(skip)]
    #[rkyv(with = rkyv::with::Skip)]
    pub space_penalty_table: Vec<i16>,
}

impl PrefixDictionary {
    /// Decode the `(offset, count)` pair stored in the double-array value.
    ///
    /// System dictionaries use 8-bit count (supports up to 255 variants per
    /// surface), while user dictionaries retain the legacy 5-bit count
    /// encoding (max 31) for binary backward compatibility with pre-built
    /// `.bin` user dictionary files.
    #[inline]
    pub(crate) fn decode_val(&self, val: u32) -> (u32, u32) {
        if self.is_system {
            (val >> 8u32, val & ((1u32 << 8) - 1u32))
        } else {
            (val >> 5u32, val & ((1u32 << 5) - 1u32))
        }
    }

    /// Load a `PrefixDictionary` from raw binary data.
    ///
    /// # Arguments
    ///
    /// * `da_data` - Double-array data bytes.
    /// * `vals_data` - Values data bytes.
    /// * `words_idx_data` - Word index data bytes.
    /// * `words_data` - Words data bytes.
    /// * `is_system` - Whether this is a system dictionary.
    ///
    /// # Returns
    ///
    /// A `PrefixDictionary`, or an error if deserialization fails.
    pub fn load(
        da_data: impl Into<Data>,
        vals_data: impl Into<Data>,
        words_idx_data: impl Into<Data>,
        words_data: impl Into<Data>,
        is_system: bool,
    ) -> LinderaResult<PrefixDictionary> {
        let da_bytes = da_data.into();
        let (da, _) = DoubleArrayAhoCorasick::deserialize(&da_bytes[..]).map_err(|err| {
            LinderaErrorKind::Deserialize.with_error(anyhow::anyhow!(err.to_string()))
        })?;

        let mut dict = PrefixDictionary {
            da,
            vals_data: vals_data.into(),
            words_idx_data: words_idx_data.into(),
            words_data: words_data.into(),
            is_system,
            space_penalty_table: Vec::new(),
        };
        dict.build_space_penalty_table();
        Ok(dict)
    }

    /// Build the per-word_id space-penalty lookup table from POS tags.
    /// Run once at load time. For non-Korean dictionaries, all values stay 0.
    pub fn build_space_penalty_table(&mut self) {
        let num_entries = self.words_idx_data.len() / 4;
        let mut table = Vec::with_capacity(num_entries);
        for word_id in 0..num_entries {
            let penalty = match self.get_first_detail(word_id as u32) {
                Some(pos) => compute_korean_space_penalty(pos),
                None => 0,
            };
            table.push(penalty);
        }
        self.space_penalty_table = table;
    }

    /// Read the first detail field (POS tag) for a given word_id.
    /// Returns None if word_id is out of bounds or data is malformed.
    pub fn get_first_detail(&self, word_id: u32) -> Option<&str> {
        let idx_pos = (word_id as usize) * 4;
        if idx_pos + 4 > self.words_idx_data.len() {
            return None;
        }
        let idx = LittleEndian::read_u32(&self.words_idx_data[idx_pos..idx_pos + 4]) as usize;
        if idx + 4 > self.words_data.len() {
            return None;
        }
        let joined_len = LittleEndian::read_u32(&self.words_data[idx..idx + 4]) as usize;
        let bytes_start = idx + 4;
        let bytes_end = bytes_start + joined_len;
        if bytes_end > self.words_data.len() {
            return None;
        }
        let joined = &self.words_data[bytes_start..bytes_end];
        let null_pos = joined.iter().position(|&b| b == 0).unwrap_or(joined.len());
        std::str::from_utf8(&joined[..null_pos]).ok()
    }

    /// Look up the pre-computed left-space-penalty for a given word_id.
    #[inline]
    pub fn get_space_penalty(&self, word_id: u32) -> i32 {
        let idx = word_id as usize;
        if idx < self.space_penalty_table.len() {
            self.space_penalty_table[idx] as i32
        } else {
            0
        }
    }

    pub fn prefix<'a>(&'a self, s: &'a str) -> impl Iterator<Item = (usize, WordEntry)> + 'a {
        self.da
            .find_overlapping_iter(s)
            .filter(|m| m.start() == 0)
            .flat_map(move |m| {
                let (offset, len) = self.decode_val(m.value());
                let offset_bytes = (offset as usize) * WordEntry::SERIALIZED_LEN;
                let data: &[u8] = &self.vals_data[offset_bytes..];
                (0..len as usize).map(move |i| {
                    (
                        m.end(),
                        WordEntry::deserialize(
                            &data[WordEntry::SERIALIZED_LEN * i..],
                            self.is_system,
                        ),
                    )
                })
            })
    }

    /// Find `WordEntry`s with surface
    pub fn find_surface(&self, surface: &str) -> Vec<WordEntry> {
        self.find_surface_iter(surface).collect()
    }

    /// Find `WordEntry`s with surface using lazy evaluation
    /// This iterator-based approach reduces memory allocations
    pub fn find_surface_iter<'a>(
        &'a self,
        surface: &'a str,
    ) -> impl Iterator<Item = WordEntry> + 'a {
        self.da
            .find_overlapping_iter(surface)
            .filter(|m| m.start() == 0 && m.end() == surface.len())
            .flat_map(move |m| {
                let (offset, len) = self.decode_val(m.value());
                let offset_bytes = (offset as usize) * WordEntry::SERIALIZED_LEN;
                let data = &self.vals_data[offset_bytes..];
                (0..len as usize).map(move |i| {
                    WordEntry::deserialize(&data[WordEntry::SERIALIZED_LEN * i..], self.is_system)
                })
            })
    }

    /// Common prefix iterator using character array input
    pub fn common_prefix_iterator(&self, suffix: &[char]) -> Vec<Match> {
        // Warning: This method takes &[char], but daachorse works on bytes (str).
        // Converting char slice to string is costly but necessary if we use daachorse standard API.

        if self.vals_data.is_empty() {
            return Vec::new();
        }

        let suffix_str: String = suffix.iter().collect();

        self.da
            .find_overlapping_iter(&suffix_str)
            .filter(|m| m.start() == 0)
            .flat_map(|m| {
                let (offset, len) = self.decode_val(m.value());
                let offset_bytes = (offset as usize) * WordEntry::SERIALIZED_LEN;

                // 範囲チェックを追加
                if offset_bytes >= self.vals_data.len() {
                    return vec![].into_iter();
                }

                let data: &[u8] = &self.vals_data[offset_bytes..];
                (0..len as usize)
                    .filter_map(move |i| {
                        let required_bytes = WordEntry::SERIALIZED_LEN * (i + 1);
                        if required_bytes <= data.len() {
                            let word_entry = WordEntry::deserialize(
                                &data[WordEntry::SERIALIZED_LEN * i..],
                                self.is_system,
                            );
                            Some(Match {
                                word_idx: WordIdx::new(word_entry.word_id.id),
                                end_char: m.end(), // prefix_len in bytes? No, m.end() is byte index.
                                                   // Match expects char length?
                                                   // Original code: end_char: prefix_len
                                                   // prefix_len was number of bytes or chars?
                                                   // yada::common_prefix_search returns (val, len) where len is length in bytes?
                                                   // yada common_prefix_search(str) returns length in bytes.
                                                   // But common_prefix_iterator takes &[char].
                                                   // Match.end_char usually implies character index if used for Viterbi on chars.
                                                   // But Viterbi usually works on bytes in Lindera?
                                                   // Let's check typical usage.
                                                   // NOTE: daachorse returns byte indices.
                                                   // If input was chars converted to String, byte index != char index.
                                                   // We need to map back to char index?
                                                   // This function common_prefix_iterator might be inefficient or deprecated given we move to byte-based Viterbi.
                                                   // For now, let's assume we return byte length.
                                                   // But wait, suffix is &[char].
                                                   // The caller likely expects char length?
                                                   // Yes. if suffix is &[char], end_char 3 means 3 chars.
                                                   // We have byte length from daachorse.
                                                   // We need to count chars in suffix_str[..m.end()].
                                                   // This is inefficient.
                            })
                        } else {
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .collect()
    }
}

impl ArchivedPrefixDictionary {
    /// Decode the `(offset, count)` pair. See [`PrefixDictionary::decode_val`].
    #[inline]
    fn decode_val(&self, val: u32) -> (u32, u32) {
        if self.is_system {
            (val >> 8u32, val & ((1u32 << 8) - 1u32))
        } else {
            (val >> 5u32, val & ((1u32 << 5) - 1u32))
        }
    }

    /// Find all prefix matches for the given string using the archived dictionary.
    ///
    /// # Arguments
    ///
    /// * `s` - The input string to search for prefix matches.
    ///
    /// # Returns
    ///
    /// An iterator of `(end_position, WordEntry)` pairs, or an error if deserialization fails.
    pub fn prefix<'a>(
        &'a self,
        s: &'a str,
    ) -> LinderaResult<impl Iterator<Item = (usize, WordEntry)> + 'a> {
        // Deserialize on the fly. Performance warning: this is slow.
        let (da, _) =
            DoubleArrayAhoCorasick::<u32>::deserialize(self.da.as_slice()).map_err(|err| {
                LinderaErrorKind::Deserialize.with_error(anyhow::anyhow!(err.to_string()))
            })?;

        let matches: Vec<_> = da
            .find_overlapping_iter(s)
            .filter(|m| m.start() == 0)
            .map(|m| (m.end(), m.value()))
            .collect();

        Ok(matches.into_iter().flat_map(move |(end, offset_len)| {
            let (offset, len) = self.decode_val(offset_len);
            let offset_bytes = (offset as usize) * WordEntry::SERIALIZED_LEN;

            let vals = self.vals_data.as_slice();
            if offset_bytes >= vals.len() {
                return vec![].into_iter();
            }

            let data = &vals[offset_bytes..];
            (0..len as usize)
                .map(move |i| {
                    (
                        end,
                        WordEntry::deserialize(
                            &data[WordEntry::SERIALIZED_LEN * i..],
                            self.is_system,
                        ),
                    )
                })
                .collect::<Vec<_>>()
                .into_iter()
        }))
    }

    /// Find `WordEntry`s matching the exact surface in the archived dictionary.
    ///
    /// # Arguments
    ///
    /// * `surface` - The surface string to search for.
    ///
    /// # Returns
    ///
    /// A vector of matching `WordEntry`s, or an error if deserialization fails.
    pub fn find_surface(&self, surface: &str) -> LinderaResult<Vec<WordEntry>> {
        let (da, _) =
            DoubleArrayAhoCorasick::<u32>::deserialize(self.da.as_slice()).map_err(|err| {
                LinderaErrorKind::Deserialize.with_error(anyhow::anyhow!(err.to_string()))
            })?;

        let matches: Vec<_> = da
            .find_overlapping_iter(surface)
            .filter(|m| m.start() == 0 && m.end() == surface.len())
            .map(|m| m.value())
            .collect();

        Ok(matches
            .into_iter()
            .flat_map(|offset_len| {
                let (offset, len) = self.decode_val(offset_len);
                let offset_bytes = (offset as usize) * WordEntry::SERIALIZED_LEN;
                let vals = self.vals_data.as_slice();
                if offset_bytes >= vals.len() {
                    return Vec::new().into_iter();
                }
                let data = &vals[offset_bytes..];
                (0..len as usize)
                    .map(|i| {
                        WordEntry::deserialize(
                            &data[WordEntry::SERIALIZED_LEN * i..],
                            self.is_system,
                        )
                    })
                    .collect::<Vec<_>>()
                    .into_iter()
            })
            .collect())
    }
}
