use std::fmt;
use std::usize;

// This compression algorithm works by taknig a chunk of bytes that all share some bit(s) in
// common, making a tritmask that contains a 0 for bits that are always 0, 1 for bits that
// are always one and 2 for bits that are not in common
// 10000010
// 10100010
// 10110010
// 10000010
// All share the first, second and last four bits, so we can create a tritmask with the shared bits as 10220010
// and only storing two bits per byte, namely 00, 10, 11, 00.

#[derive(PartialEq, Eq, Copy, Clone, Default)]
pub struct Mask([Option<bool>; 8]);

impl Mask {
    fn amount_non_common(&self) -> usize {
        (&self.0).iter().filter(|x| x.is_none()).count()
    }
}

impl fmt::Debug for Mask {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        write!(fmt, "Mask(")?;
        for bit in &self.0 {
            let ch = match bit {
                &Some(true) => '1',
                &Some(false) => '0',
                &None => '?'
            };
            write!(fmt, "{}", ch)?;
        }
        write!(fmt, ")")
    }
}

#[derive(Debug, PartialEq, Eq, Clone)]
pub struct MaskedChunk {
    mask: Mask, // Some(x) means the bit is always x, None means it's custom
    non_common: Vec<bool>
}

impl MaskedChunk {
    pub fn create_with_mask(src: &[u8], mut mask: Mask) -> MaskedChunk {
        let mut all_non_common = vec![];

        if mask.amount_non_common() == 0 {
            mask.0[0] = None;

        }

        for byte in src {
            let mut non_common = extract_non_common(*byte, &mask);
            all_non_common.append(&mut non_common);
        }

        MaskedChunk { mask: mask, non_common: all_non_common }
    }

    pub fn byte_len(&self) -> usize {
        if self.non_common.len() == 0 {
            return 0;
        }
        self.non_common.len() / self.mask.amount_non_common()
    }

    pub fn into_bytes(self) -> Vec<u8> {
        let mut res = Vec::with_capacity(self.byte_len());

        for noncommon_bits in self.non_common.chunks(self.mask.amount_non_common()) {
            res.push(insert_common(noncommon_bits.to_vec(), &self.mask));
        }

        res
    }


    pub fn encode(mut self) -> Vec<bool> {
        let mut res = encode_nonzero(self.byte_len());

        for is_common in &self.mask.0 {
            match is_common {
                &Some(ref x) => res.append(&mut vec![true, *x]),
                &None => res.push(false)
            }
        }

        res.append(&mut self.non_common);

        res
    }

    pub fn decode(bytes: Vec<bool>) -> Option<(MaskedChunk, Vec<bool>)> {
        let (amount_of_non_common, mut bytes) = decode_nonzero(bytes)?;

        let mut mask = Mask::default();

        for i in 0..8 {
            if bytes.remove(0) {
                mask.0[i] = Some(bytes.remove(0));
            }
        }

        let mut non_common = vec![];

        for _ in 0..amount_of_non_common * mask.amount_non_common() {
            non_common.push(bytes.remove(0));
        }

        Some((MaskedChunk { mask: mask, non_common: non_common }, bytes))
    }
}

pub fn encode_nonzero(num: usize) -> Vec<bool> {
    let num_len = usize::min_value().leading_zeros() - num.leading_zeros();

    let mut res: Vec<bool> = Vec::with_capacity(2 * (num_len as usize));

    for _ in 0..num_len-1 {
        res.push(false);
    }

    for i in (0..num_len).rev() {
        res.push((num >> i) & 1 == 1);
    }

    res
}

pub fn decode_nonzero(mut bytes: Vec<bool>) -> Option<(usize, Vec<bool>)> {
    let mut num_len = 1;
    while bytes.get(0) == Some(&false) {
        num_len += 1;
        bytes.remove(0);
    }
    let mut res = 0;
    let mut write_idx = num_len - 1;

    for _ in 0..num_len {
        let current_bit = bytes.remove(0);
        res |= (current_bit as usize) << write_idx;
        write_idx -= 1;
    }
    Some((res, bytes))
}

pub fn compress_to_bits(src: &[u8]) -> Vec<bool> {
    let chunks = into_chunks(src);
    chunks.into_iter().flat_map(MaskedChunk::encode).collect()
}

// As the bits might not fit exactly into a byte, we have to add some padding at the start.
// The padding consists of one bit on before the list and enough zeros before to make the amount of
// bits divisible by 8.
pub fn pad(mut src: Vec<bool>) -> Vec<bool> {
    src.insert(0, true);

    let amount_to_add = 8 - (src.len()) % 8;
    for _ in 0..amount_to_add % 8 {
        src.insert(0, false);
    }

    src
}

pub fn depad(mut src: Vec<bool>) -> Option<Vec<bool>> {
    while src.get(0) == Some(&false) {
        src.remove(0);
    }
    src.remove(0);
    Some(src)
}


// Covert a list of bits into a list of bytes.
pub fn bits_to_bytes(src: Vec<bool>) -> Vec<u8> {
    assert!(src.len() % 8 == 0);
    src.chunks(8)
        .map(|bits| bits.into_iter().fold(0u8, |num, bit| num * 2 + *bit as u8))
        .collect()
}

pub fn bytes_to_bits(src: Vec<u8>) -> Vec<bool> {
    src.into_iter()
        .flat_map(|x| {
            let mut res = vec![];
            for i in 0..8 {
                res.push((x >> (7 - i) & 1) == 1);
            }
            res
        }).collect()
}

pub fn compress(src: &[u8]) -> Vec<u8> {
    bits_to_bytes(pad(compress_to_bits(src)))
}

pub fn decompress(src: Vec<u8>) -> Option<Vec<u8>> {
    let compressed_bits = depad(bytes_to_bits(src))?;
    let chunks = bits_to_masked_chunks(compressed_bits)?;
    Some(chunks.into_iter().flat_map(MaskedChunk::into_bytes).collect::<Vec<u8>>())
}

pub fn bits_to_masked_chunks(mut bits: Vec<bool>) -> Option<Vec<MaskedChunk>> {
    let mut chunks = vec![];
    while !bits.is_empty() {
        let (chunk, new_bits) = MaskedChunk::decode(bits)?;
        bits = new_bits;
        chunks.push(chunk);
    }
    Some(chunks)
}

pub fn into_chunks(src: &[u8]) -> Vec<MaskedChunk> {
    let mut chunks: Vec<MaskedChunk> = vec![];

    let mut chunk_start = 0;

    let mut curr_mask: Option<Mask> = None;

    for i in 0..src.len()+1 {
        if let Some(curr) = curr_mask {
            let next_mask =
                if i != src.len() {
                    Some(make_mask(&src[chunk_start..i+1]))
                } else {
                    None
                };

            if next_mask.map(|next| {
                    curr.amount_non_common() as usize * (i - chunk_start) + 8 - curr.amount_non_common() + 8
                  < next.amount_non_common() as usize * (i - chunk_start + 1) + 8 - next.amount_non_common()
            }).unwrap_or(true) {

                let chunk = MaskedChunk::create_with_mask(&src[chunk_start..i], curr);
                chunks.push(chunk);

                chunk_start = i;
                if i < src.len() {
                    curr_mask = Some(make_mask(&[src[i]]));
                }
            } else {
                curr_mask = next_mask;
            }
        } else {
            curr_mask = Some(make_mask(&src[chunk_start..i+1]));
        }
    }

    chunks
}

fn nth_bit(n: u8, bit_nr: u8) -> bool {
    return (n >> (7 - bit_nr)) & 1 == 1;
}

fn extract_non_common(inp: u8, mask: &Mask) -> Vec<bool> {
    let mut res: Vec<bool> = Vec::new();

    for x in 0..8 {
        if mask.0[x].is_none() {
            res.push(nth_bit(inp, x as u8));
        }
    }

    res
}

fn insert_common(inp: Vec<bool>, mask: &Mask) -> u8 {
    let mut res = 0;

    let mut inp_idx = 0;

    for i in 0..8 {
        let bit = match mask.0[i] {
            Some(true) => 1,
            Some(false) => 0,
            None => { inp_idx += 1; inp[inp_idx - 1] as u8 }
        };

        res |= bit << (7 - i);
    }

    res
}

pub fn make_mask(src: &[u8]) -> Mask {
    let mut mask: Mask = Mask([None; 8]);
    for x in 0..8 {
        mask.0[x] = Some(nth_bit(src[0], x as u8));
    }

    for byte in src {
        for x in 0..8 {
            if let Some(bit) = mask.0[x] {
                if nth_bit(*byte, x as u8) != bit {
                    mask.0[x] = None;
                }
            }
        }
    }

    mask
}

#[test]
fn test_maskedchunk_create_with_mask() {
    assert_eq!(
        MaskedChunk {
            mask: Mask([Some(true), Some(false), None, None, Some(false), Some(false), Some(true), Some(false)]),
            non_common: vec![
                false, false,
                true,  false,
                true,  true,
                false, false
            ]
        },
        MaskedChunk::create_with_mask(&[
                   0b10000010,
                   0b10100010,
                   0b10110010,
                   0b10000010,
        ], Mask([Some(true), Some(false), None, None, Some(false), Some(false), Some(true), Some(false)]))
        );
}

#[test]
fn test_extract() {
    assert_eq!(
        vec![true, true],
        extract_non_common(0b10110010,
                           &Mask(
                               [Some(true), Some(false), None, None, Some(false), Some(false), Some(true), Some(false)]
                               )
                           )
              );
    assert_eq!(
        vec![false, true, true],
        extract_non_common(0b00110010,
                           &Mask(
                               [None, Some(false), None, None, Some(false), Some(false), Some(true), Some(false)]
                               )
                           )
              );
}

#[test]
fn test_insert_common() {
    assert_eq!(
        0b10110010,
        insert_common(vec![true, true],
                           &Mask(
                               [Some(true), Some(false), None, None, Some(false), Some(false), Some(true), Some(false)]
                               )
                           )
              );
}

#[test]
fn test_maskedchunk_into_bytes() {
    assert_eq!(
        vec![0b101, 0b111, 0b101],
        MaskedChunk {
            mask: Mask([Some(false), Some(false), Some(false), Some(false), Some(false), Some(true), None, Some(true)]),
            non_common: vec![false, true, false]
        }.into_bytes()
        )
}

#[test]
fn test_encdec_nonzero() {
    assert_eq!(Some((420, vec![])), decode_nonzero(encode_nonzero(420)));
    assert_eq!(Some((69, vec![])), decode_nonzero(encode_nonzero(69)));
}

#[test]
fn test_maskedchunk_encdec() {
    let masked = MaskedChunk::create(&[
                                     0b10000010,
                                     0b10100010,
                                     0b10110010,
                                     0b10000010,
    ]);

    let mut encoded = masked.clone().encode();

    assert_eq!(
        vec![
            false, false, true, false, false, // len = 4
            true,true,   true,false,   false,   false,   true,false,   true, false,   true, true,   true, false, // mask = 10220010
            false, false,   true, false,   true, true,   false, false // data
        ],
        encoded
        );

    encoded.append(&mut vec![true, true, true]);

    assert_eq!(
        Some((masked, vec![true, true, true])),
        MaskedChunk::decode(encoded)
        )
}

#[test]
fn test_pad() {
    assert_eq!(vec![false, false, true, false, false, true, true, true], pad(vec![false, false, true, true, true]));
    assert_eq!(vec![false, false, false, false, false, false, false, true], pad(vec![]));
    assert_eq!(vec![false, false, false, false, false, false, true, true], pad(vec![true]));
}

#[test]
fn test_bits_to_bytes() {
    assert_eq!(vec![0b01110101], bits_to_bytes(vec![false, true, true, true, false, true, false, true]));
    assert_eq!(vec![0b01110101, 0b00001100],
               bits_to_bytes(vec![
                             false, true, true, true, false, true, false, true,
                             false, false, false, false, true, true, false, false
               ])
               );
}

#[test]
fn test_bytes_to_bits() {
    assert_eq!(vec![true, false, true, true, true, false, true, false], bytes_to_bits(vec![186]));
    assert_eq!(
        vec![
            true, false, true, true, true, false, true, false,
            false, true, false, false, false, true, false, true
        ], bytes_to_bits(vec![186, 69]));
}

#[test]
fn test_depad() {
    assert_eq!(Some(vec![false]), depad(pad(vec![false])));
    assert_eq!(Some(vec![true, false, true]), depad(pad(vec![true, false, true])));
    assert_eq!(Some(vec![false, false, false, false, false, false, false, false]),
                depad(pad(vec![false, false, false, false, false, false, false, false])));
}

