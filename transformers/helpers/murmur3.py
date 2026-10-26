UINT32 = 0xFFFFFFFF
MURMUR_C1 = 0xCC9E2D51  # 3432918353
MURMUR_C2 = 0x1B873593  # 461845907


def rotl32(x, bits):
    x &= UINT32
    return ((x << bits) | (x >> (32 - bits))) & UINT32


def imul(a, b):
    return (a * b) & UINT32


def _to_int32(x):
    x &= UINT32
    return x - (1 << 32) if x & 0x80000000 else x


def murmur3_x86_32_bytes(
    byte_sequence, length=None, seed=None, mix_block=True
):
    """Core MurmurHash3 (x86_32) operating over a sequence of byte values (0-255).

    Returns a signed 32-bit integer.
    """
    if length is None:
        length = len(byte_sequence)
    if seed is None:
        seed = length

    h = seed
    body_end = length & ~3

    for i in range(0, body_end, 4):
        k = (
            byte_sequence[i]
            | (byte_sequence[i + 1] << 8)
            | (byte_sequence[i + 2] << 16)
            | (byte_sequence[i + 3] << 24)
        )
        k = imul(rotl32(imul(k, MURMUR_C1), 15), MURMUR_C2)
        h ^= k
        if mix_block:
            h = (rotl32(h, 13) * 5 + 0xE6546B64) & UINT32

    tail = length & 3
    if tail:
        k = 0
        if tail == 3:
            k |= byte_sequence[body_end + 2] << 16
        if tail >= 2:
            k |= byte_sequence[body_end + 1] << 8
        k |= byte_sequence[body_end]
        h ^= imul(rotl32(imul(k, MURMUR_C1), 15), MURMUR_C2)

    # Finalization mix
    h ^= length
    h ^= h >> 16
    h = imul(h, 0x85EBCA6B)  # 2246822507
    h ^= h >> 13
    h = imul(h, 0xC2B2AE35)  # 3266489909
    h ^= h >> 16

    return _to_int32(h)
