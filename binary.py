# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

ALIGN4_MASK = 3
BITS_PER_WORD = 32
WORD_SHIFT = 5
WORD_BIT_MASK = BITS_PER_WORD - 1
U32_WORD_MASK = 0xFFFFFFFF

def align4(value):
    return (value + ALIGN4_MASK) & ~ALIGN4_MASK

class FastBitStream:
    __slots__ = ('words', 'nwords', 'total', 'base', 'bit_offset')
    def __init__(self, words_array, base_word, word_count, total_words):
        self.words = words_array
        self.nwords = word_count
        self.total = total_words
        self.base = base_word
        self.bit_offset = 0

    def read_bits(self, count):
        off = self.bit_offset
        word_offset = off >> WORD_SHIFT
        self.bit_offset = off + count
        if word_offset >= self.nwords:
            return 0
        bit_shift = off & WORD_BIT_MASK
        wi = self.base + word_offset
        total = self.total
        low = int(self.words[wi]) if wi < total else 0
        high = int(self.words[wi + 1]) if (wi + 1) < total else 0
        return ((high << BITS_PER_WORD) | low) >> bit_shift & ((1 << count) - 1)

class BitStreamWriter:
    __slots__ = ('words', 'bit_pos')

    def __init__(self):
        self.words = []
        self.bit_pos = 0

    def reset_frame(self):
        if self.bit_pos % BITS_PER_WORD != 0:
            self.bit_pos = (self.bit_pos + WORD_BIT_MASK) & ~WORD_BIT_MASK

    def write_bits(self, value, count):
        if count == 0:
            return
        value &= (1 << count) - 1
        while count > 0:
            word_idx = self.bit_pos >> WORD_SHIFT
            bit_off = self.bit_pos & WORD_BIT_MASK
            while word_idx >= len(self.words):
                self.words.append(0)
            space = BITS_PER_WORD - bit_off
            take = min(space, count)
            chunk = value & ((1 << take) - 1)
            self.words[word_idx] |= (chunk << bit_off) & U32_WORD_MASK
            value >>= take
            count -= take
            self.bit_pos += take
