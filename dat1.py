# This module is part of the public-release split of blender_import_model_anim_release.py.
# Behavior-sensitive logic was moved mechanically; avoid algorithm changes here.

from .utils import *

DAT1_MAGIC = b'1TAD'
DAT1_TEXT_MAGIC = b'DAT1'
DAT1_FILE_ID = 0x44415431
DAT1_HEADER_SIZE = 16
DAT1_BLOCK_TABLE_ENTRY_SIZE = 12
DAT1_FIXUP_TABLE_ENTRY_SIZE = 8


def get_dat1_data(filepath):
    with open(filepath, 'rb') as f:
        file_data = f.read()

    magic_candidates = []
    for magic in (DAT1_MAGIC, DAT1_TEXT_MAGIC):
        idx = file_data.find(magic)
        if idx != -1:
            magic_candidates.append((idx, magic))
    for magic_idx, magic in sorted(magic_candidates, key=lambda item: item[0]):
        if len(file_data) - magic_idx < DAT1_HEADER_SIZE:
            continue
        if magic != DAT1_MAGIC:
            continue
        data = file_data[magic_idx:]
        try:
            data_file_id, _version, declared_size, block_count, fixup_count = struct.unpack_from("<IIIHH", data, 0)
        except struct.error:
            continue
        if data_file_id != DAT1_FILE_ID:
            continue
        table_end = (
            DAT1_HEADER_SIZE
            + (block_count * DAT1_BLOCK_TABLE_ENTRY_SIZE)
            + (fixup_count * DAT1_FIXUP_TABLE_ENTRY_SIZE)
        )
        if declared_size < table_end or declared_size > len(data):
            continue

        blocks, offset = {}, DAT1_HEADER_SIZE
        valid = True
        for _i in range(block_count):
            if offset + DAT1_BLOCK_TABLE_ENTRY_SIZE > declared_size:
                valid = False
                break
            name_hash, block_offset, block_size = struct.unpack_from("<III", data, offset)
            if name_hash in blocks:
                valid = False
                break
            if block_offset < table_end or block_offset > declared_size or block_size > declared_size - block_offset:
                valid = False
                break
            blocks[name_hash] = (block_offset, block_size)
            offset += DAT1_BLOCK_TABLE_ENTRY_SIZE
        if valid:
            return data[:declared_size], blocks, table_end
    return None, None, None
