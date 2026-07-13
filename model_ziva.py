
from __future__ import annotations

import hashlib
import math
import os
import struct

import numpy as np

from .hashes import BLOCK_HASHES, string_crc32


ZIVA_INFO_SIZE = 144
ZIVA_ELEM_SIZE = 24
ZIVA_STAGE_SIZE = 48
ZIVA_BUFFER_SIZE = 40
ZIVA_SLIDER_INFO_SIZE = 8
ZIVA_SLIDER_ID_SIZE = 4
ZIVA_SUBSET_ELEM_SIZE = 4
ZIVA_INVALID_ELEM = 0xFF
ZIVA_INVALID_VERTEX = 0xFFFFFFFF
ZIVA_STAGE_KERNEL = 1
ZIVA_STAGE_RBF = 2
ZIVA_STAGE_EIGEN = 3
ZIVA_STAGE_TENSOR = 4
ZIVA_STAGE_SCATTER = 5
ZIVA_SUPPORTED_STAGES = {
    ZIVA_STAGE_KERNEL,
    ZIVA_STAGE_RBF,
    ZIVA_STAGE_EIGEN,
    ZIVA_STAGE_TENSOR,
    ZIVA_STAGE_SCATTER,
}

MODEL_SUBSET_SIZE = 128
MODEL_SUBSET_VERTEX_COUNT_OFFSET = 4
MODEL_SUBSET_INDEX_COUNT_OFFSET = 0
MODEL_SUBSET_INDEX_DATA_OFFSET = 12
MODEL_SUBSET_FLAGS_OFFSET = 20
MODEL_SUBSET_MPU_OFFSET = 24
MODEL_SUBSET_VERTEX_STD_OFFSET = 64
MODEL_SUBSET_BASE_OFFSET = 88
MODEL_SUBSET_LOD_MASK_OFFSET = 102
MODEL_SUBSET_ORIGIN_OFFSET = 48
MODEL_VERTEX_SIZE = 16
MODEL_SUBSET_HAS_ORIGIN = 0x4000
SUBSET_CENTER_LOG_SCALE = 4

_FORMAT_DTYPES = {
    41: np.dtype("<f4"),  # DXGI_FORMAT_R32_FLOAT
    42: np.dtype("<u4"),  # DXGI_FORMAT_R32_UINT
    43: np.dtype("<i4"),  # DXGI_FORMAT_R32_SINT
    57: np.dtype("<u2"),  # DXGI_FORMAT_R16_UINT
    59: np.dtype("<i2"),  # DXGI_FORMAT_R16_SINT
    62: np.dtype("u1"),   # DXGI_FORMAT_R8_UINT
    64: np.dtype("i1"),   # DXGI_FORMAT_R8_SINT
}

_ZIVA_MODEL_CACHE = {}


def _ziva_checked_range(size, offset, length, label):
    offset = int(offset)
    length = int(length)
    if offset < 0 or length < 0 or offset + length > int(size):
        raise ValueError(f"{label} is outside the Ziva2 block")
    return offset


def _ziva_read_c_string(data, offset, label, max_length=512):
    offset = _ziva_checked_range(len(data), offset, 1, label)
    end = data.find(b"\x00", offset, min(len(data), offset + max_length))
    if end < 0:
        raise ValueError(f"{label} is not null terminated")
    return data[offset:end].decode("ascii", errors="strict")


def _ziva_parse_dat1_bytes(raw):
    dat1_offset = raw.find(b"1TAD")
    if dat1_offset < 0 or len(raw) - dat1_offset < 16:
        raise ValueError("DAT1 header was not found")
    data = raw[dat1_offset:]
    file_id, version, declared_size, block_count, fixup_count = struct.unpack_from("<IIIHH", data, 0)
    if file_id != 0x44415431 or declared_size > len(data):
        raise ValueError("invalid DAT1 header")
    table_end = 16 + block_count * 12 + fixup_count * 8
    if table_end > declared_size:
        raise ValueError("DAT1 tables are truncated")
    blocks = {}
    entries = []
    for index in range(block_count):
        name_hash, offset, size = struct.unpack_from("<III", data, 16 + index * 12)
        if name_hash in blocks:
            raise ValueError(f"duplicate DAT1 block 0x{name_hash:08X}")
        if offset < table_end or offset + size > declared_size:
            raise ValueError(f"DAT1 block 0x{name_hash:08X} is out of range")
        blocks[name_hash] = (offset, size)
        entries.append((name_hash, offset, size))
    return data[:declared_size], blocks, dat1_offset, version


def _ziva_file_fingerprint(path):
    stat = os.stat(path)
    return f"{os.path.abspath(path)}|{stat.st_size}|{stat.st_mtime_ns}"


def _ziva_quaternion_matrix(x, y, z, w):
    return np.asarray([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ], dtype=np.float32)


def engine_to_blender_vectors(values):
    values = np.asarray(values, dtype=np.float32)
    result = np.empty_like(values)
    result[..., 0] = values[..., 0]
    result[..., 1] = -values[..., 2]
    result[..., 2] = values[..., 1]
    return result


class ZivaModel:
    def __init__(self, source_path):
        self.source_path = os.path.abspath(source_path)
        self.fingerprint = _ziva_file_fingerprint(self.source_path)
        with open(self.source_path, "rb") as stream:
            raw = stream.read()
        self.data, self.blocks, self.dat1_offset, self.dat1_version = _ziva_parse_dat1_bytes(raw)
        block = self.blocks.get(BLOCK_HASHES["ModelAnimZiva2Info"])
        if not block:
            raise ValueError("model has no Model Anim Ziva2 Info block")
        offset, size = block
        if size < ZIVA_INFO_SIZE:
            raise ValueError("Ziva2 info block is truncated")
        self.ziva = self.data[offset:offset + size]
        self._parse_header()
        self._parse_model_skeleton()
        self._parse_subsets()

    def _ptr(self, field_offset, label, minimum_size=1):
        _ziva_checked_range(len(self.ziva), field_offset, 8, label)
        value = struct.unpack_from("<I", self.ziva, field_offset)[0]
        return _ziva_checked_range(len(self.ziva), value, minimum_size, label)

    def _parse_header(self):
        (
            self.flags,
            self.lod_level,
            self.elem_count,
            _pad,
            self.regular_slider_count,
            self.morph_slider_count,
            self.joint_lookup_count,
            self.stage_count,
            self.buffer_count,
            self.lod_blend_out_factor,
            self.max_vertex_count,
            self.vertex_lookup_count,
        ) = struct.unpack_from("<4B3H2Bf2I", self.ziva, 0)
        if not 1 <= self.elem_count <= 255:
            raise ValueError(f"invalid Ziva element count {self.elem_count}")
        if self.stage_count < self.elem_count or self.stage_count % self.elem_count:
            raise ValueError("Ziva stage count is not divisible by element count")
        if self.buffer_count > 255:
            raise ValueError("Ziva buffer count exceeds serialized limit")
        self.total_slider_count = self.regular_slider_count + self.morph_slider_count
        self.elems_offset = self._ptr(24, "Ziva elements", self.elem_count * ZIVA_ELEM_SIZE)
        self.joints_offset = self._ptr(32, "Ziva joint lookup", max(1, self.joint_lookup_count * 8))
        self.sliders_offset = self._ptr(40, "Ziva sliders", max(1, (self.total_slider_count + 1) * 8))
        self.slider_lookup_offset = self._ptr(48, "Ziva slider lookup")
        self.subsets_offset = self._ptr(56, "Ziva subset lookup")
        self.vertex_lookup_offset = self._ptr(64, "Ziva vertex lookup", max(1, self.vertex_lookup_count * 4))
        self.buffers_offset = self._ptr(72, "Ziva GPU buffers", self.buffer_count * ZIVA_BUFFER_SIZE)
        self.stages_offset = self._ptr(80, "Ziva GPU stages", self.stage_count * ZIVA_STAGE_SIZE)

        self.elems = []
        slider_id_count = 0
        for index in range(self.elem_count):
            values = struct.unpack_from("<IIIHHHHHH", self.ziva, self.elems_offset + index * ZIVA_ELEM_SIZE)
            elem = {
                "index": index,
                "vertex_count": int(values[0]),
                "data_offset": int(values[1]),
                "data_size": int(values[2]),
                "joint_start": int(values[3]),
                "joint_count": int(values[4]),
                "slider_start": int(values[5]),
                "slider_count": int(values[6]),
                "stage_start": int(values[7]),
                "buffer_start": int(values[8]),
            }
            if elem["joint_start"] + elem["joint_count"] > self.joint_lookup_count:
                raise ValueError(f"Ziva element {index} has an invalid joint range")
            if elem["stage_start"] + 4 > self.stage_count:
                raise ValueError(f"Ziva element {index} does not contain four GPU stages")
            slider_id_count = max(slider_id_count, elem["slider_start"] + elem["slider_count"])
            self.elems.append(elem)

        self.sliders = []
        hashes = set()
        for index in range(self.total_slider_count):
            record = self.sliders_offset + index * ZIVA_SLIDER_INFO_SIZE
            name_hash, relative_name = struct.unpack_from("<II", self.ziva, record)
            name = _ziva_read_c_string(self.ziva, record + 4 + relative_name, f"Ziva slider {index} name")
            if name_hash in hashes:
                raise ValueError(f"duplicate Ziva slider hash 0x{name_hash:08X}")
            hashes.add(name_hash)
            self.sliders.append({
                "index": index,
                "name": name,
                "hash": int(name_hash),
                "kind": "REGULAR" if index < self.regular_slider_count else "MORPH",
                "driver_status": "GAME_BRIDGE_REQUIRED" if index < self.regular_slider_count else "AUTOMATIC_MORPH",
            })
        sentinel = struct.unpack_from("<I", self.ziva, self.sliders_offset + self.total_slider_count * 8)[0]
        if sentinel != 0xFFFFFFFF:
            raise ValueError("Ziva slider lookup sentinel is missing")
        self.slider_by_name = {item["name"]: item for item in self.sliders}
        self.slider_by_hash = {item["hash"]: item for item in self.sliders}

        _ziva_checked_range(len(self.ziva), self.slider_lookup_offset, slider_id_count * 4, "Ziva slider IDs")
        self.slider_ids = [
            struct.unpack_from("<HH", self.ziva, self.slider_lookup_offset + index * 4)
            for index in range(slider_id_count)
        ]

        self.buffers = [self._parse_buffer(index) for index in range(self.buffer_count)]
        self.stages = [self._parse_stage(index) for index in range(self.stage_count)]
        for elem in self.elems:
            for local_stage in range(4):
                stage = self.stages[elem["stage_start"] + local_stage]
                if stage["work_start"] + stage["work_count"] > self.buffer_count:
                    raise ValueError(f"Ziva stage {stage['index']} has an invalid buffer range")
        self.vertex_lookup = np.frombuffer(
            self.ziva,
            dtype="<u4",
            count=self.vertex_lookup_count,
            offset=self.vertex_lookup_offset,
        ).copy()

    def _parse_buffer(self, index):
        record = self.buffers_offset + index * ZIVA_BUFFER_SIZE
        packed_format, packed_count = struct.unpack_from("<II", self.ziva, record)
        fmt = packed_format & 0xFF
        element_size = packed_format >> 8
        element_count = packed_count & 0x7FFFFFFF
        dtype = _FORMAT_DTYPES.get(fmt)
        if dtype is None:
            raise ValueError(f"Ziva buffer {index} uses unsupported DXGI format {fmt}")
        if element_size != dtype.itemsize:
            raise ValueError(
                f"Ziva buffer {index} element size {element_size} does not match format {fmt}"
            )
        data_offset = struct.unpack_from("<I", self.ziva, record + 8)[0]
        byte_count = element_size * element_count
        if byte_count:
            _ziva_checked_range(len(self.ziva), data_offset, byte_count, f"Ziva buffer {index} data")
            values = np.frombuffer(
                self.ziva,
                dtype=dtype,
                count=element_count,
                offset=data_offset,
            ).copy()
        else:
            values = np.empty(0, dtype=dtype)
        return {
            "index": index,
            "format": fmt,
            "element_size": element_size,
            "element_count": element_count,
            "data_offset": data_offset,
            "values": values,
        }

    def _parse_stage(self, index):
        record = self.stages_offset + index * ZIVA_STAGE_SIZE
        shader_id, work_count, work_start, thread_groups = struct.unpack_from("<BBHH", self.ziva, record)
        if shader_id not in ZIVA_SUPPORTED_STAGES:
            raise ValueError(f"Ziva stage {index} uses unsupported shader {shader_id}")
        output_count = struct.unpack_from("<I", self.ziva, record + 12)[0] & 0x7FFFFFFF
        return {
            "index": index,
            "shader_id": int(shader_id),
            "work_count": int(work_count),
            "work_start": int(work_start),
            "thread_groups": int(thread_groups),
            "output_count": int(output_count),
        }

    def _parse_model_skeleton(self):
        hierarchy = self.blocks.get(BLOCK_HASHES.get("ModelJointHierarchy"))
        joints = self.blocks.get(BLOCK_HASHES.get("ModelJoint"))
        bind = self.blocks.get(BLOCK_HASHES.get("ModelBindPose"))
        self.model_joint_count = 0
        self.joint_parents = []
        self.joint_flags = []
        self.bind_local_matrices = []
        if not hierarchy or not joints or not bind:
            return
        hierarchy_offset, hierarchy_size = hierarchy
        _ziva_checked_range(len(self.data), hierarchy_offset, 4, "model joint hierarchy")
        self.model_joint_count = struct.unpack_from("<H", self.data, hierarchy_offset + 2)[0]
        joints_offset, joints_size = joints
        bind_offset, bind_size = bind
        if joints_size < self.model_joint_count * 16 or bind_size < self.model_joint_count * 48:
            raise ValueError("model skeleton blocks are truncated")
        bind_values = np.frombuffer(
            self.data,
            dtype="<f4",
            count=self.model_joint_count * 12,
            offset=bind_offset,
        ).reshape(self.model_joint_count, 12)
        for index in range(self.model_joint_count):
            parent, joint_index, _subtree, flags = struct.unpack_from("<hHHH", self.data, joints_offset + index * 16)
            if joint_index != index:
                raise ValueError("model joint table is not in hierarchy order")
            values = bind_values[index]
            matrix = np.eye(4, dtype=np.float32)
            matrix[:3, :3] = _ziva_quaternion_matrix(*values[4:8]) @ np.diag(values[:3])
            matrix[:3, 3] = values[8:11]
            self.joint_parents.append(int(parent))
            self.joint_flags.append(int(flags))
            self.bind_local_matrices.append(matrix)
        if self.bind_local_matrices:
            self.bind_local_matrices[0] = np.eye(4, dtype=np.float32)

    def _parse_subsets(self):
        subset_block = self.blocks.get(BLOCK_HASHES["ModelSubset"])
        geom_block = self.blocks.get(BLOCK_HASHES["ModelSubsetGeomData"])
        if not subset_block or not geom_block:
            raise ValueError("Ziva model is missing subset geometry")
        subset_offset, subset_size = subset_block
        self.geom_offset, _geom_size = geom_block
        self.subset_count = subset_size // MODEL_SUBSET_SIZE
        _ziva_checked_range(len(self.ziva), self.subsets_offset, self.subset_count * 4, "Ziva subset elements")
        self.subsets = []
        for subset_index in range(self.subset_count):
            packed = struct.unpack_from("<I", self.ziva, self.subsets_offset + subset_index * 4)[0]
            elem_index = packed & 0xFF
            lookup_start = packed >> 8
            record = subset_offset + subset_index * MODEL_SUBSET_SIZE
            vertex_count = struct.unpack_from("<I", self.data, record + MODEL_SUBSET_VERTEX_COUNT_OFFSET)[0]
            index_count = struct.unpack_from("<I", self.data, record + MODEL_SUBSET_INDEX_COUNT_OFFSET)[0]
            lod_mask = struct.unpack_from("<H", self.data, record + MODEL_SUBSET_LOD_MASK_OFFSET)[0]
            subset = {
                "index": subset_index,
                "elem_index": int(elem_index),
                "lookup_start": int(lookup_start),
                "vertex_count": int(vertex_count),
                "index_count": int(index_count),
                "lod_mask": int(lod_mask),
                "record_offset": record,
            }
            if elem_index != ZIVA_INVALID_ELEM:
                if elem_index >= self.elem_count:
                    raise ValueError(f"subset {subset_index} references invalid Ziva element {elem_index}")
                if lookup_start + vertex_count > self.vertex_lookup_count:
                    raise ValueError(f"subset {subset_index} has an invalid Ziva vertex lookup")
            self.subsets.append(subset)

    def metadata(self):
        return {
            "fingerprint": self.fingerprint,
            "lod_level": int(self.lod_level),
            "lod_blend_out_factor": float(self.lod_blend_out_factor),
            "element_count": int(self.elem_count),
            "regular_slider_count": int(self.regular_slider_count),
            "morph_slider_count": int(self.morph_slider_count),
            "joint_lookup_count": int(self.joint_lookup_count),
            "max_vertex_count": int(self.max_vertex_count),
            "vertex_lookup_count": int(self.vertex_lookup_count),
            "channels": [dict(channel) for channel in self.sliders],
            "elements": [
                {
                    "index": elem["index"],
                    "vertex_count": elem["vertex_count"],
                    "joint_count": elem["joint_count"],
                    "slider_count": elem["slider_count"],
                    "subset_count": sum(1 for subset in self.subsets if subset["elem_index"] == elem["index"]),
                }
                for elem in self.elems
            ],
        }

    def joint_indices_for_element(self, elem_index):
        elem = self.elems[int(elem_index)]
        result = []
        for local_index in range(elem["joint_count"]):
            _name_hash, model_joint_index = struct.unpack_from(
                "<II",
                self.ziva,
                self.joints_offset + (elem["joint_start"] + local_index) * 8,
            )
            if model_joint_index != 0xFFFFFFFF:
                result.append(int(model_joint_index))
        return result

    def _matrix_metadata(self, values, matrix_index):
        start = int(matrix_index) * 5
        if start + 5 > len(values):
            raise ValueError(f"Ziva matrix {matrix_index} metadata is out of range")
        return tuple(int(value) for value in values[start:start + 5])

    def _pose_input(self, elem, slider_values, joint_local_matrices):
        pose = np.zeros(elem["slider_count"] + elem["joint_count"] * 12, dtype=np.float32)
        global_values = np.zeros(self.total_slider_count, dtype=np.float32)
        for key, value in (slider_values or {}).items():
            channel = self.slider_by_name.get(key) if isinstance(key, str) else self.slider_by_hash.get(int(key))
            if channel is not None:
                global_values[channel["index"]] = float(value)
        for lookup_index in range(elem["slider_start"], elem["slider_start"] + elem["slider_count"]):
            global_index, ziva_index = self.slider_ids[lookup_index]
            if global_index >= self.total_slider_count or ziva_index >= elem["slider_count"]:
                raise ValueError(f"Ziva element {elem['index']} has an invalid slider mapping")
            pose[ziva_index] = global_values[global_index]

        matrices = joint_local_matrices or self.bind_local_matrices
        for local_index in range(elem["joint_count"]):
            _name_hash, model_joint_index = struct.unpack_from(
                "<II",
                self.ziva,
                self.joints_offset + (elem["joint_start"] + local_index) * 8,
            )
            matrix = np.eye(4, dtype=np.float32)
            if model_joint_index != 0xFFFFFFFF and model_joint_index < len(matrices):
                matrix = np.asarray(matrices[model_joint_index], dtype=np.float32)
            flattened = np.concatenate((
                matrix[:3, 0],
                matrix[:3, 1],
                matrix[:3, 2],
                matrix[:3, 3] * 100.0,
            ))
            start = elem["slider_count"] + local_index * 12
            pose[start:start + 12] = flattened
        return pose

    def _stage_kernel(self, input_values, buffers, output_count):
        elements, matrices, matrix_values, pose_indices, kernel_scales, pose_starts, shifts, scales, vector_offsets, post_scale, constants = buffers
        count = int(constants[0])
        entries = elements[:count * 2].reshape(count, 2)
        output = np.zeros(output_count, dtype=np.float32)
        for matrix_index in np.unique(entries[:, 0]):
            rows_total, cols, values_offset, _cum_rows, _cum_cols = self._matrix_metadata(matrices, matrix_index)
            selected = entries[:, 0] == matrix_index
            rows = entries[selected, 1].astype(np.int64)
            if np.any(rows >= rows_total):
                raise ValueError("Ziva kernel row is out of range")
            values = matrix_values[values_offset:values_offset + rows_total * cols].reshape(rows_total, cols)[rows].astype(np.float32)
            pose_start = int(pose_starts[int(matrix_index)])
            pose_slice = slice(pose_start, pose_start + cols)
            pose = scales[pose_slice] * input_values[pose_indices[pose_slice].astype(np.int64)] + shifts[pose_slice]
            delta = pose[None, :] - kernel_scales[pose_slice][None, :] * values
            result = np.sqrt(np.sum(delta * delta, axis=1))
            output_indices = int(vector_offsets[int(matrix_index)]) + rows
            output[output_indices] = result * post_scale[output_indices]
        return output

    def _stage_rbf(self, input_values, buffers, output_count):
        elements, matrices, matrix_values, post_scale, constants = buffers
        count = int(constants[0])
        entries = elements[:count * 4].reshape(count, 4)
        output = np.zeros(output_count, dtype=np.float32)
        for matrix_index in np.unique(entries[:, 0]):
            rows_total, cols, values_offset, _cum_rows, _cum_cols = self._matrix_metadata(matrices, matrix_index)
            selected_entries = entries[entries[:, 0] == matrix_index]
            rows = selected_entries[:, 2].astype(np.int64)
            values = matrix_values[values_offset:values_offset + rows_total * cols].reshape(rows_total, cols)[rows].astype(np.float32)
            for row_index, entry in enumerate(selected_entries):
                input_offset = int(entry[1])
                output_offset = int(entry[3])
                output[output_offset] = float(values[row_index] @ input_values[input_offset:input_offset + cols]) * post_scale[output_offset]
        return output

    def _stage_basis(self, shader_id, input_values, buffers, output_count):
        elements, matrices, matrix_values, per_vertex_scale, constants = buffers
        output = np.zeros(output_count, dtype=np.float32)
        for matrix_index in np.unique(elements):
            rows, cols, values_offset, cumulative_rows, cumulative_cols = self._matrix_metadata(matrices, matrix_index)
            values = matrix_values[values_offset:values_offset + rows * cols].reshape(rows, cols).astype(np.float32)
            if shader_id == ZIVA_STAGE_TENSOR:
                input_matrix = input_values[cumulative_cols * 3:(cumulative_cols + cols) * 3].reshape(cols, 3)
                result = values @ input_matrix
                result *= per_vertex_scale[cumulative_rows:cumulative_rows + rows, None]
                output[cumulative_rows * 3:(cumulative_rows + rows) * 3] = result.reshape(-1)
            else:
                result = values @ input_values[cumulative_cols:cumulative_cols + cols]
                result *= per_vertex_scale[cumulative_rows:cumulative_rows + rows]
                output[cumulative_rows:cumulative_rows + rows] = result
        return output

    def _stage_scatter(self, input_values, buffers):
        scattered, scatter_indices, inverse, inverse_indices, rest, int_constants, float_constants = buffers
        vertex_count = int(int_constants[0])
        input_vertices = input_values.reshape(-1, 3)
        combined = np.zeros((vertex_count, 3), dtype=np.float32)
        for vertex in range(vertex_count):
            offset, count = (int(value) for value in scattered[vertex * 2:vertex * 2 + 2])
            if count == 1:
                combined[vertex] = input_vertices[offset]
            elif count:
                combined[vertex] = input_vertices[scatter_indices[offset:offset + count].astype(np.int64)].sum(axis=0)
        final = (combined + rest[:vertex_count * 3].reshape(vertex_count, 3)) * float_constants[:3]
        output = np.zeros(self.max_vertex_count * 3, dtype=np.float32)
        component_offsets = [int(int_constants[index]) for index in (1, 2, 3)]
        for vertex in range(vertex_count):
            offset, count = (int(value) for value in inverse[vertex * 2:vertex * 2 + 2])
            destinations = (offset,) if count == 1 else inverse_indices[offset:offset + count]
            for destination in destinations:
                destination = int(destination)
                output[destination + component_offsets[0]] = final[vertex, 0]
                output[destination + component_offsets[1]] = final[vertex, 1]
                output[destination + component_offsets[2]] = final[vertex, 2]
        return output

    def evaluate_element(self, elem_index=0, slider_values=None, joint_local_matrices=None):
        elem_index = int(elem_index)
        if not 0 <= elem_index < self.elem_count:
            raise ValueError(f"invalid Ziva element {elem_index}")
        elem = self.elems[elem_index]
        output = self._pose_input(elem, slider_values, joint_local_matrices)
        for local_stage in range(4):
            stage = self.stages[elem["stage_start"] + local_stage]
            work = [
                self.buffers[index]["values"]
                for index in range(stage["work_start"], stage["work_start"] + stage["work_count"])
            ]
            shader_id = stage["shader_id"]
            if shader_id == ZIVA_STAGE_KERNEL:
                output = self._stage_kernel(output, work, stage["output_count"])
            elif shader_id == ZIVA_STAGE_RBF:
                output = self._stage_rbf(output, work, stage["output_count"])
            elif shader_id in (ZIVA_STAGE_EIGEN, ZIVA_STAGE_TENSOR):
                output = self._stage_basis(shader_id, output, work, stage["output_count"])
            elif shader_id == ZIVA_STAGE_SCATTER:
                output = self._stage_scatter(output, work)
            else:
                raise ValueError(f"unsupported Ziva shader {shader_id}")
        return output.reshape(-1, 3)

    def subset_surface(self, subset_index, solver_positions=None):
        subset = self.subsets[int(subset_index)]
        record = subset["record_offset"]
        vertex_count = subset["vertex_count"]
        index_count = subset["index_count"]
        mpu = struct.unpack_from("<f", self.data, record + MODEL_SUBSET_MPU_OFFSET)[0]
        flags = struct.unpack_from("<H", self.data, record + MODEL_SUBSET_FLAGS_OFFSET)[0]
        subset_base = struct.unpack_from("<I", self.data, record + MODEL_SUBSET_BASE_OFFSET)[0]
        vertex_offset = struct.unpack_from("<I", self.data, record + MODEL_SUBSET_VERTEX_STD_OFFSET)[0]
        index_offset = struct.unpack_from("<I", self.data, record + MODEL_SUBSET_INDEX_DATA_OFFSET)[0]
        vertex_base = self.geom_offset + subset_base + vertex_offset
        index_base = self.geom_offset + subset_base + index_offset
        _ziva_checked_range(len(self.data), vertex_base, vertex_count * MODEL_VERTEX_SIZE, f"subset {subset_index} vertices")
        _ziva_checked_range(len(self.data), index_base, index_count * 2, f"subset {subset_index} indices")
        raw_vertices = np.frombuffer(
            self.data,
            dtype="<i2",
            count=vertex_count * 8,
            offset=vertex_base,
        ).reshape(vertex_count, 8)
        base_positions = raw_vertices[:, :3].astype(np.float32) * float(mpu)
        if flags & MODEL_SUBSET_HAS_ORIGIN:
            packed = struct.unpack_from("<iii", self.data, record + MODEL_SUBSET_ORIGIN_OFFSET)
            signed = np.asarray([
                (value & 0xFFFF) - (0x10000 if value & 0x8000 else 0)
                for value in packed
            ], dtype=np.float32)
            base_positions += signed * float(1 << SUBSET_CENTER_LOG_SCALE) * float(mpu)
        indices = np.frombuffer(self.data, dtype="<u2", count=index_count, offset=index_base).astype(np.int32)
        triangles = indices[:index_count - (index_count % 3)].reshape(-1, 3)
        lookup = None
        positions = base_positions
        if subset["elem_index"] != ZIVA_INVALID_ELEM:
            start = subset["lookup_start"]
            lookup = self.vertex_lookup[start:start + vertex_count]
            if solver_positions is not None:
                positions = base_positions.copy()
                valid = lookup != ZIVA_INVALID_VERTEX
                positions[valid] = solver_positions[lookup[valid]]
        return {
            **subset,
            "positions": positions,
            "base_positions": base_positions,
            "triangles": triangles,
            "ziva_lookup": lookup,
        }

    def element_surfaces(self, elem_index, solver_positions=None, lod=0):
        elem_index = int(elem_index)
        surfaces = []
        for subset in self.subsets:
            if subset["elem_index"] != elem_index:
                continue
            if lod is not None and subset["lod_mask"] and not (subset["lod_mask"] & (1 << int(lod))):
                continue
            surfaces.append(self.subset_surface(subset["index"], solver_positions=solver_positions))
        return surfaces


def load_ziva_model(source_path, use_cache=True):
    key = _ziva_file_fingerprint(source_path)
    if use_cache and key in _ZIVA_MODEL_CACHE:
        return _ZIVA_MODEL_CACHE[key]
    model = ZivaModel(source_path)
    if use_cache:
        _ZIVA_MODEL_CACHE.clear()
        _ZIVA_MODEL_CACHE[key] = model
    return model


def clear_ziva_cache():
    _ZIVA_MODEL_CACHE.clear()


def _ziva_object_armature_matrix(obj, armature):
    relative = armature.matrix_world.inverted() @ obj.matrix_world
    return np.asarray([[float(relative[row][column]) for column in range(4)] for row in range(4)], dtype=np.float64)


def _ziva_mesh_coordinates(obj):
    coordinates = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    obj.data.vertices.foreach_get("co", coordinates)
    return coordinates.reshape(-1, 3)


def _ziva_barycentric(point, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = point - a
    d00 = float(v0 @ v0)
    d01 = float(v0 @ v1)
    d11 = float(v1 @ v1)
    d20 = float(v2 @ v0)
    d21 = float(v2 @ v1)
    denominator = d00 * d11 - d01 * d01
    if abs(denominator) <= 1.0e-20:
        return np.asarray((1.0, 0.0, 0.0), dtype=np.float64)
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.asarray((1.0 - v - w, v, w), dtype=np.float64)


def _ziva_element_transfer_surface(model, elem_index, baseline, lod=0):
    vertices = []
    triangles = []
    solver_indices = []
    subset_ranges = {}
    for surface in model.element_surfaces(elem_index, solver_positions=baseline, lod=lod):
        positions = engine_to_blender_vectors(surface["base_positions"]).astype(np.float64)
        lookup = np.asarray(surface["ziva_lookup"], dtype=np.uint32)
        vertex_start = len(vertices)
        vertices.extend(tuple(float(value) for value in position) for position in positions)
        solver_indices.extend(int(value) for value in lookup)
        valid_triangles = []
        for triangle in surface["triangles"]:
            triangle = tuple(int(value) for value in triangle)
            if any(lookup[index] == ZIVA_INVALID_VERTEX for index in triangle):
                continue
            generated = tuple(vertex_start + index for index in triangle)
            triangles.append(generated)
            valid_triangles.append(generated)
        subset_ranges[int(surface["index"])] = {
            "positions": positions,
            "lookup": lookup,
            "valid_triangle_count": len(valid_triangles),
        }
    if not triangles:
        raise ValueError(f"Ziva element {elem_index} has no transferable triangles at LOD {lod}")
    return {
        "vertices": np.asarray(vertices, dtype=np.float64),
        "triangles": np.asarray(triangles, dtype=np.int32),
        "solver_indices": np.asarray(solver_indices, dtype=np.uint32),
        "subsets": subset_ranges,
    }


def _ziva_build_object_transfer_map(
    model, armature, obj, elem_index, baseline, max_distance, lod=0, transfer_surface=None
):
    from mathutils import Vector
    from mathutils.bvhtree import BVHTree

    surface = transfer_surface or _ziva_element_transfer_surface(model, elem_index, baseline, lod=lod)
    relative = _ziva_object_armature_matrix(obj, armature)
    object_coordinates = _ziva_mesh_coordinates(obj)
    armature_coordinates = object_coordinates @ relative[:3, :3].T + relative[:3, 3]

    source_subset_index = int(obj.get("engine_subset_index", -1))
    source_subset = surface["subsets"].get(source_subset_index)
    if source_subset is not None and len(source_subset["positions"]) == len(armature_coordinates):
        direct_error = np.linalg.norm(source_subset["positions"] - armature_coordinates, axis=1)
        if not len(direct_error) or float(direct_error.max()) <= 1.0e-4:
            return {
                "type": "DIRECT",
                "lookup": source_subset["lookup"].copy(),
                "inverse_linear": np.linalg.inv(relative[:3, :3]),
                "mapped_count": len(armature_coordinates),
                "unmapped_count": 0,
                "max_distance": float(direct_error.max()) if len(direct_error) else 0.0,
            }

    tree = BVHTree.FromPolygons(
        [Vector(tuple(value)) for value in surface["vertices"]],
        [tuple(int(value) for value in triangle) for triangle in surface["triangles"]],
        all_triangles=True,
    )
    triangle_indices = np.full(len(armature_coordinates), -1, dtype=np.int32)
    weights = np.zeros((len(armature_coordinates), 3), dtype=np.float64)
    distances = np.full(len(armature_coordinates), np.inf, dtype=np.float64)
    for vertex_index, coordinate in enumerate(armature_coordinates):
        nearest = tree.find_nearest(Vector(tuple(float(value) for value in coordinate)), float(max_distance))
        if nearest is None or nearest[0] is None or nearest[2] is None:
            continue
        nearest_position, _normal, triangle_index, distance = nearest
        triangle_index = int(triangle_index)
        triangle = surface["triangles"][triangle_index]
        a, b, c = surface["vertices"][triangle]
        triangle_indices[vertex_index] = triangle_index
        weights[vertex_index] = _ziva_barycentric(np.asarray(nearest_position, dtype=np.float64), a, b, c)
        distances[vertex_index] = float(distance)
    mapped = triangle_indices >= 0
    return {
        "type": "BARYCENTRIC",
        "triangles": surface["triangles"],
        "solver_indices": surface["solver_indices"],
        "triangle_indices": triangle_indices,
        "weights": weights,
        "inverse_linear": np.linalg.inv(relative[:3, :3]),
        "mapped_count": int(np.count_nonzero(mapped)),
        "unmapped_count": int(len(mapped) - np.count_nonzero(mapped)),
        "max_distance": float(np.max(distances[mapped])) if np.any(mapped) else math.inf,
    }


def _ziva_transfer_solver_delta(transfer_map, solver_delta):
    solver_delta_blender = engine_to_blender_vectors(solver_delta).astype(np.float64)
    if transfer_map["type"] == "DIRECT":
        lookup = transfer_map["lookup"]
        local_delta = np.zeros((len(lookup), 3), dtype=np.float64)
        valid = lookup != ZIVA_INVALID_VERTEX
        local_delta[valid] = solver_delta_blender[lookup[valid]]
    else:
        triangle_indices = transfer_map["triangle_indices"]
        local_delta = np.zeros((len(triangle_indices), 3), dtype=np.float64)
        valid = triangle_indices >= 0
        selected_triangles = transfer_map["triangles"][triangle_indices[valid]]
        selected_solver = transfer_map["solver_indices"][selected_triangles]
        triangle_deltas = solver_delta_blender[selected_solver]
        local_delta[valid] = np.sum(triangle_deltas * transfer_map["weights"][valid, :, None], axis=1)
    return local_delta @ transfer_map["inverse_linear"].T


def _ziva_object_target_metadata(obj):
    import json

    try:
        metadata = json.loads(str(obj.get("engine_morph_targets_json", "{}") or "{}"))
    except Exception:
        metadata = {}
    return metadata if isinstance(metadata, dict) else {}


def _ziva_write_shape_key(obj, channel, local_delta, overwrite=False):
    import json

    if len(local_delta) != len(obj.data.vertices):
        raise ValueError(f"{obj.name}: transfer delta count does not match mesh vertices")
    affected = np.linalg.norm(local_delta, axis=1) >= 0.0001
    if not np.any(affected):
        return "EMPTY"
    if obj.data.shape_keys is None or not obj.data.shape_keys.key_blocks:
        obj.shape_key_add(name="Basis", from_mix=False)
    key_blocks = obj.data.shape_keys.key_blocks
    key = key_blocks.get(channel["name"])
    if key is not None and not overwrite:
        return "EXISTS"
    if key is None:
        key = obj.shape_key_add(name=channel["name"], from_mix=False)
        if key.name != channel["name"]:
            obj.shape_key_remove(key)
            raise ValueError(f"{obj.name}: shape-key name {channel['name']!r} is already reserved")
    basis = key_blocks[0]
    coordinates = np.empty(len(obj.data.vertices) * 3, dtype=np.float64)
    basis.data.foreach_get("co", coordinates)
    coordinates = coordinates.reshape(-1, 3)
    coordinates += local_delta
    key.data.foreach_set("co", coordinates.reshape(-1))
    key.value = 0.0
    key.slider_min = -1.0
    key.slider_max = 1.0
    metadata = _ziva_object_target_metadata(obj)
    metadata[channel["name"]] = {
        "name": channel["name"],
        "hash": int(channel["hash"]),
        "index": int(channel["index"]),
        "ziva_kind": channel["kind"],
        "driver_status": channel["driver_status"],
    }
    obj["engine_morph_targets_json"] = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
    obj["engine_morph_shape_keys_imported"] = True
    return "CREATED"


def transfer_ziva_channels_to_objects(
    model,
    armature,
    objects,
    channels=None,
    overwrite=False,
    max_distance=0.05,
    lod=0,
    progress=None,
):
    objects = [obj for obj in objects if getattr(obj, "type", None) == 'MESH' and len(obj.data.vertices)]
    if not objects:
        raise ValueError("select at least one nonempty mesh parented to the model armature")
    channels = list(channels if channels is not None else model.sliders)
    if not channels:
        raise ValueError("this Ziva model has no named slider channels")
    elements = sorted({int(obj.get("engine_ziva_element_index", 0)) for obj in objects})
    baselines = {elem: model.evaluate_element(elem) for elem in elements}
    transfer_surfaces = {
        elem: _ziva_element_transfer_surface(model, elem, baselines[elem], lod=lod)
        for elem in elements
    }
    transfer_maps = {}
    for obj in objects:
        elem = int(obj.get("engine_ziva_element_index", 0))
        if not 0 <= elem < model.elem_count:
            raise ValueError(f"{obj.name}: invalid Ziva source element {elem}")
        transfer_maps[obj] = _ziva_build_object_transfer_map(
            model,
            armature,
            obj,
            elem,
            baselines[elem],
            max_distance,
            lod=lod,
            transfer_surface=transfer_surfaces[elem],
        )
        if transfer_maps[obj]["mapped_count"] == 0:
            raise ValueError(f"{obj.name}: no vertices are within {max_distance:g} m of Ziva element {elem}")

    created = 0
    existing = 0
    empty = 0
    for channel_index, channel in enumerate(channels):
        element_outputs = {
            elem: model.evaluate_element(elem, slider_values={channel["name"]: 1.0})
            for elem in elements
        }
        for obj in objects:
            elem = int(obj.get("engine_ziva_element_index", 0))
            solver_delta = element_outputs[elem] - baselines[elem]
            local_delta = _ziva_transfer_solver_delta(transfer_maps[obj], solver_delta)
            result = _ziva_write_shape_key(obj, channel, local_delta, overwrite=overwrite)
            created += result == "CREATED"
            existing += result == "EXISTS"
            empty += result == "EMPTY"
        if progress is not None:
            progress(channel_index + 1, len(channels), channel)
    return {
        "created": int(created),
        "existing": int(existing),
        "empty": int(empty),
        "channel_count": len(channels),
        "objects": {
            obj.name: {
                "mapping": transfer_maps[obj]["type"],
                "mapped": transfer_maps[obj]["mapped_count"],
                "unmapped": transfer_maps[obj]["unmapped_count"],
                "max_distance": transfer_maps[obj]["max_distance"],
            }
            for obj in objects
        },
    }


def transfer_ziva_pose_to_objects(
    model,
    armature,
    objects,
    joint_local_matrices,
    target_name,
    target_hash=None,
    overwrite=False,
    max_distance=0.05,
    lod=0,
):
    objects = [obj for obj in objects if getattr(obj, "type", None) == 'MESH' and len(obj.data.vertices)]
    if not objects:
        raise ValueError("select at least one nonempty mesh parented to the model armature")
    target_name = str(target_name or "").strip()
    if not target_name:
        raise ValueError("pose target name cannot be empty")
    if target_hash is None:
        target_hash = string_crc32(target_name)
    channel = {
        "name": target_name,
        "hash": int(target_hash) & 0xFFFFFFFF,
        "index": -1,
        "kind": "POSE_CAPTURE",
        "driver_status": "MANUAL_DRIVER_REQUIRED",
    }
    elements = sorted({int(obj.get("engine_ziva_element_index", 0)) for obj in objects})
    baselines = {elem: model.evaluate_element(elem) for elem in elements}
    transfer_surfaces = {
        elem: _ziva_element_transfer_surface(model, elem, baselines[elem], lod=lod)
        for elem in elements
    }
    posed = {
        elem: model.evaluate_element(elem, joint_local_matrices=joint_local_matrices)
        for elem in elements
    }
    transfer_maps = {}
    for obj in objects:
        elem = int(obj.get("engine_ziva_element_index", 0))
        if not 0 <= elem < model.elem_count:
            raise ValueError(f"{obj.name}: invalid Ziva source element {elem}")
        transfer_maps[obj] = _ziva_build_object_transfer_map(
            model,
            armature,
            obj,
            elem,
            baselines[elem],
            max_distance,
            lod=lod,
            transfer_surface=transfer_surfaces[elem],
        )
        if transfer_maps[obj]["mapped_count"] == 0:
            raise ValueError(f"{obj.name}: no vertices are within {max_distance:g} m of Ziva element {elem}")

    created = 0
    existing = 0
    empty = 0
    for obj in objects:
        elem = int(obj.get("engine_ziva_element_index", 0))
        local_delta = _ziva_transfer_solver_delta(transfer_maps[obj], posed[elem] - baselines[elem])
        result = _ziva_write_shape_key(obj, channel, local_delta, overwrite=overwrite)
        created += result == "CREATED"
        existing += result == "EXISTS"
        empty += result == "EMPTY"
    return {
        "created": int(created),
        "existing": int(existing),
        "empty": int(empty),
        "target": channel,
        "objects": {
            obj.name: {
                "mapping": transfer_maps[obj]["type"],
                "mapped": transfer_maps[obj]["mapped_count"],
                "unmapped": transfer_maps[obj]["unmapped_count"],
                "max_distance": transfer_maps[obj]["max_distance"],
            }
            for obj in objects
        },
    }


def create_empty_ziva_targets(objects, channels, overwrite=False):
    import json

    created = 0
    existing = 0
    for obj in objects:
        if getattr(obj, "type", None) != 'MESH' or not len(obj.data.vertices):
            continue
        if obj.data.shape_keys is None or not obj.data.shape_keys.key_blocks:
            obj.shape_key_add(name="Basis", from_mix=False)
        metadata = _ziva_object_target_metadata(obj)
        for channel in channels:
            key = obj.data.shape_keys.key_blocks.get(channel["name"])
            if key is None:
                key = obj.shape_key_add(name=channel["name"], from_mix=False)
                created += 1
            elif not overwrite:
                existing += 1
            metadata[channel["name"]] = {
                "name": channel["name"],
                "hash": int(channel["hash"]),
                "index": int(channel["index"]),
                "ziva_kind": channel["kind"],
                "driver_status": channel["driver_status"],
            }
        obj["engine_morph_targets_json"] = json.dumps(metadata, separators=(",", ":"), sort_keys=True)
        obj["engine_morph_shape_keys_imported"] = True
    return {"created": created, "existing": existing}
