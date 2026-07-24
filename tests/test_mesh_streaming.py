"""The large-segment memory fix: block tiling + a lazy (streaming) block source."""

import types

import numpy as np

from em_seg_morpho.config import MeshConfig
from em_seg_morpho.mesh import block_boxes, should_chunk, stream_block_masks


def test_block_boxes_tile_with_halo_and_clip():
    boxes = block_boxes((0, 0, 0, 16, 16, 16), (8, 8, 8), halo=1)
    assert len(boxes) == 8                       # 2x2x2 blocks
    # interior corner block gets a halo on the +side, clipped to the bbox on the -side
    assert boxes[0] == (0, 0, 0, 9, 9, 9)        # -side clipped to 0, +side +halo
    last = boxes[-1]
    assert last == (7, 7, 7, 16, 16, 16)         # -side -halo, +side clipped to 16


def test_stream_block_masks_is_lazy_one_block_at_a_time():
    boxes = block_boxes((0, 0, 0, 16, 16, 16), (8, 8, 8), halo=0)
    reads = []

    def read_box(b):
        reads.append(b)                          # record when each block is actually read
        return np.full((b[3] - b[0], b[4] - b[1], b[5] - b[2]), 5, dtype=np.uint64)

    gen = stream_block_masks(read_box, boxes, segment_id=5)
    assert isinstance(gen, types.GeneratorType)
    assert reads == []                           # nothing read until iterated

    first = next(gen)
    assert reads == [boxes[0]]                   # exactly one block read so far
    assert first.dtype == bool and first.all()   # (== segment_id) -> boolean mask

    rest = list(gen)
    assert len(rest) == len(boxes) - 1
    assert reads == boxes                        # each block read exactly once, in order


def test_should_chunk_threshold():
    cfg = MeshConfig(max_mask_voxels=8 ** 3)
    assert should_chunk((16, 16, 16), cfg) is True    # 4096 > 512
    assert should_chunk((8, 8, 8), cfg) is False       # 512 == 512, not over
