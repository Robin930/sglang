"""Unit tests for replicated-KV writer election across heterogeneous attention TP.

When prefill attention TP exceeds both the decode TP and the KV head count,
several prefill ranks hold the same KV head and would write identical bytes into
one decode rank. Exactly one of them must write; electing none loses KV silently
and electing two only wastes bandwidth, so both directions are asserted.

Expected replica counts are derived by hand from the head-distribution rules.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.disaggregation.common.conn import CommonKVManager
from sglang.srt.disaggregation.common.staging_buffer import (
    compute_head_slice_params,
    compute_staging_layout,
    staging_writer_ranks,
    staging_writer_slot,
)
from sglang.srt.disaggregation.common.utils import (
    kv_replicas_per_destination,
    should_send_kv_replica,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

ROOMS = [0, 1, 7, 2**31 - 1, 2**63 - 25, 914_582_337_104_221]
# (prefill attn TP, decode attn TP, total KV heads) with P TP > D TP.
GATHER_CONFIGS = [
    (src, dst, heads)
    for src in (2, 4, 8)
    for dst in (1, 2, 4)
    for heads in (1, 2, 4, 8, 16)
    if src > dst
]


def source_group(src_tp, dst_tp, dst_rank):
    size = src_tp // dst_tp
    return range(dst_rank * size, (dst_rank + 1) * size)


def source_head(src_tp, dst_tp, src_rank, dst_rank, heads):
    """Decode-side head offset this prefill rank's slice lands on."""
    return compute_head_slice_params(src_tp, dst_tp, src_rank, dst_rank, heads)[2]


def elected_writers(room, src_tp, dst_tp, dst_rank, heads):
    return [
        rank
        for rank in source_group(src_tp, dst_tp, dst_rank)
        if should_send_kv_replica(
            room=room,
            src_tp=src_tp,
            dst_tp=dst_tp,
            src_tp_rank=rank,
            dst_tp_rank=dst_rank,
            total_kv_heads=heads,
        )
    ]


class TestKvReplicasPerDestination(CustomTestCase):
    def test_replicated_heads_are_counted_per_destination_group(self):
        cases = {
            # MQA P8 -> D1: all eight ranks hold the only head.
            (8, 1, 1): 8,
            # GQA P8 -> D1 with 2 heads: ranks 0-3 hold head 0, 4-7 head 1.
            (8, 1, 2): 4,
            # P8 -> D2 with 4 heads: D0's group {0..3} holds heads 0,0,1,1.
            (8, 2, 4): 2,
            # P8 -> D4 with 2 heads: D0's group {0,1} both hold head 0.
            (8, 4, 2): 2,
            (8, 4, 4): 2,
        }
        for (src, dst, heads), expected in cases.items():
            self.assertEqual(kv_replicas_per_destination(src, dst, heads), expected)

    def test_layouts_without_replication_keep_every_writer(self):
        cases = [
            (8, 8, 1),  # Equal TP: one source per destination.
            (2, 8, 1),  # Scatter: each destination has one source.
            (8, 2, 8),  # Every prefill rank owns a distinct head.
            (4, 2, 16),
            (6, 2, 4),  # Heads do not tile the prefill ranks.
            (8, 2, 3),
        ]
        for src, dst, heads in cases:
            self.assertEqual(kv_replicas_per_destination(src, dst, heads), 1)


class TestShouldSendKvReplica(CustomTestCase):
    def test_each_destination_head_gets_exactly_one_writer(self):
        for src, dst, heads in GATHER_CONFIGS:
            for room in ROOMS:
                for dst_rank in range(dst):
                    writers = elected_writers(room, src, dst, dst_rank, heads)
                    group_heads = {
                        source_head(src, dst, r, dst_rank, heads)
                        for r in source_group(src, dst, dst_rank)
                    }
                    written_heads = [
                        source_head(src, dst, r, dst_rank, heads) for r in writers
                    ]
                    self.assertEqual(
                        sorted(written_heads),
                        sorted(group_heads),
                        f"src={src} dst={dst} heads={heads} room={room}",
                    )

    def test_ranks_outside_the_destination_group_never_write(self):
        for room in ROOMS:
            for src_rank in range(4, 8):
                self.assertFalse(
                    should_send_kv_replica(
                        room=room,
                        src_tp=8,
                        dst_tp=2,
                        src_tp_rank=src_rank,
                        dst_tp_rank=0,
                        total_kv_heads=1,
                    )
                )

    def test_rooms_spread_writes_across_replicas(self):
        chosen = {elected_writers(room, 8, 1, 0, 1)[0] for room in range(256)}
        self.assertEqual(chosen, set(range(8)))

    def test_election_ignores_the_global_rank_offset(self):
        for room in ROOMS:
            for rank in range(8):
                args = dict(room=room, src_tp=8, dst_tp=1, total_kv_heads=1)
                self.assertEqual(
                    should_send_kv_replica(src_tp_rank=rank, dst_tp_rank=0, **args),
                    should_send_kv_replica(src_tp_rank=rank + 8, dst_tp_rank=3, **args),
                )


class TestStagingLayoutWithElection(CustomTestCase):
    def test_elected_writers_fill_distinct_slots_of_their_own_head(self):
        for src, dst, heads in GATHER_CONFIGS:
            for room in ROOMS:
                for dst_rank in range(dst):
                    slot_owners = staging_writer_ranks(src, dst, dst_rank, heads)
                    writers = elected_writers(room, src, dst, dst_rank, heads)
                    slots = [staging_writer_slot(src, dst, r, heads) for r in writers]
                    self.assertEqual(sorted(slots), list(range(len(slot_owners))))
                    for writer, slot in zip(writers, slots):
                        self.assertEqual(
                            source_head(src, dst, writer, dst_rank, heads),
                            source_head(src, dst, slot_owners[slot], dst_rank, heads),
                        )

    def test_staging_reserves_one_region_per_distinct_head(self):
        tokens, bytes_per_head_token, layers = 16, 256, 3
        cases = {
            # (src, dst, heads): (writers, heads per writer)
            (8, 1, 1): (1, 1),
            (8, 1, 2): (2, 1),
            (8, 2, 4): (2, 1),
            (4, 2, 8): (2, 2),
        }
        for (src, dst, heads), (writers, heads_per_writer) in cases.items():
            num_writers, writer_bytes, total = compute_staging_layout(
                src, dst, 0, heads, tokens, bytes_per_head_token, layers
            )
            region = tokens * heads_per_writer * bytes_per_head_token * layers * 2
            self.assertEqual(num_writers, writers)
            self.assertEqual(writer_bytes, [region] * writers)
            self.assertEqual(total, region * writers)


class TestManagerShouldSendKv(CustomTestCase):
    def _manager(self, rank, *, heads=1, mla=False, hybrid_mla=False):
        mgr = object.__new__(CommonKVManager)
        mgr.attn_tp_size = 8
        mgr.is_mla_backend = mla
        mgr.is_hybrid_mla_backend = hybrid_mla
        mgr.kv_args = SimpleNamespace(
            engine_rank=rank, total_kv_head_num=heads, kv_head_num=1
        )
        return mgr

    def _senders(self, room, **kwargs):
        return [
            rank
            for rank in range(8)
            if self._manager(rank, **kwargs).should_send_kv(room, 1, 0)
        ]

    def test_gqa_elects_one_sender_per_head(self):
        for room in ROOMS:
            senders = self._senders(room, heads=2)
            self.assertEqual(len(senders), 2)
            self.assertEqual({rank // 4 for rank in senders}, {0, 1})

    def test_mla_keeps_decode_side_source_selection(self):
        # Decode marks all but one MLA source as dummy; those never reach here.
        for room in ROOMS:
            self.assertEqual(self._senders(room, mla=True), list(range(8)))

    def test_hybrid_mla_latent_counts_as_one_replicated_head(self):
        for room in ROOMS:
            self.assertEqual(len(self._senders(room, heads=0, hybrid_mla=True)), 1)


if __name__ == "__main__":
    unittest.main()
