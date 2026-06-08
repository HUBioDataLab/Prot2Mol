import pytest

from prot2mol.training.distributed import resolve_distributed_context


def test_auto_mode_defaults_to_single_gpu_without_dist_env():
    ctx = resolve_distributed_context("auto", env={})
    assert ctx.effective_mode == "single_gpu"
    assert ctx.is_distributed is False
    assert ctx.world_size == 1
    assert ctx.global_rank == 0
    assert ctx.local_rank == -1


def test_auto_mode_detects_single_node_multi_gpu():
    ctx = resolve_distributed_context(
        "auto",
        env={
            "WORLD_SIZE": "4",
            "LOCAL_WORLD_SIZE": "4",
            "RANK": "2",
            "LOCAL_RANK": "2",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        },
    )
    assert ctx.effective_mode == "multi_gpu"
    assert ctx.is_distributed is True
    assert ctx.world_size == 4
    assert ctx.local_world_size == 4
    assert ctx.global_rank == 2
    assert ctx.local_rank == 2


def test_auto_mode_detects_multi_node():
    ctx = resolve_distributed_context(
        "auto",
        env={
            "WORLD_SIZE": "8",
            "LOCAL_WORLD_SIZE": "4",
            "RANK": "5",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "node0",
            "MASTER_PORT": "29500",
        },
    )
    assert ctx.effective_mode == "multi_node"
    assert ctx.is_distributed is True
    assert ctx.world_size == 8
    assert ctx.local_world_size == 4


def test_single_gpu_mode_rejects_distributed_world_size():
    with pytest.raises(ValueError):
        resolve_distributed_context("single_gpu", env={"WORLD_SIZE": "2", "LOCAL_WORLD_SIZE": "2"})


def test_multi_gpu_mode_requires_distributed_launch():
    with pytest.raises(ValueError):
        resolve_distributed_context(
            "multi_gpu",
            env={
                "WORLD_SIZE": "1",
                "LOCAL_WORLD_SIZE": "1",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            },
        )


def test_auto_distributed_mode_requires_rank_env():
    with pytest.raises(ValueError, match="Distributed launch environment is incomplete"):
        resolve_distributed_context(
            "auto",
            env={"WORLD_SIZE": "4", "LOCAL_WORLD_SIZE": "4"},
        )


def test_multi_gpu_mode_requires_rendezvous_env():
    with pytest.raises(ValueError, match="MASTER_ADDR, MASTER_PORT"):
        resolve_distributed_context(
            "multi_gpu",
            env={
                "WORLD_SIZE": "4",
                "LOCAL_WORLD_SIZE": "4",
                "RANK": "0",
                "LOCAL_RANK": "0",
            },
        )


def test_multi_gpu_mode_rejects_multi_node_shape():
    with pytest.raises(ValueError):
        resolve_distributed_context(
            "multi_gpu",
            env={
                "WORLD_SIZE": "8",
                "LOCAL_WORLD_SIZE": "4",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "node0",
                "MASTER_PORT": "29500",
            },
        )


def test_multi_node_mode_requires_world_size_greater_than_local_world_size():
    with pytest.raises(ValueError):
        resolve_distributed_context(
            "multi_node",
            env={
                "WORLD_SIZE": "4",
                "LOCAL_WORLD_SIZE": "4",
                "RANK": "0",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "node0",
                "MASTER_PORT": "29500",
            },
        )


def test_multi_node_mode_accepts_valid_shape():
    ctx = resolve_distributed_context(
        "multi_node",
        env={
            "WORLD_SIZE": "16",
            "LOCAL_WORLD_SIZE": "8",
            "RANK": "9",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "node0",
            "MASTER_PORT": "29500",
        },
    )
    assert ctx.effective_mode == "multi_node"
    assert ctx.is_distributed is True
    assert ctx.global_rank == 9
    assert ctx.local_rank == 1


def test_invalid_mode_raises():
    with pytest.raises(ValueError):
        resolve_distributed_context("banana", env={})
