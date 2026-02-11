import os
from dataclasses import dataclass
from typing import Mapping, Optional

VALID_TRAINING_MODES = ("auto", "single_gpu", "multi_gpu", "multi_node")


def _safe_int(value, default: int) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class DistributedContext:
    requested_mode: str
    effective_mode: str
    world_size: int
    local_world_size: int
    global_rank: int
    local_rank: int
    node_rank: int
    is_distributed: bool

    @property
    def is_main_process(self) -> bool:
        return self.global_rank == 0

    def to_log_fields(self) -> dict:
        return {
            "requested_mode": self.requested_mode,
            "effective_mode": self.effective_mode,
            "world_size": self.world_size,
            "local_world_size": self.local_world_size,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "node_rank": self.node_rank,
            "is_distributed": self.is_distributed,
        }


def _infer_mode_from_env(world_size: int, local_world_size: int) -> str:
    if world_size <= 1:
        return "single_gpu"
    if world_size > local_world_size:
        return "multi_node"
    return "multi_gpu"


def resolve_distributed_context(
    requested_mode: str = "auto",
    env: Optional[Mapping[str, str]] = None,
) -> DistributedContext:
    environment = os.environ if env is None else env
    mode = (requested_mode or "auto").strip().lower()

    if mode not in VALID_TRAINING_MODES:
        raise ValueError(
            f"Invalid training_mode '{requested_mode}'. "
            f"Valid options: {VALID_TRAINING_MODES}"
        )

    world_size = max(1, _safe_int(environment.get("WORLD_SIZE"), 1))
    local_world_size_default = world_size if world_size > 1 else 1
    local_world_size = max(1, _safe_int(environment.get("LOCAL_WORLD_SIZE"), local_world_size_default))
    global_rank = _safe_int(environment.get("RANK"), 0)
    local_rank = _safe_int(environment.get("LOCAL_RANK"), 0)
    node_rank = _safe_int(
        environment.get("GROUP_RANK", environment.get("NODE_RANK")),
        0,
    )

    inferred_mode = _infer_mode_from_env(world_size=world_size, local_world_size=local_world_size)
    effective_mode = inferred_mode if mode == "auto" else mode

    if mode == "single_gpu" and world_size > 1:
        raise ValueError(
            "training_mode=single_gpu requires WORLD_SIZE=1. "
            "Launch without torchrun."
        )
    if mode == "multi_gpu":
        if world_size <= 1:
            raise ValueError(
                "training_mode=multi_gpu requires distributed launch "
                "(for example: torchrun --standalone --nproc_per_node=<N>)."
            )
        if world_size != local_world_size:
            raise ValueError(
                "training_mode=multi_gpu expects single-node launch. "
                "Detected WORLD_SIZE != LOCAL_WORLD_SIZE, which indicates multi-node."
            )
    if mode == "multi_node" and world_size <= local_world_size:
        raise ValueError(
            "training_mode=multi_node requires multi-node launch "
            "(WORLD_SIZE must be greater than LOCAL_WORLD_SIZE)."
        )

    is_distributed = effective_mode in {"multi_gpu", "multi_node"} and world_size > 1
    if is_distributed:
        if global_rank < 0 or global_rank >= world_size:
            raise ValueError(
                f"Invalid distributed rank: RANK={global_rank}, WORLD_SIZE={world_size}"
            )
        if local_rank < 0 or local_rank >= local_world_size:
            raise ValueError(
                f"Invalid local rank: LOCAL_RANK={local_rank}, LOCAL_WORLD_SIZE={local_world_size}"
            )
    else:
        world_size = 1
        local_world_size = 1
        global_rank = 0
        local_rank = -1
        node_rank = 0

    return DistributedContext(
        requested_mode=mode,
        effective_mode=effective_mode,
        world_size=world_size,
        local_world_size=local_world_size,
        global_rank=global_rank,
        local_rank=local_rank,
        node_rank=node_rank,
        is_distributed=is_distributed,
    )
