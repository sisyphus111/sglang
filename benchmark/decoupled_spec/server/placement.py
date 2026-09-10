"""Joint node-level GPU placement for decoupled-spec Engine actors."""

from __future__ import annotations

from itertools import combinations
from typing import Any

import msgspec


class CandidateNode(msgspec.Struct, frozen=True):
    """Available GPU capacity reported for one live Ray node."""

    node_id: str
    node_ip: str
    available_gpus: int


class NodeAllocation(msgspec.Struct, frozen=True):
    """Final verifier and drafter GPU accounting for one planned node."""

    node_id: str
    node_ip: str
    capacity: int
    verifier_gpus: int
    drafter_gpus: int
    free_gpus: int


class PlacementPlan(msgspec.Struct, frozen=True):
    """Joint verifier and drafter placement consumed by runtime startup."""

    verifier_nodes_per_replica: int
    verifier_gpus_per_node: int
    verifier_bundle_node_ids: list[list[str]]
    drafter_node_ids: list[str]
    node_allocations: list[NodeAllocation]

    def to_dict(self) -> dict[str, Any]:
        return msgspec.to_builtins(self)


def get_alive_gpu_nodes() -> list[CandidateNode]:
    """Return alive Ray nodes with currently available GPU capacity."""

    import ray

    available_resources = ray._private.state.available_resources_per_node()
    node_infos = {node["NodeID"]: node for node in ray.nodes() if node.get("Alive")}
    candidates = []
    for node_id, node in node_infos.items():
        resources = available_resources.get(node_id, {})
        available_gpus = int(resources.get("GPU", resources.get("NPU", 0)))
        if available_gpus <= 0:
            continue
        candidates.append(
            CandidateNode(
                node_id=str(node_id),
                node_ip=str(node["NodeManagerAddress"]),
                available_gpus=available_gpus,
            )
        )
    candidates.sort(key=lambda item: (-item.available_gpus, item.node_ip))
    return candidates


def plan_decoupled_spec_placement(
    candidate_nodes: list[CandidateNode],
    *,
    num_verifiers: int,
    target_tp_size: int,
    num_drafters: int,
    draft_tp_size: int,
) -> PlacementPlan:
    """Jointly place balanced verifier node shards and single-node drafters.

    The objective order matches the v0.5.14 orchestrator: minimize verifier
    node count first, then physical nodes, then prefer verifier/drafter
    colocation and balanced residual load.
    """

    nodes = list(candidate_nodes)
    if not nodes:
        raise ValueError("no alive Ray nodes with available GPUs")
    if num_verifiers <= 0 or target_tp_size <= 0:
        raise ValueError("num_verifiers and target_tp_size must be positive")
    if num_drafters < 0 or draft_tp_size <= 0:
        raise ValueError(
            "num_drafters must be non-negative and draft_tp_size positive"
        )

    node_ids = [node.node_id for node in nodes]
    if len(set(node_ids)) != len(node_ids):
        raise ValueError(f"candidate node ids must be unique: {node_ids}")
    capacities = [node.available_gpus for node in nodes]
    if any(capacity <= 0 for capacity in capacities):
        raise ValueError(f"candidate GPU capacities must be positive: {capacities}")

    best_plan: PlacementPlan | None = None
    best_score: tuple[Any, ...] | None = None
    placement_policies = ("pack_tight", "pack_dense", "spread", "exact")

    def exact_target_assignment(
        verifier_nodes_per_replica: int,
        verifier_gpus_per_node: int,
    ) -> tuple[list[list[int]], list[int], list[int]] | None:
        remaining = list(capacities)
        groups: list[list[int]] = []
        failed_states: set[tuple[int, tuple[int, ...]]] = set()

        def search(verifier_rank: int) -> bool:
            if verifier_rank == num_verifiers:
                return (
                    sum(free_gpus // draft_tp_size for free_gpus in remaining)
                    >= num_drafters
                )
            required_verifier_gpus = (
                (num_verifiers - verifier_rank)
                * verifier_nodes_per_replica
                * verifier_gpus_per_node
            )
            if sum(remaining) < required_verifier_gpus + num_drafters * draft_tp_size:
                return False
            eligible = [
                index
                for index, free_gpus in enumerate(remaining)
                if free_gpus >= verifier_gpus_per_node
            ]
            if len(eligible) < verifier_nodes_per_replica:
                return False

            state = (verifier_rank, tuple(sorted(remaining, reverse=True)))
            if state in failed_states:
                return False
            seen_capacity_signatures: set[tuple[int, ...]] = set()
            for selected in combinations(eligible, verifier_nodes_per_replica):
                signature = tuple(sorted(remaining[index] for index in selected))
                if signature in seen_capacity_signatures:
                    continue
                seen_capacity_signatures.add(signature)
                for index in selected:
                    remaining[index] -= verifier_gpus_per_node
                groups.append(list(selected))
                if search(verifier_rank + 1):
                    return True
                groups.pop()
                for index in selected:
                    remaining[index] += verifier_gpus_per_node

            failed_states.add(state)
            return False

        if not search(0):
            return None
        verifier_gpu_counts = [
            capacities[index] - remaining[index] for index in range(len(nodes))
        ]
        return groups, remaining, verifier_gpu_counts

    max_verifier_nodes = min(len(nodes), target_tp_size)
    for verifier_nodes_per_replica in range(1, max_verifier_nodes + 1):
        if target_tp_size % verifier_nodes_per_replica != 0:
            continue
        verifier_gpus_per_node = target_tp_size // verifier_nodes_per_replica
        if (
            sum(capacity >= verifier_gpus_per_node for capacity in capacities)
            < verifier_nodes_per_replica
        ):
            continue

        found_for_node_count = False
        for policy_index, policy in enumerate(placement_policies):
            if policy == "exact" and found_for_node_count:
                continue
            remaining = list(capacities)
            verifier_gpu_counts = [0] * len(nodes)
            verifier_node_indices: list[list[int]] = []
            feasible = True
            if policy == "exact":
                assignment = exact_target_assignment(
                    verifier_nodes_per_replica,
                    verifier_gpus_per_node,
                )
                if assignment is None:
                    continue
                verifier_node_indices, remaining, verifier_gpu_counts = assignment
            else:
                for _ in range(num_verifiers):
                    group = []
                    for _ in range(verifier_nodes_per_replica):
                        eligible = [
                            index
                            for index, free_gpus in enumerate(remaining)
                            if index not in group
                            and free_gpus >= verifier_gpus_per_node
                        ]
                        if not eligible:
                            feasible = False
                            break
                        if policy == "pack_tight":
                            selected = min(
                                eligible,
                                key=lambda index: (
                                    0 if verifier_gpu_counts[index] else 1,
                                    remaining[index] - verifier_gpus_per_node,
                                    index,
                                ),
                            )
                        elif policy == "pack_dense":
                            selected = min(
                                eligible,
                                key=lambda index: (
                                    0 if verifier_gpu_counts[index] else 1,
                                    -remaining[index],
                                    index,
                                ),
                            )
                        else:
                            selected = min(
                                eligible,
                                key=lambda index: (
                                    -remaining[index],
                                    verifier_gpu_counts[index],
                                    index,
                                ),
                            )
                        group.append(selected)
                        remaining[selected] -= verifier_gpus_per_node
                        verifier_gpu_counts[selected] += verifier_gpus_per_node
                    if not feasible:
                        break
                    verifier_node_indices.append(group)
            if not feasible:
                continue

            drafter_gpu_counts = [0] * len(nodes)
            drafter_node_indices = []
            for _ in range(num_drafters):
                eligible = [
                    index
                    for index, free_gpus in enumerate(remaining)
                    if free_gpus >= draft_tp_size
                ]
                if not eligible:
                    feasible = False
                    break
                selected = min(
                    eligible,
                    key=lambda index: (
                        0 if verifier_gpu_counts[index] else 1,
                        remaining[index] - draft_tp_size,
                        index,
                    ),
                )
                drafter_node_indices.append(selected)
                remaining[selected] -= draft_tp_size
                drafter_gpu_counts[selected] += draft_tp_size
            if not feasible:
                continue
            found_for_node_count = True

            used_indices = [
                index
                for index in range(len(nodes))
                if verifier_gpu_counts[index] or drafter_gpu_counts[index]
            ]
            mixed_nodes = sum(
                bool(verifier_gpu_counts[index] and drafter_gpu_counts[index])
                for index in used_indices
            )
            used_gpu_counts = [
                verifier_gpu_counts[index] + drafter_gpu_counts[index]
                for index in used_indices
            ]
            verifier_bundle_node_ids = [
                [node_ids[index] for index in group] for group in verifier_node_indices
            ]
            drafter_node_ids = [node_ids[index] for index in drafter_node_indices]
            score = (
                verifier_nodes_per_replica,
                len(used_indices),
                -mixed_nodes,
                max(used_gpu_counts) - min(used_gpu_counts),
                policy_index,
                verifier_bundle_node_ids,
                drafter_node_ids,
            )
            if best_score is not None and score >= best_score:
                continue

            best_score = score
            best_plan = PlacementPlan(
                verifier_nodes_per_replica=verifier_nodes_per_replica,
                verifier_gpus_per_node=verifier_gpus_per_node,
                verifier_bundle_node_ids=verifier_bundle_node_ids,
                drafter_node_ids=drafter_node_ids,
                node_allocations=[
                    NodeAllocation(
                        node_id=node_ids[index],
                        node_ip=nodes[index].node_ip,
                        capacity=capacities[index],
                        verifier_gpus=verifier_gpu_counts[index],
                        drafter_gpus=drafter_gpu_counts[index],
                        free_gpus=remaining[index],
                    )
                    for index in used_indices
                ],
            )

    if best_plan is None:
        raise ValueError(
            "unable to jointly place decoupled-spec actors: "
            f"verifiers={num_verifiers}xTP{target_tp_size}, "
            f"drafters={num_drafters}xTP{draft_tp_size}, "
            f"node_capacities={dict(zip(node_ids, capacities, strict=True))}"
        )
    return best_plan


def plan_spec_engine_placement(
    candidate_nodes: list[CandidateNode],
    *,
    num_replicas: int,
    tp_size: int,
) -> PlacementPlan:
    """Place coupled speculative engines using the primary-engine policy."""

    return plan_decoupled_spec_placement(
        candidate_nodes,
        num_verifiers=num_replicas,
        target_tp_size=tp_size,
        num_drafters=0,
        draft_tp_size=1,
    )


__all__ = [
    "CandidateNode",
    "NodeAllocation",
    "PlacementPlan",
    "get_alive_gpu_nodes",
    "plan_decoupled_spec_placement",
    "plan_spec_engine_placement",
]
