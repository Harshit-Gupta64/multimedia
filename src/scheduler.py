"""
Scheduler Module v0.4 - Dynamic Routing, Suitability Score, Node Endpoint Registry
------------------------------------------------------------------------------------
New in v0.4:
  - Each EdgeNode carries an endpoint URL used by server.py to physically
    forward obfuscated payloads via httpx.
  - Dev-mode endpoint overrides (localhost ports 8001-8003) and production
    Docker/K3s FQDNs are both supported via environment variables.
  - DynamicScheduler.get_node_endpoint(node_id) for O(1) lookup in forwarder.
"""
import os
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class EdgeNode:
    """Represents a single simulated edge compute node."""
    node_id:       str
    cpu_load:      float          # 0.0 – 100.0 %
    region:        str
    endpoint:      str            # HTTP endpoint for workload forwarding
    max_queue:     int = 10
    current_queue: int = 0

    def is_available(self) -> bool:
        return self.current_queue < self.max_queue

    def suitability_score(
        self,
        semantic_weight: int,
        high_load_threshold: float = 75.0,
    ) -> Optional[float]:
        """
        (100 - cpu_load) / (semantic_weight + 1)
        Returns None when the node is ineligible.
        Ineligibility: high-density frame (>5 obj) on overloaded node (>75% CPU)
        OR queue is full.
        """
        if semantic_weight > 5 and self.cpu_load > high_load_threshold:
            return None
        if not self.is_available():
            return None
        return (100.0 - self.cpu_load) / (semantic_weight + 1)


@dataclass
class RoutingDecision:
    """Encapsulates the outcome of a routing decision."""
    destination:       Literal["edge", "cloud", "drop"]
    node_id:           Optional[str]
    node_cpu_load:     Optional[float]
    node_endpoint:     Optional[str]          # <- NEW: forwarding URL
    priority:          int
    suitability_score: Optional[float]
    reason:            str
    semantic_weight:   int
    timestamp:         float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# DynamicScheduler
# ---------------------------------------------------------------------------

# Default endpoint templates – overridable via environment variables.
# Dev mode   : NODE_<ID>_ENDPOINT=http://127.0.0.1:800X
# Docker mode: NODE_<ID>_ENDPOINT=http://edge-node-alpha:8000
# K3s mode   : NODE_<ID>_ENDPOINT=http://edge-node-alpha.edge-analytics.svc.cluster.local:8000

_DEFAULT_ENDPOINTS: Dict[str, str] = {
    "edge-node-01": os.getenv("NODE_01_ENDPOINT", "http://127.0.0.1:8001"),
    "edge-node-02": os.getenv("NODE_02_ENDPOINT", "http://127.0.0.1:8002"),
    "edge-node-03": os.getenv("NODE_03_ENDPOINT", "http://127.0.0.1:8003"),
}


class DynamicScheduler:
    """
    Manages a pool of simulated edge nodes and routes incoming frames based on
    semantic complexity and suitability score. Each node carries a live endpoint
    URL so the server layer can forward payloads directly.
    """

    HIGH_SEMANTIC_THRESHOLD:    int   = 5
    HIGH_LOAD_THRESHOLD:        float = 75.0
    CLOUD_FALLBACK_CPU_THRESHOLD: float = 90.0

    def __init__(
        self,
        num_nodes: int = 3,
        seed: Optional[int] = None,
        node_endpoints: Optional[Dict[str, str]] = None,
    ):
        self._rng            = random.Random(seed)
        self._endpoints      = node_endpoints or _DEFAULT_ENDPOINTS
        self.nodes: List[EdgeNode]          = self._init_nodes(num_nodes)
        self._routing_history: List[RoutingDecision] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_nodes(self, n: int) -> List[EdgeNode]:
        regions = ["us-west", "eu-central", "ap-southeast"]
        nodes   = []
        for i in range(n):
            node_id  = f"edge-node-{i+1:02d}"
            endpoint = self._endpoints.get(node_id, f"http://127.0.0.1:{8001+i}")
            nodes.append(EdgeNode(
                node_id=node_id,
                cpu_load=self._rng.uniform(0, 100),
                region=regions[i % len(regions)],
                endpoint=endpoint,
            ))
        return nodes

    def _refresh_cpu_loads(self) -> None:
        for node in self.nodes:
            delta = self._rng.uniform(-15, 15)
            node.cpu_load = max(0.0, min(100.0, node.cpu_load + delta))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_node_endpoint(self, node_id: str) -> Optional[str]:
        """O(1) lookup of a node's forwarding endpoint by ID."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n.endpoint
        return None

    def get_node_status(self) -> List[Dict]:
        return [
            {
                "node_id":       n.node_id,
                "region":        n.region,
                "endpoint":      n.endpoint,
                "cpu_load":      round(n.cpu_load, 1),
                "current_queue": n.current_queue,
                "available":     n.is_available(),
            }
            for n in self.nodes
        ]

    def route_frame(
        self,
        semantic_weight: int,
        latency_ms: float = 0.0,
    ) -> RoutingDecision:
        self._refresh_cpu_loads()

        scored: List[tuple[float, EdgeNode]] = []
        for node in self.nodes:
            score = node.suitability_score(semantic_weight, self.HIGH_LOAD_THRESHOLD)
            if score is not None:
                scored.append((score, node))

        if not scored:
            decision = RoutingDecision(
                destination="cloud",
                node_id=None,
                node_cpu_load=None,
                node_endpoint=None,
                priority=2,
                suitability_score=None,
                reason=(
                    f"No eligible edge node for semantic_weight={semantic_weight} "
                    f"(all overloaded or CPU>{self.HIGH_LOAD_THRESHOLD}%)"
                ),
                semantic_weight=semantic_weight,
            )
        else:
            best_score, best_node = max(scored, key=lambda x: x[0])
            best_node.current_queue += 1

            priority = 2 if semantic_weight > self.HIGH_SEMANTIC_THRESHOLD else (
                1 if semantic_weight > 0 else 0
            )
            decision = RoutingDecision(
                destination="edge",
                node_id=best_node.node_id,
                node_cpu_load=round(best_node.cpu_load, 1),
                node_endpoint=best_node.endpoint,
                priority=priority,
                suitability_score=round(best_score, 4),
                reason=(
                    f"Node {best_node.node_id} selected "
                    f"(CPU={best_node.cpu_load:.1f}%, score={best_score:.4f}, "
                    f"semantic_weight={semantic_weight})"
                ),
                semantic_weight=semantic_weight,
            )

        self._routing_history.append(decision)
        return decision

    def get_routing_stats(self) -> Dict:
        if not self._routing_history:
            return {"total": 0}
        total      = len(self._routing_history)
        edge_count = sum(1 for d in self._routing_history if d.destination == "edge")
        cloud_count= sum(1 for d in self._routing_history if d.destination == "cloud")
        avg_weight = sum(d.semantic_weight for d in self._routing_history) / total
        node_counts: Dict[str, int] = {}
        for d in self._routing_history:
            if d.node_id:
                node_counts[d.node_id] = node_counts.get(d.node_id, 0) + 1
        return {
            "total_frames":        total,
            "edge_routed":         edge_count,
            "cloud_routed":        cloud_count,
            "edge_pct":            round(edge_count  / total * 100, 1),
            "cloud_pct":           round(cloud_count / total * 100, 1),
            "avg_semantic_weight": round(avg_weight, 2),
            "node_distribution":   node_counts,
        }


# Backward-compat alias
SemanticScheduler = DynamicScheduler
