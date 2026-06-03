#!/usr/bin/env python3
"""
charon_cluster_state.py — incremental fixed-point clusterer.

Folds NEW pull requests into the EXISTING cluster state without re-clustering
all N from scratch. The vocabulary is frozen; each new batch is first clustered
among itself (small), then each sub-cluster's mean is snapped to the nearest
existing centroid above threshold — or opens a new cluster. Centroids update as
a running mean. Cost per tick: O(new^2 + new*k), not O(n^2) full re-vectorise.

The recursive fixed-point idea applied to clustering: fold each new batch into
the existing centroids instead of re-clustering from scratch. A periodic full
refit (when new/total > ~0.15) handles vocabulary drift.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

STATE = Path(__file__).resolve().parent.parent / "data" / "cluster_state.npz"
_EPS = 1e-9


@dataclass
class ClusterState:
    centroids: np.ndarray          # (k, vocab) L2-normalised cluster means
    sizes: list                    # members per cluster
    threshold: float = 0.34

    @property
    def k(self) -> int:
        return len(self.sizes)

    def add_cluster(self, rep: np.ndarray, count: int) -> int:
        row = rep.reshape(1, -1)
        self.centroids = np.vstack([self.centroids, row]) if self.centroids.size else row
        self.sizes.append(count)
        return len(self.sizes) - 1

    def merge(self, k: int, rep: np.ndarray, count: int) -> None:
        n = self.sizes[k]
        blended = (self.centroids[k] * n + rep * count) / (n + count)
        self.centroids[k] = blended / (np.linalg.norm(blended) + _EPS)
        self.sizes[k] += count


def _find(parent: list, i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _batch_clusters(vecs: np.ndarray, threshold: float) -> list:
    """Union-find sub-clusters WITHIN a new batch. Returns list of index-lists."""
    n = len(vecs)
    if n == 0:
        return []
    sim = vecs @ vecs.T
    parent = list(range(n))
    edges = np.argwhere(np.triu(sim >= threshold, k=1))
    for a, b in edges:
        ra, rb = _find(parent, int(a)), _find(parent, int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    groups: dict[int, list] = {}
    for i in range(n):
        groups.setdefault(_find(parent, i), []).append(i)
    return list(groups.values())


def _nearest(rep: np.ndarray, centroids: np.ndarray, threshold: float):
    """Index of the nearest centroid at/above threshold, or None to spawn."""
    if not len(centroids):
        return None
    sims = centroids @ rep
    k = int(sims.argmax())
    return k if sims[k] >= threshold else None


def _place(state: ClusterState, vecs: np.ndarray, members: list, assignments: list) -> None:
    """Merge one new sub-cluster into the state; record its cluster id per member."""
    rep = vecs[members].mean(axis=0)
    rep = rep / (np.linalg.norm(rep) + _EPS)
    k = _nearest(rep, state.centroids, state.threshold)
    if k is None:
        k = state.add_cluster(rep, len(members))
    else:
        state.merge(k, rep, len(members))
    for m in members:
        assignments[m] = k


def fold_new_souls(state: ClusterState, vecs: np.ndarray) -> list:
    """Fold a new batch of L2-normalised row vectors into the state.

    Returns the per-row cluster assignments. Existing clusters are unchanged
    except where a new sub-cluster merges into them (running mean).
    """
    assignments = [-1] * len(vecs)
    for members in _batch_clusters(vecs, state.threshold):
        _place(state, vecs, members, assignments)
    return assignments


def build_initial_state(vecs: np.ndarray, labels: list, threshold: float = 0.34) -> ClusterState:
    """Seed the state from a full first-pass clustering (charon.cluster_souls)."""
    k = (max(labels) + 1) if labels else 0
    centroids = np.zeros((k, vecs.shape[1]), dtype="float32")
    sizes = [0] * k
    for i, lab in enumerate(labels):
        centroids[lab] += vecs[i]
        sizes[lab] += 1
    for j in range(k):
        if sizes[j]:
            centroids[j] = centroids[j] / sizes[j]
            centroids[j] = centroids[j] / (np.linalg.norm(centroids[j]) + _EPS)
    return ClusterState(centroids=centroids, sizes=sizes, threshold=threshold)


def save_state(state: ClusterState, path: Path = STATE) -> None:
    path.parent.mkdir(exist_ok=True)
    np.savez(path, centroids=state.centroids,
             sizes=np.array(state.sizes), threshold=state.threshold)


def load_state(path: Path = STATE) -> ClusterState:
    d = np.load(path)
    return ClusterState(centroids=d["centroids"], sizes=list(d["sizes"]),
                        threshold=float(d["threshold"]))


if __name__ == "__main__":  # tiny self-test: deterministic, no network
    rng = np.random.default_rng(67)
    base = rng.standard_normal((3, 16)).astype("float32")
    base /= np.linalg.norm(base, axis=1, keepdims=True)
    labels = [0, 0, 1, 1, 2]
    vecs = np.vstack([base[lbl] + 0.01 * rng.standard_normal(16) for lbl in labels]).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    st = build_initial_state(vecs, labels)
    # a new soul near cluster 0 should snap to 0, not spawn
    newv = (base[0] + 0.01 * rng.standard_normal(16)).astype("float32")
    newv /= np.linalg.norm(newv)
    got = fold_new_souls(st, newv.reshape(1, -1))
    print(json.dumps({"initial_k": 3, "new_assignment": got, "k_after": st.k,
                      "ok": got == [0] and st.k == 3}))
