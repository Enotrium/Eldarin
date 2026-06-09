"""
Eldarin: Sparse Distributed Memory (SDM) Layer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implements the Sparse Distributed Memory model from Kanerva (1988, 2009)
as described in Section 2.4 and Section 4 of:

  Kanerva, P. (2009). Hyperdimensional computing: An introduction to
  computing in distributed representation with high-dimensional random vectors.

SDM is a content-addressable memory consisting of two arrays:
  1. **Address array**: A large set of D-dimensional hard locations with
     fixed random hypervector addresses.
  2. **Data array**: A D-dimensional accumulator at each hard location.

**Write**: To store a hypervector x, find all hard locations within Hamming
  distance `radius` from x and accumulate x into their data vectors.

**Read**: To retrieve, find all hard locations within `radius` from the cue,
  sum their data vectors, and threshold the result.

Key properties (Kanerva 2009, §4.2):
  - Capacity scales with number of hard locations and dimensionality
  - Distributed representation → tolerant to noise and component failure
  - Cues similar to stored patterns retrieve those patterns (generalisation)
  - **One-shot learning**: a pattern is stored in a single write operation
  - Graceful capacity degradation (no catastrophic forgetting)

This module is backend-agnostic: it accepts any hypervector format
(bipolar, binary, or real-valued) and measures distances accordingly.

References
----------
* Kanerva, P. (1988). Sparse Distributed Memory. MIT Press.
* Kanerva, P. (2009). Hyperdimensional computing. Cognitive Computation, 1.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


###############################################################################
# Distance / similarity helpers
###############################################################################

def _hamming_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalised Hamming distance between bipolar {−1,+1} or binary {0,1}
    vectors.  Returns values in [0, 1] where 0 = match, 1 = opposite."""
    D = a.size(-1)
    if a.dtype in (torch.int64, torch.int32) and a.abs().max() <= 1:
        # binary {0, 1}
        return (a != b).float().sum(dim=-1) / D
    # bipolar {−1, +1}
    agreement = (a * b).sum(dim=-1)  # range [−D, +D]
    return 0.5 * (1.0 - agreement / D)


def _hamming_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Normalised Hamming similarity in [−1, 1] (1 = identical)."""
    return 1.0 - 2.0 * _hamming_distance(a, b)


def _cosine_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Cosine distance: 1 − cos(θ)."""
    a_n = F.normalize(a.float(), p=2, dim=-1)
    b_n = F.normalize(b.float(), p=2, dim=-1)
    return 1.0 - (a_n * b_n).sum(dim=-1)


###############################################################################
# Sparse Distributed Memory
###############################################################################

class SparseDistributedMemory(nn.Module):
    """Sparse Distributed Memory (SDM) following Kanerva (2009), §4.

    An SDM is defined by:
      * ``num_locations`` — number of hard locations (M)
      * ``dim`` — hypervector dimensionality (D)
      * ``radius`` — Hamming distance threshold for activation

    **Write** (``store``): Activates all hard locations whose address is
    within Hamming distance ``radius`` of the query and adds the data vector
    into each activated location's accumulator.

    **Read** (``retrieve``): Activates the same set of hard locations based on
    the cue, sums their accumulator contents, and thresholds/binarises the
    result.

    The module tracks occupancy statistics (``counter``) so the user can
    monitor graceful degradation as the memory fills.

    Parameters
    ----------
    num_locations : int
        Number of hard locations M.  Typical values range from 10³ to 10⁶.
    dim : int
        Hypervector dimensionality D.  Typical values: 1 000 – 10 000.
    radius : float
        Activation radius as a **fraction of D** (0 < radius ≤ 1).
        A radius of 0.5 means locations within 0.5·D Hamming distance
        are activated.  The paper uses absolute Hamming distance; we
        normalise for dimensionality-independence.
    dtype : torch.dtype
        Storage dtype. ``torch.float32`` for real-valued vectors,
        ``torch.int8`` for bipolar/binary.
    """

    def __init__(
        self,
        num_locations: int = 10_000,
        dim: int = 4096,
        radius: float = 0.45,
        *,
        dtype: torch.dtype = torch.float32,
        seed: int = 42,
    ):
        super().__init__()
        if not 0 < radius <= 1:
            raise ValueError(f"radius must be in (0, 1], got {radius}")
        self.num_locations = num_locations
        self.dim = dim
        self.radius = radius
        self.dtype = dtype

        # ---- Address array (hard locations) ----
        # Fixed random bipolar hypervectors.
        gen = torch.Generator()
        gen.manual_seed(seed)
        self.register_buffer(
            "address_memory",
            (torch.rand(num_locations, dim, generator=gen) > 0.5).float() * 2 - 1,
            persistent=True,
        )

        # ---- Data array (accumulators) ----
        self.register_buffer(
            "data_memory",
            torch.zeros(num_locations, dim, dtype=dtype),
            persistent=True,
        )

        # ---- Activation counter (for monitoring capacity) ----
        self.register_buffer(
            "counter",
            torch.zeros(num_locations, dtype=torch.long),
            persistent=True,
        )

        self._stored_patterns: int = 0  # bookkeeping

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _activation_mask(self, query: torch.Tensor) -> torch.Tensor:
        """Return a boolean mask [M] of hard locations within ``radius`` of
        ``query``.  ``query`` has shape (D,) or (1, D)."""
        if query.dim() == 2:
            query = query.squeeze(0)
        dist = _hamming_distance(
            query.unsqueeze(0).expand(self.num_locations, -1),
            self.address_memory,
        )  # [M]
        return dist <= self.radius

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store(
        self,
        x: torch.Tensor,
        *,
        accumulate: bool = True,
    ) -> int:
        """Store a single hypervector ``x`` in the SDM.

        Writes ``x`` into **all** hard locations whose address is within the
        activation radius.

        Parameters
        ----------
        x : torch.Tensor
            Hypervector of shape (D,) or (1, D).
        accumulate : bool
            If ``True`` (default), adds ``x`` to existing contents.
            If ``False``, overwrites each activated location.

        Returns
        -------
        int
            Number of hard locations activated by this write.
        """
        if x.dim() == 2:
            x = x.squeeze(0)
        if x.shape[-1] != self.dim:
            raise ValueError(
                f"Expected dim={self.dim}, got {x.shape[-1]}"
            )
        x = x.to(dtype=self.dtype, device=self.data_memory.device)

        mask = self._activation_mask(x)  # [M] bool
        n_activated = mask.sum().item()

        if not accumulate:
            self.data_memory[mask] = x.unsqueeze(0).expand(n_activated, -1)
        else:
            self.data_memory[mask] = self.data_memory[mask] + x.unsqueeze(0)

        self.counter[mask] += 1
        self._stored_patterns += 1
        return n_activated

    def retrieve(
        self,
        cue: torch.Tensor,
        *,
        threshold: float = 0.0,
        binarise: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """Retrieve a hypervector from the SDM using ``cue``.

        Activation is identical to ``store``: all hard locations within
        Hamming distance ``radius`` of the cue contribute.  Their
        accumulator contents are summed component-wise.

        Parameters
        ----------
        cue : torch.Tensor
            Query hypervector of shape (D,) or (1, D).
        threshold : float
            Threshold for post-retrieval binarisation.
            - For bipolar vectors: components ≥ threshold → +1, else −1.
            - For real-valued vectors: values below threshold are zeroed.
        binarise : bool
            If ``True``, the summed vector is thresholded.  If ``False``,
            the raw sum is returned (useful for graded read-out).

        Returns
        -------
        retrieved : torch.Tensor
            Retrieved hypervector of shape (D,).
        n_activated : int
            Number of hard locations that contributed to the read.
        """
        if cue.dim() == 2:
            cue = cue.squeeze(0)
        cue = cue.to(device=self.data_memory.device)

        mask = self._activation_mask(cue)  # [M] bool
        n_activated = mask.sum().item()

        if n_activated == 0:
            # No hard locations in range → return zero vector
            retrieved = torch.zeros(self.dim, dtype=self.dtype, device=cue.device)
            return retrieved, 0

        # Sum the contents of all activated locations
        accumulated = self.data_memory[mask]  # [K, D]
        retrieved = accumulated.sum(dim=0)  # [D]

        if binarise:
            retrieved = self._threshold(retrieved, threshold)

        return retrieved, n_activated

    def retrieve_batch(
        self,
        cues: torch.Tensor,
        *,
        threshold: float = 0.0,
        binarise: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve multiple hypervectors.

        Parameters
        ----------
        cues : torch.Tensor
            Shape (B, D).

        Returns
        -------
        retrieved : torch.Tensor  shape (B, D)
        counts : torch.Tensor  shape (B,)
        """
        retrieved = []
        counts = []
        for i in range(cues.size(0)):
            r, c = self.retrieve(cues[i], threshold=threshold, binarise=binarise)
            retrieved.append(r)
            counts.append(c)
        return torch.stack(retrieved, dim=0), torch.tensor(counts, device=cues.device)

    def _threshold(
        self, x: torch.Tensor, threshold: float
    ) -> torch.Tensor:
        """Post-retrieval binarisation."""
        if self.dtype in (torch.int8, torch.int16, torch.int32, torch.int64):
            # bipolar path assumed
            return torch.where(x >= threshold, 1.0, -1.0).to(self.dtype)
        # real-valued: zero out below threshold
        x = x.clone()
        x[torch.abs(x) < threshold] = 0.0
        return x

    # ------------------------------------------------------------------
    # Bookkeeping
    # ------------------------------------------------------------------

    @property
    def stored_patterns(self) -> int:
        """Number of patterns written since construction."""
        return self._stored_patterns

    @property
    def mean_occupancy(self) -> float:
        """Average number of patterns stored per hard location."""
        return self.counter.float().mean().item()

    @property
    def max_occupancy(self) -> int:
        """Maximum number of patterns stored at any single hard location."""
        return self.counter.max().item()

    def occupancy_summary(self) -> Dict[str, float]:
        """Return occupancy statistics as a dictionary."""
        c = self.counter.float()
        return {
            "stored_patterns": self._stored_patterns,
            "mean_occupancy": c.mean().item(),
            "max_occupancy": c.max().item(),
            "std_occupancy": c.std().item(),
            "fraction_active": (c > 0).float().mean().item(),
        }

    def reset(self):
        """Clear all stored data (re‑initialise data array and counter)."""
        self.data_memory.zero_()
        self.counter.zero_()
        self._stored_patterns = 0


###############################################################################
# SDM Autoencoder (demonstrates one‑shot storage & retrieval)
###############################################################################

class SDMAutoencoder(nn.Module):
    """Thin wrapper that demonstrates SDM as a one‑shot autoencoder.

    Uses a learned (or frozen) encoder to map input data into HD space,
    stores it in the SDM, and decodes via nearest‑neighbour lookup against
    a codebook.

    This is the model described in Kanerva (2009), §2.5: Item Memory (IM)
    → Mapping F → Cleanup Memory (CM), where the SDM acts as the
    distributed memory layer between encoding and cleanup.

    Parameters
    ----------
    sdm : SparseDistributedMemory
        Pre‑initialised SDM instance.
    codebook : torch.Tensor
        Shape (K, D) — the Item Memory / Cleanup Memory codebook.
    """

    def __init__(
        self,
        sdm: SparseDistributedMemory,
        codebook: torch.Tensor,
    ):
        super().__init__()
        self.sdm = sdm
        self.register_buffer("codebook", codebook)

    def store(self, hd_vector: torch.Tensor) -> int:
        """Encode and store a hypervector via the SDM.

        Returns number of activated hard locations.
        """
        return self.sdm.store(hd_vector)

    def retrieve(self, cue: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Retrieve from SDM, then cleanup via the codebook.

        Returns
        -------
        cleaned : torch.Tensor  (D,) nearest codebook entry
        similarity : torch.Tensor  scalar cosine similarity to retrieved
        """
        raw, _ = self.sdm.retrieve(cue, binarise=False)
        # Cleanup: nearest codebook entry
        sim = F.cosine_similarity(
            raw.unsqueeze(0).float(),
            self.codebook.float(),
            dim=-1,
        )
        best_idx = sim.argmax(dim=-1)
        cleaned = self.codebook[best_idx].squeeze(0)
        return cleaned, sim[best_idx]


###############################################################################
# Tests
###############################################################################

def test_sdm() -> None:
    """Smoke‑test the SDM implementation and verify Kanerva (2009) properties."""
    print("Testing Sparse Distributed Memory ...")

    dim = 1024
    num_locations = 2000
    radius = 0.48  # ~ 491 bits in 1024‑dim space

    sdm = SparseDistributedMemory(
        num_locations=num_locations,
        dim=dim,
        radius=radius,
    )

    # ---- Test 1: Store & retrieve a single pattern ----
    pattern = sdm.address_memory[0].clone()  # an existing hard location
    n_written = sdm.store(pattern)
    retrieved, n_read = sdm.retrieve(pattern)

    # Retrieved should be similar to original
    sim = _hamming_similarity(pattern, retrieved).item()
    print(f"  Test 1 — one-shot store/retrieve: similarity = {sim:.4f}")
    assert sim > 0.3, f"Expected sim > 0.3, got {sim:.4f}"

    # ---- Test 2: Noise robustness ----
    noisy = pattern.clone()
    flip_mask = torch.rand(dim) < 0.1  # flip 10 % of components
    noisy[flip_mask] = -noisy[flip_mask]
    retrieved_noisy, _ = sdm.retrieve(noisy)
    sim_noisy = _hamming_similarity(pattern, retrieved_noisy).item()
    print(f"  Test 2 — noisy cue (10 % bits flipped): similarity = {sim_noisy:.4f}")
    assert sim_noisy > 0.1, f"Expected sim > 0.1, got {sim_noisy:.4f}"

    # ---- Test 3: Graceful degradation —— multiple patterns ----
    for _ in range(50):
        random_pattern = (
            torch.bernoulli(torch.full((dim,), 0.5)) * 2 - 1
        )
        sdm.store(random_pattern)

    retrieved_multi, _ = sdm.retrieve(pattern, binarise=False)
    sim_multi = _hamming_similarity(pattern, retrieved_multi).item()
    print(
        f"  Test 3 — after 50 random writes, original recall: "
        f"similarity = {sim_multi:.4f}"
    )
    assert sim_multi > 0.1, (
        f"Graceful degradation failed: sim = {sim_multi:.4f}"
    )

    # ---- Occupancy report ----
    summary = sdm.occupancy_summary()
    print(
        f"  Occupancy: stored={summary['stored_patterns']}, "
        f"mean={summary['mean_occupancy']:.2f}, "
        f"max={summary['max_occupancy']}, "
        f"active={summary['fraction_active']:.2%}"
    )

    print("SDM tests passed!\n")


if __name__ == "__main__":
    test_sdm()