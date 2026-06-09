"""
Eldarin: One‑Shot HDC Classifier
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Implements the canonical HDC classification pipeline from Kanerva (2009),
Section 3.1.2 "Classification with HDC":

  1. **Encoding**: Map each training example to a hypervector.
  2. **Bundling**: Add all hypervectors of the same class to form a
     class-prototype hypervector.
  3. **Inference**: Encode the unknown example the same way and compare it
     with the class prototypes.  The most similar prototype wins.

This classifier requires **no backpropagation** and supports **one‑shot
and few‑shot learning**.  It is backend-agnostic and works with any VSA
representation (bipolar, binary, or real‑valued) via the ``VSAHDC``
module.

For FPGA/SNN deployment, pair with ``dtype="bipolar"`` and
``binding="xor"`` — all operations reduce to bitwise XNOR and popcount.

References
----------
* Kanerva, P. (2009). Hyperdimensional computing. Cognitive Computation, 1.
* Rahimi, A. et al. (2019). Efficient biosignal processing using
  hyperdimensional computing. Proceedings of the IEEE, 107(1).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .vsa_hdc import VSAHDC, cosine_similarity


###############################################################################
# Item Memory — Maps class labels ↔ hypervectors
###############################################################################

class ItemMemory(nn.Module):
    """Symbol ↔ Hypervector mapping consistent with Kanerva (2009), §2.5.

    The Item Memory (IM) assigns a random hypervector to each symbol
    (class label) and provides bidirectional lookup:
      * ``encode(label)`` → hypervector
      * ``decode(vector)`` → nearest label

    Parameters
    ----------
    num_symbols : int
        Initial number of symbols to register (the codebook size).
    dim : int
        Hypervector dimensionality.
    dtype : str
        ``"bipolar"`` or ``"binary"``.
    seed : int
        Random seed for reproducible hypervectors.
    """

    def __init__(
        self,
        num_symbols: int,
        dim: int = 8192,
        dtype: str = "bipolar",
        seed: int = 42,
    ):
        super().__init__()
        self.dim = dim
        self.dtype = dtype

        gen = torch.Generator()
        gen.manual_seed(seed)
        if dtype == "bipolar":
            codebook = (torch.rand(num_symbols, dim, generator=gen) > 0.5).float() * 2 - 1
        else:
            codebook = (torch.rand(num_symbols, dim, generator=gen) > 0.5).float()

        self.register_buffer("codebook", codebook, persistent=True)
        # Optional human‑readable label registry
        self._labels: Dict[int, str] = {
            i: f"class_{i}" for i in range(num_symbols)
        }

    @property
    def num_symbols(self) -> int:
        return self.codebook.size(0)

    def register_label(self, idx: int, name: str) -> None:
        """Associate a human‑readable name with a symbol index."""
        self._labels[idx] = name

    def encode(self, indices: torch.Tensor) -> torch.Tensor:
        """Map class‑index tensor → hypervectors.

        Parameters
        ----------
        indices : torch.Tensor
            LongTensor of shape (...) with values in [0, num_symbols).

        Returns
        -------
        torch.Tensor
            Shape (..., dim).
        """
        return self.codebook[indices]

    def decode(self, query: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Find the nearest symbol(s) for a query hypervector.

        Parameters
        ----------
        query : torch.Tensor
            Shape (D,) or (B, D).

        Returns
        -------
        best_idx : torch.Tensor  scalar or (B,)
        similarity : torch.Tensor  scalar or (B,)
        """
        single = query.dim() == 1
        if single:
            query = query.unsqueeze(0)

        sim = cosine_similarity(query.unsqueeze(1), self.codebook.unsqueeze(0))  # [B, K]
        best_idx = sim.argmax(dim=-1)  # [B, 1] or scalar
        best_sim = sim.gather(-1, best_idx.unsqueeze(-1)).squeeze(-1)

        if single:
            best_idx = best_idx.squeeze(0)
            best_sim = best_sim.squeeze(0)
        return best_idx, best_sim

    def decode_name(self, query: torch.Tensor) -> Tuple[str, torch.Tensor]:
        """Like ``decode`` but returns a human‑readable label name."""
        idx, sim = self.decode(query)
        if idx.dim() > 0:
            return [self._labels.get(int(i.item()), "?") for i in idx], sim
        return self._labels.get(int(idx.item()), "?"), sim


###############################################################################
# HDC Classifier — Kanerva (2009), §3.1.2
###############################################################################

class HDCClassifier(nn.Module):
    """One‑shot / few‑shot HDC classifier.

    Training is **single‑pass bundling**: every example of a class is
    encoded and added into that class's prototype hypervector.  No gradient
    descent is performed.

    Inference compares a query hypervector against all class prototypes
    via cosine similarity and returns the most similar class.

    This directly implements the three‑step recipe from Kanerva (2009) §3.1.2:

    1. Encode each training example → hypervector
    2. Bundle by class → class‑prototype hypervectors
    3. Inference: encode query → compare against prototypes

    Parameters
    ----------
    vsa : VSAHDC
        The VSA backend used to encode raw features into HD space.
        The classifier uses ``vsa.encode()`` to map features and
        ``vsa.bundle()`` to aggregate per‑class prototypes.
    num_classes : int
        Number of distinct output classes.
    item_memory : ItemMemory, optional
        Pre‑built Item Memory.  If None, one is created internally.
    """

    def __init__(
        self,
        vsa: VSAHDC,
        num_classes: int,
        *,
        item_memory: Optional[ItemMemory] = None,
    ):
        super().__init__()
        self.vsa = vsa
        self.num_classes = num_classes
        self.hd_dim = vsa.hd_dim

        # Class‑prototype hypervectors — learned via bundling
        self.register_buffer(
            "prototypes",
            torch.zeros(num_classes, self.hd_dim),
            persistent=True,
        )

        # Per‑class counters for incremental (weighted) bundling
        self.register_buffer(
            "prototype_counts",
            torch.zeros(num_classes, dtype=torch.long),
            persistent=True,
        )

        # Item Memory for label ↔ vector mapping
        if item_memory is None:
            item_memory = ItemMemory(num_classes, dim=self.hd_dim, dtype=vsa.dtype)
        self.item_memory = item_memory

        self._trained = False

    # ------------------------------------------------------------------
    # Training (one‑shot bundling)
    # ------------------------------------------------------------------

    def fit(
        self,
        features: torch.Tensor,
        labels: torch.Tensor,
        *,
        reset: bool = True,
    ) -> None:
        """One‑pass training: encode then bundle by class.

        Parameters
        ----------
        features : torch.Tensor
            Shape (N, input_dim) — raw feature vectors.
        labels : torch.Tensor
            Shape (N,) — integer class labels in [0, num_classes).
        reset : bool
            If ``True``, clear previous prototypes before fitting.
        """
        if reset:
            self.prototypes.zero_()
            self.prototype_counts.zero_()

        # Encode all examples into HD space
        with torch.no_grad():
            hd_vectors = self.vsa.encode(features)  # (N, hd_dim)

        # Accumulate per‑class
        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_vecs = hd_vectors[mask]  # (n_c, hd_dim)
                bundled = self.vsa.bundle(class_vecs)
                if self.prototype_counts[c] > 0:
                    # Merge with existing prototype (weighted average)
                    prev_count = self.prototype_counts[c].item()
                    new_count = prev_count + mask.sum().item()
                    self.prototypes[c] = (
                        self.prototypes[c] * prev_count + bundled * mask.sum().item()
                    ) / new_count
                    self.prototypes[c] = F.normalize(self.prototypes[c], p=2, dim=-1)
                else:
                    self.prototypes[c] = bundled
                self.prototype_counts[c] += mask.sum().item()

        self._trained = True

    def fit_hd(
        self,
        hd_vectors: torch.Tensor,
        labels: torch.Tensor,
        *,
        reset: bool = True,
    ) -> None:
        """Like ``fit`` but accepts pre‑encoded hypervectors.

        Parameters
        ----------
        hd_vectors : torch.Tensor
            Shape (N, hd_dim) — already‑encoded hypervectors.
        labels : torch.Tensor
            Shape (N,) — integer class labels.
        reset : bool
            Clear previous prototypes.
        """
        if reset:
            self.prototypes.zero_()
            self.prototype_counts.zero_()

        for c in range(self.num_classes):
            mask = labels == c
            if mask.any():
                class_vecs = hd_vectors[mask]
                bundled = self.vsa.bundle(class_vecs)
                if self.prototype_counts[c] > 0:
                    prev_count = self.prototype_counts[c].item()
                    new_count = prev_count + mask.sum().item()
                    self.prototypes[c] = (
                        self.prototypes[c] * prev_count + bundled * mask.sum().item()
                    ) / new_count
                    self.prototypes[c] = F.normalize(self.prototypes[c], p=2, dim=-1)
                else:
                    self.prototypes[c] = bundled
                self.prototype_counts[c] += mask.sum().item()

        self._trained = True

    def add_example(self, feature: torch.Tensor, label: int) -> None:
        """Incrementally add a **single** example (one‑shot update).

        This is the pure one‑shot learning path: no retraining needed.
        """
        with torch.no_grad():
            hd = self.vsa.encode(feature.unsqueeze(0)).squeeze(0)

        prev_count = self.prototype_counts[label].item()
        if prev_count > 0:
            self.prototypes[label] = (
                self.prototypes[label] * prev_count + hd
            ) / (prev_count + 1)
            self.prototypes[label] = F.normalize(self.prototypes[label], p=2, dim=-1)
        else:
            self.prototypes[label] = F.normalize(hd, p=2, dim=-1)
        self.prototype_counts[label] += 1
        self._trained = True

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self, features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify raw feature vectors.

        Parameters
        ----------
        features : torch.Tensor
            Shape (N, input_dim).

        Returns
        -------
        predictions : torch.Tensor  shape (N,) — predicted class indices
        confidence : torch.Tensor  shape (N,) — cosine similarity to best prototype
        """
        with torch.no_grad():
            hd = self.vsa.encode(features)  # (N, hd_dim)
        return self.predict_hd(hd)

    def predict_hd(
        self, hd_vectors: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Classify pre‑encoded hypervectors.

        Parameters
        ----------
        hd_vectors : torch.Tensor
            Shape (N, hd_dim).

        Returns
        -------
        predictions : torch.Tensor  shape (N,)
        confidence : torch.Tensor  shape (N,)
        """
        hd_norm = F.normalize(hd_vectors, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = hd_norm @ proto_norm.T  # (N, C)
        predictions = sim.argmax(dim=-1)
        confidence = sim.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
        return predictions, confidence

    def predict_topk(
        self, features: torch.Tensor, k: int = 3
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return top‑k class predictions with confidences.

        Returns
        -------
        topk_indices : torch.Tensor  shape (N, k)
        topk_sims : torch.Tensor  shape (N, k)
        """
        with torch.no_grad():
            hd = self.vsa.encode(features)
        hd_norm = F.normalize(hd, p=2, dim=-1)
        proto_norm = F.normalize(self.prototypes, p=2, dim=-1)
        sim = hd_norm @ proto_norm.T
        return sim.topk(min(k, self.num_classes), dim=-1)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @property
    def is_trained(self) -> bool:
        """Has at least one class prototype been initialised?"""
        return self._trained and self.prototype_counts.sum() > 0

    def class_support(self) -> torch.Tensor:
        """Number of examples stored per class."""
        return self.prototype_counts.clone()

    def reset(self):
        """Clear all prototypes (re‑initialise to untrained state)."""
        self.prototypes.zero_()
        self.prototype_counts.zero_()
        self._trained = False


###############################################################################
# Functional one‑shot API (no nn.Module boilerplate)
###############################################################################

def hdc_fit(
    vsa: VSAHDC,
    features: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
) -> torch.Tensor:
    """Pure‑functional one‑shot HDC fitting.

    Returns
    -------
    prototypes : torch.Tensor  shape (num_classes, hd_dim)
    """
    with torch.no_grad():
        hd = vsa.encode(features)
    prototypes = torch.zeros(num_classes, vsa.hd_dim, device=hd.device)
    for c in range(num_classes):
        mask = labels == c
        if mask.any():
            prototypes[c] = vsa.bundle(hd[mask])
    return prototypes


def hdc_predict(
    prototypes: torch.Tensor,
    query_hd: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Classify pre‑encoded queries against prototypes.

    Returns (predictions, confidences).
    """
    query_norm = F.normalize(query_hd, p=2, dim=-1)
    proto_norm = F.normalize(prototypes, p=2, dim=-1)
    sim = query_norm @ proto_norm.T
    preds = sim.argmax(dim=-1)
    confs = sim.gather(-1, preds.unsqueeze(-1)).squeeze(-1)
    return preds, confs


###############################################################################
# Tests
###############################################################################

def test_hdc_classifier() -> None:
    """Smoke‑test the HDC one‑shot classifier."""
    print("Testing HDC One‑Shot Classifier ...")

    # Tiny classification: 3 classes, 16‑dim features, 512‑dim HD
    vsa = VSAHDC(hd_dim=512, input_dim=16, dtype="bipolar", binding="xor")
    clf = HDCClassifier(vsa, num_classes=3)

    # Generate synthetic separable data
    torch.manual_seed(42)
    N = 15  # 5 per class
    features = torch.randn(N, 16)
    # Make classes separable
    features[:5] += 2.0
    features[10:] -= 2.0
    labels = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2])

    # ---- One‑shot train ----
    clf.fit(features, labels)
    assert clf.is_trained
    assert clf.prototype_counts.sum() == 15

    # ---- Evaluate on training data ----
    preds, confs = clf.predict(features)
    acc = (preds == labels).float().mean().item()
    print(f"  Training accuracy (one‑shot HDC): {acc:.2%}")
    assert acc >= 0.6, f"Expected acc >= 0.6, got {acc:.2%}"

    # ---- Incremental one‑shot add ----
    clf.add_example(torch.randn(16), 0)
    assert clf.prototype_counts[0] == 6

    # ---- Top‑k prediction ----
    topk_idx, topk_sim = clf.predict_topk(features[:3], k=2)
    print(f"  Top‑2 predictions for 3 samples: shape={topk_idx.shape}")

    # ---- Functional API ----
    protos = hdc_fit(vsa, features, labels, num_classes=3)
    hd_q = vsa.encode(features[:2])
    p, c = hdc_predict(protos, hd_q)
    print(f"  Functional API predictions: {p.tolist()}, confidences: {c.tolist()}")

    # ---- Item Memory ----
    im = ItemMemory(num_symbols=3, dim=512)
    im.register_label(0, "drone")
    im.register_label(1, "bird")
    im.register_label(2, "plane")
    idx, sim = im.decode(clf.prototypes[0])
    name, _ = im.decode_name(clf.prototypes[0])
    print(f"  Item Memory: prototype[0] decodes to idx={idx.item()}, name='{name}'")

    print("HDC classifier tests passed!\n")


if __name__ == "__main__":
    test_hdc_classifier()