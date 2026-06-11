"""
HDC-SLAM — Hyperdimensional Simultaneous Localisation and Mapping
==================================================================
Cortex-A soft real-time module. Replaces conventional dense SLAM with a
compact associative map of landmark hypervectors for place recognition
and loop-closure detection (§4.5.2 of the outline).

Architecture:
  1. Receive event frame from SNN feature tracker (FPGA)
  2. Feature Hypervector Encoding:
     • Encode landmark/appearance info from event features
     • Append depth (IR rangefinder) and altitude (barometer) as context
  3. Map Storage: Sparse Distributed Memory (SDM) or associative bundle
     of landmark HV → position/descriptor pairs
  4. Place Recognition: similarity search against stored landmarks
  5. Loop Closure: when similarity exceeds threshold → add constraint
  6. Emit position corrections and map updates

Design principle: the map is a flat associative memory — no factor graph
optimisation, no bundle adjustment. All queries are Hamming/cosine
similarity over HD vectors, computable efficiently with NEON SIMD.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
import heapq

from ..messages import (
    EventFrameMessage, DistanceMessage, BarometerMessage,
    StateEstimate, now_us,
)


@dataclass
class LandmarkConfig:
    """Configuration for the HDC-SLAM landmark mapper."""
    hd_dim: int = 8192
    image_height: int = 180
    image_width: int = 240

    # Landmark parameters
    max_landmarks: int = 5000          # maximum stored landmarks
    min_landmark_distance: float = 0.5  # metres — suppress near-duplicates
    landmark_confidence_threshold: float = 0.6  # similarity threshold for new landmark

    # Place recognition
    place_recognition_threshold: float = 0.75  # cosine similarity above which = loop closure
    num_candidate_places: int = 5       # top-K candidates for match
    min_frames_between_loop_closures: int = 100  # avoid re-triggering

    # Descriptor
    feature_dim: int = 128   # per-landmark descriptor dimension

    # HDC operation mode
    hd_dtype: str = "bipolar"   # "bipolar" | "complex"
    seed: int = 777

    # NEON SIMD hints
    use_bit_packed: bool = True
    recommended_core: int = 1  # dedicated core number for SLAM thread


@dataclass
class Landmark:
    """A single map landmark stored in the associative memory."""
    id: int
    # Hypervector representation (feature + context binding)
    hv: np.ndarray  # [hd_dim] bipolar (-1, 1) or complex
    # Geometric position (world frame, metres)
    position: np.ndarray  # [3] x, y, z
    # Appearance descriptor (lower-dimensional)
    descriptor: Optional[np.ndarray] = None  # [feature_dim]
    # Metadata
    first_seen_frame: int = 0
    last_seen_frame: int = 0
    observation_count: int = 0
    confidence: float = 0.0
    # Loop-closure graph neighbour
    loop_edges: List[int] = field(default_factory=list)


class LandmarkMemory:
    """
    Compact associative memory of landmark hypervectors.
    Implements storage, similarity search, and loop-closure detection.
    """

    def __init__(self, cfg: LandmarkConfig):
        self.cfg = cfg
        self.landmarks: List[Landmark] = []
        self._next_id: int = 0
        self._last_loop_closure_frame: int = 0
        self._landmark_hvs: Optional[np.ndarray] = None  # stacked [N, hd_dim] for batched similarity

    def insert(
        self,
        hv: np.ndarray,
        position: np.ndarray,
        descriptor: Optional[np.ndarray] = None,
        frame_id: int = 0,
    ) -> Optional[int]:
        """
        Insert a new landmark into the map.
        Returns landmark ID if inserted, None if suppressed (too similar to existing).
        """
        # Check capacity
        if len(self.landmarks) >= self.cfg.max_landmarks:
            # Prune oldest/lowest-confidence landmark
            self._prune()

        # Check similarity to existing landmarks
        if self._landmark_hvs is not None and len(self.landmarks) > 0:
            similarities = LandmarkMemory.hamming_similarity(
                hv.reshape(1, -1), self._landmark_hvs
            )[0]
            if np.max(similarities) > self.cfg.landmark_confidence_threshold:
                # Update existing landmark instead
                best_idx = int(np.argmax(similarities))
                self.landmarks[best_idx].last_seen_frame = frame_id
                self.landmarks[best_idx].observation_count += 1
                # Moving average of HV
                alpha = 0.9
                self.landmarks[best_idx].hv = (
                    alpha * self.landmarks[best_idx].hv + (1 - alpha) * hv
                )
                return self.landmarks[best_idx].id

            # Check geometric distance
            positions = np.array([lm.position for lm in self.landmarks])
            if positions.shape[0] > 0:
                dists = np.linalg.norm(positions - position, axis=1)
                if np.min(dists) < self.cfg.min_landmark_distance:
                    return None  # suppress near-duplicate

        # Create new landmark
        lm = Landmark(
            id=self._next_id,
            hv=hv.copy(),
            position=position.copy(),
            descriptor=descriptor.copy() if descriptor is not None else None,
            first_seen_frame=frame_id,
            last_seen_frame=frame_id,
            observation_count=1,
            confidence=0.5,
        )
        self.landmarks.append(lm)
        self._next_id += 1

        # Update stacked HV array
        self._rebuild_hv_stack()
        return lm.id

    def query(self, hv: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """
        Find the k most similar landmarks to the query hypervector.
        Returns list of (landmark_id, similarity).
        """
        if not self.landmarks:
            return []

        similarities = LandmarkMemory.hamming_similarity(
            hv.reshape(1, -1), self._landmark_hvs
        )[0]

        # Top-k via heap
        top_k = []
        for i, sim in enumerate(similarities):
            if len(top_k) < k:
                heapq.heappush(top_k, (sim, self.landmarks[i].id))
            elif sim > top_k[0][0]:
                heapq.heapreplace(top_k, (sim, self.landmarks[i].id))

        return [(lid, sim) for sim, lid in sorted(top_k, reverse=True)]

    def detect_loop_closure(
        self, hv: np.ndarray, frame_id: int
    ) -> Optional[Tuple[int, float, np.ndarray]]:
        """
        Check if current observation closes a loop with a previously visited place.
        Returns (landmark_id, similarity, position) if loop detected, else None.
        """
        if frame_id - self._last_loop_closure_frame < self.cfg.min_frames_between_loop_closures:
            return None

        if not self.landmarks:
            return None

        similarities = LandmarkMemory.hamming_similarity(
            hv.reshape(1, -1), self._landmark_hvs
        )[0]

        best_idx = int(np.argmax(similarities))
        best_sim = similarities[best_idx]

        if best_sim > self.cfg.place_recognition_threshold:
            lm = self.landmarks[best_idx]
            # Only consider loop closures with older landmarks
            if frame_id - lm.first_seen_frame > self.cfg.min_frames_between_loop_closures:
                self._last_loop_closure_frame = frame_id
                lm.loop_edges.append(frame_id)
                return (lm.id, best_sim, lm.position)

        return None

    def _prune(self):
        """Remove the landmark with the lowest confidence."""
        if not self.landmarks:
            return
        # Score: confidence * log(1 + observation_count)
        scores = [
            lm.confidence * np.log(1 + lm.observation_count)
            for lm in self.landmarks
        ]
        worst = int(np.argmin(scores))
        self.landmarks.pop(worst)
        self._rebuild_hv_stack()

    def _rebuild_hv_stack(self):
        """Rebuild the stacked hypervector matrix for batched similarity."""
        if not self.landmarks:
            self._landmark_hvs = None
            return
        self._landmark_hvs = np.stack([lm.hv for lm in self.landmarks], axis=0)

    @staticmethod
    def hamming_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Cosine similarity for bipolar vectors (equivalent to 1 - 2*Hamming/HD).
        Optimised for NEON SIMD in production — python fallback here.
        a: [N, HD]  b: [M, HD]  → returns [N, M]
        """
        return a @ b.T / np.sqrt((a ** 2).sum(-1, keepdims=True) * (b ** 2).sum(-1, keepdims=False))

    def get_num_landmarks(self) -> int:
        return len(self.landmarks)

    def get_all_positions(self) -> np.ndarray:
        if not self.landmarks:
            return np.zeros((0, 3))
        return np.array([lm.position for lm in self.landmarks])


class FeatureEncoder:
    """Encodes landmark/appearance information from event-frame features."""

    def __init__(self, cfg: LandmarkConfig):
        self.cfg = cfg
        rng = np.random.RandomState(cfg.seed + 200)
        if cfg.hd_dtype == "bipolar":
            self.depth_context_seed = (rng.rand(cfg.hd_dim) > 0.5).astype(np.float32) * 2 - 1
            self.altitude_context_seed = (rng.rand(cfg.hd_dim) > 0.5).astype(np.float32) * 2 - 1
        else:
            angles = rng.rand(cfg.hd_dim) * 2 * np.pi
            self.depth_context_seed = np.exp(1j * angles).astype(np.complex64)
            self.altitude_context_seed = np.exp(1j * (angles + np.pi)).astype(np.complex64)

    def encode_feature_grid(
        self,
        event_frame: EventFrameMessage,
        distance: Optional[DistanceMessage] = None,
        barometer: Optional[BarometerMessage] = None,
    ) -> np.ndarray:
        """
        Encode feature information from the event frame into a landmark
        hypervector with depth and altitude context.
        """
        hd_dim = self.cfg.hd_dim
        rng = np.random.RandomState(self.cfg.seed + 300)

        # Start with sparse feature HV
        hv = np.zeros(hd_dim, dtype=np.float32 if self.cfg.hd_dtype == "bipolar" else np.complex64)

        if event_frame.xs:
            # Use feature IDs to construct the HV
            for feat_id, activation in zip(event_frame.feature_ids, event_frame.activations):
                feat_rng = np.random.RandomState(self.cfg.seed + feat_id)
                if self.cfg.hd_dtype == "bipolar":
                    feat_hv = (feat_rng.rand(hd_dim) > 0.5).astype(np.float32) * 2 - 1
                    hv += feat_hv * activation
                else:
                    feat_hv = np.exp(1j * feat_rng.rand(hd_dim) * 2 * np.pi).astype(np.complex64)
                    hv += feat_hv * activation

        # Normalize
        norm = np.linalg.norm(hv)
        if norm > 1e-8:
            hv = hv / norm

        # Bind depth context
        if distance is not None:
            depth_norm = max(distance.down_mm / 10000.0, 0.01)
            shift = int(depth_norm * 800) % hd_dim
            ctx = np.roll(self.depth_context_seed, shift)
            hv = hv * ctx

        # Bind altitude context
        if barometer is not None:
            alt_norm = barometer.pressure_hpa / 1013.25
            shift = int(alt_norm * 500) % hd_dim
            ctx = np.roll(self.altitude_context_seed, shift)
            hv = hv * ctx

        # Re-normalise
        norm = np.linalg.norm(hv)
        if norm > 1e-8:
            hv = hv / norm

        return hv


class HDCSlamMapper:
    """
    HDC-SLAM: Hyperdimensional SLAM using compact associative memory.

    Replaces conventional dense SLAM (factor graphs, bundle adjustment) with:
      - Hyperdimensional landmark encoding
      - Associative memory for storage and retrieval
      - Hamming/cosine similarity for place recognition
      - Simple position averaging for loop closure correction
    """

    def __init__(self, config: Optional[LandmarkConfig] = None):
        self.cfg = config or LandmarkConfig()
        self.memory = LandmarkMemory(self.cfg)
        self.feature_encoder = FeatureEncoder(self.cfg)

        # State
        self.position = np.zeros(3)
        self.frame_id = 0
        self._trajectory: deque = deque(maxlen=10000)
        self._trajectory.append([0.0, 0.0, 0.0])

        # Recent observations for loop closure verification
        self._recent_hvs: deque = deque(maxlen=50)

    def step(
        self,
        event_frame: Optional[EventFrameMessage] = None,
        distance: Optional[DistanceMessage] = None,
        barometer: Optional[BarometerMessage] = None,
        position: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Process one step of HDC-SLAM.

        Args:
            event_frame: Feature information from FPGA SNN tracker
            distance: IR rangefinder
            barometer: Barometer
            position: Current position estimate (from HDC-EVIO or EKF)
        """
        self.frame_id += 1

        if position is not None:
            self.position = position.copy()
            self._trajectory.append(self.position.copy())

        result = {
            "new_landmark_id": None,
            "loop_closure": None,
            "position_correction": None,
            "map_size": self.memory.get_num_landmarks(),
        }

        if event_frame is None:
            return result

        # Encode feature hypervector
        feature_hv = self.feature_encoder.encode_feature_grid(
            event_frame, distance, barometer
        )

        # Insert landmark
        lm_id = self.memory.insert(
            hv=feature_hv,
            position=self.position,
            frame_id=self.frame_id,
        )
        if lm_id is not None:
            result["new_landmark_id"] = lm_id

        # Check for loop closure
        self._recent_hvs.append(feature_hv)
        loop = self.memory.detect_loop_closure(feature_hv, self.frame_id)
        if loop is not None:
            lm_id, sim, loop_pos = loop
            result["loop_closure"] = {
                "landmark_id": lm_id,
                "similarity": float(sim),
                "loop_position": loop_pos.tolist(),
            }
            # Compute position correction (simple average)
            correction = (loop_pos - self.position) * 0.5
            self.position += correction
            result["position_correction"] = correction.tolist()

        return result

    def query_place(self, hv: np.ndarray, k: int = 5) -> List[Tuple[int, float]]:
        """Query the map for similar places."""
        return self.memory.query(hv, k)

    def get_map(self) -> np.ndarray:
        """Return all stored landmark positions."""
        return self.memory.get_all_positions()

    def get_trajectory(self) -> np.ndarray:
        return np.array(list(self._trajectory))

    def reset(self):
        self.memory = LandmarkMemory(self.cfg)
        self.position = np.zeros(3)
        self.frame_id = 0
        self._trajectory.clear()
        self._trajectory.append([0.0, 0.0, 0.0])
        self._recent_hvs.clear()