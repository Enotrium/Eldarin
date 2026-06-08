"""
Digital Twin & Swarm Consensus Module
=======================================
Integrates concepts from Yan et al. (2026), Nature Communications Engineering:
  "Digital twin-driven swarm of autonomous underwater vehicles for marine exploration"
  https://www.nature.com/articles/s44172-025-00571-7

Adapted for Eldarin's UAV 4D tracking:
  1. Digital Twin — Synchronized virtual replica of physical UAV + tracked objects in HD space
  2. Swarm Consensus — Multi-agent collaborative fusion via VSA/HDC bundling
  3. Communication-aware weighting — Modality/agent weights based on link quality
  4. Predictive virtual model — Fills gaps when physical observations drop (occlusion, comm loss)

The digital twin maintains a hyperdimensional representation of:
  - Ego-UAV state (pose, velocity, sensor health)
  - Tracked object states (positions, velocities, trajectories)
  - Environmental context (scene semantics, weather, lighting)

Virtual ↔ Physical synchronization uses Bayesian-style updates in HD space,
consistent with the existing MixingModule architecture from VioPose.

Original references:
  VioPose: https://github.com/SeongJong-Yoo/VioPose, https://arxiv.org/pdf/2411.13607
  VSA/HDC: https://github.com/Enotrium/arthedain-1
  FPGA Event Encode: https://github.com/Enotrium/FPGA-Event-Based-encode
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .vsa_hdc import VSAHDC, bind, bundle, permute, cosine_similarity


class DigitalTwinState(nn.Module):
    """
    Hyperdimensional Digital Twin state representation.

    Maintains a synchronized virtual replica of the physical world
    in HD space. The twin state is a bundle of:
      - Ego-UAV state vector (pose, velocity, sensor status)
      - Per-object track states (positions, velocities, IDs)
      - Environmental context vector (scene type, conditions)
      - Communication graph state (neighbor UAVs, link quality)

    Uses VSA binding/permutation to encode structured relationships
    and enable efficient updates, retrieval, and cross-agent sharing.
    """

    def __init__(
        self,
        hd_dim: int = 8192,
        num_object_slots: int = 64,
        state_dim: int = 8,
        context_dim: int = 256,
    ):
        super().__init__()
        self.hd_dim = hd_dim
        self.num_object_slots = num_object_slots
        self.state_dim = state_dim

        # VSA encoder for HD operations
        self.vsa = VSAHDC(
            hd_dim=hd_dim,
            input_dim=max(state_dim, context_dim),
            dtype="bipolar",
            binding="circular",
        )

        # Role vectors for structured binding (slot-based memory)
        self.register_buffer(
            "ego_role",
            self._generate_role_vector("ego", 0),
        )
        self.register_buffer(
            "context_role",
            self._generate_role_vector("context", 1),
        )
        # Object slot roles (for addressing objects by slot index)
        self.register_buffer(
            "slot_roles",
            torch.stack([
                self._generate_role_vector(f"slot_{i}", i + 2)
                for i in range(num_object_slots)
            ]),
        )

        # Current twin state (HD bundle of all components)
        self.register_buffer(
            "twin_state",
            torch.zeros(hd_dim),
        )
        # Component states for retrieval
        self.register_buffer("ego_hd", torch.zeros(hd_dim))
        self.register_buffer("context_hd", torch.zeros(hd_dim))
        self.register_buffer("slot_states", torch.zeros(num_object_slots, hd_dim))

        # Synchronization parameters
        self.sync_decay = nn.Parameter(torch.tensor(0.95))  # Temporal smoothing
        self.confidence_threshold = nn.Parameter(torch.tensor(0.3))

    def _generate_role_vector(self, name: str, seed: int) -> torch.Tensor:
        """Generate a unique HD role vector."""
        gen = torch.Generator()
        gen.manual_seed(hash(name) % (2**31) + seed)
        return torch.bernoulli(
            torch.full((self.hd_dim,), 0.5, generator=gen)
        ) * 2 - 1  # Bipolar ±1

    def encode_ego_state(
        self, pose: torch.Tensor, velocity: torch.Tensor, sensor_health: float = 1.0
    ) -> torch.Tensor:
        """
        Encode ego-UAV state into HD space.

        Args:
            pose: [7] or [B, 7] UAV pose (quaternion + translation)
            velocity: [3] or [B, 3] UAV velocity
            sensor_health: Scalar confidence [0-1]
        Returns:
            HD ego state vector [hd_dim] or [B, hd_dim]
        """
        if pose.dim() == 1:
            pose = pose.unsqueeze(0)
            velocity = velocity.unsqueeze(0)

        # Concatenate state components
        state = torch.cat([pose, velocity, torch.full_like(velocity[:, :1], sensor_health)], dim=-1)

        # Project to match VSA input dim
        if state.shape[-1] < self.vsa.input_dim:
            state = F.pad(state, (0, self.vsa.input_dim - state.shape[-1]))
        else:
            state = state[:, :self.vsa.input_dim]

        hd_state = self.vsa.encode(state)
        # Bind with ego role
        return self.vsa.bind(hd_state, self.ego_role.unsqueeze(0))

    def encode_objects(
        self, object_states: torch.Tensor, slot_indices: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode tracked object states into HD space using slot-based binding.

        Args:
            object_states: [N, state_dim] object state vectors
            slot_indices: [N] slot assignments (or auto-assigned)
        Returns:
            HD bundle of all objects [hd_dim]
        """
        N = object_states.shape[0]
        if N == 0:
            return torch.zeros(self.hd_dim, device=object_states.device)

        # Auto-assign slots if not provided
        if slot_indices is None:
            slot_indices = torch.arange(N, device=object_states.device) % self.num_object_slots

        encoded = []
        for i in range(N):
            slot = slot_indices[i].item() % self.num_object_slots
            # Project state to VSA input dim
            state = object_states[i]
            if state.shape[-1] < self.vsa.input_dim:
                state = F.pad(state, (0, self.vsa.input_dim - state.shape[-1]))
            else:
                state = state[:self.vsa.input_dim]

            hd_obj = self.vsa.encode(state.unsqueeze(0)).squeeze(0)
            # Bind object with slot role
            bound = self.vsa.bind(hd_obj, self.slot_roles[slot])
            encoded.append(bound)

            # Update slot memory
            self.slot_states[slot] = (
                self.sync_decay * self.slot_states[slot]
                + (1 - self.sync_decay) * bound
            )

        return bundle(torch.stack(encoded)) if encoded else torch.zeros(self.hd_dim)

    def encode_context(
        self, scene_features: torch.Tensor, conditions: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode environmental context into HD space.

        Args:
            scene_features: [context_dim] or [B, context_dim] scene descriptor
            conditions: Optional condition vector (weather, lighting)
        Returns:
            HD context vector
        """
        if scene_features.dim() == 1:
            scene_features = scene_features.unsqueeze(0)

        if conditions is not None and conditions.dim() == 1:
            conditions = conditions.unsqueeze(0)

        if conditions is not None:
            features = torch.cat([scene_features, conditions], dim=-1)
        else:
            features = scene_features

        hd_ctx = self.vsa.encode(features)
        return self.vsa.bind(hd_ctx, self.context_role.unsqueeze(0))

    def update_twin(
        self,
        ego_hd: torch.Tensor,
        object_bundle: torch.Tensor,
        context_hd: torch.Tensor,
        sync_weights: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Update the full digital twin state via weighted HD bundling.

        twin_state = w_ego * ego_hd ⊕ w_obj * obj_bundle ⊕ w_ctx * context_hd

        Args:
            ego_hd: Ego state HD vector
            object_bundle: Bundled object HD vector
            context_hd: Context HD vector
            sync_weights: [3] weights for ego, objects, context
        Returns:
            Updated twin state [hd_dim]
        """
        if sync_weights is None:
            sync_weights = torch.tensor([0.3, 0.5, 0.2], device=ego_hd.device)

        components = torch.stack([ego_hd, object_bundle, context_hd])
        new_twin = bundle(components, weights=sync_weights)

        # Temporal smoothing
        self.twin_state = (
            self.sync_decay * self.twin_state.to(ego_hd.device)
            + (1 - self.sync_decay) * new_twin
        )
        self.twin_state = F.normalize(self.twin_state, p=2, dim=-1)

        # Update component caches
        self.ego_hd = ego_hd
        self.context_hd = context_hd

        return self.twin_state

    def retrieve_object(
        self, slot_idx: int, query_hd: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Retrieve object state from a specific slot.
        If query_hd is provided, uses unbinding: retrieve(bound, role) ≈ object.

        Args:
            slot_idx: Slot index
            query_hd: Optional query to verify retrieval
        Returns:
            Retrieved HD object vector
        """
        slot_binding = self.slot_states[slot_idx]
        role = self.slot_roles[slot_idx]
        # Unbind: slot_state ⊗ role^{-1} ≈ object
        retrieved = self.vsa.retrieve(slot_binding, role)
        return retrieved

    def predict_forward(self, steps: int = 1) -> torch.Tensor:
        """
        Predict future twin state using HD permutation (temporal dynamics).

        twin(t+1) ≈ ρ(twin(t))  — permutation encodes forward time step

        Args:
            steps: Number of time steps to predict forward
        Returns:
            Predicted future twin state
        """
        future = self.twin_state.clone()
        for _ in range(steps):
            future = permute(future, 1)
        return future

    def forward(
        self,
        ego_state: torch.Tensor,
        object_states: torch.Tensor,
        context_features: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Full digital twin update cycle."""
        B = ego_state.shape[0]

        ego_hd = self.encode_ego_state(
            ego_state[:, :7],
            ego_state[:, 7:10] if ego_state.shape[1] >= 10 else torch.zeros(B, 3),
        )

        obj_bundle = self.encode_objects(object_states)

        ctx_hd = torch.zeros(B, self.hd_dim, device=ego_state.device)
        if context_features is not None:
            ctx_hd = self.encode_context(context_features)

        twin = self.update_twin(ego_hd.squeeze(0), obj_bundle, ctx_hd.squeeze(0))

        return {
            "twin_state": twin,
            "ego_hd": ego_hd,
            "object_bundle": obj_bundle,
            "context_hd": ctx_hd,
        }


class SwarmConsensus(nn.Module):
    """
    Swarm consensus fusion for multi-UAV collaborative perception.
    Integrates digital twin states from multiple UAVs via HD bundling
    with communication-quality-aware weighting.

    From Yan et al. (2026): Digital twin-driven swarm coordination
    with consensus-based state fusion under communication constraints.

    Each UAV maintains its own digital twin. The swarm consensus:
      1. Exchanges compressed HD twin states with neighbors
      2. Weights contributions by link quality (SNR, latency, packet loss)
      3. Fuses via weighted HD bundling (robust to noise/sparsity)
      4. Updates local twin with consensus posterior
    """

    def __init__(
        self,
        num_agents: int = 4,
        hd_dim: int = 8192,
        consensus_rounds: int = 3,
    ):
        super().__init__()
        self.num_agents = num_agents
        self.hd_dim = hd_dim
        self.consensus_rounds = consensus_rounds

        # Agent-specific VSA modules for encoding/decoding
        self.vsa = VSAHDC(
            hd_dim=hd_dim,
            input_dim=hd_dim // 16,  # Compressed agent ID encoding
            dtype="bipolar",
            binding="circular",
        )

        # Agent identity vectors (for distinguishing sources in bundle)
        self.register_buffer(
            "agent_ids",
            torch.stack([
                self._generate_agent_id(i)
                for i in range(num_agents)
            ]),
        )

        # Communication link quality estimator
        self.link_quality = nn.Parameter(torch.ones(num_agents, num_agents) * 0.5)

    def _generate_agent_id(self, agent_idx: int) -> torch.Tensor:
        """Generate unique HD identity for each agent."""
        gen = torch.Generator()
        gen.manual_seed(agent_idx * 1000 + 42)
        return torch.bernoulli(
            torch.full((self.hd_dim,), 0.5, generator=gen)
        ) * 2 - 1

    def compress_twin(self, twin_state: torch.Tensor) -> torch.Tensor:
        """
        Compress HD twin state for communication-efficient sharing.
        Uses random projection to reduce bandwidth while preserving
        similarity structure (Johnson-Lindenstrauss property of HD vectors).
        """
        # Simple compression: sign of random projection
        compressed = torch.sign(twin_state @ torch.randn_like(twin_state).T[:self.hd_dim // 8])
        return compressed

    def decompress_twin(self, compressed: torch.Tensor) -> torch.Tensor:
        """Decompress received twin state (approximate reconstruction)."""
        return F.normalize(compressed.float(), p=2, dim=-1).expand(self.hd_dim)

    def consensus_update(
        self,
        local_twin: torch.Tensor,
        neighbor_twins: List[torch.Tensor],
        neighbor_ids: List[int],
        link_qualities: Optional[List[float]] = None,
    ) -> torch.Tensor:
        """
        Perform consensus-based fusion of multiple digital twin states.

        Args:
            local_twin: Local UAV's twin state [hd_dim]
            neighbor_twins: List of neighbor twin states [hd_dim]
            neighbor_ids: List of neighbor agent indices
            link_qualities: Optional per-neighbor link quality [0-1]
        Returns:
            Consensus twin state [hd_dim]
        """
        all_twins = [local_twin] + list(neighbor_twins)

        if link_qualities is None:
            # Use learned link qualities
            link_qualities = [1.0] + [
                self.link_quality[0, nid].item() for nid in neighbor_ids
            ]
        else:
            link_qualities = [1.0] + list(link_qualities)

        for _ in range(self.consensus_rounds):
            # Weighted HD bundling
            weighted_bundle = torch.zeros(self.hd_dim, device=local_twin.device)

            for twin, quality in zip(all_twins, link_qualities):
                if quality > self.vsa.uncertainty_threshold if hasattr(self.vsa, 'uncertainty_threshold') else 0.1:
                    weighted_bundle += quality * twin

            # Normalize (majority vote in bipolar space)
            consensus = torch.sign(weighted_bundle)
            consensus = F.normalize(consensus.float(), p=2, dim=-1)

            # Update all twins toward consensus
            all_twins = [
                (0.8 * twin + 0.2 * consensus).to(twin.device)
                for twin in all_twins
            ]

        return all_twins[0]  # Return updated local twin

    def forward(
        self,
        local_twin: torch.Tensor,
        neighbor_data: Dict[int, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Swarm consensus forward pass.

        Args:
            local_twin: Local twin state
            neighbor_data: Dict mapping agent_id → twin_state
        Returns:
            Dict with consensus_twin and agent_weights
        """
        neighbor_ids = list(neighbor_data.keys())
        neighbor_twins = [neighbor_data[i] for i in neighbor_ids]

        consensus = self.consensus_update(
            local_twin, neighbor_twins, neighbor_ids
        )

        return {
            "consensus_twin": consensus,
            "consensus_delta": F.cosine_similarity(
                consensus.unsqueeze(0), local_twin.unsqueeze(0)
            ).item(),
        }


class CommunicationAwareMixing(nn.Module):
    """
    Communication-aware modality mixing.
    Extends the MixingModule with communication quality awareness
    from the digital twin swarm framework.

    When communication is degraded, the model relies more on:
      - Local sensor data (higher weight)
      - Predictive virtual model (twin forward prediction)
      - Historical HD memory (temporal smoothing)
    """

    def __init__(
        self,
        hd_dim: int = 8192,
        feature_dim: int = 1024,
    ):
        super().__init__()
        self.hd_dim = hd_dim

        # Communication quality estimator
        self.comm_quality_net = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid(),
        )

        # Adaptive weighting based on comm quality
        self.local_weight_scale = nn.Parameter(torch.tensor(0.7))
        self.virtual_weight_scale = nn.Parameter(torch.tensor(0.3))

    def forward(
        self,
        local_features: torch.Tensor,
        virtual_features: torch.Tensor,
        comm_quality: Optional[float] = None,
    ) -> Tuple[torch.Tensor, float]:
        """
        Mix local and virtual features based on communication quality.

        When comm_quality is low → trust virtual model more
        When comm_quality is high → trust local sensors more

        Args:
            local_features: Features from local sensors
            virtual_features: Features from digital twin prediction
            comm_quality: Estimated communication quality [0-1]
        Returns:
            (mixed_features, effective_comm_quality)
        """
        if comm_quality is None:
            comm_quality = self.comm_quality_net(local_features).squeeze(-1)

        w_local = self.local_weight_scale * comm_quality
        w_virtual = self.virtual_weight_scale * (1 - comm_quality)

        mixed = w_local * local_features + w_virtual * virtual_features

        return mixed, comm_quality.item() if comm_quality.numel() == 1 else comm_quality