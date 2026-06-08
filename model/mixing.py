"""
Mixing Module — Bayesian-style Cross-modal Fusion with VSA/HDC
================================================================
's mixing module that performs Bayesian-style updates
across modalities to handle uncertainty, missing data, and causal relationships.

Enhanced with VSA/HDC operations from arthedain-1:
  - Prior: HD bundle of previously processed modalities
  - Likelihood: HD encoding of new modality features
  - Posterior: Weighted bundle with uncertainty gating
  - Natural handling of missing modalities (sparse sensor data)

This preserves Eldarin's strength in leveraging cross-modal cues for
robust estimation under partial observations.


Paper: https://arxiv.org/pdf/2411.13607
VSA/HDC: https://github.com/Enotrium/arthedain-1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
from .vsa_hdc import VSAHDC


class UncertaintyGate(nn.Module):
    """
    Learns to gate modality contributions based on uncertainty.
    Produces per-modality confidence weights for the Bayesian update.
    """

    def __init__(self, feature_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [B, N_modalities, feature_dim] or [B, feature_dim]
        Returns:
            confidence weight [B, N_modalities] or [B]
        """
        return self.gate(features).squeeze(-1)


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention for propagating information between modalities.
    Used in the mixing iterations to share context.
    """

    def __init__(self, embed_dim: int, num_heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """
        Cross-modal attention: query from modality A, key/value from modality B.

        Args:
            query: [B, N, D] query modality features
            key: [B, M, D] key modality features
            value: [B, M, D] value modality features

        Returns:
            [B, N, D] attended features
        """
        attn_out, _ = self.attn(query, key, value)
        return self.norm(query + attn_out)


class MixingModule(nn.Module):
    """
    Bayesian-style mixing module for cross-modal fusion.
    Iteratively updates a unified representation using all available modalities.

    From Eldarin: Models priors from earlier/higher-level features and
    uses likelihoods from each modality to produce posterior estimates.
    The HD-enhanced version operates in hyperdimensional space for
    robustness to noise and sparsity.

    Process:
        Initialize prior from first available modality
        For each iteration:
            For each modality:
                Likelihood = encode(modality_features)
                Posterior = bayesian_update(prior, likelihood, uncertainty)
                Cross-modal attention sharing
        Output fused representation

    Args:
        feature_dims: Dict mapping modality names to feature dimensions
        hd_dim: Hyperdimensional dimension for VSA operations
        num_iterations: Number of Bayesian update iterations
        use_uncertainty_gating: Enable learned uncertainty weighting
        temporal_window: Number of past frames to incorporate
        prior_weight: Default prior weight in Bayesian update
    """

    def __init__(
        self,
        feature_dims: Dict[str, int],
        hd_dim: int = 8192,
        num_iterations: int = 3,
        use_uncertainty_gating: bool = True,
        temporal_window: int = 16,
        prior_weight: float = 0.7,
    ):
        super().__init__()
        self.feature_dims = feature_dims
        self.hd_dim = hd_dim
        self.num_iterations = num_iterations
        self.use_uncertainty_gating = use_uncertainty_gating
        self.temporal_window = temporal_window
        self.prior_weight = prior_weight

        # Create VSA module for each modality
        self.vsa_modules = nn.ModuleDict()
        for mod_name, mod_dim in feature_dims.items():
            self.vsa_modules[mod_name] = VSAHDC(
                hd_dim=hd_dim,
                input_dim=mod_dim,
                dtype="bipolar",
                binding="circular",
            )

        # Uncertainty gates per modality
        if use_uncertainty_gating:
            self.uncertainty_gates = nn.ModuleDict()
            for mod_name, mod_dim in feature_dims.items():
                self.uncertainty_gates[mod_name] = UncertaintyGate(mod_dim)

        # Joint projection for all modalities
        max_dim = max(feature_dims.values())
        self.joint_dim = max_dim

        self.modality_projections = nn.ModuleDict()
        for mod_name, mod_dim in feature_dims.items():
            if mod_dim != self.joint_dim:
                self.modality_projections[mod_name] = nn.Sequential(
                    nn.Linear(mod_dim, self.joint_dim),
                    nn.LayerNorm(self.joint_dim),
                )

        # Cross-modal attention for each iteration
        self.cross_attn_layers = nn.ModuleList([
            CrossModalAttention(self.joint_dim) for _ in range(num_iterations)
        ])

        # Final fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(self.joint_dim, self.joint_dim),
            nn.LayerNorm(self.joint_dim),
            nn.SiLU(),
            nn.Linear(self.joint_dim, self.joint_dim),
        )

        # Temporal memory (exponential moving average)
        self.register_buffer(
            "temporal_memory",
            torch.zeros(1, self.joint_dim),
        )
        self.memory_decay = nn.Parameter(torch.tensor(0.9))

    def _align_modality(
        self, features: torch.Tensor, mod_name: str
    ) -> torch.Tensor:
        """Align modality features to joint dimension."""
        if mod_name in self.modality_projections:
            return self.modality_projections[mod_name](features)
        return features

    def forward(
        self,
        modality_features: Dict[str, torch.Tensor],
        available_modalities: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Perform Bayesian-style cross-modal mixing.

        Args:
            modality_features: Dict mapping modality name → features [B, D]
            available_modalities: Optional list of available modality names
                                  (for handling missing data)

        Returns:
            dict with:
                - "fused": Fused representation [B, joint_dim]
                - "hd_representation": HD fusion result [B, hd_dim]
                - "modality_weights": Per-modality confidence [B, N_mod]
                - "uncertainties": Per-modality uncertainty
                - "iteration_outputs": List of intermediate states
        """
        B = list(modality_features.values())[0].shape[0]
        device = next(iter(modality_features.values())).device

        if available_modalities is None:
            available_modalities = list(modality_features.keys())

        # Align all features to joint dimension
        aligned_features = {}
        for mod_name in available_modalities:
            if mod_name in modality_features:
                aligned_features[mod_name] = self._align_modality(
                    modality_features[mod_name], mod_name
                )

        if not aligned_features:
            return {
                "fused": torch.zeros(B, self.joint_dim, device=device),
                "hd_representation": torch.zeros(B, self.hd_dim, device=device),
                "modality_weights": torch.zeros(B, len(available_modalities), device=device),
                "uncertainties": {},
            }

        # Compute uncertainties
        uncertainties = {}
        if self.use_uncertainty_gating:
            for mod_name, feats in aligned_features.items():
                uncertainties[mod_name] = self.uncertainty_gates[mod_name](feats)

        # Encode to HD space
        hd_features = {}
        for mod_name, feats in aligned_features.items():
            hd_features[mod_name] = self.vsa_modules[mod_name].encode(feats)

        # Initialize prior: bundle of all HD features
        all_hd = torch.stack(list(hd_features.values()), dim=0)  # [N_mod, B, hd_dim]
        prior = torch.mean(all_hd, dim=0)  # [B, hd_dim]
        prior = F.normalize(prior, p=2, dim=-1)

        # Bayesian iterations
        iteration_outputs = []
        fused = torch.stack(list(aligned_features.values()), dim=1).mean(dim=1)  # [B, joint_dim]

        for iter_idx in range(self.num_iterations):
            modality_posteriors = []

            for mod_name in available_modalities:
                if mod_name not in aligned_features:
                    continue

                # Likelihood in HD space
                likelihood = hd_features[mod_name]

                # Uncertainty
                uncert = uncertainties.get(mod_name, None)

                # Bayesian update
                posterior_hd, new_uncert = self.vsa_modules[mod_name].bayesian_update(
                    prior=prior,
                    likelihood=likelihood,
                    prior_weight=self.prior_weight,
                    uncertainty=uncert.unsqueeze(-1) if uncert is not None else None,
                )
                modality_posteriors.append(posterior_hd)

                if new_uncert is not None and uncert is not None:
                    uncertainties[mod_name] = new_uncert

            # Update prior as bundle of all posteriors
            if modality_posteriors:
                posteriors_stack = torch.stack(modality_posteriors, dim=0)
                prior = torch.mean(posteriors_stack, dim=0)
                prior = F.normalize(prior, p=2, dim=-1)

            # Cross-modal attention in feature space
            feats_stack = torch.stack(
                [aligned_features[m] for m in available_modalities], dim=1
            )  # [B, N_mod, D]
            attended = self.cross_attn_layers[iter_idx](feats_stack, feats_stack, feats_stack)
            fused = attended.mean(dim=1)  # [B, D]

            iteration_outputs.append(fused)

        # Final fusion
        fused = self.fusion_proj(fused)

        # Temporal memory update
        self.temporal_memory = (
            self.memory_decay * self.temporal_memory
            + (1 - self.memory_decay) * fused.detach().mean(dim=0, keepdim=True)
        )

        # Modality weights
        modality_weights = torch.zeros(B, len(available_modalities), device=device)
        for i, mod_name in enumerate(available_modalities):
            if mod_name in uncertainties:
                modality_weights[:, i] = 1 - uncertainties[mod_name]

        return {
            "fused": fused,
            "hd_representation": prior,
            "modality_weights": modality_weights,
            "uncertainties": uncertainties,
            "iteration_outputs": iteration_outputs,
        }

    def forward_with_temporal(
        self,
        modality_features: Dict[str, torch.Tensor],
        available_modalities: Optional[List[str]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Mix with temporal memory from previous frames.
        Uses exponential moving average of past fused representations
        as an additional prior.
        """
        result = self.forward(modality_features, available_modalities)

        # Blend with temporal memory
        temporal_weight = 0.1  # Weight for past information
        result["fused"] = (
            (1 - temporal_weight) * result["fused"]
            + temporal_weight * self.temporal_memory.to(result["fused"].device)
        )

        return result

    def missing_modality_imputation(
        self,
        available_features: Dict[str, torch.Tensor],
        missing_modality: str,
    ) -> torch.Tensor:
        """
        Impute missing modality features from available ones using HD retrieval.
        Demonstrates VSA's ability to handle missing data.

        Args:
            available_features: Available modality features
            missing_modality: Name of missing modality

        Returns:
            Imputed features for missing modality [B, D]
        """
        # Encode all available
        hd_available = []
        for mod_name, feats in available_features.items():
            if mod_name in self.vsa_modules:
                hd_available.append(self.vsa_modules[mod_name].encode(feats))

        if not hd_available:
            return torch.zeros(
                list(available_features.values())[0].shape[0],
                self.feature_dims.get(missing_modality, self.joint_dim),
                device=list(available_features.values())[0].device,
            )

        # Bundle available → prior
        hd_bundle = torch.stack(hd_available, dim=0).mean(dim=0)

        # Retrieve missing from HD space
        if missing_modality in self.vsa_modules:
            # Project HD back to modality feature space
            imputed = hd_bundle @ self.vsa_modules[missing_modality].projection
            imputed = self._align_modality(imputed, missing_modality)
            return imputed
        return hd_bundle[:, :self.joint_dim]