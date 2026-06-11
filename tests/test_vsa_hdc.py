"""
Tests for VSA/HDC algebra operations.

Covers:
  - Binding invertibility: unbind(bind(a, b), b) ≈ a
  - Bundling similarity: sim(bundle([a, b, c]), a) > sim(bundle([a, b, c]), d)
  - FPE translation equivariance: encode(x + Δ) ≈ encode(x) ⊗ seed^Δ
  - Resonator network convergence on synthetic translation/rotation data
"""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from model.vsa_hdc import (
    VSAHDC,
    bind,
    bundle,
    permute,
    cosine_similarity,
    hamming_similarity,
    ResonatorNetwork,
    HierarchicalResonatorNetwork,
    HDCKalmanFilter,
)


@pytest.mark.vsa
class TestVSABinding:
    """Test VSA binding and unbinding operations."""

    def test_binding_invertibility_bipolar(self):
        """bind(a, b) should be invertible: unbind(bind(a, b), b) ≈ a."""
        vsa = VSAHDC(hd_dim=1024, input_dim=64, dtype="bipolar", binding="xor")
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)

        a_hd = vsa.encode(a)
        b_hd = vsa.encode(b)

        # Bind then unbind
        bound = vsa.bind(a_hd, b_hd)
        recovered = vsa.retrieve(bound, b_hd)

        sim = vsa.similarity(recovered, a_hd).item()
        assert sim > 0.5, f"Binding invertibility failed: sim={sim:.4f}"

    def test_binding_invertibility_circular(self):
        """Circular (FHRR) binding should be invertible."""
        vsa = VSAHDC(hd_dim=1024, input_dim=64, dtype="bipolar", binding="circular")
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)

        a_hd = vsa.encode(a)
        b_hd = vsa.encode(b)

        bound = vsa.bind(a_hd, b_hd)
        recovered = vsa.retrieve(bound, b_hd)

        sim = vsa.similarity(recovered, a_hd).item()
        assert sim > 0.3, f"Circular binding invertibility failed: sim={sim:.4f}"

    def test_binding_commutativity(self):
        """Binding should be commutative: a ⊗ b ≈ b ⊗ a."""
        vsa = VSAHDC(hd_dim=1024, input_dim=64, dtype="bipolar")
        a = torch.randn(1, 64)
        b = torch.randn(1, 64)

        a_hd = vsa.encode(a)
        b_hd = vsa.encode(b)

        ab = vsa.bind(a_hd, b_hd)
        ba = vsa.bind(b_hd, a_hd)

        sim = vsa.similarity(ab, ba).item()
        assert sim > 0.9, f"Binding commutativity failed: sim={sim:.4f}"

    def test_functional_bind(self):
        """Test the functional `bind()` API."""
        a = torch.randn(1, 128)
        b = torch.randn(1, 128)
        bound = bind(a, b, mode="circular")

        assert bound.shape == (1, 128)
        assert torch.isfinite(bound).all()


@pytest.mark.vsa
class TestVSABundling:
    """Test VSA bundling operations."""

    def test_bundling_similarity_to_constituents(self):
        """A bundle should be more similar to its constituents than random vectors."""
        vsa = VSAHDC(hd_dim=1024, input_dim=64, dtype="bipolar")
        a = vsa.encode(torch.randn(1, 64))
        b = vsa.encode(torch.randn(1, 64))
        c = vsa.encode(torch.randn(1, 64))
        d = vsa.encode(torch.randn(1, 64))

        bundled = vsa.bundle(torch.cat([a, b, c], dim=0))

        sim_to_constituent = vsa.similarity(bundled.unsqueeze(0), a).item()
        sim_to_outsider = vsa.similarity(bundled.unsqueeze(0), d).item()

        # Bundle should be more similar to constituent than outsider
        assert sim_to_constituent > sim_to_outsider, (
            f"sim(constituent)={sim_to_constituent:.4f} <= sim(outsider)={sim_to_outsider:.4f}"
        )

    def test_bundle_dimensions(self):
        """Bundling preserves dimensions."""
        vsa = VSAHDC(hd_dim=1024, input_dim=64, dtype="bipolar")
        vecs = [vsa.encode(torch.randn(1, 64)) for _ in range(5)]
        stacked = torch.cat(vecs, dim=0)
        bundled = vsa.bundle(stacked)

        assert bundled.shape == (1024,)
        assert torch.isfinite(bundled).all()

    def test_functional_bundle(self):
        """Test the functional `bundle()` API."""
        vecs = torch.randn(5, 128)
        bundled = bundle(vecs)

        assert bundled.shape == (128,)
        assert torch.isfinite(bundled).all()


@pytest.mark.vsa
class TestVSASimilarity:
    """Test VSA similarity metrics."""

    def test_self_similarity(self):
        """A vector should have maximum similarity with itself."""
        vsa = VSAHDC(hd_dim=1024, input_dim=64, dtype="bipolar")
        v = vsa.encode(torch.randn(1, 64))

        sim = vsa.similarity(v, v).item()
        assert sim > 0.99, f"Self-similarity should be ~1.0, got {sim:.4f}"

    def test_orthogonal_similarity(self):
        """High-dimensional random vectors are nearly orthogonal."""
        vsa = VSAHDC(hd_dim=8192, input_dim=64, dtype="bipolar")
        a = vsa.encode(torch.randn(1, 64))
        b = vsa.encode(torch.randn(1, 64))

        sim = vsa.similarity(a, b).item()
        # At 8192 dimensions, random bipolar vectors should have near-zero cosine sim
        assert abs(sim) < 0.1, f"Random vectors should be near-orthogonal: sim={sim:.4f}"

    def test_hamming_similarity(self):
        """Hamming similarity measures binary overlap."""
        vsa = VSAHDC(hd_dim=1024, input_dim=64, dtype="binary", similarity="hamming")
        a = vsa.encode(torch.randn(1, 64))

        sim = hamming_similarity(a, a).item()
        assert sim > 0.9, f"Self-hamming similarity failed: {sim:.4f}"


@pytest.mark.vsa
class TestFPETranslationEquivariance:
    """Test Fractional Power Encoding properties."""

    def test_binding_equals_addition(self):
        """encode(x) ⊗ encode(Δ) ≈ encode(x + Δ)."""
        from model.fpe import FractionalPowerEncoder

        fpe = FractionalPowerEncoder(hd_dim=1024, min_val=0.0, max_val=1.0, dtype="bipolar")

        x = torch.tensor([0.3])
        delta = torch.tensor([0.2])

        enc_x = fpe.encode_1d(x)
        enc_delta = fpe.encode_1d(delta)
        enc_sum = fpe.encode_1d(torch.tensor([0.5]))

        bound = enc_x * enc_delta  # binding for bipolar
        bound = torch.nn.functional.normalize(bound, p=2, dim=-1)
        enc_sum = torch.nn.functional.normalize(enc_sum, p=2, dim=-1)

        sim = cosine_similarity(bound, enc_sum).item()
        assert sim > 0.7, f"FPE binding=addition failed: sim={sim:.4f}"

    def test_codebook_encoding(self):
        """Codebook encoding produces correct shape."""
        from model.fpe import FractionalPowerEncoder

        fpe = FractionalPowerEncoder(hd_dim=1024, min_val=0.0, max_val=100.0, dtype="bipolar")
        codebook = fpe.build_codebook(16, 16)

        assert codebook.shape == (256, 1024), f"Expected (256, 1024), got {codebook.shape}"

        image = torch.zeros(16, 16)
        image[4:12, 4:12] = 1.0
        encoded = fpe.encode_image(image, codebook)

        assert encoded.shape == (1024,), f"Expected (1024,), got {encoded.shape}"
        assert torch.isfinite(encoded).all()


@pytest.mark.vsa
class TestResonatorNetwork:
    """Test resonator network convergence."""

    def test_resonator_factorization_translation(self):
        """Resonator can factorize a composite made from known factors."""
        hd_dim = 512
        factor_sizes = [8, 8]  # 8x8 translation space

        rn = ResonatorNetwork(
            factor_sizes=factor_sizes,
            hd_dim=hd_dim,
            gamma=0.5,
            nonlinearity="softmax",
            seed=42,
        )

        # Create a known composite: bind codebook[0][3] and codebook[1][5]
        h_idx, v_idx = 3, 5
        composite = rn.codebooks[0][h_idx] * rn.codebooks[1][v_idx]
        composite = torch.nn.functional.normalize(composite.unsqueeze(0), p=2, dim=-1)

        result = rn(composite, num_iterations=20)

        h_conf = result["factors"][0]  # [1, 8]
        v_conf = result["factors"][1]  # [1, 8]

        h_pred = h_conf.argmax(dim=-1).item()
        v_pred = v_conf.argmax(dim=-1).item()

        assert h_pred == h_idx, f"Resonator failed for h: predicted {h_pred}, expected {h_idx}"
        assert v_pred == v_idx, f"Resonator failed for v: predicted {v_pred}, expected {v_idx}"

    def test_resonator_hierarchical(self):
        """Hierarchical resonator initializes without error."""
        hrn = HierarchicalResonatorNetwork(
            cartesian_factors=[32, 32],
            logpolar_factors=[16],
            hd_dim=512,
            gamma=0.3,
            nonlinearity="phasor",
            seed=42,
        )

        # Create a dummy input
        encoded = torch.randn(1, 512)
        encoded = torch.nn.functional.normalize(encoded, p=2, dim=-1)

        result = hrn(encoded, num_iterations=3)

        assert "translation" in result
        assert "rotation" in result
        assert "cartesian_factors" in result
        assert "logpolar_factors" in result

    def test_population_vector_readout(self):
        """Population vector readout gives sub-index precision."""
        rn = ResonatorNetwork(
            factor_sizes=[16],
            hd_dim=512,
            seed=42,
        )

        # Create a confidence vector peaked at index 7 with Gaussian spread
        conf = torch.zeros(1, 16)
        for i in range(16):
            conf[0, i] = torch.exp(torch.tensor(-((i - 7) ** 2) / 4.0))

        readout = rn.population_vector_readout(conf)

        # Readout should be close to 7.0 (within ±2 due to variance)
        assert 5.0 < readout.item() < 9.0, f"Population readout off: {readout.item():.2f}"


@pytest.mark.vsa
class TestHDKalmanFilter:
    """Test Hyperdimensional Kalman filter."""

    def test_kalman_predict_update(self):
        """HD Kalman filter predict and update operations."""
        kf = HDCKalmanFilter(state_dim=4, hd_dim=512, dtype="bipolar")

        state = torch.randn(1, 4)
        measurement = torch.randn(1, 4)

        # Test encode
        hd_state = kf.encode_state(state)
        assert hd_state.shape == (1, 512)

        # Test predict
        pred = kf.predict(hd_state)
        assert pred.shape == (1, 512)
        assert torch.isfinite(pred).all()

        # Test update
        updated = kf.update(pred, kf.encode_state(measurement))
        assert updated.shape == (1, 512)
        assert torch.isfinite(updated).all()

    def test_kalman_forward(self):
        """Forward pass with measurement."""
        kf = HDCKalmanFilter(state_dim=4, hd_dim=512, dtype="bipolar")

        prev_state = torch.randn(1, 4)
        measurement = torch.randn(1, 4)

        hd_prev = kf.encode_state(prev_state)
        hd_meas = kf.encode_state(measurement)

        posterior, predicted = kf(hd_prev, hd_meas)

        assert posterior.shape == (1, 512)
        assert predicted.shape == (1, 512)

    def test_kalman_forward_no_measurement(self):
        """Forward pass without measurement (pure prediction)."""
        kf = HDCKalmanFilter(state_dim=4, hd_dim=512, dtype="bipolar")

        prev_state = torch.randn(1, 4)
        hd_prev = kf.encode_state(prev_state)

        posterior, predicted = kf(prev_hd_state=hd_prev, measurement=None)

        assert posterior.shape == (1, 512)
        assert predicted.shape == (1, 512)
        # Without measurement, posterior should equal predicted
        sim = cosine_similarity(posterior, predicted).item()
        assert sim > 0.99, f"Without measurement, posterior ≈ predicted: sim={sim:.4f}"