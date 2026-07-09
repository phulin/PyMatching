# Copyright 2022 PyMatching Contributors

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#      http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
from scipy.sparse import csc_matrix

import pymatching
from pymatching import Matching


class TestEdgeReweighting:
    """Test suite for edge reweighting functionality."""

    def test_basic_edge_reweighting(self):
        """Test basic edge reweighting with simple graph."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0, fault_ids=0)
        m.add_edge(1, 2, weight=2.0, fault_ids=1)
        m.add_boundary_edge(0, weight=0.5, fault_ids=2)
        m.add_boundary_edge(2, weight=0.5, fault_ids=3)

        # Test without reweighting
        syndrome = np.array([1, 0, 1])
        correction_orig = m.decode(syndrome)

        # Test with edge reweighting - make edge (0,1) much heavier
        edge_reweights = np.array([
            [0, 1, 10.0],  # Make edge (0,1) very heavy
        ], dtype=np.float64)

        correction_reweighted = m.decode(syndrome, edge_reweights=edge_reweights)

        # The reweighted solution should potentially be different
        # (depending on the specific graph structure and syndrome)
        assert correction_reweighted is not None
        assert len(correction_reweighted) >= 1

    def test_boundary_edge_reweighting(self):
        """Test reweighting of boundary edges."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(1, weight=1.0)

        # Reweight boundary edge
        edge_reweights = np.array([
            [0, -1, 0.1],  # Make boundary edge (0,) very light
        ], dtype=np.float64)

        syndrome = np.array([1, 0])
        correction = m.decode(syndrome, edge_reweights=edge_reweights)

        assert correction is not None

    def test_multiple_edge_reweighting(self):
        """Test reweighting multiple edges simultaneously."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_edge(1, 2, weight=1.0)
        m.add_edge(2, 3, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(3, weight=1.0)

        # Reweight multiple edges
        edge_reweights = np.array([
            [0, 1, 0.1],   # Make edge (0,1) light
            [1, 2, 10.0],  # Make edge (1,2) heavy
            [0, -1, 5.0],  # Make boundary edge (0,) heavy
        ], dtype=np.float64)

        syndrome = np.array([1, 0, 0, 1])
        correction = m.decode(syndrome, edge_reweights=edge_reweights)

        assert correction is not None

    def test_reweighting_without_regeneration(self):
        """Tier-1 (reweight <= max, no regeneration): in-place weight updates must
        match a matcher built from scratch with the same weights. boundary(2)=5.0
        pins the max weight so both share the same discretization."""
        def build(w01, w12):
            m = Matching()
            m.add_edge(0, 1, weight=w01, fault_ids=0)
            m.add_edge(1, 2, weight=w12, fault_ids=1)
            m.add_boundary_edge(0, weight=1.5, fault_ids=2)
            m.add_boundary_edge(2, weight=5.0, fault_ids=3)  # pins max at 5.0
            return m

        syndrome = np.array([1, 0, 1])
        corr, w = build(2.0, 1.0).decode(
            syndrome,
            edge_reweights=np.array([[0, 1, 1.5], [1, 2, 0.5]], dtype=np.float64),
            return_weight=True,
        )
        corr_oracle, w_oracle = build(1.5, 0.5).decode(syndrome, return_weight=True)
        np.testing.assert_array_equal(corr, corr_oracle)
        assert np.isclose(w, w_oracle)

    def test_reweighting_with_regeneration(self):
        """Tier-2 (reweight > max, full regeneration): the rebuilt graph must match a
        matcher built from scratch with the reweighted edge (both regenerate to the
        same weights/normalisation)."""
        def build(w01):
            m = Matching()
            m.add_edge(0, 1, weight=w01, fault_ids=0)
            m.add_edge(1, 2, weight=0.5, fault_ids=1)
            m.add_boundary_edge(0, weight=0.8, fault_ids=2)
            return m

        syndrome = np.array([1, 0, 1])
        corr, w = build(1.0).decode(
            syndrome,
            edge_reweights=np.array([[0, 1, 5.0]], dtype=np.float64),  # > max 1.0 -> Tier 2
            return_weight=True,
        )
        corr_oracle, w_oracle = build(5.0).decode(syndrome, return_weight=True)
        np.testing.assert_array_equal(corr, corr_oracle)
        assert np.isclose(w, w_oracle)

    def test_batch_decoding_with_reweights(self):
        """Test batch decoding with edge reweights."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_edge(1, 2, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(2, weight=1.0)

        # Create batch of syndromes
        shots = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0]
        ], dtype=np.uint8)

        # Different reweights for each shot
        reweights1 = np.array([[0, 1, 0.1]], dtype=np.float64)
        reweights2 = np.array([[1, 2, 0.1]], dtype=np.float64)
        reweights3 = np.array([[0, -1, 0.1]], dtype=np.float64)

        edge_reweights = [reweights1, reweights2, reweights3]

        corrections = m.decode_batch(shots, edge_reweights=edge_reweights)

        assert corrections.shape[0] == 3  # Three shots
        assert corrections is not None

    def test_weight_restoration(self):
        """Test that original weights are restored after decoding."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_edge(1, 2, weight=2.0)

        # Store original correction
        syndrome = np.array([1, 0, 1])
        correction_orig = m.decode(syndrome)

        # Decode with reweights
        edge_reweights = np.array([[0, 1, 10.0]], dtype=np.float64)
        correction_reweighted = m.decode(syndrome, edge_reweights=edge_reweights)

        # Decode again without reweights - should match original
        correction_restored = m.decode(syndrome)

        np.testing.assert_array_equal(correction_orig, correction_restored)

    def test_return_weight_with_reweights(self):
        """return_weight reports the reweighted solution weight -- the exact expected
        value, matching an oracle (not merely != the un-reweighted weight)."""
        def build(w01):
            m = Matching()
            m.add_edge(0, 1, weight=w01, fault_ids=0)
            m.add_boundary_edge(0, weight=3.0, fault_ids=1)  # pins max; boundary pair costs 6.0
            m.add_boundary_edge(1, weight=3.0, fault_ids=2)
            return m

        syndrome = np.array([1, 1])  # edge(0,1) route beats the boundary pair
        _, weight_orig = build(1.0).decode(syndrome, return_weight=True)
        corr, weight_rw = build(1.0).decode(
            syndrome, edge_reweights=np.array([[0, 1, 0.5]], dtype=np.float64), return_weight=True
        )
        _, weight_oracle = build(0.5).decode(syndrome, return_weight=True)

        assert np.isclose(weight_orig, 1.0)
        assert np.isclose(weight_rw, weight_oracle)
        assert np.isclose(weight_rw, 0.5)  # exact reweighted value, not just "different"

    def test_correlations_with_reweights(self):
        """enable_correlations reweights the search graph too; the result must match a
        freshly-built matcher with the reweighted edge (same normalisation via a pinned max)."""
        def build(w01):
            m = Matching()
            m.add_edge(0, 1, weight=w01, fault_ids=0)
            m.add_edge(1, 2, weight=1.0, fault_ids=1)
            m.add_boundary_edge(0, weight=3.0, fault_ids=2)  # pins max at 3.0
            m.add_boundary_edge(2, weight=3.0, fault_ids=3)
            return m

        syndrome = np.array([1, 0, 1])
        corr, w = build(2.0).decode(
            syndrome,
            edge_reweights=np.array([[0, 1, 0.5]], dtype=np.float64),
            enable_correlations=True,
            return_weight=True,
        )
        corr_oracle, w_oracle = build(0.5).decode(
            syndrome, enable_correlations=True, return_weight=True
        )
        np.testing.assert_array_equal(corr, corr_oracle)
        assert np.isclose(w, w_oracle)

    def test_invalid_edge_specification(self):
        """Test error handling for invalid edge specifications."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)

        syndrome = np.array([1, 1])

        # Test non-existent edge
        with pytest.raises(ValueError, match="does not exist"):
            edge_reweights = np.array([[0, 2, 1.0]], dtype=np.float64)  # Edge (0,2) doesn't exist
            m.decode(syndrome, edge_reweights=edge_reweights)

    def test_negative_weight_reweighting(self):
        """Test error handling for negative reweight values."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)

        syndrome = np.array([1, 1])

        # Test negative weight
        with pytest.raises(ValueError, match="non-negative"):
            edge_reweights = np.array([[0, 1, -1.0]], dtype=np.float64)  # Negative weight
            m.decode(syndrome, edge_reweights=edge_reweights)

    def test_invalid_boundary_edge_format(self):
        """Test error handling for invalid boundary edge format."""
        m = Matching()
        m.add_boundary_edge(0, weight=1.0)

        syndrome = np.array([1])

        # Test invalid boundary edge format (should use -1, not -2)
        with pytest.raises(ValueError, match="exactly -1"):
            edge_reweights = np.array([[0, -2, 1.0]], dtype=np.float64)
            m.decode(syndrome, edge_reweights=edge_reweights)

    def test_reweighting_with_check_matrix(self):
        """Reweighting on a graph built from a check matrix yields the expected
        correction, exercising the check-matrix edge indexing (not just 'no crash').

        The rep code has boundary edge (0,) = fault 0, edge (0,1) = fault 1, boundary
        edge (1,) = fault 2 (all unit weight). Syndrome [1,0] normally pairs detector 0
        to its boundary (fault 0). Making that boundary edge expensive forces the route
        via edge (0,1) + boundary (1,) instead (faults 1 and 2).
        """
        H = csc_matrix(([1, 1, 1, 1], ([0, 0, 1, 1], [0, 1, 1, 2])), shape=(2, 3))
        m = Matching(H)  # edges: (0,2)=fault 0, (0,1)=fault 1, (1,2)=fault 2 (node 2 is boundary)
        syndrome = np.array([1, 1])

        # Default: pair the two defects directly via edge (0,1) = fault 1.
        np.testing.assert_array_equal(m.decode(syndrome), [0, 1, 0])
        # Make edge (0,1) expensive -> route each defect via node 2 instead (faults 0 and 2).
        corr = m.decode(syndrome, edge_reweights=np.array([[0, 1, 3.0]], dtype=np.float64))
        np.testing.assert_array_equal(corr, [1, 0, 1])

    def test_empty_reweights_array(self):
        """Test behavior with empty reweights array."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(1, weight=1.0)

        syndrome = np.array([1, 1])

        # Empty reweights array should work like no reweights
        edge_reweights = np.array([], dtype=np.float64).reshape(0, 3)
        correction_empty = m.decode(syndrome, edge_reweights=edge_reweights)
        correction_none = m.decode(syndrome)

        np.testing.assert_array_equal(correction_empty, correction_none)

    def test_batch_with_different_reweight_sizes(self):
        """Test batch decoding with different numbers of reweights per shot."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_edge(1, 2, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(2, weight=1.0)

        shots = np.array([
            [1, 0, 1],
            [0, 1, 0]
        ], dtype=np.uint8)

        # Different numbers of reweights per shot
        reweights1 = np.array([[0, 1, 0.5], [1, 2, 0.5]], dtype=np.float64)  # 2 reweights
        reweights2 = np.array([[0, -1, 0.1]], dtype=np.float64)  # 1 reweight

        edge_reweights = [reweights1, reweights2]

        corrections = m.decode_batch(shots, edge_reweights=edge_reweights)
        assert corrections.shape[0] == 2

    def test_exception_safety_single_decode(self):
        """When decoding raises *after* reweights are applied, the weights must be
        restored. Force a real exception (a boundaryless odd-parity syndrome has no
        matching) and check the graph is pristine afterwards."""
        def build():
            m = Matching()
            m.add_edge(0, 1, weight=1.0, fault_ids=0)  # no boundary -> [1,0,0] is unmatchable
            m.add_edge(1, 2, weight=1.0, fault_ids=1)
            return m

        m = build()
        # Reweight is applied, then decoding the unmatchable syndrome raises.
        with pytest.raises(ValueError):
            m.decode(np.array([1, 0, 0]), edge_reweights=np.array([[0, 1, 0.5]], dtype=np.float64))

        # After the exception the graph must be restored: a valid decode matches a fresh matcher.
        syndrome = np.array([1, 1, 0])
        corr, w = m.decode(syndrome, return_weight=True)
        corr_fresh, w_fresh = build().decode(syndrome, return_weight=True)
        np.testing.assert_array_equal(corr, corr_fresh)
        assert np.isclose(w, w_fresh)

    def test_exception_safety_batch_decode(self):
        """When a batch decode raises after reweights are applied, weights are restored."""
        def build():
            m = Matching()
            m.add_edge(0, 1, weight=1.0, fault_ids=0)  # no boundary -> [1,0,0] unmatchable
            m.add_edge(1, 2, weight=1.0, fault_ids=1)
            return m

        m = build()
        with pytest.raises(ValueError):
            m.decode_batch(
                np.array([[1, 0, 0]], dtype=np.uint8),
                edge_reweights=[np.array([[0, 1, 0.5]], dtype=np.float64)],
            )
        # Graph pristine afterwards.
        syndrome = np.array([1, 1, 0])
        corr, w = m.decode(syndrome, return_weight=True)
        corr_fresh, w_fresh = build().decode(syndrome, return_weight=True)
        np.testing.assert_array_equal(corr, corr_fresh)
        assert np.isclose(w, w_fresh)

    def test_batch_decoding_with_stride(self):
        """Test batch decoding with stride > 1."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_edge(1, 2, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(2, weight=1.0)

        # 6 shots, 2 rules, stride=3
        shots = np.array([
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 0],
            [1, 1, 0]
        ], dtype=np.uint8)

        reweights1 = np.array([[0, 1, 0.1]], dtype=np.float64)  # Shots 0-2
        reweights2 = np.array([[1, 2, 0.1]], dtype=np.float64)  # Shots 3-5

        corrections = m.decode_batch(
            shots,
            edge_reweights=[reweights1, reweights2],
            reweight_stride=3
        )

        assert corrections.shape[0] == 6

    def test_stride_one_backward_compatible(self):
        """Test that stride=1 behaves identically to current implementation."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(1, weight=1.0)

        shots = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        reweights = [
            np.array([[0, 1, 0.5]], dtype=np.float64),
            np.array([[0, 1, 0.5]], dtype=np.float64)
        ]

        # Explicit stride=1
        result1 = m.decode_batch(shots, edge_reweights=reweights, reweight_stride=1)

        # Default stride (should be 1)
        result2 = m.decode_batch(shots, edge_reweights=reweights)

        np.testing.assert_array_equal(result1, result2)

    def test_invalid_stride_zero(self):
        """Test that stride=0 raises an error."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(1, weight=1.0)

        shots = np.array([[1, 1]], dtype=np.uint8)
        reweights = [np.array([[0, 1, 0.5]], dtype=np.float64)]

        with pytest.raises(ValueError, match="positive integer"):
            m.decode_batch(shots, edge_reweights=reweights, reweight_stride=0)

    def test_stride_mismatch(self):
        """Test that stride × rules != shots raises an error."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(1, weight=1.0)

        shots = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.uint8)  # 3 shots
        reweights = [np.array([[0, 1, 0.5]], dtype=np.float64)]  # 1 rule

        with pytest.raises(ValueError, match="must be equal"):
            m.decode_batch(shots, edge_reweights=reweights, reweight_stride=2)  # 2 × 1 = 2 ≠ 3

    def test_stride_with_none_rules(self):
        """Test stride with some None rules (no reweighting for those blocks)."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(1, weight=1.0)

        shots = np.array([
            [1, 0], [0, 1],  # Block 0: reweighted
            [1, 1], [1, 0],  # Block 1: no reweighting
        ], dtype=np.uint8)

        reweights = [
            np.array([[0, 1, 0.5]], dtype=np.float64),  # Block 0
            None,  # Block 1: no reweighting
        ]

        corrections = m.decode_batch(
            shots,
            edge_reweights=reweights,
            reweight_stride=2
        )

        assert corrections.shape[0] == 4

    def test_stride_with_regeneration(self):
        """Test stride mode when regeneration is needed."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)  # Max weight is 1.0
        m.add_boundary_edge(0, weight=0.5)
        m.add_boundary_edge(1, weight=0.5)

        shots = np.array([
            [1, 0], [0, 1],
            [1, 1], [1, 0],
        ], dtype=np.uint8)

        # Weight 5.0 > max weight 1.0, triggers regeneration
        reweights = [
            np.array([[0, 1, 5.0]], dtype=np.float64),
            np.array([[0, 1, 0.5]], dtype=np.float64),
        ]

        corrections = m.decode_batch(
            shots,
            edge_reweights=reweights,
            reweight_stride=2
        )

        assert corrections.shape[0] == 4

    def test_stride_weight_restoration(self):
        """Test that weights are properly restored after stride-based decoding."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(1, weight=1.0)

        syndrome = np.array([1, 1])

        # Get original result
        original = m.decode(syndrome)

        # Decode batch with stride
        shots = np.array([[1, 0], [0, 1]], dtype=np.uint8)
        reweights = [np.array([[0, 1, 10.0]], dtype=np.float64)]
        m.decode_batch(shots, edge_reweights=reweights, reweight_stride=2)

        # Verify weights are restored
        restored = m.decode(syndrome)
        np.testing.assert_array_equal(original, restored)

    def test_tier1_reweight_matches_freshly_built_matcher(self):
        """Regression for the factor-of-2 being applied twice in Tier 1.

        A Tier-1 reweight (new weight <= original max, no regeneration) must
        produce the same correction *and* solution weight as a matcher built
        from scratch with those weights. boundary(0)=5.0 pins the maximum weight
        so the reweighted graph and the oracle share the same normalisation
        (a fair comparison; Tier-1 reweighting deliberately reuses the original
        graph's normalisation). Non-integer weights give fine discretisation so
        the reported weight is exact. Reweighting edge (0, 1) from 2.7 to 0.9
        keeps it the cheapest match for syndrome [1, 1]; if the weight is
        silently doubled (~1.8) the reported solution weight no longer matches
        the oracle.
        """
        def build(edge01):
            m = Matching()
            m.add_edge(0, 1, weight=edge01, fault_ids=0)
            m.add_boundary_edge(0, weight=5.0, fault_ids=1)  # pins max_abs_weight
            m.add_boundary_edge(1, weight=1.3, fault_ids=2)
            return m

        syndrome = np.array([1, 1])

        reweighted = build(2.7)
        corr, weight = reweighted.decode(
            syndrome,
            edge_reweights=np.array([[0, 1, 0.9]], dtype=np.float64),
            return_weight=True,
        )

        # Fair oracle: built from scratch with the same weights and the same max
        # weight (5.0), hence the same discretisation as the reweighted graph.
        oracle = build(0.9)
        corr_oracle, weight_oracle = oracle.decode(syndrome, return_weight=True)

        np.testing.assert_array_equal(corr, corr_oracle)
        assert np.isclose(weight, weight_oracle)
        # Concretely: edge (0, 1) (fault 0) wins at weight 0.9, not a doubled 1.8.
        np.testing.assert_array_equal(corr, np.array([1, 0, 0]))
        assert np.isclose(weight, 0.9)

    def test_reweight_does_not_contaminate_later_shots(self):
        """Regression for reweight contamination across shots in a batch.

        The restore path must return a reweighted edge to its *exact* original
        discretized weight, not a doubled value. A reweight applied to shot 0
        must not change the result of shot 1, which carries no reweight.
        """
        def build():
            m = Matching()
            m.add_edge(0, 1, weight=1.0, fault_ids=0)
            m.add_edge(1, 2, weight=1.0, fault_ids=1)
            m.add_boundary_edge(0, weight=1.2, fault_ids=2)
            m.add_boundary_edge(2, weight=1.2, fault_ids=3)
            return m

        # Shot 0 is trivial but carries a Tier-1 reweight of edge (0, 1) -> 0.0.
        # Shot 1 carries no reweight; its optimal matching uses edge (0, 1).
        shots = np.array([[0, 0, 0], [1, 1, 0]], dtype=np.uint8)

        reweighted = build()
        _, w_reweighted = reweighted.decode_batch(
            shots,
            edge_reweights=[np.array([[0, 1, 0.0]], dtype=np.float64), None],
            reweight_stride=1,
            return_weights=True,
        )

        # Ground truth: shot 1 decoded on its own with a pristine matcher.
        ref = build()
        _, w_ref = ref.decode(shots[1], return_weight=True)

        # Shot 1 has no reweight, so its weight must be independent of shot 0.
        assert np.isclose(w_reweighted[1], w_ref)
        assert np.isclose(w_reweighted[1], 1.0)

    def test_batch_regeneration_does_not_persist_reweights(self):
        """Regression for bug #2: a multi-block batch that triggers regeneration.

        When a batch contains a reweight above the original max weight
        (batch regeneration) AND more than one block, the apply and restore of
        block 0 must use the *same* tier. Otherwise block 0's reweight is written
        into the UserGraph (Tier-2 apply) but never undone (Tier-1 restore) and
        persists forever. Here block 0 reweights edge (0, 1) -> 5.0 (> max 1.0);
        after the batch every edge must equal its original weight.
        """
        def build():
            m = Matching()
            m.add_edge(0, 1, weight=1.0, fault_ids=0)   # original max weight = 1.0
            m.add_edge(1, 2, weight=1.0, fault_ids=1)
            m.add_boundary_edge(0, weight=1.0, fault_ids=2)
            m.add_boundary_edge(2, weight=1.0, fault_ids=3)
            return m

        def w01(m):
            return next(a["weight"] for x, y, a in m.edges() if {x, y} == {0, 1})

        m = build()
        shots = np.array([[1, 0, 1], [1, 0, 1]], dtype=np.uint8)
        # block 0: edge (0,1) -> 5.0 (> max => regeneration); block 1: None.
        m.decode_batch(
            shots,
            edge_reweights=[np.array([[0, 1, 5.0]], dtype=np.float64), None],
            reweight_stride=1,
        )

        assert np.isclose(w01(m), 1.0)          # UserGraph reverted
        m.decode(np.array([1, 0, 1]))           # a later regeneration
        assert np.isclose(w01(m), 1.0)          # still reverted

    def test_batch_regeneration_does_not_contaminate_later_shots(self):
        """Regression for bug #2: later shots must decode on the original graph.

        Even a shot with no reweight, following a regeneration-triggering block,
        must decode exactly as on the original graph -- the block-0 reweight must
        not leak into it (previously it saw the reweighted edge at weight 0).
        """
        def build(w01=2.0):
            m = Matching()
            m.add_edge(0, 1, weight=w01, fault_ids=0)
            m.add_edge(1, 2, weight=2.0, fault_ids=1)
            m.add_boundary_edge(0, weight=1.0, fault_ids=2)
            m.add_boundary_edge(2, weight=1.0, fault_ids=3)
            return m

        syn = np.array([1, 1, 0])  # edge (0,1) route competes with the boundary route
        shots = np.tile(syn, (2, 1))

        m = build()
        preds, ws = m.decode_batch(
            shots,
            edge_reweights=[np.array([[0, 1, 5.0]], dtype=np.float64), None],
            reweight_stride=1,
            return_weights=True,
        )

        # Shot 0 uses the reweighted graph (edge=5.0): the boundary route wins.
        corr0, w0 = build(w01=5.0).decode(syn, return_weight=True)
        np.testing.assert_array_equal(preds[0], corr0)
        assert np.isclose(ws[0], w0)

        # Shot 1 has NO reweight: must decode as the ORIGINAL graph (edge=2.0),
        # i.e. edge (0,1) wins at weight 2.0 -- not a zeroed/contaminated edge.
        corr1, w1 = build().decode(syn, return_weight=True)
        np.testing.assert_array_equal(preds[1], corr1)
        assert np.isclose(ws[1], w1)
        assert np.isclose(ws[1], 2.0)

    def test_fractional_reweight_on_integer_graph(self):
        """Regression for the discretization-resolution gap.

        On an all-integer-weight graph the normalising constant collapses to 1.0,
        so the matching graph has integer-only resolution. A fractional reweight
        applied in place (Tier 1) would be silently rounded to the nearest integer
        (e.g. 0.1 -> a free, weight-0 edge). needs_regeneration must instead force a
        full regeneration (Tier 2) so the graph is rebuilt at fine resolution, and
        the result must match a matcher built from scratch with the fractional edge.
        """
        def build(edge01):
            m = Matching()
            m.add_edge(0, 1, weight=edge01, fault_ids=0)   # all-integer base -> nc == 1
            m.add_boundary_edge(0, weight=1.0, fault_ids=1)
            m.add_boundary_edge(1, weight=1.0, fault_ids=2)
            return m

        syndrome = np.array([1, 1])
        for v in (0.1, 0.6, 1.4):
            corr, weight = build(1.0).decode(
                syndrome,
                edge_reweights=np.array([[0, 1, v]], dtype=np.float64),
                return_weight=True,
            )
            # Oracle: built from scratch with the fractional edge (fine resolution).
            corr_oracle, weight_oracle = build(v).decode(syndrome, return_weight=True)
            np.testing.assert_array_equal(corr, corr_oracle)
            assert np.isclose(weight, weight_oracle, atol=1e-5)
            assert np.isclose(weight, v, atol=1e-5)  # not rounded to an integer

    def test_reweight_after_set_boundary_matches_oracle(self):
        """Tier-1 reweighting after set_boundary must operate on the rebuilt graph.

        Regression for stale-index writes after a topology change (originally a
        segfault via a since-removed edge-index cache). The reweighted edge (2,3)
        is NOT incident to the new boundary node, so it keeps a dedicated Tier-1
        slot; the weight assertions discriminate a dropped or misdirected
        reweight (0.5 applied vs 1.3 plain), unlike the earlier version of this
        test whose reweight targeted a boundary-incident edge and was silently a
        no-op for subject and oracle alike.
        """
        def build(w23=1.3):
            m = Matching()
            m.add_edge(0, 1, weight=1.3, fault_ids=0)
            m.add_edge(1, 2, weight=1.3, fault_ids=1)
            m.add_edge(2, 3, weight=w23, fault_ids=2)
            m.add_boundary_edge(0, weight=1.3, fault_ids=3)
            return m

        syndrome = np.array([0, 0, 1, 1])
        rw = np.array([[2, 3, 0.5]], dtype=np.float64)

        m = build()
        m.decode(syndrome, edge_reweights=np.array([[2, 3, 0.9]], dtype=np.float64))  # pre-change reweighted decode
        m.set_boundary_nodes({1})                                                     # rebuild graph
        corr, weight = m.decode(syndrome, edge_reweights=rw, return_weight=True)

        # Oracle: fresh matcher built directly with the reweighted value.
        oracle = build(0.5)
        oracle.set_boundary_nodes({1})
        corr_o, weight_o = oracle.decode(syndrome, return_weight=True)
        np.testing.assert_array_equal(corr, corr_o)
        assert np.isclose(weight, weight_o)

        # Discriminating check: the reweight must actually have taken effect.
        plain = build()
        plain.set_boundary_nodes({1})
        _, weight_plain = plain.decode(syndrome, return_weight=True)
        assert not np.isclose(weight, weight_plain)

    def test_reweights_invalid_shape_raises(self):
        """C2: an edge_reweights array without exactly 3 columns must raise a clear
        error instead of reading out of bounds."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        syndrome = np.array([1, 1])
        for bad in (np.zeros((1, 2)), np.zeros((1, 4)), np.zeros((2, 2))):
            with pytest.raises(ValueError, match="shape"):
                m.decode(syndrome, edge_reweights=bad.astype(np.float64))
        with pytest.raises(ValueError, match="shape"):
            m.decode_batch(
                np.array([[1, 1]], dtype=np.uint8),
                edge_reweights=[np.zeros((1, 2), dtype=np.float64)],
            )

    def test_reweight_stride_zero_raises(self):
        """C3: reweight_stride=0 must raise even when edge_reweights is None
        (previously integer division by zero -> hard crash)."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        m.add_boundary_edge(0, weight=1.0)
        m.add_boundary_edge(1, weight=1.0)
        with pytest.raises(ValueError, match="positive integer"):
            m.decode_batch(np.array([[1, 1]], dtype=np.uint8), reweight_stride=0)

    def test_reweight_negative_node_raises(self):
        """C7: a negative node index must raise, not invoke undefined behaviour
        casting a negative double to size_t."""
        m = Matching()
        m.add_edge(0, 1, weight=1.0)
        with pytest.raises(ValueError):
            m.decode(np.array([1, 1]), edge_reweights=np.array([[-1.0, 1.0, 0.5]]))

    def test_negative_weight_in_graph_rejected(self):
        """Reweighting is unsupported (and must raise) when the graph itself contains a
        negative edge weight -- distinct from a negative reweight *value*."""
        m = Matching()
        m.add_edge(0, 1, weight=-1.0, fault_ids=0)
        m.add_boundary_edge(0, weight=1.0, fault_ids=1)
        m.add_boundary_edge(1, weight=1.0, fault_ids=2)
        with pytest.raises(ValueError, match="negative"):
            m.decode(np.array([1, 1]), edge_reweights=np.array([[0, 1, 0.5]], dtype=np.float64))

    def test_stride_value_correctness(self):
        """With stride>1, each block's rule applies to its shots and the per-shot
        weights match per-block oracles (not just a shape check)."""
        def build(w01=2.0):
            m = Matching()
            m.add_edge(0, 1, weight=w01, fault_ids=0)
            m.add_boundary_edge(0, weight=3.0, fault_ids=1)  # pins max at 3.0
            m.add_boundary_edge(1, weight=3.0, fault_ids=2)
            return m

        shots = np.tile(np.array([1, 1], dtype=np.uint8), (4, 1))
        _, ws = build().decode_batch(
            shots,
            edge_reweights=[np.array([[0, 1, 0.5]]), np.array([[0, 1, 2.5]])],  # blocks 0-1, 2-3
            reweight_stride=2,
            return_weights=True,
        )
        _, w0 = build(0.5).decode(np.array([1, 1]), return_weight=True)
        _, w1 = build(2.5).decode(np.array([1, 1]), return_weight=True)
        assert np.allclose(ws, [w0, w0, w1, w1])

    def test_regeneration_in_non_first_block(self):
        """A regeneration-triggering reweight in a block that is NOT block 0 decodes
        correctly and does not corrupt the (un-reweighted) earlier block."""
        def build(w01=1.0):
            m = Matching()
            m.add_edge(0, 1, weight=w01, fault_ids=0)
            m.add_boundary_edge(0, weight=1.0, fault_ids=1)
            m.add_boundary_edge(1, weight=1.0, fault_ids=2)
            return m

        shots = np.tile(np.array([1, 1], dtype=np.uint8), (2, 1))
        _, ws = build().decode_batch(
            shots,
            edge_reweights=[None, np.array([[0, 1, 5.0]])],  # block 1 (>max) regenerates
            reweight_stride=1,
            return_weights=True,
        )
        _, w0 = build().decode(np.array([1, 1]), return_weight=True)       # original graph
        _, w1 = build(5.0).decode(np.array([1, 1]), return_weight=True)    # regenerated oracle
        assert np.isclose(ws[0], w0)
        assert np.isclose(ws[1], w1)


class TestReweightInputValidation:
    """Malformed reweight specs must raise cleanly, leaving the matcher untouched.

    Before these guards, non-finite or out-of-range doubles hit undefined
    double->size_t / double->weight_int casts: on arm64 a NaN node index silently
    reweighted node 0, 1e30 saturated to SIZE_MAX (the boundary sentinel) and
    silently reweighted the boundary edge, a fractional index truncated to the
    wrong node, and a NaN weight made the edge decode as free (weight 0.0).
    """

    SYNDROME = np.array([1, 1, 0])

    @staticmethod
    def build():
        m = Matching()
        m.add_edge(0, 1, weight=1.3, fault_ids=0)
        m.add_edge(1, 2, weight=1.1, fault_ids=1)
        m.add_boundary_edge(0, weight=2.7, fault_ids=2)
        m.add_boundary_edge(2, weight=2.7, fault_ids=3)
        return m

    def assert_unchanged(self, m):
        pristine_pred, pristine_w = self.build().decode(self.SYNDROME, return_weight=True)
        pred, w = m.decode(self.SYNDROME, return_weight=True)
        assert np.array_equal(pred, pristine_pred)
        assert np.isclose(w, pristine_w)

    @pytest.mark.parametrize("bad_weight", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_weight_raises(self, bad_weight):
        m = self.build()
        with pytest.raises(ValueError, match="finite|non-negative"):
            m.decode(self.SYNDROME, edge_reweights=np.array([[0, 1, bad_weight]]))
        self.assert_unchanged(m)

    def test_over_max_weight_raises_and_matcher_survives(self):
        # 2**25 > MAX_USER_EDGE_WEIGHT (16777215). Previously this took Tier-2,
        # mutated the UserGraph, and threw during regeneration BEFORE the
        # restore-guaranteeing try block -- permanently bricking the matcher.
        m = self.build()
        with pytest.raises(ValueError, match="maximum edge weight"):
            m.decode(self.SYNDROME, edge_reweights=np.array([[0, 1, 2.0**25]]))
        self.assert_unchanged(m)

    @pytest.mark.parametrize("bad_node", [float("nan"), float("inf"), 1e30, 2.7])
    @pytest.mark.parametrize("position", [0, 1])
    def test_bad_node_index_raises(self, bad_node, position):
        m = self.build()
        spec = [0.0, 1.0, 0.5]
        spec[position] = bad_node
        with pytest.raises(ValueError, match="non-negative integer|exactly -1"):
            m.decode(self.SYNDROME, edge_reweights=np.array([spec]))
        self.assert_unchanged(m)

    def test_node_index_out_of_range_raises(self):
        # In-range-castable but nonexistent nodes keep the pre-existing error.
        m = self.build()
        with pytest.raises(ValueError, match="does not exist"):
            m.decode(self.SYNDROME, edge_reweights=np.array([[0, 999, 0.5]]))
        self.assert_unchanged(m)

    def test_failed_validation_leaves_no_stale_reweights(self):
        # Regression test for a stale-state replay: a validation throw used to
        # leave partially-validated entries in the active-reweight list (no
        # restore runs, since decode's apply threw before any decode), and a
        # LATER exception-path restore_weights replayed them into the UserGraph,
        # silently overwriting a weight the user had changed in the meantime.
        m = Matching()
        m.add_edge(0, 1, weight=1.0, fault_ids=0)
        m.add_boundary_edge(0, weight=3.0, fault_ids=1)
        m.add_boundary_edge(1, weight=3.0, fault_ids=2)

        # Call 1: second spec fails validation -> stale entry for edge (0,1)
        # holding original_weight=1.0 (before the fix).
        with pytest.raises(ValueError, match="does not exist"):
            m.decode(np.array([1, 1]),
                     edge_reweights=np.array([[0, 1, 1.5], [7, 8, 1.5]]))

        # User then changes the edge's weight.
        m.add_edge(0, 1, weight=5.0, fault_ids=0, merge_strategy="replace")

        # A later reweight call that throws mid-apply must NOT replay the stale
        # entry. (An out-of-range weight throws at validation, exercising the
        # exception-path restore in decode's catch.)
        with pytest.raises(ValueError):
            m.decode(np.array([1, 1]),
                     edge_reweights=np.array([[0, 1, 2.0**25]]))

        # Edge (0,1) must still hold the user's 5.0, not the stale 1.0.
        _, w = m.decode(np.array([1, 1]), return_weight=True)
        oracle = Matching()
        oracle.add_edge(0, 1, weight=5.0, fault_ids=0)
        oracle.add_boundary_edge(0, weight=3.0, fault_ids=1)
        oracle.add_boundary_edge(1, weight=3.0, fault_ids=2)
        _, w_oracle = oracle.decode(np.array([1, 1]), return_weight=True)
        assert np.isclose(w, w_oracle)

    def test_decode_error_then_decode_sequence(self):
        # decode -> rejected reweight -> decode must behave as if the rejected
        # call never happened, in both tiers' precondition paths.
        m = self.build()
        pred0, w0 = m.decode(self.SYNDROME, return_weight=True)
        for bad in ([[0, 1, float("nan")]], [[0, 1, 2.0**25]], [[0, 999, 0.5]]):
            with pytest.raises(ValueError):
                m.decode(self.SYNDROME, edge_reweights=np.array(bad))
            pred, w = m.decode(self.SYNDROME, return_weight=True)
            assert np.array_equal(pred, pred0)
            assert np.isclose(w, w0)
        # And a good reweight still works after all the failures.
        _, w_rw = m.decode(self.SYNDROME, edge_reweights=np.array([[0, 1, 0.4]]),
                           return_weight=True)
        assert not np.isclose(w_rw, w0)
        self.assert_unchanged(m)

    def test_stride_overflow_rejected(self):
        # stride = 2**63 + 4 with 2 rules: the product wraps mod 2**64 to 8,
        # which used to pass the multiplication-based validation and silently
        # decode every shot under rule 0.
        m = self.build()
        shots = np.zeros((8, 3), dtype=np.uint8)
        rules = [np.array([[0, 1, 0.5]]), np.array([[1, 2, 0.9]])]
        with pytest.raises(ValueError, match="must be equal"):
            m.decode_batch(shots, edge_reweights=rules, reweight_stride=2**63 + 4)
        self.assert_unchanged(m)


class TestSlotAmbiguousReweights:
    """Edges without a dedicated Tier-1 slot must be honored via regeneration.

    Previously these Tier-1 reweights silently no-opped (edges incident to a
    set_boundary node map to a nullptr boundary slot the pointer lookup misses)
    or stomped a shared min-merged slot (parallel boundary candidates), while
    identical specs above the max weight (Tier-2) worked -- results depended on
    the reweight magnitude alone.
    """

    def test_check_matrix_boundary_edge_reweight_matches_oracle(self):
        # Matching(H) creates a real boundary node via set_boundary; column-0
        # faults become edge (0, 2) with node 2 a boundary node.
        H = csc_matrix(np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8))
        m = Matching(H, spacelike_weights=np.array([1.3, 1.3, 1.3]))
        syndrome = np.array([1, 0])

        # 0.1 < max 1.3: previously Tier-1 -> silently dropped (weight stayed 1.3).
        corr, w = m.decode(syndrome, edge_reweights=np.array([[0, 2, 0.1]]),
                           return_weight=True)
        oracle = Matching(H, spacelike_weights=np.array([0.1, 1.3, 1.3]))
        corr_o, w_o = oracle.decode(syndrome, return_weight=True)
        np.testing.assert_array_equal(corr, corr_o)
        assert np.isclose(w, w_o)

        # Restore: plain decode matches a pristine matcher.
        pristine = Matching(H, spacelike_weights=np.array([1.3, 1.3, 1.3]))
        _, w_plain = m.decode(syndrome, return_weight=True)
        _, w_pristine = pristine.decode(syndrome, return_weight=True)
        assert np.isclose(w_plain, w_pristine)

    def test_min_merged_boundary_slot_reweight_matches_oracle(self):
        # Node 0 has an explicit boundary edge AND an edge to a boundary node;
        # both min-merge into ONE discretized boundary slot (0.6 wins). A Tier-1
        # write used to stomp that slot with 1.5, erasing the smaller parallel
        # candidate and mis-attributing the cost to the slot's build-time
        # fault_ids; a fresh build keeps min(1.5, 0.6) = 0.6.
        def build(bweight=2.0):
            m = Matching()
            m.add_boundary_edge(0, weight=bweight, fault_ids=0)
            m.add_edge(0, 1, weight=1.3, fault_ids=1)
            m.add_edge(0, 3, weight=0.6, fault_ids=3)
            m.set_boundary_nodes({3})
            return m

        syndrome = np.array([1, 0, 0, 0])
        m = build()
        corr, w = m.decode(syndrome, edge_reweights=np.array([[0, -1, 1.5]]),
                           return_weight=True)
        corr_o, w_o = build(1.5).decode(syndrome, return_weight=True)
        np.testing.assert_array_equal(corr, corr_o)
        assert np.isclose(w, w_o)

        # Restore: the shared slot must return to the build-time min (0.6), not
        # keep the reweighted or original explicit-boundary value.
        _, w_plain = m.decode(syndrome, return_weight=True)
        _, w_pristine = build().decode(syndrome, return_weight=True)
        assert np.isclose(w_plain, w_pristine)


class TestNegativeWeightGraphReweights:
    """Reweighting is uniformly rejected on graphs with negative edge weights.

    The old guard checked negative_weight_detection_events, which is
    parity-toggled per endpoint and cancels to EMPTY on cycles of negative
    edges; a Tier-1 reweight then wrote a plain positive weight over a slot
    holding abs(w) while the baked-in compensation (virtual detection events,
    pre-flipped observables, negative_weight_sum) still assumed the negative
    weight -- silently wrong corrections and weights. The guard now checks
    negative_weight_sum, which cannot cancel.
    """

    @staticmethod
    def build():
        m = Matching()
        m.add_edge(0, 1, weight=-1, fault_ids=0)
        m.add_edge(1, 2, weight=-1, fault_ids=1)
        m.add_edge(0, 2, weight=-1, fault_ids=2)
        m.add_boundary_edge(0, weight=2, fault_ids=3)
        return m

    def test_baseline_decode_unaffected(self):
        # Sanity: negative cycles decode fine without reweights.
        m = self.build()
        pred, w = m.decode(np.array([0, 0, 0]), return_weight=True)
        np.testing.assert_array_equal(pred, [1, 1, 1, 0])
        assert np.isclose(w, -3.0)

    @pytest.mark.parametrize("spec", [[0, 1, 2.0], [0, 1, 0.5], [0, -1, 1.0]])
    def test_reweight_rejected_despite_cancelled_detection_events(self, spec):
        # [0,1,2.0] was the silently-corrupting Tier-1 case; [0,1,0.5] took
        # Tier-2 (fractional on integral graph); [0,-1,1.0] targets a positive
        # edge. All are now rejected uniformly, before any state is touched.
        m = self.build()
        with pytest.raises(ValueError, match="negative edge weights"):
            m.decode(np.array([0, 0, 0]),
                     edge_reweights=np.array([spec], dtype=np.float64))
        pred, w = m.decode(np.array([0, 0, 0]), return_weight=True)
        assert np.isclose(w, -3.0)


class TestImpliedWeightAwareTier:
    def test_reweight_between_edge_max_and_implied_max_matches_oracle(self):
        """needs_regeneration compares against the max INCLUDING implied
        correlation weights (the max the normalising constant is actually sized
        by), so a reweight in (edge_max, implied_max] stays Tier-1 instead of
        paying two full rebuilds. Either tier must match the oracle; this pins
        the decode result across that classification change."""
        import stim
        c = stim.Circuit.generated(
            "surface_code:rotated_memory_z", distance=3, rounds=2,
            after_clifford_depolarization=0.01)
        dem = c.detector_error_model(decompose_errors=True)
        m = Matching.from_detector_error_model(dem, enable_correlations=True)
        edges = [(a, b, d) for a, b, d in m.edges() if b is not None]
        max_w = max(d["weight"] for _, _, d in m.edges())
        a, b, d0 = edges[0]
        new_w = max_w * 1.02  # just above the edge-only max

        syndrome = np.zeros(dem.num_detectors, dtype=np.uint8)
        syndrome[a] = 1
        syndrome[b] = 1
        pred, w = m.decode(syndrome, edge_reweights=np.array([[a, b, new_w]]),
                           return_weight=True, enable_correlations=True)

        oracle = Matching.from_detector_error_model(dem, enable_correlations=True)
        oracle.add_edge(a, b, weight=new_w, fault_ids=d0["fault_ids"],
                        merge_strategy="replace")
        pred_o, w_o = oracle.decode(syndrome, return_weight=True,
                                    enable_correlations=True)
        np.testing.assert_array_equal(pred, pred_o)
        assert np.isclose(w, w_o)

        # Restore: plain decode matches a pristine matcher.
        pristine = Matching.from_detector_error_model(dem, enable_correlations=True)
        pred_p, w_p = pristine.decode(syndrome, return_weight=True,
                                      enable_correlations=True)
        pred_after, w_after = m.decode(syndrome, return_weight=True,
                                       enable_correlations=True)
        np.testing.assert_array_equal(pred_after, pred_p)
        assert np.isclose(w_after, w_p)


if __name__ == "__main__":
    pytest.main([__file__])
