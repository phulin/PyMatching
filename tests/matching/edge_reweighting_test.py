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

    def test_reweight_after_set_boundary_uses_fresh_cache(self):
        """Regression for a stale edge-index cache (a segfault).

        set_boundary -- and any graph regeneration -- rebuilds the per-node
        neighbor arrays that Tier-1 reweighting indexes into. The edge-index cache
        must be invalidated; otherwise a later reweighted decode reuses indices
        computed against the old graph and writes to a stale / out-of-bounds slot.
        Uses non-integer base weights so the reweight stays on the Tier-1 (cached)
        path rather than regenerating.
        """
        def build():
            m = Matching()
            m.add_edge(0, 1, weight=1.3, fault_ids=0)
            m.add_edge(1, 2, weight=1.3, fault_ids=1)
            m.add_edge(2, 3, weight=1.3, fault_ids=2)
            m.add_boundary_edge(0, weight=1.3, fault_ids=3)
            return m

        syndrome = np.array([0, 0, 1, 1])
        rw = np.array([[1, 2, 0.5]], dtype=np.float64)

        m = build()
        m.decode(syndrome, edge_reweights=np.array([[2, 3, 0.5]], dtype=np.float64))  # build cache
        m.set_boundary_nodes({1})                                                     # rebuild graph
        corr, weight = m.decode(syndrome, edge_reweights=rw, return_weight=True)       # must not crash

        # Oracle: a fresh matcher whose cache is built after the same topology change.
        oracle = build()
        oracle.set_boundary_nodes({1})
        corr_o, weight_o = oracle.decode(syndrome, edge_reweights=rw, return_weight=True)

        np.testing.assert_array_equal(corr, corr_o)
        assert np.isclose(weight, weight_o)

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


if __name__ == "__main__":
    pytest.main([__file__])
