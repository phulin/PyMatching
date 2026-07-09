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

from pymatching import Matching


def _boundaryless_triangle():
    """A boundaryless triangle: a syndrome with a single defect (odd parity, no
    boundary to absorb it) has no perfect matching."""
    m = Matching()
    m.add_edge(0, 1, weight=1.0, fault_ids=0)
    m.add_edge(1, 2, weight=1.0, fault_ids=1)
    m.add_edge(0, 2, weight=1.0, fault_ids=2)
    return m


class TestNoMatching:
    def test_default_raises_on_no_matching(self):
        m = _boundaryless_triangle()
        with pytest.raises(ValueError, match="No perfect matching"):
            m.decode_batch(np.array([[1, 0, 0]], dtype=np.uint8))

    def test_return_no_matching_flags_and_does_not_raise(self):
        m = _boundaryless_triangle()
        shots = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)  # shot 0 matchable, shot 1 not
        preds, weights, no_matching = m.decode_batch(
            shots, return_weights=True, return_no_matching=True
        )
        assert no_matching.dtype == bool
        np.testing.assert_array_equal(no_matching, [False, True])
        # Matchable shot decodes exactly as a standalone decode; finite weight.
        np.testing.assert_array_equal(preds[0], _boundaryless_triangle().decode(shots[0]))
        assert np.isfinite(weights[0])
        # Unmatchable shot: all observables flipped, weight +inf (not 0.0).
        np.testing.assert_array_equal(preds[1], [1, 1, 1])
        assert np.isinf(weights[1])

    def test_all_matchable_flags_all_false(self):
        m = _boundaryless_triangle()
        shots = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.uint8)
        _, no_matching = m.decode_batch(shots, return_no_matching=True)
        np.testing.assert_array_equal(no_matching, [False, False])

    def test_return_shape_matrix(self):
        m = _boundaryless_triangle()
        shots = np.array([[1, 1, 0]], dtype=np.uint8)  # matchable
        assert isinstance(m.decode_batch(shots), np.ndarray)
        r = m.decode_batch(shots, return_weights=True)
        assert isinstance(r, tuple) and len(r) == 2
        r = m.decode_batch(shots, return_no_matching=True)
        assert isinstance(r, tuple) and len(r) == 2 and r[1].dtype == bool
        r = m.decode_batch(shots, return_weights=True, return_no_matching=True)
        assert isinstance(r, tuple) and len(r) == 3 and r[2].dtype == bool

    def test_bit_packed_predictions_are_binary_bits(self):
        m = _boundaryless_triangle()  # 3 observables
        preds, nm = m.decode_batch(
            np.array([[1, 0, 0]], dtype=np.uint8),
            bit_packed_predictions=True,
            return_no_matching=True,
        )
        assert nm[0]
        # low 3 bits set (0b111 == 7); no garbage high bits.
        assert preds[0, 0] == 0b111
        assert set(np.unique(preds)).issubset({0, 0b111})

    def test_unrelated_error_still_raises_with_flag(self):
        # A different error (here: a shots array wider than the graph) must still
        # propagate even with return_no_matching=True -- the flag only catches
        # the specific no-perfect-matching condition.
        m = _boundaryless_triangle()  # 3 nodes
        with pytest.raises(ValueError):
            m.decode_batch(np.array([[0, 0, 0, 0]], dtype=np.uint8), return_no_matching=True)

    def test_no_matching_independent_of_reweights(self):
        m = _boundaryless_triangle()
        shots = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.uint8)
        # A no-op (Tier-1) reweight per shot; the unmatchable shot must still be flagged.
        reweights = [np.array([[0, 1, 1.0]], dtype=np.float64)] * 2
        _, no_matching = m.decode_batch(
            shots, edge_reweights=reweights, reweight_stride=1, return_no_matching=True
        )
        np.testing.assert_array_equal(no_matching, [False, True])


if __name__ == "__main__":
    pytest.main([__file__])
