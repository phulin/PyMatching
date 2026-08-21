// Copyright 2022 PyMatching Contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "pymatching/sparse_blossom/flooder/detector_node.h"

#include <limits>
#include <optional>
#include <stdexcept>

#include "pymatching/sparse_blossom/flooder/graph_fill_region.h"

namespace pm {

GraphFillRegion *DetectorNode::resolve_top_region() const {
    GraphFillRegion *result = region_that_arrived_top;
    if (result == nullptr) {
        return nullptr;
    }

    cumulative_time_int new_wrapped_radius = wrapped_radius_cached;
    while (result->blossom_parent != nullptr) {
        // A region can only become a non-top child at the instant it is frozen
        // into a new blossom. Crossing a varying region would mean the geometry
        // and the dual state have diverged.
        if (!result->radius.is_frozen()) {
            throw std::logic_error("A lazily nested blossom region was not frozen.");
        }
        new_wrapped_radius += result->radius.y_intercept();
        result = result->blossom_parent;
    }

    if (new_wrapped_radius < std::numeric_limits<int32_t>::min() ||
        new_wrapped_radius > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("Wrapped blossom radius exceeded int32_t range.");
    }
    wrapped_radius_cached = (int32_t)new_wrapped_radius;
    region_that_arrived_top = result;
    return result;
}

int32_t DetectorNode::compute_wrapped_radius() const {
    if (reached_from_source == nullptr) {
        return 0;
    }
    GraphFillRegion *top = top_region();
    cumulative_time_int total = 0;
    auto r = region_that_arrived;
    while (r != top) {
        if (r == nullptr) {
            throw std::logic_error("Detector ownership chain did not reach its top region.");
        }
        total += r->radius.y_intercept();
        r = r->blossom_parent;
    }
    total -= radius_of_arrival;
    if (total < std::numeric_limits<int32_t>::min() || total > std::numeric_limits<int32_t>::max()) {
        throw std::overflow_error("Wrapped blossom radius exceeded int32_t range.");
    }
    return (int32_t)total;
}

void DetectorNode::reset() {
    observables_crossed_from_source = 0;
    reached_from_source = nullptr;
    radius_of_arrival = 0;
    region_that_arrived = nullptr;
    region_that_arrived_top = nullptr;
    wrapped_radius_cached = 0;
    node_event_tracker.clear();
}

size_t DetectorNode::index_of_neighbor(DetectorNode *target) const {
    for (size_t k = 0; k < neighbors.size(); k++) {
        if (neighbors[k] == target) {
            return k;
        }
    }
    throw std::invalid_argument("Failed to find neighbor.");
}

GraphFillRegion *DetectorNode::heir_region_on_shatter() const {
    GraphFillRegion *top = top_region();
    GraphFillRegion *r = region_that_arrived;
    while (r != nullptr) {
        GraphFillRegion *p = r->blossom_parent;
        if (p == top) {
            return r;
        }
        r = p;
    }
    throw std::logic_error("Detector node had no heir region inside the shattering blossom.");
}

cumulative_time_int DetectorNode::compute_local_radius_at_time_bounded_by_region(
    cumulative_time_int time, const GraphFillRegion &bounding_region) const {
    if (region_that_arrived == nullptr) {
        // Nodes not in an region have zero local radius.
        return 0;
    }
    if (*region_that_arrived > bounding_region) {
        // If the target region is a descendant of the region that reached this node, then
        // bounding to that region effectively means this node has not yet been reached. Act
        // like a node not in a region.
        return 0;
    }

    const GraphFillRegion *container = region_that_arrived;
    cumulative_time_int container_radius = 0;
    while (true) {
        if (container == nullptr) {
            // Wasn't inside the bounding region.
            break;
        }
        if (*container > bounding_region) {
            // Was a cousin of some sort of the bounding region, and have just reached the
            // common ancestor.
            break;
        }
        container_radius += container->radius.get_distance_at_time(time);
        if (*container == bounding_region) {
            // Don't go beyond the limits of the bounding region.
            break;
        }
        container = container->blossom_parent;
    }
    return container_radius - radius_of_arrival;
}

std::optional<float> DetectorNode::compute_stitch_radius_at_time_bounded_by_region_towards_neighbor(
    cumulative_time_int time, const GraphFillRegion &bounding_region, size_t neighbor_index) const {
    DetectorNode *neighbor = neighbors[neighbor_index];
    cumulative_time_int max_w = neighbor_weights[neighbor_index];
    auto r1 = compute_local_radius_at_time_bounded_by_region(time, bounding_region);
    if (neighbor == nullptr) {
        return (weight_int)std::min(max_w, r1);
    }

    auto r2 = neighbor->compute_local_radius_at_time_bounded_by_region(time, bounding_region);

    // If the nodes at either side of the edge have regions that aren't linked according to the
    // state the mwpm, then the transition must be happening exactly at the local radius.
    if (r1 + r2 < max_w || neighbor->top_region() != top_region()) {
        return (weight_int)r1;
    }
    if (r1 == max_w && *neighbor->region_that_arrived > *region_that_arrived) {
        return max_w;
    }

    // Now we are in the complicated case, where for example we are trying to determine exactly
    // where two child blossom collided along an edge far in the past.

    // There is an additional complication in that *actually* we are focusing on a specific source
    // node, and we want to see the boundaries between the parts of the graph reached by one source
    // node and another source node, even if those nodes are part of the same region.

    // If the edge is between the same two sources, there is no stitch.
    if (reached_from_source == neighbor->reached_from_source) {
        return {};
    }

    // Back up to the time when they collided. Because they must grow in tandem past that point,
    // they each grew half of the extra width.
    float growth_past_collision = r1 + r2 - max_w;
    float growth_past_collision_of_r1 = growth_past_collision / 2;
    return r1 - growth_past_collision_of_r1;
}

}  // namespace pm
