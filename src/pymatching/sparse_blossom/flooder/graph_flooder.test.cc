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

#include "pymatching/sparse_blossom/flooder/graph_flooder.h"

#include "gtest/gtest.h"

#include "pymatching/sparse_blossom/flooder/graph_fill_region.h"
#include "pymatching/sparse_blossom/flooder_matcher_interop/mwpm_event.h"
#include "pymatching/sparse_blossom/ints.h"

using namespace pm;

TEST(GraphFlooder, PriorityQueue) {
    GraphFlooder flooder(MatchingGraph(10, 64));
    auto &graph = flooder.graph;
    graph.add_edge(0, 1, 10, {}, {});
    graph.add_edge(1, 2, 10, {}, {});
    graph.add_edge(2, 3, 10, {}, {});
    graph.add_edge(3, 4, 10, {}, {});
    graph.add_edge(4, 5, 10, {}, {});
    graph.add_edge(5, 0, 10, {}, {});

    auto qn = [&](int i, int t) {
        graph.nodes[i].node_event_tracker.set_desired_event({&graph.nodes[i], cyclic_time_int{t}}, flooder.queue);
    };

    qn(0, 10);
    qn(1, 8);
    qn(2, 5);

    GraphFillRegion gfr;
    gfr.shrink_event_tracker.set_desired_event({&gfr, cyclic_time_int{70}}, flooder.queue);

    qn(4, 100);
    auto e = flooder.dequeue_valid();
    ASSERT_EQ(e.time, 5);
    ASSERT_EQ(e.tentative_event_type, LOOK_AT_NODE);
    ASSERT_EQ(e.data_look_at_node, &graph.nodes[2]);
    ASSERT_EQ(flooder.dequeue_valid().time, 8);
    ASSERT_EQ(flooder.dequeue_valid().time, 10);
    auto e6 = flooder.dequeue_valid();
    ASSERT_EQ(e6.time, 70);
    ASSERT_EQ(e6.tentative_event_type, LOOK_AT_SHRINKING_REGION);
    ASSERT_EQ(e6.data_look_at_shrinking_region, &gfr);
    ASSERT_EQ(flooder.dequeue_valid().time, 100);
}

TEST(GraphFlooder, blossom_frontier_inheritance_only_rebuilds_changed_slopes) {
    GraphFlooder flooder(MatchingGraph(3, 0));
    flooder.queue.cur_time = 10;

    GraphFillRegion growing_left;
    GraphFillRegion frozen_middle;
    GraphFillRegion growing_right;
    auto &left_node = flooder.graph.nodes[0];
    auto &middle_node = flooder.graph.nodes[1];
    auto &right_node = flooder.graph.nodes[2];

    auto initialize_node = [&](DetectorNode &node, GraphFillRegion &region) {
        node.reached_from_source = &node;
        node.region_that_arrived = &region;
        node.region_that_arrived_top = &region;
        node.radius_of_arrival = 0;
        node.wrapped_radius_cached = 0;
        region.shell_area.push_back(&node);
        node.node_event_tracker.set_desired_event({&node, cyclic_time_int{20}}, flooder.queue);
    };
    initialize_node(left_node, growing_left);
    initialize_node(middle_node, frozen_middle);
    initialize_node(right_node, growing_right);

    growing_left.radius = VaryingCT::growing_value_at_time(3, 10);
    frozen_middle.radius = VaryingCT::frozen(7);
    growing_right.radius = VaryingCT::growing_value_at_time(5, 10);

    std::vector<RegionEdge> children{
        {&growing_left, {}},
        {&frozen_middle, {}},
        {&growing_right, {}},
    };
    auto *blossom = flooder.create_blossom(children);

    // Growing children retain their already-valid frontier events. The frozen
    // child changed slope and was therefore rescheduled (to no event here,
    // because this test graph has no edges).
    ASSERT_TRUE(left_node.node_event_tracker.has_desired_time);
    ASSERT_EQ(left_node.node_event_tracker.desired_time, cyclic_time_int{20});
    ASSERT_FALSE(middle_node.node_event_tracker.has_desired_time);
    ASSERT_TRUE(right_node.node_event_tracker.has_desired_time);
    ASSERT_EQ(right_node.node_event_tracker.desired_time, cyclic_time_int{20});

    // Wrapping a growing child into a growing blossom preserves its trajectory.
    ASSERT_EQ(left_node.local_radius().get_distance_at_time(25), 18);
    ASSERT_EQ(right_node.local_radius().get_distance_at_time(25), 20);
    // The frozen child acquires the new blossom's growth.
    ASSERT_EQ(middle_node.local_radius().get_distance_at_time(25), 22);
    ASSERT_EQ(left_node.top_region(), blossom);
    ASSERT_EQ(middle_node.top_region(), blossom);
    ASSERT_EQ(right_node.top_region(), blossom);
}
