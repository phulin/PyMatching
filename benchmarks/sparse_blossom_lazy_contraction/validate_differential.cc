// Copyright 2026 PyMatching Contributors
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

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <random>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "pymatching/sparse_blossom/matcher/mwpm.h"

static pm::GraphFillRegion *node_top(pm::DetectorNode &node) {
#ifdef PM_LAZY_TOP
    return node.top_region();
#else
    return node.region_that_arrived_top;
#endif
}

struct Outcome {
    bool success;
    pm::total_weight_int weight;
    pm::obs_int mask;
    std::string error;
};

static Outcome decode(pm::MatchingGraph graph, const std::vector<size_t> &syndrome) {
    try {
        pm::Mwpm mwpm(pm::GraphFlooder(std::move(graph)));
        mwpm.flooder.queue.cur_time = 0;
        for (size_t d : syndrome) {
            mwpm.create_detection_event(&mwpm.flooder.graph.nodes[d]);
        }
        while (true) {
            auto event = mwpm.flooder.run_until_next_mwpm_notification();
            if (event.event_type == pm::NO_EVENT) break;
            mwpm.process_event(event);
        }
        if (mwpm.node_arena.allocated.size() != mwpm.node_arena.available.size()) {
            return {false, 0, 0, "no-perfect-matching"};
        }
        pm::MatchingResult result;
        for (size_t d : syndrome) {
            auto &node = mwpm.flooder.graph.nodes[d];
            if (node.region_that_arrived != nullptr) {
                result += mwpm.shatter_blossom_and_extract_matches(node_top(node));
            }
        }
        return {true, result.weight, result.obs_mask, {}};
    } catch (const std::exception &ex) {
        return {false, 0, 0, std::string("exception:") + ex.what()};
    } catch (...) {
        return {false, 0, 0, "unknown-exception"};
    }
}

static std::vector<size_t> random_observables(std::mt19937_64 &rng, size_t num_obs) {
    std::vector<size_t> result;
    for (size_t k = 0; k < num_obs; k++) {
        if (rng() % 9 == 0) result.push_back(k);
    }
    return result;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "usage: differential SEED CASES positive|zero\n";
        return 2;
    }
    uint64_t seed = std::strtoull(argv[1], nullptr, 10);
    size_t cases = std::strtoull(argv[2], nullptr, 10);
    bool zero_mode = std::string(argv[3]) == "zero";
    if (!zero_mode && std::string(argv[3]) != "positive") return 2;

    std::mt19937_64 rng(seed);
    for (size_t case_id = 0; case_id < cases; case_id++) {
        size_t n = 4 + rng() % 57;
        constexpr size_t num_obs = 2;
        pm::MatchingGraph graph(n, num_obs);
        std::set<std::pair<size_t, size_t>> used;
        auto add_edge = [&](size_t u, size_t v, pm::weight_int w) {
            if (u == v) return;
            if (u > v) std::swap(u, v);
            if (!used.insert({u, v}).second) return;
            graph.add_edge(u, v, w, random_observables(rng, num_obs), {});
        };
        auto random_weight = [&]() -> pm::weight_int {
            if (zero_mode && rng() % 5 == 0) return 0;
            if (case_id % 7 == 0) return 2;  // many ties / dense zero-slack phases
            return 2 * (1 + (rng() % 9));
        };

        // Connected backbone, plus enough chords to create odd cycles and blossoms.
        for (size_t u = 1; u < n; u++) add_edge(u - 1, u, random_weight());
        size_t extra = n + rng() % (3 * n + 1);
        for (size_t k = 0; k < extra; k++) add_edge(rng() % n, rng() % n, random_weight());

        // Some components have no boundary, deliberately exercising failure cases.
        bool allow_boundary = rng() % 5 != 0;
        if (allow_boundary) {
            size_t boundary_count = 1 + rng() % std::max<size_t>(1, n / 5);
            std::set<size_t> boundary_nodes;
            while (boundary_nodes.size() < boundary_count) boundary_nodes.insert(rng() % n);
            for (size_t u : boundary_nodes) {
                pm::weight_int w = zero_mode && rng() % 5 == 0 ? 0 : 2 * (1 + rng() % 7);
                graph.add_boundary_edge(u, w, random_observables(rng, num_obs), {});
            }
        }

        double p = 0.08 + (rng() % 63) / 100.0;
        std::bernoulli_distribution hit(p);
        std::vector<size_t> syndrome;
        for (size_t u = 0; u < n; u++) if (hit(rng)) syndrome.push_back(u);
        if (syndrome.empty()) syndrome.push_back(rng() % n);

        auto out = decode(std::move(graph), syndrome);
        std::cout << case_id << ',' << (out.success ? 'S' : 'F') << ',' << out.weight << ',' << out.mask;
        if (!out.success) std::cout << ',' << out.error;
        std::cout << '\n';
    }
}
