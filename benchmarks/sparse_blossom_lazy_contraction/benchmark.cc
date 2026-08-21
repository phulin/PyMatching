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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
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

static pm::MatchingResult decode(pm::Mwpm &mwpm, const std::vector<size_t> &syndrome) {
    mwpm.flooder.queue.cur_time = 0;
    for (size_t d : syndrome) mwpm.create_detection_event(&mwpm.flooder.graph.nodes[d]);
    while (true) {
        auto event = mwpm.flooder.run_until_next_mwpm_notification();
        if (event.event_type == pm::NO_EVENT) break;
        mwpm.process_event(event);
    }
    if (mwpm.node_arena.allocated.size() != mwpm.node_arena.available.size()) {
        throw std::runtime_error("unfinished matching");
    }
    pm::MatchingResult result;
    for (size_t d : syndrome) {
        auto &node = mwpm.flooder.graph.nodes[d];
        if (node.region_that_arrived != nullptr) {
            result += mwpm.shatter_blossom_and_extract_matches(node_top(node));
        }
    }
    return result;
}

static uint64_t mix64(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}


template <typename T>
static T read_binary(std::ifstream &in) {
    T value{};
    in.read(reinterpret_cast<char *>(&value), sizeof(value));
    if (!in) throw std::runtime_error("truncated benchmark input");
    return value;
}

struct LoadedSurface {
    pm::MatchingGraph graph;
    std::vector<std::vector<size_t>> shots;
    double normalising_constant;
};

static LoadedSurface load_surface(const std::string &graph_path, const std::string &shots_path) {
    std::ifstream graph_file(graph_path, std::ios::binary);
    if (!graph_file) throw std::runtime_error("failed to open graph input");
    char magic[8];
    graph_file.read(magic, sizeof(magic));
    if (!graph_file || std::string(magic, magic + 6) != "PMDEM1") {
        throw std::runtime_error("bad graph input magic");
    }
    uint64_t num_nodes = read_binary<uint64_t>(graph_file);
    uint64_t num_observables = read_binary<uint64_t>(graph_file);
    uint64_t num_edges = read_binary<uint64_t>(graph_file);
    double normalising_constant = read_binary<double>(graph_file);
    pm::MatchingGraph graph(num_nodes, num_observables, normalising_constant);
    for (uint64_t i = 0; i < num_edges; i++) {
        uint64_t u = read_binary<uint64_t>(graph_file);
        int64_t v = read_binary<int64_t>(graph_file);
        int32_t weight = read_binary<int32_t>(graph_file);
        uint64_t obs_mask = read_binary<uint64_t>(graph_file);
        std::vector<size_t> observables;
        for (size_t b = 0; b < 64; b++) {
            if ((obs_mask >> b) & 1) observables.push_back(b);
        }
        if (v < 0) {
            graph.add_boundary_edge(u, weight, observables, {});
        } else {
            graph.add_edge(u, (uint64_t)v, weight, observables, {});
        }
    }

    std::ifstream shots_file(shots_path, std::ios::binary | std::ios::ate);
    if (!shots_file) throw std::runtime_error("failed to open shot input");
    auto bytes = shots_file.tellg();
    shots_file.seekg(0);
    size_t shot_bytes = (num_nodes + num_observables + 7) / 8;
    if (bytes < 0 || (size_t)bytes % shot_bytes != 0) throw std::runtime_error("bad b8 shot size");
    std::vector<uint8_t> packed((size_t)bytes);
    shots_file.read(reinterpret_cast<char *>(packed.data()), packed.size());
    if (!shots_file) throw std::runtime_error("truncated b8 shot input");
    size_t num_shots = packed.size() / shot_bytes;
    std::vector<std::vector<size_t>> shots;
    shots.reserve(num_shots);
    for (size_t s = 0; s < num_shots; s++) {
        std::vector<size_t> syndrome;
        const uint8_t *row = packed.data() + s * shot_bytes;
        for (size_t d = 0; d < num_nodes; d++) {
            if ((row[d >> 3] >> (d & 7)) & 1) syndrome.push_back(d);
        }
        shots.push_back(std::move(syndrome));
    }
    return {std::move(graph), std::move(shots), normalising_constant};
}

static void run_surface(
    const std::string &graph_path, const std::string &shots_path, size_t loops, size_t dump_count) {
    auto surface = load_surface(graph_path, shots_path);
    size_t nodes = surface.graph.nodes.size();
    double mean_detection = 0;
    for (const auto &s : surface.shots) mean_detection += s.size();
    mean_detection /= surface.shots.size();
    pm::Mwpm mwpm(pm::GraphFlooder(std::move(surface.graph)));
    for (size_t i = 0; i < std::min<size_t>(surface.shots.size(), 20); i++) decode(mwpm, surface.shots[i]);
    if (dump_count != 0) {
        for (size_t i = 0; i < std::min(dump_count, surface.shots.size()); i++) {
            auto r = decode(mwpm, surface.shots[i]);
            std::cout << i << ',' << r.weight << ',' << r.weight / surface.normalising_constant << ','
                      << r.obs_mask << '\n';
        }
        return;
    }
    uint64_t weight_checksum = 0;
    uint64_t mask_checksum = 0;
    auto start = std::chrono::steady_clock::now();
    for (size_t loop = 0; loop < loops; loop++) {
        for (size_t i = 0; i < surface.shots.size(); i++) {
            auto r = decode(mwpm, surface.shots[i]);
            weight_checksum += mix64((uint64_t)r.weight + i * 17 + loop * 131);
            mask_checksum += mix64(r.obs_mask + i * 19 + loop * 137);
        }
    }
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    size_t decodes = loops * surface.shots.size();
    std::cout << "surface_d13_p001," << nodes << ',' << surface.shots.size() << ',' << mean_detection << ','
              << loops << ',' << us / decodes << ',' << weight_checksum << ',' << mask_checksum << '\n';
}

struct Scenario {
    std::string name;
    pm::MatchingGraph graph;
    std::vector<std::vector<size_t>> shots;
    size_t loops;
};

static std::vector<std::vector<size_t>> make_shots(size_t n, size_t count, double p, uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::bernoulli_distribution hit(p);
    std::vector<std::vector<size_t>> shots;
    shots.reserve(count);
    for (size_t s = 0; s < count; s++) {
        std::vector<size_t> syndrome;
        for (size_t u = 0; u < n; u++) if (hit(rng)) syndrome.push_back(u);
        if (syndrome.empty()) syndrome.push_back(rng() % n);
        shots.push_back(std::move(syndrome));
    }
    return shots;
}

static pm::MatchingGraph make_grid(size_t side, int kind) {
    const size_t n = side * side;
    pm::MatchingGraph graph(n, 1);
    auto node = [side](size_t r, size_t c) { return r * side + c; };
    auto weight = [&](size_t u, size_t v) {
        uint64_t h = mix64(u * 0x9e3779b97f4a7c15ULL + v);
        if (kind == 1) return 2;  // equal weight
        if (kind == 2 && h % 7 == 0) return 0;  // erasure islands
        return 2 * (1 + (int)(h % 7));
    };
    for (size_t r = 0; r < side; r++) {
        for (size_t c = 0; c < side; c++) {
            size_t u = node(r, c);
            if (r + 1 < side) {
                size_t v = node(r + 1, c);
                std::vector<size_t> obs;
                if ((u + v) % 29 == 0) obs.push_back(0);
                graph.add_edge(u, v, weight(u, v), obs, {});
            }
            if (c + 1 < side) {
                size_t v = node(r, c + 1);
                std::vector<size_t> obs;
                if ((u + v) % 31 == 0) obs.push_back(0);
                graph.add_edge(u, v, weight(u, v), obs, {});
            }
        }
    }
    for (size_t r = 0; r < side; r++) {
        graph.add_boundary_edge(node(r, 0), kind == 2 && r % 9 == 0 ? 0 : 4, {}, {});
        graph.add_boundary_edge(node(r, side - 1), kind == 2 && r % 11 == 0 ? 0 : 4, {0}, {});
    }
    return graph;
}

static pm::MatchingGraph make_random_blossom(size_t n, bool zero) {
    pm::MatchingGraph graph(n, 1);
    auto w = [&](size_t i, size_t salt) {
        if (zero && mix64(i + salt) % 6 == 0) return 0;
        return 2;
    };
    for (size_t i = 0; i < n; i++) {
        graph.add_edge(i, (i + 1) % n, w(i, 1), {}, {});
        graph.add_edge(i, (i + 2) % n, w(i, 2), (i % 37 == 0) ? std::vector<size_t>{0} : std::vector<size_t>{}, {});
        graph.add_edge(i, (i + 17) % n, zero ? w(i, 17) : 4, {}, {});
        if (i % 29 == 0) graph.add_boundary_edge(i, zero && i % 58 == 0 ? 0 : 6, {}, {});
    }
    return graph;
}

static Scenario make_scenario(const std::string &name) {
    if (name == "grid_lowp_mixed") {
        return {name, make_grid(28, 0), make_shots(28 * 28, 400, 0.02, 1), 30};
    }
    if (name == "grid_highp_mixed") {
        return {name, make_grid(28, 0), make_shots(28 * 28, 250, 0.25, 2), 10};
    }
    if (name == "grid_highp_equal") {
        return {name, make_grid(28, 1), make_shots(28 * 28, 250, 0.25, 3), 10};
    }
    if (name == "grid_erasure_zero") {
        return {name, make_grid(28, 2), make_shots(28 * 28, 180, 0.20, 4), 8};
    }
    if (name == "random_blossom") {
        return {name, make_random_blossom(600, false), make_shots(600, 180, 0.35, 5), 1};
    }
    if (name == "random_zero") {
        return {name, make_random_blossom(600, true), make_shots(600, 150, 0.30, 6), 2};
    }
    throw std::invalid_argument("unknown scenario");
}

static void run_scenario(const std::string &name, size_t loop_multiplier) {
    auto scenario = make_scenario(name);
    if (loop_multiplier == 0 || scenario.loops > SIZE_MAX / loop_multiplier) {
        throw std::invalid_argument("invalid loop multiplier");
    }
    scenario.loops *= loop_multiplier;
    double mean_detection = 0;
    for (const auto &s : scenario.shots) mean_detection += s.size();
    mean_detection /= scenario.shots.size();
    pm::Mwpm mwpm(pm::GraphFlooder(std::move(scenario.graph)));

    // Warm caches and allocator arenas.
    for (size_t i = 0; i < std::min<size_t>(scenario.shots.size(), 20); i++) decode(mwpm, scenario.shots[i]);

    uint64_t weight_checksum = 0;
    uint64_t mask_checksum = 0;
    auto start = std::chrono::steady_clock::now();
    for (size_t loop = 0; loop < scenario.loops; loop++) {
        for (size_t i = 0; i < scenario.shots.size(); i++) {
            auto r = decode(mwpm, scenario.shots[i]);
            weight_checksum += mix64((uint64_t)r.weight + i * 17 + loop * 131);
            mask_checksum += mix64(r.obs_mask + i * 19 + loop * 137);
        }
    }
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count();
    size_t decodes = scenario.loops * scenario.shots.size();
    std::cout << name << ',' << mwpm.flooder.graph.nodes.size() << ',' << scenario.shots.size() << ','
              << mean_detection << ',' << scenario.loops << ',' << us / decodes << ','
              << weight_checksum << ',' << mask_checksum << '\n';
}

static pm::MatchingGraph make_chain(size_t k) {
    pm::MatchingGraph graph(2 * k + 2, 0);
    auto x = [](size_t i) { return 2 * i - 1; };
    auto y = [](size_t i) { return 2 * i; };
    for (size_t i = 1; i <= k; i++) {
        size_t a = i == 1 ? 0 : x(i - 1);
        graph.add_edge(a, x(i), 2, {}, {});
        graph.add_edge(a, y(i), 2, {}, {});
        graph.add_edge(x(i), y(i), 2, {}, {});
    }
    graph.add_edge(x(k), 2 * k + 1, 4, {}, {});
    return graph;
}

static void run_chain(size_t k, size_t repeats) {
    pm::Mwpm mwpm(pm::GraphFlooder(make_chain(k)));
    std::vector<size_t> syndrome(2 * k + 2);
    for (size_t i = 0; i < syndrome.size(); i++) syndrome[i] = i;
    decode(mwpm, syndrome);
    auto start = std::chrono::steady_clock::now();
    uint64_t checksum = 0;
    for (size_t r = 0; r < repeats; r++) checksum += decode(mwpm, syndrome).weight;
    auto end = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(end - start).count() / repeats;
    std::cout << "chain," << k << ',' << repeats << ',' << us << ',' << checksum << '\n';
}

int main(int argc, char **argv) {
    if (argc >= 2 && std::string(argv[1]) == "chain") {
        if (argc != 4) return 2;
        run_chain(std::strtoull(argv[2], nullptr, 10), std::strtoull(argv[3], nullptr, 10));
    } else if (argc >= 2 && std::string(argv[1]) == "surface") {
        if (argc != 5) return 2;
        run_surface(argv[2], argv[3], std::strtoull(argv[4], nullptr, 10), 0);
    } else if (argc >= 2 && std::string(argv[1]) == "surface_dump") {
        if (argc != 5) return 2;
        run_surface(argv[2], argv[3], 0, std::strtoull(argv[4], nullptr, 10));
    } else {
        if (argc != 2 && argc != 3) return 2;
        size_t loop_multiplier = argc == 3 ? std::strtoull(argv[2], nullptr, 10) : 1;
        run_scenario(argv[1], loop_multiplier);
    }
}
