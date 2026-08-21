#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <random>
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

static void run_scenario(const std::string &name) {
    auto scenario = make_scenario(name);
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
    } else {
        if (argc != 2) return 2;
        run_scenario(argv[1]);
    }
}
