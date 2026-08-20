#ifndef PYMATCHING_FAST_DECODE_PYBIND_H
#define PYMATCHING_FAST_DECODE_PYBIND_H

#include <algorithm>
#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"
#include "pymatching/sparse_blossom/driver/mwpm_decoding.h"
#include "pymatching/sparse_blossom/driver/user_graph.h"

namespace pm_pybind_fast {
namespace py = pybind11;

inline void process_timeline(pm::Mwpm &mwpm, const std::vector<uint64_t> &detection_events) {
    if (!mwpm.flooder.queue.empty()) {
        throw std::invalid_argument("!mwpm.flooder.queue.empty()");
    }
    mwpm.flooder.queue.cur_time = 0;
    if (mwpm.flooder.negative_weight_detection_events.empty()) {
        for (auto detection : detection_events) {
            if (detection >= mwpm.flooder.graph.nodes.size()) {
                throw std::invalid_argument("detection event index out of range");
            }
            if (detection + 1 > mwpm.flooder.graph.is_user_graph_boundary_node.size() ||
                !mwpm.flooder.graph.is_user_graph_boundary_node[detection]) {
                mwpm.create_detection_event(&mwpm.flooder.graph.nodes[detection]);
            }
        }
    } else {
        for (auto det : mwpm.flooder.negative_weight_detection_events) {
            mwpm.flooder.graph.nodes[det].radius_of_arrival = 1;
        }
        for (auto detection : detection_events) {
            if (detection >= mwpm.flooder.graph.nodes.size()) {
                throw std::invalid_argument("detection event index out of range");
            }
            if (!mwpm.flooder.graph.nodes[detection].radius_of_arrival) {
                if (detection + 1 > mwpm.flooder.graph.is_user_graph_boundary_node.size() ||
                    !mwpm.flooder.graph.is_user_graph_boundary_node[detection]) {
                    mwpm.create_detection_event(&mwpm.flooder.graph.nodes[detection]);
                }
            } else {
                mwpm.flooder.graph.nodes[detection].radius_of_arrival = 0;
            }
        }
        for (auto det : mwpm.flooder.negative_weight_detection_events) {
            if (mwpm.flooder.graph.nodes[det].radius_of_arrival) {
                mwpm.flooder.graph.nodes[det].radius_of_arrival = 0;
                mwpm.create_detection_event(&mwpm.flooder.graph.nodes[det]);
            }
        }
    }
    while (true) {
        auto event = mwpm.flooder.run_until_next_mwpm_notification();
        if (event.event_type == pm::NO_EVENT) {
            break;
        }
        mwpm.process_event(event);
    }
    if (mwpm.node_arena.allocated.size() != mwpm.node_arena.available.size()) {
        mwpm.reset();
        throw pm::NoPerfectMatchingError(
            "No perfect matching could be found. This likely means that the syndrome has odd parity in a "
            "connected component without a boundary.");
    }
}

inline void shatter_to_match_edges(
    pm::Mwpm &mwpm,
    const std::vector<uint64_t> &detection_events,
    std::vector<pm::CompressedEdge> &out) {
    for (auto i : detection_events) {
        if (mwpm.flooder.graph.nodes[i].region_that_arrived) {
            mwpm.shatter_blossom_and_extract_match_edges(
                mwpm.flooder.graph.nodes[i].region_that_arrived_top, out);
        }
    }
}

inline void flip_edge(const pm::SearchGraphEdge &edge) {
    edge.detector_node->neighbor_markers[edge.neighbor_index] ^= pm::FLIPPED;
    auto neighbor = edge.detector_node->neighbors[edge.neighbor_index];
    if (neighbor) {
        auto idx = neighbor->index_of_neighbor(edge.detector_node);
        neighbor->neighbor_markers[idx] ^= pm::FLIPPED;
    }
}

inline pm::total_weight_int decode_edges_and_weight(
    pm::Mwpm &mwpm,
    const std::vector<uint64_t> &detection_events,
    std::vector<int64_t> &edges) {
    if (mwpm.flooder.graph.nodes.size() != mwpm.search_flooder.graph.nodes.size()) {
        throw std::invalid_argument("Mwpm object does not contain a search flooder");
    }
    process_timeline(mwpm, detection_events);
    mwpm.flooder.match_edges.clear();
    shatter_to_match_edges(mwpm, detection_events, mwpm.flooder.match_edges);
    if (!mwpm.flooder.negative_weight_detection_events.empty()) {
        shatter_to_match_edges(
            mwpm, mwpm.flooder.negative_weight_detection_events, mwpm.flooder.match_edges);
    }

    std::vector<uint8_t> ignored_observables(mwpm.flooder.graph.num_observables, 0);
    pm::total_weight_int weight = 0;
    mwpm.extract_paths_from_match_edges(
        mwpm.flooder.match_edges, ignored_observables.data(), weight);
    weight += mwpm.flooder.negative_weight_sum;

    for (const auto &neg_node_pair : mwpm.search_flooder.graph.negative_weight_edges) {
        auto node1_ptr = &mwpm.search_flooder.graph.nodes[neg_node_pair.first];
        auto node2_ptr = neg_node_pair.second != SIZE_MAX
            ? &mwpm.search_flooder.graph.nodes[neg_node_pair.second]
            : nullptr;
        pm::SearchGraphEdge neg_edge = {node1_ptr, node1_ptr->index_of_neighbor(node2_ptr)};
        flip_edge(neg_edge);
        edges.push_back(neg_edge.detector_node - &mwpm.search_flooder.graph.nodes[0]);
        edges.push_back(node2_ptr ? node2_ptr - &mwpm.search_flooder.graph.nodes[0] : -1);
    }

    for (const auto &match_edge : mwpm.flooder.match_edges) {
        size_t node_from = match_edge.loc_from - &mwpm.flooder.graph.nodes[0];
        size_t node_to = match_edge.loc_to
            ? match_edge.loc_to - &mwpm.flooder.graph.nodes[0]
            : SIZE_MAX;
        mwpm.search_flooder.iter_edges_on_shortest_path_from_middle(
            node_from, node_to, [&](const pm::SearchGraphEdge &e) {
                flip_edge(e);
                edges.push_back(e.detector_node - &mwpm.search_flooder.graph.nodes[0]);
                auto node2_ptr = e.detector_node->neighbors[e.neighbor_index];
                edges.push_back(node2_ptr ? node2_ptr - &mwpm.search_flooder.graph.nodes[0] : -1);
            });
    }

    for (size_t i = 0; i < edges.size() / 2;) {
        int64_t u = edges[2 * i];
        int64_t v = edges[2 * i + 1];
        auto &u_node = mwpm.search_flooder.graph.nodes[u];
        size_t idx;
        if (v == -1) {
            idx = 0;
        } else {
            idx = u_node.index_of_neighbor(&mwpm.search_flooder.graph.nodes[v]);
        }
        if (!(u_node.neighbor_markers[idx] & pm::FLIPPED)) {
            edges[2 * i] = edges[edges.size() - 2];
            edges[2 * i + 1] = edges[edges.size() - 1];
            edges.resize(edges.size() - 2);
        } else {
            flip_edge({&u_node, idx});
            i++;
        }
    }
    return weight;
}

inline void register_fast_decode(py::module_ &m) {
    m.def(
        "decode_to_edges_array_and_weight",
        [](pm::UserGraph &self,
           const py::array_t<uint64_t, py::array::c_style | py::array::forcecast> &detection_events,
           py::object edge_reweights) {
            auto &mwpm = self.get_mwpm_with_search_graph();
            bool has_reweights = !edge_reweights.is_none();
            std::vector<std::array<double, 3>> raw;
            if (has_reweights) {
                auto arr = edge_reweights.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
                if (arr.ndim() != 2 || arr.shape(1) != 3) {
                    throw std::invalid_argument("edge_reweights must have shape (N, 3)");
                }
                auto a = arr.unchecked<2>();
                raw.reserve(a.shape(0));
                for (py::ssize_t i = 0; i < a.shape(0); i++) {
                    raw.push_back({a(i, 0), a(i, 1), a(i, 2)});
                }
                has_reweights = !raw.empty();
            }
            try {
                if (has_reweights) {
                    auto parsed = self.parse_reweight_specs(raw);
                    bool regen = self.needs_regeneration(parsed);
                    self.apply_reweights(std::move(parsed), regen, true);
                }
                std::vector<uint64_t> dets(
                    detection_events.data(), detection_events.data() + detection_events.size());
                std::vector<int64_t> edges;
                edges.reserve(dets.size());
                auto weight = decode_edges_and_weight(mwpm, dets, edges);
                double scaled = (double)weight / mwpm.flooder.graph.normalising_constant;
                if (has_reweights) {
                    self.restore_weights();
                }
                py::array_t<int64_t> out({(py::ssize_t)(edges.size() / 2), (py::ssize_t)2});
                if (!edges.empty()) {
                    std::copy(edges.begin(), edges.end(), out.mutable_data());
                }
                return py::make_tuple(std::move(out), scaled);
            } catch (...) {
                if (has_reweights) {
                    self.restore_weights();
                }
                throw;
            }
        },
        py::arg("graph"),
        py::arg("detection_events"),
        py::arg("edge_reweights") = py::none());
}

}  // namespace pm_pybind_fast

#endif
