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

#include "pymatching/sparse_blossom/driver/user_graph.h"

#include "pymatching/rand/rand_gen.h"
#include "pymatching/sparse_blossom/driver/implied_weights.h"

namespace {

double bernoulli_xor(double p1, double p2) {
    return p1 * (1 - p2) + p2 * (1 - p1);
}

}  // namespace


double pm::to_weight_for_correlations(double probability) {
    return std::log((1 - probability) / probability);
}

double pm::merge_weights(double a, double b) {
    auto sgn = std::copysign(1, a) * std::copysign(1, b);
    auto signed_min = sgn * std::min(std::abs(a), std::abs(b));
    return signed_min + std::log(1 + std::exp(-std::abs(a + b))) - std::log(1 + std::exp(-std::abs(a - b)));
}

pm::UserNode::UserNode() : is_boundary(false) {
}

size_t pm::UserNode::index_of_neighbor(size_t node) const {
    auto it = std::find_if(neighbors.begin(), neighbors.end(), [&](const UserNeighbor& neighbor) {
        if (neighbor.pos == 0) {
            return neighbor.edge_it->node1 == node;
        } else if (neighbor.pos == 1) {
            return neighbor.edge_it->node2 == node;
        } else {
            throw std::runtime_error("`neighbor.pos` should be 0 or 1, but got: " + std::to_string(neighbor.pos));
        }
    });
    if (it == neighbors.end())
        return SIZE_MAX;

    return it - neighbors.begin();
}

bool is_valid_probability(double p) {
    return (p >= 0 && p <= 1);
}

void pm::UserGraph::merge_edge_or_boundary_edge(
    size_t node,
    size_t neighbor_index,
    const std::vector<size_t>& parallel_observables,
    double parallel_weight,
    double parallel_error_probability,
    pm::MERGE_STRATEGY merge_strategy) {
    auto& neighbor = nodes[node].neighbors[neighbor_index];
    if (merge_strategy == DISALLOW) {
        throw std::invalid_argument(
            "Edge (" + std::to_string(neighbor.edge_it->node1) + ", " + std::to_string(neighbor.edge_it->node2) +
            ") already exists in the graph. "
            "Parallel edges not permitted with the provided `disallow` `merge_strategy`. Please provide a "
            "different `merge_strategy`.");
    } else if (
        merge_strategy == KEEP_ORIGINAL ||
        (merge_strategy == SMALLEST_WEIGHT && parallel_weight >= neighbor.edge_it->weight)) {
        return;
    } else {
        double new_weight, new_error_probability;
        bool use_new_observables;
        if (merge_strategy == REPLACE || merge_strategy == SMALLEST_WEIGHT) {
            new_weight = parallel_weight;
            new_error_probability = parallel_error_probability;
            use_new_observables = true;
        } else if (merge_strategy == INDEPENDENT) {
            new_weight = pm::merge_weights(parallel_weight, neighbor.edge_it->weight);
            new_error_probability = -1;
            if (is_valid_probability(neighbor.edge_it->error_probability) &&
                is_valid_probability(parallel_error_probability))
                new_error_probability = parallel_error_probability * (1 - neighbor.edge_it->error_probability) +
                                        neighbor.edge_it->error_probability * (1 - parallel_error_probability);
            // We do not need to update the observables. If they do not match up, then the code has distance 2.
            use_new_observables = false;
        } else {
            throw std::invalid_argument("Merge strategy not recognised.");
        }
        // Update the existing edge weight and probability in the adjacency list of `node`
        neighbor.edge_it->weight = new_weight;
        neighbor.edge_it->error_probability = new_error_probability;
        if (use_new_observables)
            neighbor.edge_it->observable_indices = parallel_observables;

        _mwpm_needs_updating = true;
        // Invalidate caches when edge weights change
        invalidate_max_weight_cache();
        if (new_error_probability < 0 || new_error_probability > 1)
            _all_edges_have_error_probabilities = false;
    }
}

void pm::UserGraph::add_or_merge_edge(
    size_t node1,
    size_t node2,
    const std::vector<size_t>& observables,
    double weight,
    double error_probability,
    MERGE_STRATEGY merge_strategy) {
    auto max_id = std::max(node1, node2);
    if (max_id + 1 > nodes.size())
        nodes.resize(max_id + 1);

    size_t idx = nodes[node1].index_of_neighbor(node2);

    if (idx == SIZE_MAX) {
        pm::UserEdge edge = {node1, node2, observables, weight, error_probability};
        edges.push_back(edge);
        nodes[node1].neighbors.push_back({std::prev(edges.end()), 1});
        if (node1 != node2)
            nodes[node2].neighbors.push_back({std::prev(edges.end()), 0});

        for (auto& obs : observables) {
            if (obs + 1 > _num_observables)
                _num_observables = obs + 1;
        }
        _mwpm_needs_updating = true;
        // Invalidate caches when graph structure changes
        invalidate_max_weight_cache();
        if (error_probability < 0 || error_probability > 1)
            _all_edges_have_error_probabilities = false;
    } else {
        merge_edge_or_boundary_edge(node1, idx, observables, weight, error_probability, merge_strategy);
    }
}

void pm::UserGraph::add_or_merge_boundary_edge(
    size_t node,
    const std::vector<size_t>& observables,
    double weight,
    double error_probability,
    MERGE_STRATEGY merge_strategy) {
    if (node + 1 > nodes.size())
        nodes.resize(node + 1);

    size_t idx = nodes[node].index_of_neighbor(SIZE_MAX);

    if (idx == SIZE_MAX) {
        pm::UserEdge edge = {node, SIZE_MAX, observables, weight, error_probability};
        edges.push_back(edge);
        nodes[node].neighbors.push_back({std::prev(edges.end()), 1});

        for (auto& obs : observables) {
            if (obs + 1 > _num_observables)
                _num_observables = obs + 1;
        }
        _mwpm_needs_updating = true;
        // Invalidate caches when graph structure changes
        invalidate_max_weight_cache();
        if (error_probability < 0 || error_probability > 1)
            _all_edges_have_error_probabilities = false;
    } else {
        merge_edge_or_boundary_edge(node, idx, observables, weight, error_probability, merge_strategy);
    }
}

pm::UserGraph::UserGraph()
    : _num_observables(0), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true),
      _edge_index_cache_valid(false), _cached_max_abs_weight(0), _max_weight_cache_valid(false) {
}

pm::UserGraph::UserGraph(size_t num_nodes)
    : _num_observables(0), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true),
      _edge_index_cache_valid(false), _cached_max_abs_weight(0), _max_weight_cache_valid(false) {
    nodes.resize(num_nodes);
}

pm::UserGraph::UserGraph(size_t num_nodes, size_t num_observables)
    : _num_observables(num_observables), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true),
      _edge_index_cache_valid(false), _cached_max_abs_weight(0), _max_weight_cache_valid(false) {
    nodes.resize(num_nodes);
}

void pm::UserGraph::set_boundary(const std::set<size_t>& boundary) {
    for (auto& n : boundary_nodes)
        nodes[n].is_boundary = false;
    boundary_nodes = boundary;
    for (auto& n : boundary_nodes) {
        if (n >= nodes.size())
            nodes.resize(n + 1);
        nodes[n].is_boundary = true;
    }
    _mwpm_needs_updating = true;
    // Topology changed: a node switching boundary status resizes/reorders its neighbor
    // arrays when the graph is rebuilt, so any cached edge-neighbor indices are stale.
    _edge_index_cache_valid = false;
}

std::set<size_t> pm::UserGraph::get_boundary() {
    return boundary_nodes;
}

size_t pm::UserGraph::get_num_observables() {
    return _num_observables;
}

size_t pm::UserGraph::get_num_nodes() {
    return nodes.size();
}

size_t pm::UserGraph::get_num_detectors() {
    return get_num_nodes() - boundary_nodes.size();
}

bool pm::UserGraph::is_boundary_node(size_t node_id) {
    return (node_id == SIZE_MAX) || nodes[node_id].is_boundary;
}

void pm::UserGraph::update_mwpm() {
    _mwpm = to_mwpm(pm::NUM_DISTINCT_WEIGHTS, false);
    _mwpm_needs_updating = false;
    // The matching/search graphs were rebuilt; any cached edge-neighbor indices point
    // into the old, now-freed neighbor arrays.
    _edge_index_cache_valid = false;
}

pm::Mwpm& pm::UserGraph::get_mwpm() {
    if (_mwpm_needs_updating)
        update_mwpm();
    return _mwpm;
}

void pm::UserGraph::add_noise(uint8_t* error_arr, uint8_t* syndrome_arr) const {
    if (!_all_edges_have_error_probabilities)
        return;

    for (auto& e : edges) {
        auto p = e.error_probability;
        if (rand_float(0.0, 1.0) < p) {
            // Flip the observables
            for (auto& obs : e.observable_indices) {
                *(error_arr + obs) ^= 1;
            }
            // Flip the syndrome bits
            *(syndrome_arr + e.node1) ^= 1;
            if (e.node2 != SIZE_MAX)
                *(syndrome_arr + e.node2) ^= 1;
        }
    }

    for (auto& b : boundary_nodes)
        *(syndrome_arr + b) = 0;
}

size_t pm::UserGraph::get_num_edges() {
    return edges.size();
}

bool pm::UserGraph::all_edges_have_error_probabilities() {
    return _all_edges_have_error_probabilities;
}

double pm::UserGraph::max_abs_weight() {
    // Use cached value if available (Optimization 5)
    if (_max_weight_cache_valid) {
        return _cached_max_abs_weight;
    }

    double max_weight = 0;
    for (auto& e : edges) {
        if (std::abs(e.weight) > max_weight) {
            max_weight = std::abs(e.weight);
        }
    }

    // Cache the result
    _cached_max_abs_weight = max_weight;
    _max_weight_cache_valid = true;

    return max_weight;
}

double pm::UserGraph::get_cached_max_abs_weight() {
    return max_abs_weight();  // Uses caching internally
}

void pm::UserGraph::invalidate_max_weight_cache() {
    _max_weight_cache_valid = false;
    _edge_index_cache_valid = false;  // Also invalidate edge index cache when graph changes
}

pm::MatchingGraph pm::UserGraph::to_matching_graph(pm::weight_int num_distinct_weights) {
    pm::MatchingGraph matching_graph(nodes.size(), _num_observables);

    double normalising_constant = to_matching_or_search_graph_helper(
        num_distinct_weights,
        [&](size_t u,
            size_t v,
            pm::signed_weight_int weight,
            const std::vector<size_t>& observables,
            const std::vector<ImpliedWeightUnconverted>& implied_weights_for_other_edges) {
            matching_graph.add_edge(u, v, weight, observables, implied_weights_for_other_edges);
        },
        [&](size_t u,
            pm::signed_weight_int weight,
            const std::vector<size_t>& observables,
            const std::vector<ImpliedWeightUnconverted>& implied_weights_for_other_edges) {
            matching_graph.add_boundary_edge(u, weight, observables, implied_weights_for_other_edges);
        });

    matching_graph.normalising_constant = normalising_constant;
    if (boundary_nodes.size() > 0) {
        matching_graph.is_user_graph_boundary_node.clear();
        matching_graph.is_user_graph_boundary_node.resize(nodes.size(), false);
        for (auto& i : boundary_nodes)
            matching_graph.is_user_graph_boundary_node[i] = true;
    }

    matching_graph.convert_implied_weights(normalising_constant);
    return matching_graph;
}

pm::SearchGraph pm::UserGraph::to_search_graph(pm::weight_int num_distinct_weights) {
    /// Identical to to_matching_graph but for constructing a pm::SearchGraph
    pm::SearchGraph search_graph(nodes.size());

    double normalizing_constant = to_matching_or_search_graph_helper(
        num_distinct_weights,
        [&](size_t u,
            size_t v,
            pm::signed_weight_int weight,
            const std::vector<size_t>& observables,
            const std::vector<ImpliedWeightUnconverted>& implied_weights_for_other_edges) {
            search_graph.add_edge(u, v, weight, observables, implied_weights_for_other_edges);
        },
        [&](size_t u,
            pm::signed_weight_int weight,
            const std::vector<size_t>& observables,
            const std::vector<ImpliedWeightUnconverted>& implied_weights_for_other_edges) {
            search_graph.add_boundary_edge(u, weight, observables, implied_weights_for_other_edges);
        });

    search_graph.convert_implied_weights(normalizing_constant);
    return search_graph;
}

pm::Mwpm pm::UserGraph::to_mwpm(pm::weight_int num_distinct_weights, bool ensure_search_graph_included) {
    if (_num_observables > sizeof(pm::obs_int) * 8 || ensure_search_graph_included) {
        auto mwpm = pm::Mwpm(
            pm::GraphFlooder(to_matching_graph(num_distinct_weights)),
            pm::SearchFlooder(to_search_graph(num_distinct_weights)));
        mwpm.flooder.sync_negative_weight_observables_and_detection_events();
        mwpm.flooder.graph.loaded_from_dem_without_correlations = loaded_from_dem_without_correlations;
        return mwpm;
    } else {
        auto mwpm = pm::Mwpm(pm::GraphFlooder(to_matching_graph(num_distinct_weights)));
        mwpm.flooder.sync_negative_weight_observables_and_detection_events();
        mwpm.flooder.graph.loaded_from_dem_without_correlations = loaded_from_dem_without_correlations;
        return mwpm;
    }
}

pm::Mwpm& pm::UserGraph::get_mwpm_with_search_graph() {
    if (!_mwpm_needs_updating && _mwpm.flooder.graph.nodes.size() == _mwpm.search_flooder.graph.nodes.size()) {
        return _mwpm;
    } else {
        _mwpm = to_mwpm(pm::NUM_DISTINCT_WEIGHTS, true);
        _mwpm_needs_updating = false;
        _edge_index_cache_valid = false;  // rebuilt graphs invalidate cached neighbor indices
        return _mwpm;
    }
}

void pm::UserGraph::handle_dem_instruction(
    double p, const std::vector<size_t>& detectors, const std::vector<size_t>& observables) {
    if (detectors.size() == 2) {
        add_or_merge_edge(detectors[0], detectors[1], observables, std::log((1 - p) / p), p, INDEPENDENT);
    } else if (detectors.size() == 1) {
        add_or_merge_boundary_edge(detectors[0], observables, std::log((1 - p) / p), p, INDEPENDENT);
    }
}

void pm::UserGraph::handle_dem_instruction_include_correlations(
    double p, const std::vector<size_t>& detectors, const std::vector<size_t>& observables) {
    if (detectors.size() == 2) {
        add_or_merge_edge(detectors[0], detectors[1], observables, pm::to_weight_for_correlations(p), p, INDEPENDENT);
    } else if (detectors.size() == 1) {
        add_or_merge_boundary_edge(detectors[0], observables, pm::to_weight_for_correlations(p), p, INDEPENDENT);
    }
}

void pm::UserGraph::get_nodes_on_shortest_path_from_source(size_t src, size_t dst, std::vector<size_t>& out_nodes) {
    auto& mwpm = get_mwpm_with_search_graph();
    bool src_is_boundary = is_boundary_node(src);
    bool dst_is_boundary = is_boundary_node(dst);
    if (src != SIZE_MAX && src >= nodes.size())
        throw std::invalid_argument("node " + std::to_string(src) + " is not in the graph");
    if (dst != SIZE_MAX && dst >= nodes.size())
        throw std::invalid_argument("node " + std::to_string(dst) + " is not in the graph");
    if (!src_is_boundary) {
        size_t dst_tmp = dst_is_boundary ? SIZE_MAX : dst;
        mwpm.search_flooder.iter_edges_on_shortest_path_from_source(src, dst_tmp, [&](const pm::SearchGraphEdge edge) {
            out_nodes.push_back(edge.detector_node - &mwpm.search_flooder.graph.nodes[0]);
        });
        if (!dst_is_boundary)
            out_nodes.push_back(dst);
    } else if (!dst_is_boundary) {
        std::vector<size_t> temp_out_nodes;
        get_nodes_on_shortest_path_from_source(dst, src, temp_out_nodes);
        for (size_t i = 0; i < temp_out_nodes.size(); i++) {
            out_nodes.push_back(temp_out_nodes[temp_out_nodes.size() - 1 - i]);
        }
    } else {
        throw std::invalid_argument("Both the source and destination vertices provided are boundary nodes");
    }
}

bool pm::UserGraph::has_edge(size_t node1, size_t node2) {
    if (node1 >= nodes.size())
        return false;
    return nodes[node1].index_of_neighbor(node2) != SIZE_MAX;
}

bool pm::UserGraph::has_boundary_edge(size_t node) {
    if (node >= nodes.size())
        return false;
    return nodes[node].index_of_neighbor(SIZE_MAX) != SIZE_MAX;
}

bool pm::UserGraph::get_edge_or_boundary_edge_weight(size_t node1, size_t node2, double& weight_out) {
    if (node1 >= nodes.size()) {
        return false;
    }
    size_t neighbor_idx = nodes[node1].index_of_neighbor(node2);
    if (neighbor_idx == SIZE_MAX) {
        return false;
    }
    weight_out = nodes[node1].neighbors[neighbor_idx].edge_it->weight;
    return true;
}

void pm::UserGraph::set_min_num_observables(size_t num_observables) {
    if (num_observables > _num_observables)
        _num_observables = num_observables;
}

double pm::UserGraph::get_edge_weight_normalising_constant(size_t max_num_distinct_weights) {
    double max_abs_weight = 0;
    bool all_integral_weight = true;
    for (auto& e : edges) {
        if (std::abs(e.weight) > max_abs_weight)
            max_abs_weight = std::abs(e.weight);

        if (round(e.weight) != e.weight) {
            all_integral_weight = false;
        }

        for (auto implied : e.implied_weights_for_other_edges) {
            if (std::abs(implied.implied_weight) > max_abs_weight) {
                max_abs_weight = std::abs(implied.implied_weight);
            }

            if (round(implied.implied_weight) != implied.implied_weight) {
                all_integral_weight = false;
            }

            double current_weight;
            bool has_edge = get_edge_or_boundary_edge_weight(implied.node1, implied.node2, current_weight);
            if (!has_edge) {
                throw std::invalid_argument(
                    "Edge rewrite rule refers to non-existent edge (" + std::to_string(implied.node1) + ", " +
                    std::to_string(implied.node2) + ")");
            }
            bool same_sign = (current_weight * implied.implied_weight) >= 0.;
            if (!same_sign) {
                throw std::invalid_argument(
                    "Edge weight rewrite rules that change the sign of an edge weight are not currently supported.");
            }
        }
    }

    if (max_abs_weight > pm::MAX_USER_EDGE_WEIGHT)
        throw std::invalid_argument(
            "maximum absolute edge weight of " + std::to_string(pm::MAX_USER_EDGE_WEIGHT) + " exceeded.");

    if (all_integral_weight) {
        return 1.0;
    } else {
        pm::weight_int max_half_edge_weight = max_num_distinct_weights - 1;
        return (double)max_half_edge_weight / max_abs_weight;
    }
}

void pm::add_decomposed_error_to_joint_probabilities(
    DecomposedDemError& error,
    std::map<std::pair<size_t, size_t>, std::map<std::pair<size_t, size_t>, double>>& joint_probabilites) {
    if (error.components.size() > 1) {
        for (size_t k0 = 0; k0 < error.components.size(); k0++) {
            for (size_t k1 = k0 + 1; k1 < error.components.size(); k1++) {
                auto& c0 = error.components[k0];
                auto& c1 = error.components[k1];
                std::pair<size_t, size_t> e0 = std::minmax(c0.node1, c0.node2);
                std::pair<size_t, size_t> e1 = std::minmax(c1.node1, c1.node2);
                double& p01 = joint_probabilites[e0][e1];
                double& p10 = joint_probabilites[e1][e0];
                p01 = bernoulli_xor(p01, error.probability);
                p10 = bernoulli_xor(p10, error.probability);
            }
        }
    }

    for (auto& e : error.components) {
        double& p = joint_probabilites[std::minmax(e.node1, e.node2)][std::minmax(e.node1, e.node2)];
        p = bernoulli_xor(p, error.probability);
    }
}

pm::UserGraph pm::detector_error_model_to_user_graph(
    const stim::DetectorErrorModel& detector_error_model,
    const bool enable_correlations,
    pm::weight_int num_distinct_weights) {
    pm::UserGraph user_graph(detector_error_model.count_detectors(), detector_error_model.count_observables());
    std::map<std::pair<size_t, size_t>, std::map<std::pair<size_t, size_t>, double>> joint_probabilites;
    if (enable_correlations) {
        pm::iter_dem_instructions_include_correlations(
            detector_error_model,
            [&](double p, const std::vector<size_t>& detectors, std::vector<size_t>& observables) {
                user_graph.handle_dem_instruction_include_correlations(p, detectors, observables);
            },
            joint_probabilites);

        user_graph.populate_implied_edge_weights(joint_probabilites);
    } else {
        pm::iter_detector_error_model_edges(
            detector_error_model,
            [&](double p, const std::vector<size_t>& detectors, std::vector<size_t>& observables) {
                user_graph.handle_dem_instruction(p, detectors, observables);
            });
        user_graph.loaded_from_dem_without_correlations = true;
    }
    return user_graph;
}

// Optimization 1: Create a canonical key for edge lookup
std::pair<size_t, size_t> pm::UserGraph::make_edge_key(size_t node1, size_t node2) {
    // For boundary edges (node2 == SIZE_MAX), always put the real node first
    if (node2 == SIZE_MAX) {
        return {node1, SIZE_MAX};
    }
    // For regular edges, use min/max to ensure consistent ordering
    return std::minmax(node1, node2);
}

// Optimization 1: Build the edge index cache
void pm::UserGraph::build_edge_index_cache(pm::Mwpm& mwpm) {
    _edge_index_cache.clear();

    // Iterate through all edges in the UserGraph
    for (size_t node_idx = 0; node_idx < nodes.size(); node_idx++) {
        for (size_t neighbor_idx = 0; neighbor_idx < nodes[node_idx].neighbors.size(); neighbor_idx++) {
            const auto& neighbor = nodes[node_idx].neighbors[neighbor_idx];
            size_t other_node;
            if (neighbor.pos == 0) {
                other_node = neighbor.edge_it->node1;
            } else {
                other_node = neighbor.edge_it->node2;
            }

            auto key = make_edge_key(node_idx, other_node);

            // Only process each edge once (when node_idx is the smaller node, or for boundary edges)
            if (key.first != node_idx && other_node != SIZE_MAX) {
                continue;
            }

            EdgeIndexCache cache;
            cache.user_graph_neighbor_idx = neighbor_idx;

            // Find matching graph indices
            cache.matching_graph_node1_neighbor_idx = find_neighbor_index_in_matching_graph(key.first, key.second);
            if (key.second != SIZE_MAX) {
                cache.matching_graph_node2_neighbor_idx = find_neighbor_index_in_matching_graph(key.second, key.first);
            } else {
                cache.matching_graph_node2_neighbor_idx = SIZE_MAX;
            }

            // Find search graph indices (if search graph exists)
            if (mwpm.search_flooder.graph.nodes.size() > 0) {
                cache.search_graph_node1_neighbor_idx = find_neighbor_index_in_search_graph(key.first, key.second);
                if (key.second != SIZE_MAX) {
                    cache.search_graph_node2_neighbor_idx = find_neighbor_index_in_search_graph(key.second, key.first);
                } else {
                    cache.search_graph_node2_neighbor_idx = SIZE_MAX;
                }
            } else {
                cache.search_graph_node1_neighbor_idx = SIZE_MAX;
                cache.search_graph_node2_neighbor_idx = SIZE_MAX;
            }

            _edge_index_cache[key] = cache;
        }
    }

    _edge_index_cache_valid = true;
}

// Optimization 1: Get or create cached edge indices
const pm::EdgeIndexCache* pm::UserGraph::get_edge_index_cache(size_t node1, size_t node2, pm::Mwpm& mwpm) {
    // Build cache if not valid
    if (!_edge_index_cache_valid) {
        build_edge_index_cache(mwpm);
    }

    auto key = make_edge_key(node1, node2);
    auto it = _edge_index_cache.find(key);
    if (it != _edge_index_cache.end()) {
        return &it->second;
    }
    return nullptr;
}

// Optimization 4: Prepare batch reweights
void pm::UserGraph::prepare_batch_reweights(const std::vector<std::vector<std::array<double, 3>>>& all_reweight_specs, pm::Mwpm& mwpm) {
    // Build edge index cache if not already built
    if (!_edge_index_cache_valid) {
        build_edge_index_cache(mwpm);
    }

    // Pre-validate all unique edges across the batch
    std::unordered_map<std::pair<size_t, size_t>, bool, PairHash> validated_edges;

    for (const auto& shot_specs : all_reweight_specs) {
        for (const auto& spec : shot_specs) {
            size_t node1 = (size_t)spec[0];
            double node2_raw = spec[1];
            size_t node2 = (node2_raw < 0) ? SIZE_MAX : (size_t)node2_raw;

            auto key = make_edge_key(node1, node2);
            if (validated_edges.find(key) == validated_edges.end()) {
                // Validate this edge exists
                double original_weight;
                if (!get_edge_or_boundary_edge_weight(node1, node2, original_weight)) {
                    std::string node2_str = (node2 == SIZE_MAX) ? "-1" : std::to_string(node2);
                    throw std::invalid_argument("Edge (" + std::to_string(node1) + ", " +
                                              node2_str + ") does not exist");
                }
                validated_edges[key] = true;
            }
        }
    }
}

void pm::UserGraph::apply_reweights(const std::vector<std::array<double, 3>>& reweight_specs, pm::Mwpm& mwpm, bool needs_regeneration) {
    // Check graph has no negative weights
    if (!mwpm.flooder.negative_weight_detection_events.empty()) {
        throw std::invalid_argument("Edge reweighting not supported with negative edge weights");
    }

    // Optimization 3: Reuse the reweight buffer to avoid heap allocations
    _reweight_buffer.clear();
    if (_reweight_buffer.capacity() < reweight_specs.size()) {
        _reweight_buffer.reserve(reweight_specs.size() * 2);  // Amortized growth
    }

    // Validate and prepare reweights
    for (const auto& spec : reweight_specs) {
        size_t node1 = (size_t)spec[0];
        double node2_raw = spec[1];
        size_t node2;

        // Validate boundary edge format first
        if (node2_raw < 0) {
            if (node2_raw != -1.0) {
                throw std::invalid_argument("Boundary edges must use exactly -1 as second node");
            }
            node2 = SIZE_MAX;
        } else {
            node2 = (size_t)node2_raw;
        }

        double new_weight = spec[2];

        // Validate new weight is non-negative
        if (new_weight < 0) {
            throw std::invalid_argument("Reweight values must be non-negative");
        }

        // Find existing edge and store original weight
        double original_weight;
        if (!get_edge_or_boundary_edge_weight(node1, node2, original_weight)) {
            std::string node2_str = (node2 == SIZE_MAX) ? "-1" : std::to_string(node2);
            throw std::invalid_argument("Edge (" + std::to_string(node1) + ", " +
                                      node2_str + ") does not exist");
        }

        EdgeReweight reweight;
        reweight.node1 = node1;
        reweight.node2 = node2;
        reweight.original_weight = original_weight;
        reweight.new_weight = new_weight;

        // Optimization 1: Use cached edge indices if available
        const EdgeIndexCache* cache = get_edge_index_cache(node1, node2, mwpm);
        if (cache != nullptr) {
            // Determine which indices to use based on node ordering
            auto key = make_edge_key(node1, node2);
            if (key.first == node1) {
                reweight.matching_graph_node1_neighbor_idx = cache->matching_graph_node1_neighbor_idx;
                reweight.matching_graph_node2_neighbor_idx = cache->matching_graph_node2_neighbor_idx;
                reweight.search_graph_node1_neighbor_idx = cache->search_graph_node1_neighbor_idx;
                reweight.search_graph_node2_neighbor_idx = cache->search_graph_node2_neighbor_idx;
            } else {
                // Swap indices since node1/node2 are reversed from the cache key
                reweight.matching_graph_node1_neighbor_idx = cache->matching_graph_node2_neighbor_idx;
                reweight.matching_graph_node2_neighbor_idx = cache->matching_graph_node1_neighbor_idx;
                reweight.search_graph_node1_neighbor_idx = cache->search_graph_node2_neighbor_idx;
                reweight.search_graph_node2_neighbor_idx = cache->search_graph_node1_neighbor_idx;
            }
        } else {
            // Fallback: indices will be computed lazily
            reweight.matching_graph_node1_neighbor_idx = SIZE_MAX;
            reweight.matching_graph_node2_neighbor_idx = SIZE_MAX;
            reweight.search_graph_node1_neighbor_idx = SIZE_MAX;
            reweight.search_graph_node2_neighbor_idx = SIZE_MAX;
        }
        reweight.original_normalized_weight = 0;
        reweight.new_normalized_weight = 0;

        _reweight_buffer.push_back(reweight);
    }

    // Move buffer to active reweights (avoids copy)
    _active_reweights = std::move(_reweight_buffer);

    if (needs_regeneration) {
        // Full regeneration - need to update UserGraph edges
        for (const auto& rw : _active_reweights) {
            size_t neighbor_idx = nodes[rw.node1].index_of_neighbor(rw.node2);
            nodes[rw.node1].neighbors[neighbor_idx].edge_it->weight = rw.new_weight;
        }
        _mwpm_needs_updating = true;
    } else {
        // Optimization 2: Skip UserGraph updates in Tier 1 mode
        // Note: edges() will return original weights during reweighted decode
        // This is documented behavior for performance optimization

        // Direct graph weight updates
        // Calculate normalized weights using the mwpm's normalization constant.
        // NOTE: graph.normalising_constant already has the factor of 2 baked in
        // (see iter_discretized_edges, which returns `normalising_constant * 2`).
        // Build-time discretization is `round(weight * nc_base) * 2` where
        // nc_base == graph.normalising_constant / 2, so we must divide by 2 inside
        // the round() to match it. Multiplying by graph.normalising_constant and
        // then by 2 would apply the factor of 2 twice, doubling every reweighted
        // edge's integer weight.
        for (auto& rw : _active_reweights) {
            rw.original_normalized_weight = (weight_int)round(rw.original_weight * mwpm.flooder.graph.normalising_constant / 2) * 2;
            rw.new_normalized_weight = (weight_int)round(rw.new_weight * mwpm.flooder.graph.normalising_constant / 2) * 2;
        }

        // Update weights directly in existing graphs
        update_existing_graph_weights();

        // DO NOT set _mwpm_needs_updating = true
    }
}

void pm::UserGraph::restore_weights(bool needs_regeneration) {
    if (_active_reweights.empty()) return;

    if (needs_regeneration) {
        // Full regeneration path - need to restore UserGraph edges
        for (const auto& rw : _active_reweights) {
            size_t neighbor_idx = nodes[rw.node1].index_of_neighbor(rw.node2);
            nodes[rw.node1].neighbors[neighbor_idx].edge_it->weight = rw.original_weight;
        }
        // Trigger regeneration to restore original normalization
        _mwpm_needs_updating = true;
    } else {
        // Optimization 2: Skip UserGraph restoration in Tier 1 mode
        // UserGraph was never modified, so no need to restore it
        // Only restore weights directly in existing graphs

        // Restore MatchingGraph weights
        if (_mwpm.flooder.graph.nodes.size() > 0) {
            for (const auto& rw : _active_reweights) {
                if (rw.matching_graph_node1_neighbor_idx != SIZE_MAX) {
                    _mwpm.flooder.graph.nodes[rw.node1].neighbor_weights[rw.matching_graph_node1_neighbor_idx] =
                        rw.original_normalized_weight;
                }
                if (rw.node2 != SIZE_MAX && rw.matching_graph_node2_neighbor_idx != SIZE_MAX) {
                    _mwpm.flooder.graph.nodes[rw.node2].neighbor_weights[rw.matching_graph_node2_neighbor_idx] =
                        rw.original_normalized_weight;
                }
            }
        }

        // Restore SearchGraph weights
        if (_mwpm.search_flooder.graph.nodes.size() > 0) {
            for (const auto& rw : _active_reweights) {
                if (rw.search_graph_node1_neighbor_idx != SIZE_MAX) {
                    _mwpm.search_flooder.graph.nodes[rw.node1].neighbor_weights[rw.search_graph_node1_neighbor_idx] =
                        rw.original_normalized_weight;
                }
                if (rw.node2 != SIZE_MAX && rw.search_graph_node2_neighbor_idx != SIZE_MAX) {
                    _mwpm.search_flooder.graph.nodes[rw.node2].neighbor_weights[rw.search_graph_node2_neighbor_idx] =
                        rw.original_normalized_weight;
                }
            }
        }

        // DO NOT set _mwpm_needs_updating = true
    }

    // Reset state
    _active_reweights.clear();
}

bool pm::UserGraph::all_edges_integral() const {
    for (const auto& e : edges) {
        if (round(e.weight) != e.weight)
            return false;
        for (const auto& implied : e.implied_weights_for_other_edges) {
            if (round(implied.implied_weight) != implied.implied_weight)
                return false;
        }
    }
    return true;
}

bool pm::UserGraph::needs_regeneration(const std::vector<std::array<double, 3>>& reweight_specs) {
    double original_max_abs_weight = max_abs_weight();
    double max_new_abs_weight = original_max_abs_weight;

    // When the current graph is all-integral, get_edge_weight_normalising_constant
    // collapses to 1.0, so the matching graph is discretized at integer resolution.
    // An in-place (Tier-1) reweight to a non-integer value would then be silently
    // rounded to the nearest integer (e.g. 0.1 -> a free edge). Force a full
    // regeneration in that case: the reweighted, now non-integral edge makes
    // all_integral_weight false, rescaling the constant to fine resolution.
    bool graph_all_integral = all_edges_integral();

    // Find the maximum reweight magnitude (regeneration is also required when a
    // reweight exceeds the original max, which changes the normalising constant).
    for (const auto& spec : reweight_specs) {
        double new_weight = spec[2];
        if (graph_all_integral && round(new_weight) != new_weight)
            return true;
        max_new_abs_weight = std::max(max_new_abs_weight, std::abs(new_weight));
    }

    return max_new_abs_weight > original_max_abs_weight;
}

bool pm::UserGraph::batch_needs_regeneration(const std::vector<std::vector<std::array<double, 3>>>& all_reweight_specs) {
    double original_max_abs_weight = max_abs_weight();
    double global_max_abs_weight = original_max_abs_weight;

    // Find global maximum weight across all shots
    for (const auto& shot_reweights : all_reweight_specs) {
        for (const auto& spec : shot_reweights) {
            double new_weight = spec[2];
            global_max_abs_weight = std::max(global_max_abs_weight, std::abs(new_weight));
        }
    }

    return global_max_abs_weight > original_max_abs_weight;
}

void pm::UserGraph::update_existing_graph_weights() {
    // Update MatchingGraph weights
    if (_mwpm.flooder.graph.nodes.size() > 0) {
        for (auto& rw : _active_reweights) {
            // Find neighbor indices if not already found
            if (rw.matching_graph_node1_neighbor_idx == SIZE_MAX) {
                rw.matching_graph_node1_neighbor_idx = find_neighbor_index_in_matching_graph(rw.node1, rw.node2);
            }
            if (rw.node2 != SIZE_MAX && rw.matching_graph_node2_neighbor_idx == SIZE_MAX) {
                rw.matching_graph_node2_neighbor_idx = find_neighbor_index_in_matching_graph(rw.node2, rw.node1);
            }

            // Update forward edge (node1 -> node2)
            if (rw.matching_graph_node1_neighbor_idx != SIZE_MAX) {
                _mwpm.flooder.graph.nodes[rw.node1].neighbor_weights[rw.matching_graph_node1_neighbor_idx] =
                    rw.new_normalized_weight;
            }

            // Update reverse edge (node2 -> node1) if not boundary edge
            if (rw.node2 != SIZE_MAX && rw.matching_graph_node2_neighbor_idx != SIZE_MAX) {
                _mwpm.flooder.graph.nodes[rw.node2].neighbor_weights[rw.matching_graph_node2_neighbor_idx] =
                    rw.new_normalized_weight;
            }
        }
    }

    // Update SearchGraph weights (if it exists)
    if (_mwpm.search_flooder.graph.nodes.size() > 0) {
        for (auto& rw : _active_reweights) {
            // Find neighbor indices if not already found
            if (rw.search_graph_node1_neighbor_idx == SIZE_MAX) {
                rw.search_graph_node1_neighbor_idx = find_neighbor_index_in_search_graph(rw.node1, rw.node2);
            }
            if (rw.node2 != SIZE_MAX && rw.search_graph_node2_neighbor_idx == SIZE_MAX) {
                rw.search_graph_node2_neighbor_idx = find_neighbor_index_in_search_graph(rw.node2, rw.node1);
            }

            // Update forward edge (node1 -> node2)
            if (rw.search_graph_node1_neighbor_idx != SIZE_MAX) {
                _mwpm.search_flooder.graph.nodes[rw.node1].neighbor_weights[rw.search_graph_node1_neighbor_idx] =
                    rw.new_normalized_weight;
            }

            // Update reverse edge (node2 -> node1) if not boundary edge
            if (rw.node2 != SIZE_MAX && rw.search_graph_node2_neighbor_idx != SIZE_MAX) {
                _mwpm.search_flooder.graph.nodes[rw.node2].neighbor_weights[rw.search_graph_node2_neighbor_idx] =
                    rw.new_normalized_weight;
            }
        }
    }
}

size_t pm::UserGraph::find_neighbor_index_in_matching_graph(size_t node1, size_t node2) {
    if (node1 >= _mwpm.flooder.graph.nodes.size()) {
        return SIZE_MAX;
    }

    const auto& neighbors = _mwpm.flooder.graph.nodes[node1].neighbors;
    for (size_t i = 0; i < neighbors.size(); i++) {
        if (node2 == SIZE_MAX) {
            // Looking for boundary edge
            if (neighbors[i] == nullptr) {
                return i;
            }
        } else {
            // Looking for regular edge
            if (neighbors[i] != nullptr && (neighbors[i] - &_mwpm.flooder.graph.nodes[0]) == node2) {
                return i;
            }
        }
    }
    return SIZE_MAX;
}

size_t pm::UserGraph::find_neighbor_index_in_search_graph(size_t node1, size_t node2) {
    if (node1 >= _mwpm.search_flooder.graph.nodes.size()) {
        return SIZE_MAX;
    }

    const auto& neighbors = _mwpm.search_flooder.graph.nodes[node1].neighbors;
    for (size_t i = 0; i < neighbors.size(); i++) {
        if (node2 == SIZE_MAX) {
            // Looking for boundary edge
            if (neighbors[i] == nullptr) {
                return i;
            }
        } else {
            // Looking for regular edge
            if (neighbors[i] != nullptr && (neighbors[i] - &_mwpm.search_flooder.graph.nodes[0]) == node2) {
                return i;
            }
        }
    }
    return SIZE_MAX;
}


void pm::UserGraph::populate_implied_edge_weights(
    std::map<std::pair<size_t, size_t>, std::map<std::pair<size_t, size_t>, double>>& joint_probabilites) {
    for (auto& edge : edges) {
        std::pair<size_t, size_t> current_edge_nodes = std::minmax(edge.node1, edge.node2);
        auto it = joint_probabilites.find(current_edge_nodes);
        if (it != joint_probabilites.end()) {
            const auto& pf = *it;
            std::pair<size_t, size_t> causal_edge = pf.first;
            double marginal_probability = pf.second.at(causal_edge);
            if (marginal_probability == 0)
                continue;

            for (const auto& affected_edge_and_probability : pf.second) {
                std::pair<size_t, size_t> affected_edge = affected_edge_and_probability.first;
                if (affected_edge != causal_edge) {
                    // Since edge weights are computed as std::log((1-p)/p), a probability of more than 0.5 for an
                    // error, would lead to a negatively weighted error. We do not support this (yet), and use a
                    // minimum of 0.5 as an implied probability for an edge to be reweighted.
                    double implied_probability_for_other_edge =
                        std::min(0.5, affected_edge_and_probability.second / marginal_probability);
                    double w = pm::to_weight_for_correlations(implied_probability_for_other_edge);
                    ImpliedWeightUnconverted implied{affected_edge.first, affected_edge.second, w};
                    edge.implied_weights_for_other_edges.push_back(implied);
                }
            }
        }
    }
}
