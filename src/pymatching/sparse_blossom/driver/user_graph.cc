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

/// Returns the index of `node2` in `node1`'s neighbor list within `graph` (a
/// matching or search graph; `node2 == SIZE_MAX` finds the boundary slot), or
/// SIZE_MAX if the graph or slot is absent. Absence is not an error here,
/// unlike the nodes' own index_of_neighbor. NOTE: only edges with a dedicated
/// slot are addressable this way -- edges incident to a boundary NODE are
/// stored as a nullptr boundary slot at the other endpoint (or dropped
/// entirely), so needs_regeneration routes reweights of such edges through a
/// full regeneration instead of this lookup.
template <typename Graph>
size_t find_neighbor_index_in_graph(const Graph& graph, size_t node1, size_t node2) {
    if (node1 >= graph.nodes.size() || (node2 != SIZE_MAX && node2 >= graph.nodes.size()))
        return SIZE_MAX;
    const auto* target = node2 == SIZE_MAX ? nullptr : &graph.nodes[node2];
    const auto& neighbors = graph.nodes[node1].neighbors;
    for (size_t i = 0; i < neighbors.size(); i++) {
        if (neighbors[i] == target)
            return i;
    }
    return SIZE_MAX;
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
        invalidate_edge_stats();
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
        invalidate_edge_stats();
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
        invalidate_edge_stats();
        if (error_probability < 0 || error_probability > 1)
            _all_edges_have_error_probabilities = false;
    } else {
        merge_edge_or_boundary_edge(node, idx, observables, weight, error_probability, merge_strategy);
    }
}

pm::UserGraph::UserGraph()
    : _num_observables(0), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true) {
}

pm::UserGraph::UserGraph(size_t num_nodes)
    : _num_observables(0), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true) {
    nodes.resize(num_nodes);
}

pm::UserGraph::UserGraph(size_t num_nodes, size_t num_observables)
    : _num_observables(num_observables), _mwpm_needs_updating(true), _all_edges_have_error_probabilities(true) {
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

const pm::UserGraph::EdgeStats& pm::UserGraph::edge_stats() const {
    // One traversal computes every per-edge aggregate the reweighting hot path
    // needs. Crucially, max_abs_weight_incl_implied is the SAME maximum
    // get_edge_weight_normalising_constant sizes the discretization by, and
    // all_integral matches its integer-resolution collapse -- computing them in
    // one place means the tier classifier and the discretization can never
    // silently desynchronise. Cached because needs_regeneration runs per
    // reweighted decode; an uncached O(E) list traversal measurably slows
    // Tier-1 (e.g. +35us at E=26k for the previously-uncached integral scan).
    if (_edge_stats.valid) {
        return _edge_stats;
    }
    EdgeStats stats;
    for (const auto& e : edges) {
        double abs_w = std::abs(e.weight);
        if (abs_w > stats.max_abs_weight)
            stats.max_abs_weight = abs_w;
        if (e.weight < 0)
            stats.has_negative_weight = true;
        if (round(e.weight) != e.weight)
            stats.all_integral = false;
        for (const auto& implied : e.implied_weights_for_other_edges) {
            double abs_iw = std::abs(implied.implied_weight);
            if (abs_iw > stats.max_abs_weight_incl_implied)
                stats.max_abs_weight_incl_implied = abs_iw;
            if (round(implied.implied_weight) != implied.implied_weight)
                stats.all_integral = false;
        }
    }
    stats.max_abs_weight_incl_implied = std::max(stats.max_abs_weight_incl_implied, stats.max_abs_weight);
    stats.valid = true;
    _edge_stats = stats;
    return _edge_stats;
}

double pm::UserGraph::max_abs_weight() {
    return edge_stats().max_abs_weight;
}

double pm::UserGraph::max_abs_weight_including_implied() {
    return edge_stats().max_abs_weight_incl_implied;
}

void pm::UserGraph::invalidate_edge_stats() {
    _edge_stats.valid = false;
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
    // Validate implied-weight rewrite rules (edge existence, no sign change).
    for (auto& e : edges) {
        for (const auto& implied : e.implied_weights_for_other_edges) {
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

    // Size the constant from the shared edge stats -- the SAME values
    // needs_regeneration classifies tiers against, so the two cannot
    // desynchronise.
    const EdgeStats& stats = edge_stats();

    if (stats.max_abs_weight_incl_implied > pm::MAX_USER_EDGE_WEIGHT)
        throw std::invalid_argument(
            "maximum absolute edge weight of " + std::to_string(pm::MAX_USER_EDGE_WEIGHT) + " exceeded.");

    if (stats.all_integral) {
        return 1.0;
    } else {
        pm::weight_int max_half_edge_weight = max_num_distinct_weights - 1;
        return (double)max_half_edge_weight / stats.max_abs_weight_incl_implied;
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

void pm::UserGraph::apply_reweights(
    const std::vector<std::array<double, 3>>& reweight_specs, bool needs_regeneration, bool ensure_search_graph) {
    // Reject reweighting on graphs with any negative edge weight: the matching
    // graph stores abs(weight) in the slot plus baked-in compensation (virtual
    // detection events, pre-flipped observables, negative_weight_sum) that an
    // in-place Tier-1 write cannot maintain. The check reads the UserGraph FLOAT
    // weights, so it is uniform: it cannot be parity-cancelled by cycles of
    // negative edges (the old detection-events check) and does not depend on
    // whether a tiny negative weight happens to discretize to 0 (the discretized
    // negative_weight_sum). Checked before materialising, so a rejected call
    // does no graph work.
    if (edge_stats().has_negative_weight) {
        throw std::invalid_argument("Edge reweighting not supported with negative edge weights");
    }

    // Materialise the graph in the requested shape before touching anything, so
    // any regeneration still pending (e.g. from a previous Tier-2 restore or a
    // graph mutation) is applied and the Tier-1 snapshots below target the
    // up-to-date graph. O(1) when nothing is pending.
    pm::Mwpm& mwpm = ensure_search_graph ? get_mwpm_with_search_graph() : get_mwpm();

    // Validate into a local vector and commit to _active_reweights only after the
    // whole spec list has passed (strong exception guarantee). A throw part-way
    // through must leave _active_reweights untouched: stale partial entries from a
    // failed call would otherwise be replayed into the graph by a later
    // exception-path restore_weights, silently overwriting weights the user set
    // in the meantime.
    std::vector<EdgeReweight> validated;
    validated.reserve(reweight_specs.size());

    // Validate and prepare reweights
    for (const auto& spec : reweight_specs) {
        // Node indices arrive as user-supplied doubles; ensure the value is safely
        // castable BEFORE any cast. Casting a negative, non-finite, non-integral,
        // or too-large double to size_t is undefined behaviour and silently
        // "succeeds" with garbage on most platforms (e.g. on arm64 NaN converts to
        // node 0, 1e30 saturates to SIZE_MAX -- the boundary sentinel -- and 2.7
        // truncates to node 2), reweighting the wrong edge with no error. The
        // bound is derived from the platform: (double)SIZE_MAX rounds UP to 2^64
        // on 64-bit size_t (the exact boundary for defined casts) and is exactly
        // representable on 32-bit, where it also rejects the SIZE_MAX boundary
        // sentinel itself. In-range but nonexistent nodes fall through to the
        // "Edge does not exist" check below, preserving the pre-existing error.
        auto validate_node_index = [](double v, const char* which) {
            if (!std::isfinite(v) || v < 0 || round(v) != v || v >= (double)SIZE_MAX)
                throw std::invalid_argument(
                    std::string("Reweight ") + which + " must be a finite non-negative integer, got " +
                    std::to_string(v));
        };
        validate_node_index(spec[0], "node1");
        size_t node1 = (size_t)spec[0];
        double node2_raw = spec[1];
        size_t node2;

        // Exactly -1 is the boundary sentinel; other FINITE negatives get the
        // boundary-format error; everything else (including NaN and +/-inf) goes
        // through validate_node_index for the castability diagnostic.
        if (node2_raw == -1.0) {
            node2 = SIZE_MAX;
        } else if (std::isfinite(node2_raw) && node2_raw < 0) {
            throw std::invalid_argument("Boundary edges must use exactly -1 as second node");
        } else {
            validate_node_index(node2_raw, "node2");
            node2 = (size_t)node2_raw;
        }

        double new_weight = spec[2];

        // Reject negative, non-finite, and over-max weights before any state is
        // touched. NaN would otherwise pass a plain `< 0` check and reach an
        // out-of-range double->weight_int cast (UB) in the Tier-1 discretization;
        // weights above MAX_USER_EDGE_WEIGHT would throw later, mid-regeneration.
        if (new_weight < 0 || !std::isfinite(new_weight)) {
            throw std::invalid_argument("Reweight values must be finite and non-negative");
        }
        if (new_weight > pm::MAX_USER_EDGE_WEIGHT) {
            throw std::invalid_argument(
                "Reweight value " + std::to_string(new_weight) + " exceeds the maximum edge weight " +
                std::to_string(pm::MAX_USER_EDGE_WEIGHT));
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

        // Neighbor indices are computed in the Tier-1 branch below
        // (the O(degree) lookup is negligible next to a decode).
        reweight.matching_graph_node1_neighbor_idx = SIZE_MAX;
        reweight.matching_graph_node2_neighbor_idx = SIZE_MAX;
        reweight.search_graph_node1_neighbor_idx = SIZE_MAX;
        reweight.search_graph_node2_neighbor_idx = SIZE_MAX;
        reweight.original_normalized_weight = 0;
        reweight.new_normalized_weight = 0;

        validated.push_back(reweight);
    }

    // Every spec validated; commit, recording the tier so restore_weights can
    // only ever undo with the tier that was applied. From here on nothing throws
    // before the apply completes (Tier-1 writes discretized ints; Tier-2 writes
    // UserGraph floats for edges the validation loop just resolved).
    _active_reweights = std::move(validated);
    _active_reweights_regen = needs_regeneration;

    if (needs_regeneration) {
        // Full regeneration - need to update UserGraph edges
        for (const auto& rw : _active_reweights) {
            size_t neighbor_idx = nodes[rw.node1].index_of_neighbor(rw.node2);
            nodes[rw.node1].neighbors[neighbor_idx].edge_it->weight = rw.new_weight;
        }
        _mwpm_needs_updating = true;
        // UserGraph edge weights changed, so the cached edge stats are stale.
        invalidate_edge_stats();
        // Materialise the regeneration now, in the same shape, so callers decode
        // against the reweighted graph without needing a get_mwpm() refresh.
        (void)(ensure_search_graph ? get_mwpm_with_search_graph() : get_mwpm());
    } else {
        // Optimization 2: Skip UserGraph updates in Tier 1 mode
        // Note: edges() will return original weights during reweighted decode
        // This is documented behavior for performance optimization

        // Locate every discretized-weight slot for each reweighted edge, snapshot
        // the value it currently holds (so restore_weights can write it back
        // verbatim rather than re-deriving it, which risked mismatching the
        // build-time discretization), and discretize the new weight.
        // NOTE: graph.normalising_constant already has the factor of 2 baked in
        // (see iter_discretized_edges, which returns `normalising_constant * 2`).
        // Build-time discretization is `round(weight * nc_base) * 2` where
        // nc_base == graph.normalising_constant / 2, so we must divide by 2 inside
        // the round() to match it. Multiplying by graph.normalising_constant and
        // then by 2 would apply the factor of 2 twice, doubling every reweighted
        // edge's integer weight.
        const auto& matching_graph = _mwpm.flooder.graph;
        const auto& search_graph = _mwpm.search_flooder.graph;
        for (auto& rw : _active_reweights) {
            rw.matching_graph_node1_neighbor_idx = find_neighbor_index_in_graph(matching_graph, rw.node1, rw.node2);
            rw.search_graph_node1_neighbor_idx = find_neighbor_index_in_graph(search_graph, rw.node1, rw.node2);
            if (rw.node2 != SIZE_MAX) {
                rw.matching_graph_node2_neighbor_idx = find_neighbor_index_in_graph(matching_graph, rw.node2, rw.node1);
                rw.search_graph_node2_neighbor_idx = find_neighbor_index_in_graph(search_graph, rw.node2, rw.node1);
            }
            if (rw.matching_graph_node1_neighbor_idx != SIZE_MAX) {
                rw.original_normalized_weight =
                    _mwpm.flooder.graph.nodes[rw.node1].neighbor_weights[rw.matching_graph_node1_neighbor_idx];
            } else if (rw.search_graph_node1_neighbor_idx != SIZE_MAX) {
                rw.original_normalized_weight =
                    _mwpm.search_flooder.graph.nodes[rw.node1].neighbor_weights[rw.search_graph_node1_neighbor_idx];
            }
            rw.new_normalized_weight =
                (weight_int)round(rw.new_weight * mwpm.flooder.graph.normalising_constant / 2) * 2;
        }

        // Write the new weights only after every snapshot is taken, so duplicate
        // reweights of the same edge in one call snapshot the true original.
        for (const auto& rw : _active_reweights) {
            write_reweight_slots(rw, rw.new_normalized_weight);
        }

        // DO NOT set _mwpm_needs_updating = true
    }
}

void pm::UserGraph::restore_weights() {
    if (_active_reweights.empty()) return;
    bool regen = _active_reweights_regen;

    if (regen) {
        // Full regeneration path - need to restore UserGraph edges
        for (const auto& rw : _active_reweights) {
            size_t neighbor_idx = nodes[rw.node1].index_of_neighbor(rw.node2);
            nodes[rw.node1].neighbors[neighbor_idx].edge_it->weight = rw.original_weight;
        }
        // Trigger regeneration to restore original normalization
        _mwpm_needs_updating = true;
        // UserGraph edge weights changed, so the cached edge stats are stale.
        invalidate_edge_stats();
    } else {
        // Optimization 2: Skip UserGraph restoration in Tier 1 mode
        // UserGraph was never modified, so no need to restore it.
        // Write the snapshotted discretized weights back verbatim.
        for (const auto& rw : _active_reweights) {
            write_reweight_slots(rw, rw.original_normalized_weight);
        }

        // DO NOT set _mwpm_needs_updating = true
    }

    // Reset state BEFORE materialising: the graph is fully restored at this
    // point, so even if the rebuild below throws (e.g. bad_alloc), a repeated
    // restore is a clean no-op rather than a replay.
    _active_reweights.clear();
    _active_reweights_regen = false;

    if (regen) {
        // Materialise the regeneration back to the original graph, preserving
        // the current shape (search graph present iff it is present now), so a
        // following block or decode -- including one that applies no reweights
        // -- sees the restored graph without needing a get_mwpm() refresh.
        bool has_search_graph = _mwpm.flooder.graph.nodes.size() > 0 &&
                                _mwpm.search_flooder.graph.nodes.size() == _mwpm.flooder.graph.nodes.size();
        (void)(has_search_graph ? get_mwpm_with_search_graph() : get_mwpm());
    }
}

bool pm::UserGraph::all_edges_integral() const {
    return edge_stats().all_integral;
}

bool pm::UserGraph::needs_regeneration(const std::vector<std::array<double, 3>>& reweight_specs) {
    // Compare against the SAME maximum the normalising constant is sized by:
    // get_edge_weight_normalising_constant takes the max over edges AND implied
    // correlation weights. Using the edge-only max would needlessly classify
    // reweights in (edge_max, implied_max] as Tier-2 (two full rebuilds per
    // decode) even though an in-place write at the actual constant is exact.
    double original_max_abs_weight = max_abs_weight_including_implied();
    double max_new_abs_weight = original_max_abs_weight;

    // When the current graph is all-integral, get_edge_weight_normalising_constant
    // collapses to 1.0, so the matching graph is discretized at integer resolution.
    // An in-place (Tier-1) reweight to a non-integer value would then be silently
    // rounded to the nearest integer (e.g. 0.1 -> a free edge). Force a full
    // regeneration in that case: the reweighted, now non-integral edge makes
    // all_integral_weight false, rescaling the constant to fine resolution.
    bool graph_all_integral = all_edges_integral();

    // Returns true iff node u has an edge to a boundary NODE (as opposed to the
    // virtual boundary): such edges min-merge with u's explicit boundary edge
    // into a single discretized boundary slot at build time.
    auto has_boundary_node_neighbor = [&](size_t u) {
        for (const auto& neighbor : nodes[u].neighbors) {
            size_t other = neighbor.pos == 0 ? neighbor.edge_it->node1 : neighbor.edge_it->node2;
            if (other != SIZE_MAX && nodes[other].is_boundary)
                return true;
        }
        return false;
    };

    for (const auto& spec : reweight_specs) {
        double new_weight = spec[2];
        if (graph_all_integral && round(new_weight) != new_weight)
            return true;
        // Regeneration is also required when a reweight exceeds the original max,
        // which changes the normalising constant.
        max_new_abs_weight = std::max(max_new_abs_weight, std::abs(new_weight));

        // Slot-ambiguity rules: force regeneration when the spec touches an edge
        // with no dedicated in-place slot, where a Tier-1 write would silently
        // no-op or corrupt shared state. Regeneration re-derives the discretized
        // graph from the UserGraph (including boundary min-merging), so it
        // handles all of these exactly. Node values are interpreted only when
        // castable; malformed specs are left for apply_reweights to reject.
        double n1_raw = spec[0], n2_raw = spec[1];
        bool n1_castable =
            std::isfinite(n1_raw) && n1_raw >= 0 && round(n1_raw) == n1_raw && n1_raw < (double)nodes.size();
        if (!n1_castable)
            continue;
        size_t node1 = (size_t)n1_raw;
        if (n2_raw == -1.0) {
            // Explicit boundary edge (u,-1): its discretized slot is shared with
            // any edge (u,v) where v is a boundary node (min-merged), and it has
            // no slot at all if u is itself a boundary node.
            if (nodes[node1].is_boundary || has_boundary_node_neighbor(node1))
                return true;
        } else if (
            std::isfinite(n2_raw) && n2_raw >= 0 && round(n2_raw) == n2_raw && n2_raw < (double)nodes.size()) {
            // Edge (u,v): if either endpoint is a boundary node, the edge is
            // stored as a nullptr boundary slot at the other endpoint (or not at
            // all), which the Tier-1 pointer lookup cannot address.
            size_t node2 = (size_t)n2_raw;
            if (nodes[node1].is_boundary || nodes[node2].is_boundary)
                return true;
        }
    }

    return max_new_abs_weight > original_max_abs_weight;
}

void pm::UserGraph::write_reweight_slots(const EdgeReweight& rw, pm::weight_int value) {
    // A SIZE_MAX index means the slot does not exist (boundary edge reverse
    // direction, edge absent from the graph, or the graph itself is absent).
    if (rw.matching_graph_node1_neighbor_idx != SIZE_MAX) {
        _mwpm.flooder.graph.nodes[rw.node1].neighbor_weights[rw.matching_graph_node1_neighbor_idx] = value;
    }
    if (rw.node2 != SIZE_MAX && rw.matching_graph_node2_neighbor_idx != SIZE_MAX) {
        _mwpm.flooder.graph.nodes[rw.node2].neighbor_weights[rw.matching_graph_node2_neighbor_idx] = value;
    }
    if (rw.search_graph_node1_neighbor_idx != SIZE_MAX) {
        _mwpm.search_flooder.graph.nodes[rw.node1].neighbor_weights[rw.search_graph_node1_neighbor_idx] = value;
    }
    if (rw.node2 != SIZE_MAX && rw.search_graph_node2_neighbor_idx != SIZE_MAX) {
        _mwpm.search_flooder.graph.nodes[rw.node2].neighbor_weights[rw.search_graph_node2_neighbor_idx] = value;
    }
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
    // Implied weights feed the cached edge stats.
    invalidate_edge_stats();
}
