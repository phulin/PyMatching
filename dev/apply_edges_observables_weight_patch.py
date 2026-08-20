from pathlib import Path


def insert_once(text: str, needle: str, insertion: str, *, before: bool = True) -> str:
    if text.count(needle) != 1:
        raise SystemExit(f"expected exactly one occurrence of {needle!r}, found {text.count(needle)}")
    return text.replace(needle, insertion + needle if before else needle + insertion)


# Public C++ declaration.
h = Path("src/pymatching/sparse_blossom/driver/mwpm_decoding.h")
text = h.read_text()
needle = "void decode_detection_events_to_edges_with_edge_correlations(\n"
decl = """void decode_detection_events_to_edges_observables_weight(\n    pm::Mwpm& mwpm,\n    const std::vector<uint64_t>& detection_events,\n    std::vector<int64_t>& edges,\n    uint8_t* obs_begin_ptr,\n    pm::total_weight_int& weight);\n\n"""
if "decode_detection_events_to_edges_observables_weight" not in text:
    text = insert_once(text, needle, decl)
    h.write_text(text)


# Refactor edge extraction into a shared single-solve core that can also
# accumulate the exact observable vector and matching objective weight.
cc = Path("src/pymatching/sparse_blossom/driver/mwpm_decoding.cc")
text = cc.read_text()
start_marker = "void pm::decode_detection_events_to_edges(\n"
end_marker = "\nvoid pm::decode_detection_events_to_edges_with_edge_correlations(\n"
if "decode_detection_events_to_edges_observables_weight" not in text:
    start = text.index(start_marker)
    end = text.index(end_marker, start)
    new_block = r'''static void decode_detection_events_to_edges_core(
    pm::Mwpm& mwpm,
    const std::vector<uint64_t>& detection_events,
    std::vector<int64_t>& edges,
    uint8_t* obs_begin_ptr,
    pm::total_weight_int* weight_ptr) {
    if (mwpm.flooder.graph.nodes.size() != mwpm.search_flooder.graph.nodes.size()) {
        throw std::invalid_argument(
            "Mwpm object does not contain search flooder, which is required to decode to edges.");
    }
    process_timeline_until_completion(mwpm, detection_events);
    mwpm.flooder.match_edges.clear();
    shatter_blossoms_for_all_detection_events_and_extract_match_edges(mwpm, detection_events);
    if (!mwpm.flooder.negative_weight_detection_events.empty())
        shatter_blossoms_for_all_detection_events_and_extract_match_edges(
            mwpm, mwpm.flooder.negative_weight_detection_events);

    if (obs_begin_ptr != nullptr) {
        for (auto obs : mwpm.flooder.negative_weight_observables) {
            obs_begin_ptr[obs] ^= 1;
        }
        *weight_ptr += mwpm.flooder.negative_weight_sum;
    }

    // Flip edges with negative weights and add to edges vector.
    for (const auto& neg_node_pair : mwpm.search_flooder.graph.negative_weight_edges) {
        auto node1_ptr = &mwpm.search_flooder.graph.nodes[neg_node_pair.first];
        auto node2_ptr =
            neg_node_pair.second != SIZE_MAX ? &mwpm.search_flooder.graph.nodes[neg_node_pair.second] : nullptr;
        SearchGraphEdge neg_edge = {node1_ptr, node1_ptr->index_of_neighbor(node2_ptr)};
        flip_edge(neg_edge);
        int64_t node1 = neg_edge.detector_node - &mwpm.search_flooder.graph.nodes[0];
        int64_t node2 = node2_ptr ? node2_ptr - &mwpm.search_flooder.graph.nodes[0] : -1;
        edges.push_back(node1);
        edges.push_back(node2);
    }

    // Walk each matched shortest path once. Besides building the correction
    // edge set, optionally accumulate the same observable flips and exact path
    // weight returned by decode(). This removes the second sparse-blossom solve
    // that callers previously needed to obtain both products.
    for (const auto& match_edge : mwpm.flooder.match_edges) {
        size_t node_from = match_edge.loc_from - &mwpm.flooder.graph.nodes[0];
        size_t node_to = match_edge.loc_to ? match_edge.loc_to - &mwpm.flooder.graph.nodes[0] : SIZE_MAX;
        mwpm.search_flooder.iter_edges_on_shortest_path_from_middle(node_from, node_to, [&](const SearchGraphEdge& e) {
            flip_edge(e);
            int64_t node1 = e.detector_node - &mwpm.search_flooder.graph.nodes[0];
            auto node2_ptr = e.detector_node->neighbors[e.neighbor_index];
            int64_t node2 = node2_ptr ? node2_ptr - &mwpm.search_flooder.graph.nodes[0] : -1;
            edges.push_back(node1);
            edges.push_back(node2);
            if (obs_begin_ptr != nullptr) {
                for (auto obs : e.detector_node->neighbor_observable_indices[e.neighbor_index]) {
                    obs_begin_ptr[obs] ^= 1;
                }
                *weight_ptr += e.detector_node->neighbor_weights[e.neighbor_index];
            }
        });
    }

    // Remove edges occurring an even number of times, and clear all temporary
    // FLIPPED markers before returning.
    for (size_t i = 0; i < edges.size() / 2;) {
        int64_t u = edges[2 * i];
        int64_t v = edges[2 * i + 1];
        auto& u_node = mwpm.search_flooder.graph.nodes[u];
        size_t idx;
        if (v == -1) {
            idx = 0;
        } else {
            auto v_ptr = &mwpm.search_flooder.graph.nodes[v];
            idx = u_node.index_of_neighbor(v_ptr);
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
}

void pm::decode_detection_events_to_edges(
    pm::Mwpm& mwpm, const std::vector<uint64_t>& detection_events, std::vector<int64_t>& edges) {
    decode_detection_events_to_edges_core(mwpm, detection_events, edges, nullptr, nullptr);
}

void pm::decode_detection_events_to_edges_observables_weight(
    pm::Mwpm& mwpm,
    const std::vector<uint64_t>& detection_events,
    std::vector<int64_t>& edges,
    uint8_t* obs_begin_ptr,
    pm::total_weight_int& weight) {
    decode_detection_events_to_edges_core(mwpm, detection_events, edges, obs_begin_ptr, &weight);
}
'''
    cc.write_text(text[:start] + new_block + text[end:])


# Native Python binding, including the existing hardened temporary-reweight
# transaction from the Allenator fork.
pybind = Path("src/pymatching/sparse_blossom/driver/user_graph.pybind.cc")
text = pybind.read_text()
needle = '''    g.def(\n        "decode_to_matched_detection_events_array",'''
if "decode_to_edges_array_with_observables_and_weight" not in text:
    block = r'''    g.def(
        "decode_to_edges_array_with_observables_and_weight",
        [](pm::UserGraph &self,
           const py::array_t<uint64_t> &detection_events,
           py::object edge_reweights = py::none()) -> py::tuple {
            auto &mwpm = self.get_mwpm_with_search_graph();
            bool has_reweights = !edge_reweights.is_none();
            std::vector<std::array<double, 3>> reweight_specs;
            if (has_reweights) {
                py::array_t<double> reweights_array = edge_reweights.cast<py::array_t<double>>();
                validate_reweights_array(reweights_array);
                auto reweights_unchecked = reweights_array.unchecked<2>();
                for (py::ssize_t i = 0; i < reweights_unchecked.shape(0); i++) {
                    reweight_specs.push_back({
                        reweights_unchecked(i, 0),
                        reweights_unchecked(i, 1),
                        reweights_unchecked(i, 2)
                    });
                }
                has_reweights = !reweight_specs.empty();
            }

            try {
                if (has_reweights) {
                    auto parsed = self.parse_reweight_specs(reweight_specs);
                    bool regen = self.needs_regeneration(parsed);
                    self.apply_reweights(std::move(parsed), regen, /*ensure_search_graph=*/true);
                }

                std::vector<uint64_t> detection_events_vec(
                    detection_events.data(), detection_events.data() + detection_events.size());
                auto edges = std::make_unique<std::vector<int64_t>>();
                edges->reserve(detection_events_vec.size() / 2);
                auto observables = std::make_unique<std::vector<uint8_t>>(self.get_num_observables(), 0);
                pm::total_weight_int weight = 0;
                pm::decode_detection_events_to_edges_observables_weight(
                    mwpm, detection_events_vec, *edges, observables->data(), weight);
                double rescaled_weight = (double)weight / mwpm.flooder.graph.normalising_constant;

                if (has_reweights) {
                    self.restore_weights();
                }

                auto num_edges = edges->size() / 2;
                auto edges_arr = pm_pybind::vec_to_array<int64_t>(edges.release());
                edges_arr.resize({(py::ssize_t)num_edges, (py::ssize_t)2});

                auto *obs_raw = observables.release();
                auto obs_capsule = py::capsule(obs_raw, [](void *x) {
                    delete reinterpret_cast<std::vector<uint8_t> *>(x);
                });
                py::array_t<uint8_t> obs_arr =
                    py::array_t<uint8_t>(obs_raw->size(), obs_raw->data(), obs_capsule);
                return py::make_tuple(edges_arr, obs_arr, rescaled_weight);
            } catch (...) {
                if (has_reweights) {
                    self.restore_weights();
                }
                throw;
            }
        },
        "detection_events"_a,
        "edge_reweights"_a = py::none());
'''
    pybind.write_text(insert_once(text, needle, block))


# High-level Matching wrapper.
matching = Path("src/pymatching/matching.py")
text = matching.read_text()
needle = "    def decode_to_matched_dets_array(self,\n"
if "def decode_to_edges_array_with_observables_and_weight" not in text:
    method = '''    def decode_to_edges_array_with_observables_and_weight(\n            self,\n            syndrome: Union[np.ndarray, List[bool], List[int]],\n            *,\n            edge_reweights: np.ndarray = None):\n        """Decode once and return (correction edges, observables, exact weight)."""\n        detection_events = self._syndrome_array_to_detection_events(syndrome)\n        return self._matching_graph.decode_to_edges_array_with_observables_and_weight(\n            detection_events, edge_reweights=edge_reweights)\n\n'''
    matching.write_text(insert_once(text, needle, method))

print("patched one-solve edge/observable/weight API")
