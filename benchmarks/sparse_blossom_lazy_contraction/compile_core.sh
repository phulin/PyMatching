#!/usr/bin/env bash
set -euo pipefail
root=$(cd "$1" && pwd)
out=$2
src=$3
shift 3
sources=(
    src/pymatching/sparse_blossom/flooder/graph.cc
    src/pymatching/sparse_blossom/flooder/detector_node.cc
    src/pymatching/sparse_blossom/flooder_matcher_interop/compressed_edge.cc
    src/pymatching/sparse_blossom/flooder/graph_fill_region.cc
    src/pymatching/sparse_blossom/flooder/match.cc
    src/pymatching/sparse_blossom/flooder/graph_flooder.cc
    src/pymatching/sparse_blossom/matcher/alternating_tree.cc
    src/pymatching/sparse_blossom/matcher/mwpm.cc
    src/pymatching/sparse_blossom/flooder_matcher_interop/region_edge.cc
    src/pymatching/sparse_blossom/flooder_matcher_interop/mwpm_event.cc
    src/pymatching/sparse_blossom/tracker/flood_check_event.cc
    src/pymatching/sparse_blossom/search/search_graph.cc
    src/pymatching/sparse_blossom/search/search_detector_node.cc
    src/pymatching/sparse_blossom/search/search_flooder.cc
)
for i in "${!sources[@]}"; do sources[$i]="$root/${sources[$i]}"; done
g++ -std=c++20 -I"$root/src" "$@" "$src" "${sources[@]}" -o "$out"
