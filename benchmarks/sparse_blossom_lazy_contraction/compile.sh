#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 CHECKOUT OUTPUT [baseline|lazy]" >&2
    exit 2
fi

root=$(cd "$1" && pwd)
out=$2
mode=${3:-lazy}
defines=()
case "$mode" in
    baseline) ;;
    lazy) defines+=(-DPM_LAZY_TOP) ;;
    *) echo "mode must be baseline or lazy" >&2; exit 2 ;;
esac

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
for i in "${!sources[@]}"; do
    sources[$i]="$root/${sources[$i]}"
done

g++ -std=c++20 -O3 -DNDEBUG -march=native \
    "${defines[@]}" \
    -I"$root/src" \
    "$(dirname "$0")/benchmark.cc" \
    "${sources[@]}" \
    -o "$out"
