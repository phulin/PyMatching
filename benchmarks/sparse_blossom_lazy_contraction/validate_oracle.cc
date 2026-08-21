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
#include <array>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pymatching/sparse_blossom/matcher/mwpm.h"

static constexpr int OBS = 2;
static constexpr int MASKS = 1 << OBS;
static constexpr int64_t INF = std::numeric_limits<int64_t>::max() / 8;

struct SimpleEdge { size_t u, v; int w, mask; };
struct BoundaryEdge { size_t u; int w, mask; };

static std::vector<size_t> obs_vec(int mask) {
    std::vector<size_t> r;
    for (int k = 0; k < OBS; k++) if ((mask >> k) & 1) r.push_back(k);
    return r;
}

static pm::GraphFillRegion *node_top(pm::DetectorNode &node) {
#ifdef PM_LAZY_TOP
    return node.top_region();
#else
    return node.region_that_arrived_top;
#endif
}

struct Outcome { bool success; int64_t weight; int mask; std::string error; };

static Outcome decode(pm::MatchingGraph graph, const std::vector<size_t> &syndrome) {
    try {
        pm::Mwpm mwpm(pm::GraphFlooder(std::move(graph)));
        for (size_t d : syndrome) mwpm.create_detection_event(&mwpm.flooder.graph.nodes[d]);
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
        return {true, result.weight, (int)result.obs_mask, {}};
    } catch (const std::exception &ex) {
        return {false, 0, 0, ex.what()};
    }
}

static std::array<int64_t, MASKS> shortest_between(
    size_t n,
    const std::vector<SimpleEdge> &edges,
    size_t src,
    size_t dst) {
    std::vector<std::vector<std::tuple<size_t,int,int>>> adj(n);
    for (auto &e : edges) {
        adj[e.u].push_back({e.v,e.w,e.mask});
        adj[e.v].push_back({e.u,e.w,e.mask});
    }
    std::vector<std::array<int64_t,MASKS>> dist(n);
    for (auto &a : dist) a.fill(INF);
    using Q = std::tuple<int64_t,size_t,int>;
    std::priority_queue<Q,std::vector<Q>,std::greater<Q>> pq;
    dist[src][0]=0; pq.push({0,src,0});
    while (!pq.empty()) {
        auto [d,u,m]=pq.top(); pq.pop();
        if (d!=dist[u][m]) continue;
        for (auto [v,w,em] : adj[u]) {
            int nm=m^em; int64_t nd=d+w;
            if (nd<dist[v][nm]) { dist[v][nm]=nd; pq.push({nd,v,nm}); }
        }
    }
    return dist[dst];
}

static std::array<int64_t, MASKS> shortest_to_boundary(
    size_t n,
    const std::vector<SimpleEdge> &edges,
    const std::vector<BoundaryEdge> &boundaries,
    size_t src) {
    std::vector<std::vector<std::tuple<size_t,int,int>>> adj(n);
    for (auto &e : edges) {
        adj[e.u].push_back({e.v,e.w,e.mask});
        adj[e.v].push_back({e.u,e.w,e.mask});
    }
    std::vector<std::vector<std::pair<int,int>>> b(n);
    for (auto &e : boundaries) b[e.u].push_back({e.w,e.mask});
    std::vector<std::array<int64_t,MASKS>> dist(n);
    for (auto &a : dist) a.fill(INF);
    std::array<int64_t,MASKS> ans; ans.fill(INF);
    using Q = std::tuple<int64_t,size_t,int>;
    std::priority_queue<Q,std::vector<Q>,std::greater<Q>> pq;
    dist[src][0]=0; pq.push({0,src,0});
    while (!pq.empty()) {
        auto [d,u,m]=pq.top(); pq.pop();
        if (d!=dist[u][m]) continue;
        for (auto [w,bm] : b[u]) ans[m^bm]=std::min(ans[m^bm],d+w);
        for (auto [v,w,em] : adj[u]) {
            int nm=m^em; int64_t nd=d+w;
            if (nd<dist[v][nm]) { dist[v][nm]=nd; pq.push({nd,v,nm}); }
        }
    }
    return ans;
}

static std::array<int64_t,MASKS> oracle(
    size_t n,
    const std::vector<SimpleEdge> &edges,
    const std::vector<BoundaryEdge> &boundaries,
    const std::vector<size_t> &syndrome) {
    size_t k=syndrome.size();
    std::vector<std::vector<std::array<int64_t,MASKS>>> pair(
        k, std::vector<std::array<int64_t,MASKS>>(k));
    std::vector<std::array<int64_t,MASKS>> bcost(k);
    for (size_t i=0;i<k;i++) {
        bcost[i]=shortest_to_boundary(n,edges,boundaries,syndrome[i]);
        for (size_t j=i+1;j<k;j++) pair[i][j]=shortest_between(n,edges,syndrome[i],syndrome[j]);
    }
    size_t states=1ULL<<k;
    std::vector<std::array<int64_t,MASKS>> dp(states);
    for (auto &a:dp) a.fill(INF);
    dp[0][0]=0;
    for (size_t s=1;s<states;s++) {
        size_t i=__builtin_ctzll(s);
        size_t rest=s^(1ULL<<i);
        for (int tail=0;tail<MASKS;tail++) if (dp[rest][tail]<INF) {
            for (int em=0;em<MASKS;em++) if (bcost[i][em]<INF)
                dp[s][tail^em]=std::min(dp[s][tail^em],dp[rest][tail]+bcost[i][em]);
        }
        for (size_t j=i+1;j<k;j++) if ((s>>j)&1) {
            size_t rest2=rest^(1ULL<<j);
            for (int tail=0;tail<MASKS;tail++) if (dp[rest2][tail]<INF) {
                for (int em=0;em<MASKS;em++) if (pair[i][j][em]<INF)
                    dp[s][tail^em]=std::min(dp[s][tail^em],dp[rest2][tail]+pair[i][j][em]);
            }
        }
    }
    return dp.back();
}

int main(int argc,char **argv) {
    if (argc!=3) { std::cerr<<"usage: oracle SEED CASES\n"; return 2; }
    uint64_t seed=std::strtoull(argv[1],nullptr,10);
    size_t cases=std::strtoull(argv[2],nullptr,10);
    std::mt19937_64 rng(seed);
    size_t successes=0, failures=0, tied_masks=0;
    for (size_t cid=0;cid<cases;cid++) {
        size_t n=3+rng()%8; // at most 10 syndrome bits -> exact subset DP is small
        bool zero_mode=(cid&1)!=0;
        std::vector<SimpleEdge> edges;
        std::vector<BoundaryEdge> boundaries;
        std::set<std::pair<size_t,size_t>> used;
        auto weight=[&](){ if (zero_mode && rng()%4==0) return 0; return 2*(1+(int)(rng()%6)); };
        auto add=[&](size_t u,size_t v){
            if(u==v)return; if(u>v)std::swap(u,v); if(!used.insert({u,v}).second)return;
            edges.push_back({u,v,weight(),(int)(rng()%MASKS)});
        };
        for(size_t u=1;u<n;u++) add(u-1,u);
        for(size_t z=0;z<n+rng()%(2*n+1);z++) add(rng()%n,rng()%n);
        if(rng()%4!=0) {
            std::set<size_t> bn;
            size_t c=1+rng()%std::max<size_t>(1,n/3);
            while(bn.size()<c)bn.insert(rng()%n);
            for(size_t u:bn) boundaries.push_back({u,weight(),(int)(rng()%MASKS)});
        }
        pm::MatchingGraph graph(n,OBS);
        for(auto &e:edges) graph.add_edge(e.u,e.v,e.w,obs_vec(e.mask),{});
        for(auto &e:boundaries) graph.add_boundary_edge(e.u,e.w,obs_vec(e.mask),{});
        std::vector<size_t> syndrome;
        std::bernoulli_distribution hit(0.15+(rng()%70)/100.0);
        for(size_t u=0;u<n;u++)if(hit(rng))syndrome.push_back(u);
        if(syndrome.empty())syndrome.push_back(rng()%n);

        auto expected=oracle(n,edges,boundaries,syndrome);
        int64_t best=*std::min_element(expected.begin(),expected.end());
        auto got=decode(std::move(graph),syndrome);
        if(best>=INF) {
            if(got.success) {
                std::cerr<<"case "<<cid<<": decoder succeeded but oracle has no matching, got "<<got.weight<<" mask "<<got.mask<<"\n";
                return 1;
            }
            failures++;
        } else {
            if(!got.success) {
                std::cerr<<"case "<<cid<<": decoder failed but oracle optimum is "<<best<<"; "<<got.error<<"\n";
                return 1;
            }
            if(got.weight!=best || got.mask<0 || got.mask>=MASKS || expected[got.mask]!=got.weight) {
                std::cerr<<"case "<<cid<<": got weight/mask "<<got.weight<<'/'<<got.mask<<", best "<<best
                         <<", mask optimum "<<(got.mask>=0&&got.mask<MASKS?expected[got.mask]:-1)<<"\n";
                return 1;
            }
            int nm=0; for(int m=0;m<MASKS;m++) if(expected[m]==best) nm++;
            if(nm>1)tied_masks++;
            successes++;
        }
    }
    std::cout<<"oracle_cases="<<cases<<" successes="<<successes<<" failures="<<failures<<" tied_optima="<<tied_masks<<"\n";
}
