/*
 * cpp_ext/rule_miner.cpp
 * ======================
 * C++ / pybind11 extension that accelerates Horn rule mining from a
 * knowledge graph.
 *
 * Implements:
 *   mine_rules(triples, num_relations, min_support, min_confidence,
 *              max_rules, max_rule_len)
 *
 * The function returns (rules, weights) compatible with the Python
 * interface in mln_builder.py.
 *
 * Build:
 *   cd cpp_ext && python setup.py build_ext --inplace
 *
 * Requires:
 *   pybind11  (pip install pybind11)
 *   A C++14 (or later) compiler
 *
 * Design notes
 * ------------
 * The bottleneck in the Python rule miner is the O(R^2) loop over
 * relation pairs and the O(|pairs_r1| * |pairs_r2|) set intersection.
 * We speed this up by:
 *
 *   1. Building rel -> sorted-pair-set mappings in C++.
 *   2. Using std::set_intersection for overlap counting (O(n log n)
 *      instead of Python set operations).
 *   3. For 2-hop rules, enumerating chains via an entity-indexed
 *      reverse mapping in C++.
 *   4. Parallelising the outer relation loop with OpenMP (optional).
 */

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

// -------------------------------------------------------------------------
// Type aliases
// -------------------------------------------------------------------------

using Pair      = std::pair<int32_t, int32_t>;   // (head, tail)
using PairSet   = std::vector<Pair>;              // sorted vector for intersection
using RelPairs  = std::unordered_map<int32_t, PairSet>;

// A rule: (head_rel, {body_rel_1, body_rel_2, ...})
struct Rule {
    int32_t              head_rel;
    std::vector<int32_t> body_rels;
    double               confidence{0.0};
};

// -------------------------------------------------------------------------
// Build rel -> sorted pairs mapping
// -------------------------------------------------------------------------

RelPairs build_rel_pairs(py::array_t<int32_t> triples) {
    auto buf = triples.unchecked<2>();
    RelPairs rp;

    for (py::ssize_t i = 0; i < buf.shape(0); ++i) {
        int32_t h = buf(i, 0);
        int32_t r = buf(i, 1);
        int32_t t = buf(i, 2);
        rp[r].emplace_back(h, t);
    }

    // Sort each list for std::set_intersection
    for (auto& [rel, pairs] : rp) {
        std::sort(pairs.begin(), pairs.end());
        pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());
    }

    return rp;
}

// -------------------------------------------------------------------------
// Count overlap between two sorted pair vectors
// -------------------------------------------------------------------------

static size_t count_overlap(const PairSet& a, const PairSet& b) {
    std::vector<Pair> inter;
    inter.reserve(std::min(a.size(), b.size()));
    std::set_intersection(a.begin(), a.end(),
                          b.begin(), b.end(),
                          std::back_inserter(inter));
    return inter.size();
}

// -------------------------------------------------------------------------
// Main rule mining function
// -------------------------------------------------------------------------

/*
 * mine_rules
 * ----------
 * Parameters
 *   triples        : numpy int32 array of shape (N, 3)  [h, r, t]
 *   num_relations  : total number of distinct relations
 *   min_support    : minimum overlap count
 *   min_confidence : minimum confidence in [0,1]
 *   max_rules      : stop after this many rules
 *   max_rule_len   : maximum rule body length (1 or 2)
 *
 * Returns
 *   (rules, weights)
 *   rules   : list of [head_rel, body_rel_1, (body_rel_2)]
 *   weights : list of confidence values (doubles)
 */
std::tuple<std::vector<std::vector<int32_t>>, std::vector<double>>
mine_rules(py::array_t<int32_t> triples,
           int32_t               num_relations,
           int32_t               min_support,
           double                min_confidence,
           int32_t               max_rules,
           int32_t               max_rule_len)
{
    RelPairs rel_pairs = build_rel_pairs(triples);

    // Sort relations by frequency (descending)
    std::vector<std::pair<int32_t, size_t>> rel_freq;
    rel_freq.reserve(rel_pairs.size());
    for (const auto& [r, pairs] : rel_pairs)
        rel_freq.emplace_back(r, pairs.size());
    std::sort(rel_freq.begin(), rel_freq.end(),
              [](const auto& a, const auto& b){ return a.second > b.second; });

    std::vector<int32_t> ordered_rels;
    ordered_rels.reserve(rel_freq.size());
    for (const auto& [r, _] : rel_freq)
        ordered_rels.push_back(r);

    std::vector<Rule>  found_rules;
    auto start = std::chrono::steady_clock::now();

    // ---- 1-hop rules ----
    for (int32_t head_rel : ordered_rels) {
        if (static_cast<int32_t>(found_rules.size()) >= max_rules) break;

        const PairSet& head_pairs = rel_pairs[head_rel];
        if (static_cast<int32_t>(head_pairs.size()) < min_support) continue;

        for (int32_t body_rel : ordered_rels) {
            if (body_rel == head_rel) continue;

            const PairSet& body_pairs = rel_pairs[body_rel];
            if (static_cast<int32_t>(body_pairs.size()) < min_support) continue;

            size_t overlap    = count_overlap(head_pairs, body_pairs);
            double confidence = static_cast<double>(overlap) /
                                static_cast<double>(body_pairs.size());

            if (confidence >= min_confidence &&
                static_cast<int32_t>(overlap) >= min_support) {
                Rule r;
                r.head_rel  = head_rel;
                r.body_rels = {body_rel};
                r.confidence = confidence;
                found_rules.push_back(r);

                if (static_cast<int32_t>(found_rules.size()) >= max_rules)
                    break;
            }
        }
    }

    // ---- 2-hop chain rules ----
    if (max_rule_len >= 2 &&
        static_cast<int32_t>(found_rules.size()) < max_rules)
    {
        // Build entity -> [(r, tail)] mapping for chain enumeration
        using EntityRelTail = std::unordered_map<int32_t,
                              std::vector<std::pair<int32_t,int32_t>>>;
        EntityRelTail head_to_rt;  // head entity -> list of (r, tail)
        {
            auto buf = triples.unchecked<2>();
            for (py::ssize_t i = 0; i < buf.shape(0); ++i)
                head_to_rt[buf(i,0)].emplace_back(buf(i,1), buf(i,2));
        }

        // For top relations only (to limit time)
        size_t top_n = std::min(static_cast<size_t>(30), ordered_rels.size());
        auto   top   = std::vector<int32_t>(
                           ordered_rels.begin(),
                           ordered_rels.begin() + top_n);

        for (int32_t head_rel : top) {
            if (static_cast<int32_t>(found_rules.size()) >= max_rules) break;

            const PairSet& head_pairs = rel_pairs[head_rel];
            if (static_cast<int32_t>(head_pairs.size()) < min_support) continue;

            // Count chains (r1, r2) -> head_rel
            std::map<Pair, int32_t> chain_counts;
            size_t max_pairs_sample = std::min(head_pairs.size(),
                                                static_cast<size_t>(500));
            for (size_t pi = 0; pi < max_pairs_sample; ++pi) {
                auto [h, t] = head_pairs[pi];
                // Find mid nodes reachable from h
                auto it = head_to_rt.find(h);
                if (it == head_to_rt.end()) continue;
                for (auto& [r1, mid] : it->second) {
                    if (r1 == head_rel) continue;
                    // Check if mid can reach t via some r2
                    auto it2 = head_to_rt.find(mid);
                    if (it2 == head_to_rt.end()) continue;
                    for (auto& [r2, t2] : it2->second) {
                        if (r2 == head_rel) continue;
                        if (t2 == t)
                            chain_counts[{r1, r2}]++;
                    }
                }
            }

            // Convert chain counts to rules
            for (auto& [chain, cnt] : chain_counts) {
                if (cnt < min_support) continue;
                auto [r1, r2] = chain;
                if (r1 == r2) continue;
                size_t body_size = std::min(rel_pairs[r1].size(),
                                            rel_pairs[r2].size());
                if (body_size == 0) continue;
                double conf = static_cast<double>(cnt) /
                              static_cast<double>(body_size);
                if (conf >= min_confidence) {
                    Rule ru;
                    ru.head_rel   = head_rel;
                    ru.body_rels  = {r1, r2};
                    ru.confidence = conf;
                    found_rules.push_back(ru);
                    if (static_cast<int32_t>(found_rules.size()) >= max_rules)
                        break;
                }
            }
        }
    }

    // Sort by confidence descending
    std::sort(found_rules.begin(), found_rules.end(),
              [](const Rule& a, const Rule& b){
                  return a.confidence > b.confidence;
              });

    // Serialise to Python lists
    std::vector<std::vector<int32_t>> out_rules;
    std::vector<double>               out_weights;

    for (const Rule& r : found_rules) {
        std::vector<int32_t> row = {r.head_rel};
        row.insert(row.end(), r.body_rels.begin(), r.body_rels.end());
        out_rules.push_back(row);
        out_weights.push_back(r.confidence);
    }

    return {out_rules, out_weights};
}

// -------------------------------------------------------------------------
// pybind11 module definition
// -------------------------------------------------------------------------

PYBIND11_MODULE(rule_miner, m) {
    m.doc() = "C++ accelerated Horn rule miner for knowledge graphs";
    m.def("mine_rules", &mine_rules,
          py::arg("triples"),
          py::arg("num_relations"),
          py::arg("min_support")    = 3,
          py::arg("min_confidence") = 0.2,
          py::arg("max_rules")      = 200,
          py::arg("max_rule_len")   = 2,
          R"doc(
Mine Horn rules from a knowledge graph triple array.

Parameters
----------
triples        : numpy int32 array (N, 3) – columns are [head, relation, tail]
num_relations  : int – total number of distinct relation types
min_support    : int – minimum grounding count for a rule
min_confidence : float – minimum confidence (overlap / body_size)
max_rules      : int – stop after this many rules are found
max_rule_len   : int – maximum rule body length (1 or 2)

Returns
-------
(rules, weights)
rules   : list[list[int]]  – each inner list is [head_rel, body_rel_1, ...]
weights : list[float]      – confidence of the corresponding rule
)doc");
}
