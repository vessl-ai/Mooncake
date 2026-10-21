#pragma once

// Placement keys (fork-only, master flag --placement_key_hrw, default off).
//
// A put may name a placement key instead of a segment: an entry
// "placement-key:<key>" in ReplicateConfig::preferred_segments. With the flag
// on, the master replaces that entry by every serving memory segment ranked
// by rendezvous (highest random weight, HRW) hash of <key> and the segment
// name. Puts that share a key then land on the same segment while it has
// room, spill to the next one in the same order when it does not, and when a
// segment joins or leaves only the keys ranked first on it move. With the
// flag off the entry is an ordinary preferred segment name that no segment
// has, so the allocator skips it and places the put as before. No wire
// format changes: the key travels in an existing string field.

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace mooncake {

inline constexpr std::string_view kPlacementKeyPrefix = "placement-key:";

// The key named by a preferred_segments entry, or nullopt when the entry
// names a segment. An empty key is not a placement key.
inline std::optional<std::string_view> ParsePlacementKey(
    std::string_view entry) {
    if (entry.size() <= kPlacementKeyPrefix.size() ||
        entry.substr(0, kPlacementKeyPrefix.size()) != kPlacementKeyPrefix) {
        return std::nullopt;
    }
    return entry.substr(kPlacementKeyPrefix.size());
}

// HRW weight of `segment` for `key`. Spelled out (FNV-1a 64 over key, a zero
// byte and the segment name, then the splitmix64 finalizer) rather than
// std::hash, so the ranking is the same across builds and master restarts.
inline uint64_t PlacementWeight(std::string_view key,
                                std::string_view segment) {
    uint64_t h = 0xcbf29ce484222325ULL;
    auto mix = [&h](unsigned char c) {
        h ^= c;
        h *= 0x100000001b3ULL;
    };
    for (unsigned char c : key) mix(c);
    mix(0);
    for (unsigned char c : segment) mix(c);
    h ^= h >> 30;
    h *= 0xbf58476d1ce4e5b9ULL;
    h ^= h >> 27;
    h *= 0x94d049bb133111ebULL;
    h ^= h >> 31;
    return h;
}

// `segments` ordered by descending HRW weight for `key` (ties by name, so the
// order is total).
inline std::vector<std::string> RankSegmentsForPlacementKey(
    std::string_view key, const std::vector<std::string>& segments) {
    std::vector<std::pair<uint64_t, const std::string*>> weighted;
    weighted.reserve(segments.size());
    for (const auto& segment : segments) {
        weighted.emplace_back(PlacementWeight(key, segment), &segment);
    }
    std::sort(weighted.begin(), weighted.end(),
              [](const auto& a, const auto& b) {
                  if (a.first != b.first) return a.first > b.first;
                  return *a.second < *b.second;
              });
    std::vector<std::string> ranked;
    ranked.reserve(weighted.size());
    for (const auto& [weight, segment] : weighted) ranked.push_back(*segment);
    return ranked;
}

}  // namespace mooncake
